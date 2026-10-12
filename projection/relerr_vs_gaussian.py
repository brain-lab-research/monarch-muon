#!/usr/bin/env python
"""Relative Frobenius error of the closed-form Monarch projection (Dao et al., 2022; MonarchLinear layout, --nblocks
blocks) for every linear projection of a local HF checkpoint, divided by the error of an i.i.d. Gaussian matrix of the
same shape (mean over --gauss-seeds seeds).

Matrices: model.layers.<l>.<path>.<x>_proj[_with_mqa].weight with
    self_attn                  -> group "attn"            (Qwen2.5: q, k, v, o; Moonlight/MLA: q, kv_a, kv_b, o)
    mlp                        -> group "dense FFN"       (gate, up, down; Moonlight: layer 0 only)
    mlp.shared_experts         -> group "shared experts"
    mlp.experts.<e>            -> group "routed experts"  pooled per (layer, type) over all experts:
                                  sqrt(sum_e ||W_e - M_e||_F^2 / sum_e ||W_e||_F^2), as run_joint_blocks.py reports a
                                  routed unit; the Gaussian reference is that of one expert's shape.
Routers, norms, embeddings and the LM head are not touched.

Usage:
    python relerr_vs_gaussian.py --model-dir /data/Moonlight-16B-A3B --model-name muon --nblocks 4 --out relerr_muon.tsv
    python relerr_vs_gaussian.py --model-dir /data/Qwen2.5-7B --model-name qwen --nblocks 2 --out relerr_qwen.tsv

Writes a TSV with columns model, layer, group, mtype, rel_err, gauss_rel_err, ratio (= rel_err / gauss_rel_err).
"""
import argparse
import csv
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

PATTERN = re.compile(r"^model\.layers\.(?P<layer>\d+)\.(?P<path>.+)\.(?P<leaf>\w+_proj(?:_with_mqa)?)\.weight$")


def lost_and_total(w, nb):
    # Same slices and ranks as the MonarchLinear(nblocks=nb) closed-form fit (monarch_blocks.fit_svd):
    # slice (ob, ib) = W[ob::nb, ib*p:(ib+1)*p], kept rank = min(p, s) / nb; the error is the discarded singular mass.
    out_f, in_f = w.shape
    p, s = in_f // nb, out_f // nb
    assert out_f % nb == 0 and in_f % nb == 0 and min(p, s) % nb == 0, f"nblocks={nb} does not divide {tuple(w.shape)}"
    rank = min(p, s) // nb
    lost = torch.zeros((), dtype=torch.float64, device=w.device)
    for ib in range(nb):
        for ob in range(nb):
            sv = torch.linalg.svdvals(w[ob::nb, ib * p:(ib + 1) * p])
            lost += (sv[rank:].double() ** 2).sum()
    return lost.item(), (w.double() ** 2).sum().item()


def rel_err(w, nb):
    lost, total = lost_and_total(w, nb)
    return (lost / total) ** 0.5


def self_check():
    """The singular-value formula against a dense reconstruction on a small random matrix."""
    torch.manual_seed(0)
    w = torch.randn(48, 32, dtype=torch.float64)
    nb = 2; p, s = 16, 24; q = min(p, s); r = q // nb
    recon = torch.zeros_like(w)
    for ib in range(nb):
        for ob in range(nb):
            blk = w[ob::nb, ib * p:(ib + 1) * p]
            u, sv, vh = torch.linalg.svd(blk, full_matrices=False)
            recon[ob::nb, ib * p:(ib + 1) * p] = (u[:, :r] * sv[:r]) @ vh[:r]
    assert abs(rel_err(w, nb) - ((w - recon).norm() / w.norm()).item()) < 1e-10
    print("SELFCHECK ok", flush=True)


def classify(path):
    if path == "self_attn":
        return "attn"
    if path == "mlp":
        return "dense FFN"
    if path == "mlp.shared_experts":
        return "shared experts"
    if re.fullmatch(r"mlp\.experts\.\d+", path):
        return "routed experts"
    raise ValueError(f"unknown projection path {path!r}")


def weight_files(model_dir):
    """{tensor name: safetensors file} of a local HF checkpoint (sharded or not)."""
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        return {n: model_dir / f for n, f in json.loads(index.read_text())["weight_map"].items()}
    out = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as st:
            out.update({n: f for n in st.keys()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--model-name", required=True, help="Value of the `model` column (e.g. muon, adamw, qwen).")
    ap.add_argument("--nblocks", type=int, required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gauss-seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev, nb = args.device, args.nblocks
    self_check()

    files = weight_files(args.model_dir)
    targets = {}                                        # file -> [(name, layer, group, mtype)]
    for name, f in files.items():
        m = PATTERN.match(name)
        if m:
            mtype = m["leaf"].replace("_with_mqa", "").removesuffix("_proj")
            targets.setdefault(f, []).append((name, int(m["layer"]), classify(m["path"]), mtype))
    print("TARGETS", sum(len(v) for v in targets.values()), flush=True)

    acc, shapes = {}, {}                                # (layer, group, mtype) -> [lost, total]; -> shape
    for f in sorted(targets):
        with safe_open(str(f), framework="pt") as st:
            for name, layer, group, mtype in targets[f]:
                w = st.get_tensor(name).to(dev, torch.float32)
                key = (layer, group, mtype)
                lost, total = lost_and_total(w, nb)
                a = acc.setdefault(key, [0.0, 0.0])
                a[0] += lost
                a[1] += total
                assert shapes.setdefault(key, tuple(w.shape)) == tuple(w.shape), (name, w.shape)
        print("done", f.name, flush=True)

    gauss = {}
    for shape in sorted(set(shapes.values())):
        errs = []
        for seed in range(args.gauss_seeds):
            g = torch.Generator(device=dev).manual_seed(seed)
            errs.append(rel_err(torch.randn(*shape, generator=g, device=dev), nb))
        gauss[shape] = sum(errs) / len(errs)
        print("GAUSS", list(shape), round(gauss[shape], 4), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["model", "layer", "group", "mtype", "rel_err", "gauss_rel_err", "ratio"])
        for key in sorted(acc):
            layer, group, mtype = key
            err = (acc[key][0] / acc[key][1]) ** 0.5
            ref = gauss[shapes[key]]
            w.writerow([args.model_name, layer, group, mtype, f"{err:.6f}", f"{ref:.6f}", f"{err / ref:.6f}"])
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
