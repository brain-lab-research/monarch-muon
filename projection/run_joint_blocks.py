#!/usr/bin/env python
"""Monarch replacement of single blocks and of joint sets of blocks, measured against the original model.

Every physical matrix type of a layer (attn.q_proj, ffn.down_proj, the 64 routed experts' up_proj together, ...) is fitted
once with the closed-form MonarchLinear(nblocks=B) projection (monarch_blocks.fit_svd) and kept on the host as its dense
reconstruction in the model dtype. Blocks are unions of those units:
    attn    the layer's attention matrices          ffn     its FFN (dense, or shared + routed experts)
    routed  the routed experts only (MoE)
Every configuration is patched into the model, measured, and restored bit-exactly before the next one.

Plans (L layers; for Moonlight, L = 27)
    sweep   every FFN block alone (L<l>_ffn), then the joint sets `all` (every block of every layer) and ffn_2_<L-2>
    arms    FFN blocks from a depth on (ffn_<c>_<L-2>, c = round(0.7 L), round(0.5 L), round(0.3 L)) and the routed
            experts of two depth ranges (routed_2_<round(0.5 L)-1>, routed_<round(0.5 L)>_<L-2>)

Outputs
    configs.csv      one row per configuration: KL / NLL / top-1 / final-state error on the calibration evaluation
                     tokens, the sum of the single-block KLs of its members (kl_sum_singles), parameters saved
    mmlu.csv         MMLU 5-shot on the dense model and the joint sets (sweep: all; arms: every set)
    gsm8k.csv        GSM8K 4-shot (arms only) on the dense model, ffn_<round(0.7 L)>_<L-2> and ffn_<round(0.5 L)>_<L-2>
    matrices.csv     per-matrix error of the fit (rel_err, mse) and parameter fraction
    joint_sets.json  the members of every joint set
    run_info.json    settings, the dense model's metrics and the baseline check

Usage (tokens and reference cache from prepare_calibration.py):
    python run_joint_blocks.py --model-dir /data/Moonlight-16B-A3B --model-tag moonlight_muon \\
        --tokens-dir /data/calib --cache-dir /data/calib/refs/moonlight_muon_grouped_mm --accel grouped_mm \\
        --out-dir out/joint/moonlight_muon/sweep --plan sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from calibration_data import ReferenceCache, load_tokens
from matrix_patching import MatrixPatcher
from model_loading import load_model
from model_registry import build_registry, read_checkpoint_shapes, read_config
from moe_accel import effective_accel
from monarch_blocks import BlockShape, fit_svd
from run_downstream_eval import run_lm_eval
from sensitivity_metrics import ForwardEvaluator

BLOCK_OF_GROUP = {"attn": "attn", "ffn": "ffn", "moe_shared": "ffn", "moe_routed": "ffn"}


def append(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def fit_unit(model, unit, nblocks: int, fit_device):
    """Fit every matrix of the unit. Returns (assignments [(ref, dense in the model dtype)], per-matrix rows, error of
    the whole unit). The errors are those of the fp32 reconstruction, before the cast."""
    W = torch.stack([ref.read(model) for ref in unit.refs])
    model_dtype = W.dtype
    res = fit_svd(W.to(fit_device, torch.float32), nblocks)
    sh = BlockShape(unit.shape[0], unit.shape[1], nblocks)
    n = unit.shape[0] * unit.shape[1]
    assignments, rows = [], []
    sq_err = sq_norm = 0.0
    for i, (ref, eid) in enumerate(zip(unit.refs, unit.expert_ids)):
        assignments.append((ref, res.dense[i].detach().to(model_dtype)))            # what the patched model will hold
        rows.append(dict(method=res.method, mtype=unit.mtype, expert="" if eid is None else eid, param_name=ref.param_name,
                         n_out=unit.shape[0], n_in=unit.shape[1], params_dense=n, params_monarch=sh.n_params,
                         fraction=sh.fraction, mse=float(res.mse[i]), rel_err=float(res.rel_err[i]), w_rms=float(res.w_rms[i])))
        sq_err += float(res.mse[i]) * n
        sq_norm += float(res.w_rms[i]) ** 2 * n
    return assignments, rows, math.sqrt(sq_err / sq_norm) if sq_norm > 0 else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--tokens-dir", required=True)
    ap.add_argument("--cache-dir", required=True, help="The reference cache (prepare_calibration.py --stages reference).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--accel", default="none")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nblocks", type=int, default=4)
    ap.add_argument("--plan", choices=["sweep", "arms"], default="sweep")
    ap.add_argument("--eval-sequences", type=int, default=32)
    ap.add_argument("--mmlu-limit", type=int, default=100)
    ap.add_argument("--gsm8k-limit", type=int, default=400)
    ap.add_argument("--bench-batch-size", type=int, default=16)
    ap.add_argument("--gen-batch-size", type=int, default=16)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log = lambda *a: print(f"[{(time.time() - t_start) / 60:6.1f} min]", *a, flush=True)

    registry = build_registry(read_config(args.model_dir), read_checkpoint_shapes(args.model_dir))
    L = registry.n_layers
    units = [u for u in registry.units if u.group in BLOCK_OF_GROUP]
    if args.plan == "arms":
        if not any(u.group == "moe_routed" for u in units):
            raise SystemExit("--plan arms needs an MoE model (routed experts)")
        units = [u for u in units if u.group in ("moe_shared", "moe_routed")]
    log(f"{len(units)} matrix units over {L} layers")

    model = load_model(args.model_dir, dtype=args.dtype, device=args.device, accel=args.accel)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log("model loaded, accel =", effective_accel(model), gpu)
    dev = next(model.parameters()).device

    cache = ReferenceCache(args.cache_dir)
    _, eval_tok, _ = load_tokens(args.tokens_dir)
    # the batch size the reference was computed with: in bf16 a token's MoE output depends on the batch it runs in
    evaluator = ForwardEvaluator(model, eval_tok, cache, batch_size=int(cache.meta["batch_size"]), token_chunk=512)
    base = evaluator.baseline_check(4)
    log(f"baseline check: KL(orig || unpatched) = {base.kl:.2e}")
    dense = evaluator.evaluate(args.eval_sequences).as_row()
    log(f"dense: ppl {dense['ppl_orig']:.4f}")
    (out / "run_info.json").write_text(json.dumps(dict(vars(args), accel_effective=effective_accel(model), gpu=gpu,
                                                       torch=torch.__version__, baseline_kl=base.kl, dense=dense), indent=1))

    def bench(name: str, task: str, batch: int, limit: int) -> tuple[list[dict], float]:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        rows = run_lm_eval(model, tok, task, {"mmlu": 5, "gsm8k": 4}, SimpleNamespace(
            batch_size=batch, limit=limit, samples_path=out / "samples" / f"{name}_{task}_b{batch}.jsonl"))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return rows, time.time() - t0

    # 1. fit every matrix unit once; keep the dense reconstructions on the host
    store, saved, unit_of = {}, {}, {}
    for u in units:
        t0 = time.time()
        assignments, mrows, rel_err = fit_unit(model, u, args.nblocks, dev)
        key = (u.layer, u.mtype)
        store[key] = [(ref, W.cpu()) for ref, W in assignments]
        saved[key] = u.n_dense_params - BlockShape(u.shape[0], u.shape[1], args.nblocks).n_params * len(u.refs)
        unit_of[key] = u
        del assignments
        for m in mrows:
            append(out / "matrices.csv", dict(model_tag=args.model_tag, layer=u.layer, unit=u.mtype, **m))
        log(f"fit L{u.layer:02d} {u.mtype}: rel_err {rel_err:.4f} ({time.time() - t0:.0f}s)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # blocks (and the routed experts) as unions of matrix units
    for key, u in list(unit_of.items()):
        for kind in [BLOCK_OF_GROUP[u.group]] + (["routed"] if u.group == "moe_routed" else []):
            k = (u.layer, kind)
            store[k] = store.get(k, []) + store[key]
            saved[k] = saved.get(k, 0) + saved[key]

    single_kl = {}

    def run(name: str, kind: str, keys) -> dict:
        keys = sorted(set(keys))
        patcher = MatrixPatcher(model, save_on_cpu=True)
        try:
            for k in keys:
                patcher.apply(store[k])
            m = evaluator.evaluate(args.eval_sequences).as_row()
        finally:
            patcher.restore()
        row = dict(model_tag=args.model_tag, config=name, kind=kind, n_units=len(keys), params_saved=sum(saved[k] for k in keys),
                   kl_sum_singles=sum(single_kl.get(k, float("nan")) for k in keys),
                   units=" ".join(f"{l}:{t}" for l, t in keys), **m)
        append(out / "configs.csv", row)
        log(f"{kind} {name}: {len(keys)} units, KL {m['kl']:.4f} (sum of singles {row['kl_sum_singles']:.4f}), "
            f"NLL +{m['nll_delta']:.4f}, top1 {m['top1_agree']:.3f}")
        return row

    # 2. singles
    blocks = sorted({(l, BLOCK_OF_GROUP[u.group]) for (l, _), u in unit_of.items()})
    singles = [k for k in blocks if k[1] == "ffn"] if args.plan == "sweep" else []
    for k in singles:
        single_kl[k] = run(f"L{k[0]:02d}_{k[1]}", "single", [k])["kl"]

    # 3. joint sets
    Fb = lambda ls: {(l, "ffn") for l in ls}
    R = lambda ls: {(l, "routed") for l in ls}
    last = L - 1
    if args.plan == "sweep":
        joints = {"all": set(blocks), f"ffn_2_{last - 1}": Fb(range(2, last))}
        bench_sets = ["dense", "all"]
        gsm_sets = []
    else:
        deep, mid, shallow = round(0.7 * L), round(0.5 * L), round(0.3 * L)
        joints = {
            f"ffn_{deep}_{last - 1}": Fb(range(deep, last)),
            f"ffn_{mid}_{last - 1}": Fb(range(mid, last)),
            f"ffn_{shallow}_{last - 1}": Fb(range(shallow, last)),
            f"routed_2_{mid - 1}": R(range(2, mid)),
            f"routed_{mid}_{last - 1}": R(range(mid, last)),
        }
        bench_sets = ["dense"] + list(joints)
        gsm_sets = ["dense", f"ffn_{deep}_{last - 1}", f"ffn_{mid}_{last - 1}"]
    joints = {n: {k for k in ks if k in store} for n, ks in joints.items()}
    (out / "joint_sets.json").write_text(json.dumps({n: sorted(f"{l}:{t}" for l, t in s) for n, s in joints.items()}, indent=1))
    for name, keys in joints.items():
        if keys:
            run(name, "joint", keys)

    # 4. benchmarks on chosen sets
    for task, batch, limit, names in [("mmlu", args.bench_batch_size, args.mmlu_limit, bench_sets),
                                      ("gsm8k", args.gen_batch_size, args.gsm8k_limit, gsm_sets)]:
        for name in names:
            keys = sorted(joints.get(name, set()))
            if name != "dense" and not keys:
                continue
            patcher = MatrixPatcher(model, save_on_cpu=True)
            try:
                for k in keys:
                    patcher.apply(store[k])
                rows, sec = bench(name, task, batch, limit)
            finally:
                patcher.restore()
            for r in rows:
                append(out / f"{task}.csv", dict(model_tag=args.model_tag, config=name, n_units=len(keys), seconds=round(sec, 1),
                                                 **{k: r.get(k) for k in ("task", "metric", "filter", "value", "stderr", "n")}))
            head = {r["metric"] + "," + r["filter"]: round(r["value"], 4) for r in rows if r.get("task") == task}
            log(f"{task} {name}: {head} ({sec:.0f}s)")
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
