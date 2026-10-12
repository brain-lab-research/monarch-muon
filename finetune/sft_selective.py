"""Selective dense-to-Monarch SFT of Qwen2.5-7B: the protocol of train_sft.py -- same data split, seed, order, schedule,
optimizers (SFTOptimizer), loss normalization and NLL evaluation -- but only the SELECTED projections are replaced by
2-block Monarch factors (closed-form projection of torch_projection.py), and every trial also reports its test NLL
(the LR is still selected by validation NLL only). No checkpoint is saved: the result JSON is the deliverable.

    --select mlp@2-25 self_attn.o_proj@14-27      rules <what>@<first>-<last>; what = all | mlp | self_attn |
                                                    <parent>.<proj> | <proj>
    --trials muon:0.001 muon:0.0003 adamw:3e-05    optimizer:lr, run one after another
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from sft_data import read_split
from sft_model import TARGETS, load_base
from torch_projection import MonarchLinear, materialize, project_dense
from train_sft import evaluate, initialize, train_trial


def parse_select(specs, n_layers):
    names = set()
    for spec in specs:
        what, _, layers = spec.partition("@")
        first, _, last = (layers or f"0-{n_layers - 1}").partition("-")
        for layer in range(int(first), int(last or first) + 1):
            for parent, projections in TARGETS.items():
                for name in projections:
                    if what in ("all", parent, f"{parent}.{name}", name):
                        names.add(f"model.layers.{layer}.{parent}.{name}")
    return names


@torch.no_grad()
def replace_selected(model, blocks, selected):
    records = []
    for index, layer in enumerate(model.model.layers):
        for parent_name, names in TARGETS.items():
            parent = getattr(layer, parent_name)
            for name in names:
                full = f"model.layers.{index}.{parent_name}.{name}"
                if full not in selected:
                    continue
                dense = getattr(parent, name)
                first, second = project_dense(dense.weight, blocks)
                error = ((materialize(first, second) - dense.weight.float()).norm() / dense.weight.float().norm()).item()
                setattr(parent, name, MonarchLinear(first.to(dense.weight.dtype), second.to(dense.weight.dtype), dense.bias))
                records.append({"name": full, "relative_projection_error": error})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sft_config.json")
    parser.add_argument("--model-dir", required=True, help="Local Qwen2.5-7B directory.")
    parser.add_argument("--data", required=True, help="Directory with train / validation / test .jsonl (sft_data.py).")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--select", nargs="+", required=True)
    parser.add_argument("--trials", nargs="+", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    rank, world, device = initialize()
    if world != config["gpus"]:
        raise RuntimeError(f"expected {config['gpus']} ranks, got {world}")
    data, out = Path(args.data), Path(args.out)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    rows, validation, test = (read_split(data / f"{s}.jsonl") for s in ("train", "validation", "test"))
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    for trial in args.trials:
        algorithm, lr = trial.split(":")
        lr = float(lr)
        name = f"{args.label}-{algorithm}-lr{lr:g}-g{world}"
        started = time.monotonic()
        model = load_base(args.model_dir, device)
        selected = parse_select(args.select, len(model.model.layers))
        records = replace_selected(model, config["blocks"], selected)
        torch.cuda.empty_cache()
        result = train_trial(model, rows, validation, config, tokenizer, algorithm, lr, rank, world, device, out, name,
                             False)
        result["test"] = evaluate(model, test, config, tokenizer, rank, world, device)
        result.update(select=args.select, projected=len(records), projections=records,
                      wall_seconds=time.monotonic() - started)
        if rank == 0:
            (out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
            print("SFT_SELECTIVE_RESULT " + json.dumps({k: result[k] for k in ("run", "baseline", "validation", "test",
                                                                              "projected", "parameters", "training_seconds")}),
                  flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
