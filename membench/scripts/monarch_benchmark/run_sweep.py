#!/usr/bin/env python3
"""Run the complete large-model sweep on one GPU, over models and optimizers."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.monarch_benchmark.common import (
    DEFAULT_ACCUMULATION_STEPS,
    DEFAULT_MICROBATCH,
    DEFAULT_SEQUENCE_LENGTH,
    MODEL_SPECS,
    VARIANTS,
    atomic_write_json,
    requested_controls,
    result_is_recorded,
    result_matches_request,
)

ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] {message}", flush=True)


def result_path(output_dir: Path, model_name: str, variant_name: str, microbatch: int) -> Path:
    return output_dir / "runs" / f"{model_name}-{variant_name}-bs{microbatch}.json"


def load_complete(path: Path, model: dict, variant: dict, controls: dict) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return payload if result_matches_request(
        payload,
        model_name=model["name"],
        variant_name=variant["name"],
        controls=controls,
    ) else None


def worker_command(model: dict, variant: dict, microbatch: int, accumulation: int,
                   output: Path, args) -> list[str]:
    return [
        args.python,
        "-m",
        "scripts.monarch_benchmark.benchmark_train_step",
        "--variant", variant["name"],
        "--model-size", model["name"],
        "--sequence-length", str(args.sequence_length),
        "--microbatch", str(microbatch),
        "--accumulation-steps", str(accumulation),
        "--warmup-steps", str(args.warmup_steps),
        "--measured-steps", str(args.measured_steps),
        "--resample-steps", str(args.resample_steps),
        "--density", str(args.density),
        "--update-proj-gap", str(args.update_proj_gap),
        "--lr", str(args.lr),
        "--momentum", str(args.momentum),
        "--beta1", str(args.beta1),
        "--beta2", str(args.beta2),
        "--weight-decay", str(args.weight_decay),
        "--eps", str(args.eps),
        "--seed", str(args.seed),
        "--output", str(output),
    ]


def run_worker(model: dict, variant: dict, args) -> dict:
    microbatch = DEFAULT_MICROBATCH
    accumulation = DEFAULT_ACCUMULATION_STEPS
    controls = requested_controls(args, microbatch, accumulation)
    output = result_path(args.output_dir, model["name"], variant["name"], microbatch)
    existing = load_complete(output, model, variant, controls)
    if existing is not None and not args.rerun:
        log(f"Reusing {output}")
        return existing

    command = worker_command(model, variant, microbatch, accumulation, output, args)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    log(f"Running {model['label']} / {variant['label']} / microbatch {microbatch} x {accumulation}")
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if not output.is_file():
        raise RuntimeError(f"worker exited {completed.returncode} without producing {output}")
    payload = json.loads(output.read_text())
    if payload.get("status") == "oom":
        # An out-of-memory point is a result, not a failure.
        log(f"Out of memory for {model['label']} / {variant['label']}")
        return payload
    if completed.returncode != 0 or not result_is_recorded(payload):
        raise RuntimeError(
            f"worker failed for {model['label']} / {variant['label']}: "
            f"{payload.get('error', payload.get('status'))}"
        )
    return payload


def summary(results: list[dict], models, variants, args) -> dict:
    return {
        "status": "complete",
        "sweep": {
            "sequence_length": args.sequence_length,
            "microbatch": DEFAULT_MICROBATCH,
            "accumulation_steps": DEFAULT_ACCUMULATION_STEPS,
            "tokens_per_step": args.sequence_length * DEFAULT_MICROBATCH * DEFAULT_ACCUMULATION_STEPS,
            "models": [model["name"] for model in models],
            "variants": [variant["name"] for variant in variants],
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "resample_steps": args.resample_steps,
            "update_proj_gap": args.update_proj_gap,
            "density": args.density,
        },
        "results": results,
    }


def rotated(items: list[dict], offset: int) -> list[dict]:
    position = offset % len(items)
    return items[position:] + items[:position]


def selected(specs, names: str | None, label: str) -> list[dict]:
    if not names:
        return list(specs)
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    by_name = {spec["name"]: spec for spec in specs}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}")
    return [by_name[name] for name in wanted]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/monarch-muon-large"))
    parser.add_argument("--python", default=sys.executable, help="interpreter used for the workers")
    parser.add_argument("--models", default=None, help="comma-separated subset of model sizes")
    parser.add_argument("--variants", default=None, help="comma-separated subset of variants")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=12)
    parser.add_argument("--resample-steps", type=int, default=3)
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--update-proj-gap", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    models = selected(MODEL_SPECS, args.models, "model size")
    variants = selected(VARIANTS, args.variants, "variant")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for model_index, model in enumerate(models):
        for variant in rotated(variants, model_index + DEFAULT_MICROBATCH):
            results.append(run_worker(model, variant, args))
            # Snapshot after every point so an interrupted sweep still has a
            # readable matrix.
            atomic_write_json(args.output_dir / "results.json", summary(results, models, variants, args))
    log(f"Sweep complete: {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
