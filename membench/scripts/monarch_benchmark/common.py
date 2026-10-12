from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from pathlib import Path


# Monarch factorises every Linear except the LM head into four blocks, so at the
# same geometry it carries 43-64 % of the dense parameters.
MODEL_SPECS = (
    {
        "name": "257m",
        "label": "257M",
        "n_layer": 12,
        "n_embd": 1024,
        "n_head": 8,
        "intermediate_size": 0,
        "dense_params_expected": 257_188_864,
        "monarch_params_expected": 163_603_456,
    },
    {
        "name": "834m",
        "label": "834M",
        "n_layer": 24,
        "n_embd": 1536,
        "n_head": 12,
        "intermediate_size": 0,
        "dense_params_expected": 834_086_400,
        "monarch_params_expected": 423_568_896,
    },
    {
        "name": "1p4b",
        "label": "1.44B",
        "n_layer": 24,
        "n_embd": 2048,
        "n_head": 16,
        "intermediate_size": 0,
        "dense_params_expected": 1_439_270_912,
        "monarch_params_expected": 690_587_648,
    },
    {
        "name": "3p5b",
        "label": "3.48B",
        "n_layer": 28,
        "n_embd": 3072,
        "n_head": 24,
        "intermediate_size": 0,
        "dense_params_expected": 3_480_136_704,
        "monarch_params_expected": 1_564_388_352,
    },
    {
        "name": "6p9b",
        "label": "6.89B",
        "n_layer": 32,
        "n_embd": 4096,
        "n_head": 32,
        "intermediate_size": 11008,
        "dense_params_expected": 6_888_361_984,
        "monarch_params_expected": 2_970_882_048,
    },
)

VARIANTS = (
    {"name": "monarch_muon", "label": "Monarch-Muon", "family": "monarch"},
    {"name": "dense_adamw", "label": "Dense AdamW", "family": "dense"},
    {"name": "dense_muon", "label": "Dense Muon", "family": "dense"},
    {"name": "galore", "label": "GaLore", "family": "memory_efficient"},
    {"name": "frugal", "label": "FRUGAL", "family": "memory_efficient"},
    {"name": "apollo", "label": "APOLLO", "family": "memory_efficient"},
    {"name": "fira", "label": "Fira", "family": "memory_efficient"},
)

DEFAULT_SEQUENCE_LENGTH = 1024
DEFAULT_MICROBATCH = 4
DEFAULT_ACCUMULATION_STEPS = 4

HARNESS_REVISION = 2


def model_spec(name: str) -> dict:
    for spec in MODEL_SPECS:
        if spec["name"] == name:
            return dict(spec)
    raise KeyError(f"unknown model size {name!r}")


def model_geometry(spec: dict, variant_name: str) -> dict:
    """The geometry a variant is built at, and the parameter count it must hit."""
    return {
        "n_layer": spec["n_layer"],
        "n_embd": spec["n_embd"],
        "n_head": spec["n_head"],
        "intermediate_size": spec["intermediate_size"],
        "params_expected": (
            spec["monarch_params_expected"]
            if variant_name == "monarch_muon"
            else spec["dense_params_expected"]
        ),
    }


def variant_spec(name: str) -> dict:
    for spec in VARIANTS:
        if spec["name"] == name:
            return dict(spec)
    raise KeyError(f"unknown benchmark variant {name!r}")


def requested_controls(args, microbatch: int, accumulation: int) -> dict:
    """The control surface a result must reproduce to be reused instead of rerun."""
    return {
        "harness_revision": HARNESS_REVISION,
        "storage_dtype": "bfloat16",
        "autocast_dtype": "bfloat16",
        "optimizer_moment_dtype": "bfloat16",
        "sequence_length": args.sequence_length,
        "microbatch": microbatch,
        "accumulation_steps": accumulation,
        "tokens_per_step": args.sequence_length * microbatch * accumulation,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "compile_model": False,
        "density": args.density,
        "update_proj_gap": args.update_proj_gap,
        "resample_steps": args.resample_steps,
        "lr": args.lr,
        "momentum": args.momentum,
        "betas": [args.beta1, args.beta2],
        "weight_decay": args.weight_decay,
        "eps": args.eps,
        "seed": args.seed,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "std": statistics.pstdev(numeric),
        "min": min(numeric),
        "p10": percentile(numeric, 0.1),
        "p90": percentile(numeric, 0.9),
        "max": max(numeric),
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def result_is_complete(payload: dict) -> bool:
    if payload.get("status") != "complete":
        return False
    samples = payload.get("samples", [])
    requested = payload.get("benchmark", {}).get("measured_steps")
    return bool(samples) and requested == len(samples)


def result_is_recorded(payload: dict) -> bool:
    """A point is finished once it either measured cleanly or ran out of memory."""
    return payload.get("status") == "oom" or result_is_complete(payload)


def result_matches_request(
    payload: dict,
    *,
    model_name: str,
    variant_name: str,
    controls: dict,
) -> bool:
    if not result_is_recorded(payload):
        return False
    if payload.get("status") == "oom":
        # An OOM payload carries only the identity and the controls it was asked for.
        return (
            payload.get("model_size") == model_name
            and payload.get("variant") == variant_name
            and payload.get("requested_controls") == controls
        )
    if payload.get("model", {}).get("name") != model_name:
        return False
    if payload.get("variant", {}).get("name") != variant_name:
        return False
    benchmark = payload.get("benchmark", {})
    return all(benchmark.get(key) == value for key, value in controls.items())
