#!/usr/bin/env python3
"""Benchmark one end-to-end Llama training-step configuration on one GPU."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from models.llama import Llama  # noqa: E402
from models.monarch import (  # noqa: E402
    MonarchMuonOptimizer,
    apply_monarch,
    patch_monarch_linear,
)
from optim.memory_efficient.apollo import APOLLOAdamW  # noqa: E402
from optim.memory_efficient.fira import FiraAdamW, GaLoreAdamW  # noqa: E402
from optim.memory_efficient.frugal import BlockAdamW  # noqa: E402
from optim.sota_opt import dion  # noqa: E402
from scripts.monarch_benchmark.common import (  # noqa: E402
    DEFAULT_ACCUMULATION_STEPS,
    DEFAULT_MICROBATCH,
    DEFAULT_SEQUENCE_LENGTH,
    VARIANTS,
    atomic_write_json,
    model_geometry,
    model_spec,
    requested_controls,
    summarize,
    variant_spec,
)

OPTIMIZER_BACKENDS = {
    "monarch_muon": "MonarchMuonOptimizer",
    "dense_adamw": "torch.optim.AdamW",
    "dense_muon": "dion.Muon",
    "galore": "fira.GaLoreAdamW",
    "frugal": "frugal.BlockAdamW",
    "apollo": "apollo.APOLLOAdamW",
    "fira": "fira.FiraAdamW",
}


def make_config(geometry: dict, sequence_length: int) -> SimpleNamespace:
    return SimpleNamespace(
        model="llama",
        vocab_size=50304,
        sequence_length=sequence_length,
        dropout=0.0,
        n_layer=geometry["n_layer"],
        n_embd=geometry["n_embd"],
        n_head=geometry["n_head"],
        intermediate_size=geometry["intermediate_size"],
        multiple_of=256,
        rmsnorm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        init_std=0.02,
    )


def instantiate_model(config: SimpleNamespace, device: torch.device) -> Llama:
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(device):
            return Llama(config)
    finally:
        torch.set_default_dtype(previous_dtype)


def dense_parameter_groups(model: Llama) -> list[dict]:
    by_name = dict(model.named_parameters())
    groups = []
    for specification in model.get_parameter_group_specs():
        group = {key: value for key, value in specification.items() if key != "params"}
        group["params"] = [by_name[name] for name in specification["params"]]
        groups.append(group)
    return groups


def projection_rank(args, spec: dict) -> int:
    """rank = density x hidden size, as in the training optimizer factory."""
    return max(1, int(args.density * spec["n_embd"]))


def build_memory_efficient_optimizer(args, spec: dict, model: Llama):
    """Mirrors the group wiring the training code's optimizer factory applies,
    driven by the is_proj_params flag from model.get_parameter_group_specs()."""
    groups = dense_parameter_groups(model)
    adamw_kwargs = {
        "lr": args.lr,
        "betas": (args.beta1, args.beta2),
        "eps": args.eps,
        "weight_decay": args.weight_decay,
        "no_deprecation_warning": True,
    }

    if args.variant == "frugal":
        # FRUGAL holds moments only for the currently active blocks and updates
        # the rest with SignSGD, so it carries no projection matrices at all.
        return BlockAdamW(
            groups,
            update_gap=args.update_proj_gap,
            density=args.density,
            **adamw_kwargs,
        )

    for group in groups:
        if not group.get("is_proj_params", False):
            continue
        group["rank"] = projection_rank(args, spec)
        group["update_proj_gap"] = args.update_proj_gap
        if args.variant == "galore":
            group["alpha"] = 1.0
        elif args.variant == "fira":
            group["alpha"] = 1.0
            group["reset_statistics"] = True
        else:
            group["scale"] = 1.0

    if args.variant == "galore":
        return GaLoreAdamW(groups, **adamw_kwargs)
    if args.variant == "fira":
        return FiraAdamW(groups, **adamw_kwargs)
    return APOLLOAdamW(groups, **adamw_kwargs)


def build_model_and_optimizer(args, spec: dict, geometry: dict, device: torch.device):
    config = make_config(geometry, args.sequence_length)
    model = instantiate_model(config, device)
    model.train()

    if args.variant == "monarch_muon":
        apply_monarch(model, nblocks=4, verbose=False)
        patch_monarch_linear()
        muon_params = []
        adamw_params = []
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2 and not any(
                excluded in name for excluded in ("wte", "lm_head", "embed")
            ):
                muon_params.append(parameter)
            else:
                adamw_params.append(parameter)
        optimizer = MonarchMuonOptimizer(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=args.lr,
            momentum=args.momentum,
            adamw_betas=(args.beta1, args.beta2),
            adamw_weight_decay=args.weight_decay,
            adamw_eps=args.eps,
        )
    elif args.variant == "dense_adamw":
        groups = dense_parameter_groups(model)
        for group in groups:
            group.pop("is_proj_params", None)
        optimizer = torch.optim.AdamW(
            groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=args.eps,
            fused=True,
        )
    elif args.variant == "dense_muon":
        groups = dense_parameter_groups(model)
        for group in groups:
            if not group.get("is_proj_params", False):
                group["algorithm"] = "adamw"
        optimizer = dion.Muon(
            groups,
            lr=args.lr,
            mu=args.momentum,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            epsilon=args.eps,
        )
    elif variant_spec(args.variant)["family"] == "memory_efficient":
        optimizer = build_memory_efficient_optimizer(args, spec, model)
    else:
        raise ValueError(args.variant)

    return model, optimizer


def _attribute_values(value, seen: set[int]):
    """Projectors are plain objects that keep their matrices in attributes
    (GradientProjector.ortho_matrix), so walking only tensors and containers
    would count them as zero bytes."""
    attributes = getattr(value, "__dict__", None)
    if attributes is None or id(value) in seen:
        return ()
    seen.add(id(value))
    return attributes.values()


def tensor_bytes(value, seen: set[int] | None = None) -> int:
    if isinstance(value, torch.Tensor):
        local = value.to_local() if hasattr(value, "to_local") else value
        return local.numel() * local.element_size()
    seen = set() if seen is None else seen
    if isinstance(value, dict):
        return sum(tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item, seen) for item in value)
    return sum(tensor_bytes(item, seen) for item in _attribute_values(value, seen))


def tensor_dtypes(value, *, nonscalar_only: bool = False, seen: set[int] | None = None) -> set[str]:
    if isinstance(value, torch.Tensor):
        local = value.to_local() if hasattr(value, "to_local") else value
        if nonscalar_only and local.numel() <= 1:
            return set()
        return {str(local.dtype).removeprefix("torch.")}
    seen = set() if seen is None else seen
    if isinstance(value, dict):
        items = value.values()
    elif isinstance(value, (tuple, list)):
        items = value
    else:
        items = _attribute_values(value, seen)
    collections = [
        tensor_dtypes(item, nonscalar_only=nonscalar_only, seen=seen) for item in items
    ]
    return set().union(*collections) if collections else set()


def moment_state(optimizer) -> dict:
    """Optimizer state without the projectors, i.e. the moments whose storage
    dtype the benchmark controls."""
    return {
        parameter: {key: value for key, value in state.items() if key != "projector"}
        for parameter, state in optimizer.state.items()
    }


def projector_state(optimizer) -> list:
    return [state["projector"] for state in optimizer.state.values() if "projector" in state]


def force_projector_resample(optimizer) -> None:
    """Make every following step rebuild its projection. GaLore/Fira/APOLLO hold
    the gap on the projector and on the group; FRUGAL holds it on the group only."""
    for group in optimizer.param_groups:
        for key in ("update_proj_gap", "update_gap"):
            if key in group:
                group[key] = 1
    for projector in projector_state(optimizer):
        if hasattr(projector, "update_proj_gap"):
            projector.update_proj_gap = 1


def timed_step(model, optimizer, batches, stream: torch.cuda.Stream) -> dict:
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    forward_pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in batches
    ]
    backward_pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in batches
    ]
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)

    host_start = time.perf_counter_ns()
    losses = []
    with torch.cuda.stream(stream):
        total_start.record(stream)
        for index, (inputs, targets) in enumerate(batches):
            forward_pairs[index][0].record(stream)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(inputs, targets=targets)["loss"] / len(batches)
            forward_pairs[index][1].record(stream)
            backward_pairs[index][0].record(stream)
            loss.backward()
            backward_pairs[index][1].record(stream)
            losses.append(loss.detach())

        optimizer_start.record(stream)
        optimizer.step()
        optimizer_end.record(stream)
        optimizer.zero_grad(set_to_none=True)
        total_end.record(stream)

    torch.cuda.synchronize()
    host_total_ms = (time.perf_counter_ns() - host_start) / 1e6
    return {
        "host_total_ms": host_total_ms,
        "gpu_total_ms": total_start.elapsed_time(total_end),
        "forward_ms": sum(start.elapsed_time(end) for start, end in forward_pairs),
        "backward_ms": sum(start.elapsed_time(end) for start, end in backward_pairs),
        "optimizer_ms": optimizer_start.elapsed_time(optimizer_end),
        "loss": sum(float(loss.float().item()) for loss in losses),
    }


METRICS = (
    "host_total_ms",
    "gpu_total_ms",
    "forward_ms",
    "backward_ms",
    "optimizer_ms",
    "tokens_per_second",
)


def measure(model, optimizer, batches, stream, steps: int, tokens_per_step: int, label: str) -> list[dict]:
    samples = []
    for iteration in range(steps):
        sample = timed_step(model, optimizer, batches, stream)
        if not math.isfinite(sample["loss"]):
            raise RuntimeError(f"non-finite {label} loss at {iteration}: {sample['loss']}")
        sample["iteration"] = iteration
        sample["tokens_per_second"] = tokens_per_step / (sample["host_total_ms"] / 1000.0)
        samples.append(sample)
    return samples


def run(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    spec = model_spec(args.model_size)
    variant = variant_spec(args.variant)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gpu_uuid = f"GPU-{torch.cuda.get_device_properties(device).uuid}"

    geometry = model_geometry(spec, args.variant)
    model, optimizer = build_model_and_optimizer(args, spec, geometry, device)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != geometry["params_expected"]:
        raise RuntimeError(
            f"parameter-count mismatch: actual={actual_parameters}, "
            f"expected={geometry['params_expected']}"
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 17)
    batches = [
        (
            torch.randint(
                0,
                50304,
                (args.microbatch, args.sequence_length),
                generator=generator,
                device=device,
            ),
            torch.randint(
                0,
                50304,
                (args.microbatch, args.sequence_length),
                generator=generator,
                device=device,
            ),
        )
        for _ in range(args.accumulation_steps)
    ]
    tokens_per_step = args.microbatch * args.sequence_length * args.accumulation_steps
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    stream.synchronize()

    for _ in range(args.warmup_steps):
        sample = timed_step(model, optimizer, batches, stream)
        if not math.isfinite(sample["loss"]):
            raise RuntimeError(f"non-finite warmup loss: {sample['loss']}")

    torch.cuda.reset_peak_memory_stats(device)

    # The projections were built during warmup and update_proj_gap is larger
    # than this window, so these steps are the steady state between rebuilds.
    samples = measure(
        model, optimizer, batches, stream, args.measured_steps, tokens_per_step, "measured"
    )
    memory = {
        "model_bytes": sum(tensor_bytes(parameter) for parameter in model.parameters()),
        "gradient_bytes_nominal": sum(
            tensor_bytes(parameter) for parameter in model.parameters()
        ),
        "optimizer_state_bytes": tensor_bytes(optimizer.state),
        "optimizer_moment_bytes": tensor_bytes(moment_state(optimizer)),
        "optimizer_projector_bytes": tensor_bytes(projector_state(optimizer)),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }

    # Rebuilding a projection costs both time and a transient FP32 workspace
    # that the steady-state window never sees, so it gets its own window.
    resample_samples = []
    resample_memory = None
    if variant["family"] == "memory_efficient":
        force_projector_resample(optimizer)
        timed_step(model, optimizer, batches, stream)
        torch.cuda.reset_peak_memory_stats(device)
        resample_samples = measure(
            model, optimizer, batches, stream, args.resample_steps, tokens_per_step, "resample"
        )
        resample_memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }

    moments = moment_state(optimizer)
    projectors = projector_state(optimizer)
    optimizer_state_dtypes = sorted(tensor_dtypes(optimizer.state))
    optimizer_nonscalar_state_dtypes = sorted(
        tensor_dtypes(moments, nonscalar_only=True)
    )
    if optimizer_nonscalar_state_dtypes != ["bfloat16"]:
        raise RuntimeError(
            "unexpected nonscalar optimizer-state dtypes: "
            f"{optimizer_nonscalar_state_dtypes}"
        )

    memory.update(
        {
            "optimizer_state_dtypes": optimizer_state_dtypes,
            "optimizer_nonscalar_state_dtypes": optimizer_nonscalar_state_dtypes,
            "optimizer_projector_dtypes": sorted(tensor_dtypes(projectors)),
        }
    )
    summary = {metric: summarize([sample[metric] for sample in samples]) for metric in METRICS}
    resample_summary = (
        {metric: summarize([sample[metric] for sample in resample_samples]) for metric in METRICS}
        if resample_samples
        else None
    )
    result = {
        "status": "complete",
        "model": {
            **spec,
            "geometry": geometry,
            "dense_equivalent_parameters": spec["dense_params_expected"],
            "actual_parameters": actual_parameters,
        },
        "variant": variant,
        "benchmark": {
            **requested_controls(args, args.microbatch, args.accumulation_steps),
            "adamw_fused": args.variant == "dense_adamw",
            "resample_measured": bool(resample_samples),
            "optimizer_backend": OPTIMIZER_BACKENDS[args.variant],
            "projection_rank": (
                projection_rank(args, spec)
                if variant["family"] == "memory_efficient" and args.variant != "frugal"
                else None
            ),
            "newton_schulz_steps": (
                5 if args.variant in ("monarch_muon", "dense_muon") else None
            ),
            "cuda_timing": "events on a dedicated stream; device-wide sync at step end",
        },
        "gpu": {
            "uuid": gpu_uuid,
            "logical_device": str(device),
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "memory": memory,
        "resample_memory": resample_memory,
        "summary": summary,
        "resample_summary": resample_summary,
        "samples": samples,
        "resample_samples": resample_samples,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "pid": os.getpid(),
        },
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=[variant["name"] for variant in VARIANTS])
    parser.add_argument("--model-size", required=True, choices=("257m", "834m", "1p4b", "3p5b", "6p9b"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--microbatch", type=int, default=DEFAULT_MICROBATCH)
    parser.add_argument("--accumulation-steps", type=int, default=DEFAULT_ACCUMULATION_STEPS)
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
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.microbatch, args.sequence_length, args.accumulation_steps) <= 0:
        raise ValueError("microbatch, sequence length, and accumulation steps must be positive")
    if args.warmup_steps < 1 or args.measured_steps < 1 or args.resample_steps < 1:
        raise ValueError("warmup, measured and resample steps must all be positive")
    if args.update_proj_gap <= args.warmup_steps + args.measured_steps:
        raise ValueError(
            "--update-proj-gap must exceed warmup + measured steps, otherwise a "
            "projection rebuild lands inside the steady-state window"
        )
    if not 0.0 < args.density <= 1.0:
        raise ValueError("--density must be in (0, 1]")
    started = time.time()
    try:
        payload = run(args)
        payload["wall_started_unix"] = started
        payload["wall_finished_unix"] = time.time()
        atomic_write_json(args.output, payload)
        print(json.dumps({
            "status": payload["status"],
            "model": payload["model"]["name"],
            "variant": payload["variant"]["name"],
            "median_ms": payload["summary"]["host_total_ms"]["median"],
            "tokens_per_second": payload["summary"]["tokens_per_second"]["median"],
            "peak_gb": payload["memory"]["peak_allocated_bytes"] / 1e9,
        }, sort_keys=True))
    except BaseException as error:
        status = "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed"
        payload = {
            "status": status,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "model_size": args.model_size,
            "variant": args.variant,
            "requested_controls": requested_controls(args, args.microbatch, args.accumulation_steps),
            "wall_started_unix": started,
            "wall_finished_unix": time.time(),
        }
        atomic_write_json(args.output, payload)
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
