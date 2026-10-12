"""Faster MoE experts for the DeepSeek-V3-style models (Moonlight): one grouped GEMM instead of a Python loop.

The checkpoint's own MoE code (`moe_infer` of modeling_deepseek.py; `moe` of the native transformers 4.x class) sorts the
tokens by expert and then loops over the experts in Python -- 3 small matmuls per expert, 64 experts, a host
synchronisation per layer (`tokens_per_expert.cpu()`). `apply_moe_accel` replaces that loop:

    sort the (token, expert) pairs by expert -> offsets = cumsum(tokens per expert), computed on the device
    -> one grouped GEMM with the stacked gate+up weights (E, 2*I, H) -> act(gate) * up
    -> one grouped GEMM with the stacked down weights (E, H, I) -> un-sort, weight by the router, sum over top-k

using `torch._grouped_mm` (torch >= 2.9 on SM80+/A100, torch 2.8 on Hopper only, bf16), and a plain per-expert loop
elsewhere (CPU, other dtypes), so that the same code path is testable anywhere. Only SM90 / SM100 get a fused (CUTLASS)
grouped GEMM: on SM80 torch 2.9 runs `_grouped_mm_fallback` -- a device-to-host copy of the offsets and one mm per expert,
twice per MoE layer. That is still one sync per call instead of the remote loop's per-expert work, but not a single kernel.

The weights stay patchable. Every expert's `gate_proj` / `up_proj` / `down_proj` weight becomes a VIEW into the stacked
tensor: MatrixPatcher (and anything else that writes in place) changes what the grouped GEMM reads, and restoring works
unchanged. Nothing is copied per forward. (Do not move / re-cast the model afterwards -- `.to()` gives every parameter its
own storage again and the views stop feeding the grouped GEMM; load_model applies this after placing the model.)

Numerics: a different GEMM kernel rounds differently in bf16, and bf16 MoE outputs are sensitive (near-tied routing
decisions can flip). So the acceleration is part of the identity of a measurement: the reference cache is built with the
same setting (prepare_calibration.py --accel, recorded in its meta.json), and prepare_calibration.py refuses to reuse a
cache built with another one. At load time a self-check runs on a small random input: for a few MoE layers the input they
really receive goes through the original loop, the grouped GEMM and a float32 computation, and the kernel must be about
as close to float32 as the loop is; if it is not, or it raises, the original code path is kept (load_model then stops,
since the run was asked for the other numerics) and the numbers are logged. The whole-model difference of the final
hidden states is logged but decides nothing: through ~27 bf16 layers it is rounding noise and flipped routing decisions
(4-5% was seen on the real checkpoints between two paths that are equal in exact arithmetic).
"""
from __future__ import annotations

import functools
import os
import sys
import types

import torch
import torch.nn.functional as F

ACCEL_CHOICES = ("none", "grouped_mm")


def add_accel_argument(parser) -> None:
    parser.add_argument(
        "--accel", choices=ACCEL_CHOICES, default=os.environ.get("MONARCH_ACCEL", "none"),
        help="grouped_mm: MoE experts as one grouped GEMM instead of a Python loop over the experts (moe_accel.py). "
             "Changes bf16 rounding, so the reference cache must be built with the same value (it is recorded in it; "
             "default: $MONARCH_ACCEL or none).")


def effective_accel(model) -> str:
    """What the loaded model actually runs: 'grouped_mm' only if it was requested AND passed the load-time check."""
    return "grouped_mm" if getattr(model, "moe_accel_report", {}).get("active") else "none"


def accel_note(model) -> str:
    rep = getattr(model, "moe_accel_report", None)
    return f" (grouped_mm was requested but is OFF: {rep.get('reason')})" if rep and not rep.get("active") else ""


def check_cache_accel(cache_meta: dict, accel: str, what: str = "the reference cache", note: str = "") -> None:
    """Refuse to compare a model run with `accel` against a cache built with another setting (caches written before
    this setting existed count as 'none')."""
    built = cache_meta.get("accel", "none")
    if built != accel:
        raise SystemExit(
            f"{what} was built with accel={built}, this run uses accel={accel}{note}: the two numerics differ (bf16 rounding of "
            f"the MoE experts), so KL against it would mix them. Rebuild it with the same setting "
            f"(prepare_calibration.py --accel {accel} --force-reference, or another --cache-dir), "
            f"or run with --accel {built}.")


# ---------------------------------------------------------------------------
# Grouped GEMM
# ---------------------------------------------------------------------------


def _torch_version() -> tuple[int, int]:
    major, minor = torch.__version__.split("+")[0].split(".")[:2]
    return int(major), int(minor)


@functools.lru_cache(maxsize=None)
def kernel_available(device: torch.device, dtype: torch.dtype) -> bool:
    """torch._grouped_mm usable here? bf16 on CUDA: SM80+ with torch >= 2.9, Hopper only before (as transformers checks)."""
    if not hasattr(torch, "_grouped_mm") or device.type != "cuda" or dtype != torch.bfloat16:
        return False
    cap = torch.cuda.get_device_capability(device)
    return cap >= (8, 0) if _torch_version() >= (2, 9) else cap[0] == 9


KERNEL_FAILURES: list[str] = []       # first line of every error torch._grouped_mm raised; after the first, the loop is used


def grouped_mm_loop(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Reference / fallback: per-expert F.linear over the groups given by the cumulative offsets. w: (E, N, K)."""
    out = x.new_empty(x.shape[0], w.shape[1])
    start = 0
    for e, end in enumerate(offs.tolist()):
        if end > start:
            out[start:end] = F.linear(x[start:end], w[e])
        start = end
    return out


def grouped_mm(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """x (S, K) sorted by expert, w (E, N, K) in nn.Linear layout, offs (E,) int32 cumulative counts -> (S, N).
    A kernel error (e.g. a shape / alignment it does not support) is reported once and the per-expert loop takes over for
    the rest of the process: a run of hours must not die on it (the loop differs from the kernel by bf16 rounding only)."""
    if kernel_available(x.device, w.dtype) and not KERNEL_FAILURES:
        try:
            return torch._grouped_mm(x.to(w.dtype), w.transpose(-2, -1), offs=offs)
        except RuntimeError as exc:
            KERNEL_FAILURES.append(str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__)
            print(f"accel: torch._grouped_mm failed ({KERNEL_FAILURES[-1]}); using the per-expert loop from now on "
                  f"(x {tuple(x.shape)}, w {tuple(w.shape)})", file=sys.stderr, flush=True)
    return grouped_mm_loop(x, w, offs)


def _offsets(flat_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """cumsum of the tokens per expert as int32, without a host synchronisation (histc, like transformers)."""
    src = flat_ids.float() if flat_ids.device.type in ("cpu", "mps") else flat_ids.int()
    counts = torch.histc(src, bins=num_experts, min=0, max=num_experts - 1)
    return torch.cumsum(counts, dim=0, dtype=torch.int32)


@torch.no_grad()
def grouped_moe(module, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
    """Replacement of the experts loop: x (T, H), topk_ids / topk_weight (T, K) -> (T, H) (what moe_infer returns)."""
    st = module._grouped_moe
    if not st.on:
        if st.capture is not None:                                 # the self-check records what this layer is fed
            st.capture.append((x.clone(), topk_ids.clone(), topk_weight.clone()))
        return st.orig(x, topk_ids, topk_weight)
    t, k = topk_ids.shape
    flat = topk_ids.reshape(-1)
    order = torch.argsort(flat)                                    # (token, expert) pairs grouped by expert
    sorted_x = x[order // k]
    offs = _offsets(flat, st.num_experts)
    gate, up = grouped_mm(sorted_x, st.gate_up, offs).chunk(2, dim=-1)
    out = grouped_mm(st.act(gate) * up, st.down, offs)
    unsorted = torch.empty_like(out)
    unsorted[order] = out
    return unsorted.view(t, k, -1).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(-1)).sum(dim=1).type(unsorted.dtype)


# ---------------------------------------------------------------------------
# Restructuring the model
# ---------------------------------------------------------------------------


class _State:
    def __init__(self, orig, gate_up, down, act, num_experts):
        self.orig, self.gate_up, self.down, self.act, self.num_experts = orig, gate_up, down, act, num_experts
        self.on = True
        self.capture: list | None = None


def moe_modules(model) -> list[tuple[str, torch.nn.Module, str]]:
    """(name, module, method) of every MoE block whose experts loop can be replaced: a module with a `.experts`
    ModuleList of gated MLPs (gate_proj / up_proj / down_proj, no bias) and a `moe_infer` (remote code) or `moe` (native
    transformers 4.x) method taking (x, topk_ids, topk_weight)."""
    found = []
    for name, m in model.named_modules():
        experts = getattr(m, "experts", None)
        method = "moe_infer" if hasattr(m, "moe_infer") else "moe" if hasattr(m, "moe") else None
        if method is None or not isinstance(experts, torch.nn.ModuleList) or len(experts) == 0:
            continue
        if getattr(m, "ep_size", 1) != 1:
            continue                                               # expert parallelism: not handled
        e0 = experts[0]
        if not all(isinstance(getattr(e0, n, None), torch.nn.Linear) and getattr(e0, n).bias is None
                   for n in ("gate_proj", "up_proj", "down_proj")):
            continue
        found.append((name, m, method))
    return found


@torch.no_grad()
def _stack_experts(module) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack the experts' weights and turn every expert's weight into a view of the stack. Returns (gate_up, down)."""
    experts = module.experts
    e = len(experts)
    g0, d0 = experts[0].gate_proj.weight, experts[0].down_proj.weight
    inter, hidden = g0.shape
    gate_up = torch.empty(e, 2 * inter, hidden, dtype=g0.dtype, device=g0.device)
    down = torch.empty(e, *d0.shape, dtype=d0.dtype, device=d0.device)
    for i, ex in enumerate(experts):
        gate_up[i, :inter].copy_(ex.gate_proj.weight)
        gate_up[i, inter:].copy_(ex.up_proj.weight)
        down[i].copy_(ex.down_proj.weight)
    for i, ex in enumerate(experts):                               # views: in-place writes reach the stacked tensors
        ex.gate_proj.weight = torch.nn.Parameter(gate_up[i, :inter], requires_grad=False)
        ex.up_proj.weight = torch.nn.Parameter(gate_up[i, inter:], requires_grad=False)
        ex.down_proj.weight = torch.nn.Parameter(down[i], requires_grad=False)
    return gate_up, down


def _set_enabled(modules, on: bool) -> None:
    for _, m, _ in modules:
        m._grouped_moe.on = on


MODEL_DIFF_CAP = 0.5        # final hidden states of the whole model: only a gross error (a wrong kernel) is above this
LAYERS_CHECKED = 4          # MoE layers compared one by one, evenly spread over the depth


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


@torch.no_grad()
def _experts_fp32(st: _State, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
    """The experts' output in float32 from the same (stacked) weights: the yardstick both bf16 paths are measured against."""
    inter = st.down.shape[2]
    x32, out = x.float(), torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in range(st.num_experts):
        tok, slot = (topk_ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        gate, up = F.linear(x32[tok], st.gate_up[e].float()).split(inter, dim=-1)
        y = F.linear(st.act(gate) * up, st.down[e].float())
        out.index_add_(0, tok, y * topk_weight[tok, slot].float().unsqueeze(-1))
    return out


@torch.no_grad()
def _selfcheck(model, modules, tol: float, floor: float, tokens: int = 256, seed: int = 0) -> tuple[float, list[dict]]:
    """Returns (relative L2 difference of the model's final hidden states with the accelerated and the original experts,
    per-layer checks).

    The model-level number is only informational (and a gross-error cap): through ~27 layers of bf16 it is dominated by
    rounding noise and routing decisions that flip, the same size as the noise of running a batch differently -- a
    4-5% difference there is not evidence against the kernel. The decision rests on the per-layer checks: for a few
    MoE layers the input actually reaching them (captured from the original path) goes through (a) the original loop,
    (b) the grouped GEMM, (c) a float32 computation. Both bf16 paths round the same way (every projection output is
    rounded), so the kernel is as good as the loop when its error against (c) is about the loop's:
    err_grouped <= 2 * err_loop + floor, and it must stay within `tol` of the loop's output."""
    from model_loading import final_hidden, input_device

    vocab = model.get_input_embeddings().weight.shape[0]
    ids = torch.randint(0, vocab, (1, tokens), generator=torch.Generator().manual_seed(seed)).to(input_device(model))
    n = len(modules)
    k = min(LAYERS_CHECKED, n)
    picks = sorted({round(i * (n - 1) / max(k - 1, 1)) for i in range(k)})
    _set_enabled(modules, False)
    for i in picks:
        modules[i][1]._grouped_moe.capture = []
    ref = final_hidden(model, ids).float()
    _set_enabled(modules, True)
    got = final_hidden(model, ids).float()
    checks = []
    for i in picks:
        name, m, _ = modules[i]
        st = m._grouped_moe
        x, topk_ids, topk_weight = st.capture[0]
        st.capture = None
        st.on = False
        loop = st.orig(x, topk_ids, topk_weight)
        st.on = True
        acc = grouped_moe(m, x, topk_ids, topk_weight)
        truth = _experts_fp32(st, x, topk_ids, topk_weight)
        c = dict(layer=name, err_loop=_rel(loop, truth), err_grouped=_rel(acc, truth), grouped_vs_loop=_rel(acc, loop))
        c["ok"] = bool(c["err_grouped"] <= 2 * c["err_loop"] + floor and c["grouped_vs_loop"] <= tol)    # (False for NaN)
        checks.append(c)
    return _rel(got, ref), checks


def _describe(checks: list[dict]) -> str:
    return "; ".join(f"{c['layer'] or 'moe'}: err(grouped) {c['err_grouped']:.2e} vs err(loop) {c['err_loop']:.2e}, "
                     f"grouped-vs-loop {c['grouped_vs_loop']:.2e}" for c in checks)


def apply_moe_accel(model, *, tol: float | None = None, check: bool = True, log=print) -> dict:
    """Replace the experts loop of every MoE block by grouped GEMMs (see the module docstring). Returns a report
    {"requested", "active", "layers", "kernel", "reason", "selfcheck_rel_diff", "layer_checks"}; the original path is
    kept when the check fails (see `_selfcheck` for what is checked)."""
    modules = moe_modules(model)
    report = dict(requested="grouped_mm", active=False, layers=len(modules), kernel=False, reason="", selfcheck_rel_diff=None,
                  layer_checks=[])
    if not modules:
        report["reason"] = "no MoE block with a replaceable experts loop found"
        log(f"accel: grouped_mm not applied: {report['reason']}")
        return report
    sample = modules[0][1].experts[0].gate_proj.weight
    report["kernel"] = kernel_available(sample.device, sample.dtype)
    for _, m, method in modules:
        if hasattr(m, "_grouped_moe"):                             # applied before: keep the existing stacks
            m._grouped_moe.on = True
            continue
        act = getattr(m.experts[0], "act_fn", None) or F.silu
        orig = getattr(m, method)
        gate_up, down = _stack_experts(m)
        m._grouped_moe = _State(orig, gate_up, down, act, len(m.experts))
        setattr(m, method, types.MethodType(grouped_moe, m))
    if tol is None:                                                # float32: accumulation order only; bf16/fp16: rounding too
        tol = 1e-4 if sample.dtype == torch.float32 else 3e-2
    floor = tol / 10
    diff, checks = 0.0, []
    try:
        if check:
            diff, checks = _selfcheck(model, modules, tol, floor)
    except Exception as exc:                                       # kernel missing / failing on this GPU
        _set_enabled(modules, False)
        report["reason"] = f"the accelerated forward failed ({type(exc).__name__}: {str(exc).splitlines()[0][:160]})"
        log(f"accel: grouped_mm OFF, keeping the original experts loop: {report['reason']}")
        return report
    report["selfcheck_rel_diff"], report["layer_checks"] = diff, checks
    bad = [c for c in checks if not c["ok"]]
    if bad or not diff <= MODEL_DIFF_CAP:
        _set_enabled(modules, False)
        report["reason"] = (f"self-check: {len(bad)} of {len(checks)} MoE layers differ from the original loop by more than the "
                            f"loop's own bf16 error ({_describe(bad)})" if bad else
                            f"self-check: final hidden states differ by {diff:.3g} > {MODEL_DIFF_CAP}")
        log(f"accel: grouped_mm OFF, keeping the original experts loop: {report['reason']}")
        return report
    report["active"] = True
    report["kernel"] = report["kernel"] and not KERNEL_FAILURES
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    how = (("torch._grouped_mm kernel" if cap[0] in (9, 10) else
            f"torch._grouped_mm, which on SM{cap[0]}{cap[1]} is torch's per-expert fallback, not a fused kernel")
           if report["kernel"] else
           f"per-expert loop over the sorted layout (the kernel failed: {KERNEL_FAILURES[0]})" if KERNEL_FAILURES else
           "per-expert loop over the sorted layout (no torch._grouped_mm for this device / dtype)")
    log(f"accel: grouped_mm ON for {len(modules)} MoE layers ({len(modules[0][1].experts)} experts, gate+up fused; {how})")
    if checks:
        log(f"accel: self-check per layer against float32 -- {_describe(checks)}")
        log(f"accel: final hidden states, whole model, accelerated vs original: relative difference {diff:.2e} "
            "(informational: bf16 noise through all layers, not a pass/fail number)")
    return report
