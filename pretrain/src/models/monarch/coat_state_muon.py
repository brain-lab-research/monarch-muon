"""Monarch-Muon with COAT FP8 optimizer states (BF16 GEMMs are untouched).

Every state tensor -- Muon's momentum and the embedded AdamW's two moments -- is stored the way
COAT stores AdamW moments: E4M3 in groups of 128 with dynamic range expansion. The codec is the
first-order (signed) and second-order (unsigned) half of `triton_fp8_adamw_expand_kernel`, split
into a dequantize and a quantize kernel because Muon's update is not an elementwise kernel. A step
dequantizes the states to FP32, runs the unchanged Monarch-Muon update and quantizes them back.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .monarch_muon import MonarchMuonOptimizer

GROUP_SIZE = 128
FP8 = {"E4M3": torch.float8_e4m3fn, "E5M2": torch.float8_e5m2}
# state name -> signed (the embedded AdamW's second moment is the only unsigned state)
STATES = {"momentum_buffer": True, "exp_avg": True, "exp_avg_sq": False}


@triton.jit
def _dequantize_kernel(data_ptr, out_ptr, scale_ptr, expand_ptr, sqrt_minmax_ptr, numel,
                       SIGNED: tl.constexpr, GROUP: tl.constexpr):
    quant_eps = 1e-30
    pid = tl.program_id(0)
    offs = pid * GROUP + tl.arange(0, GROUP)
    mask = offs < numel
    s = tl.load(scale_ptr + pid).to(tl.float32)
    raw = tl.load(data_ptr + offs, mask=mask, other=0.0)
    expand = tl.maximum(tl.load(expand_ptr + pid).to(tl.float32), quant_eps)
    sqrtmm = tl.maximum(tl.load(sqrt_minmax_ptr + pid).to(tl.float32), quant_eps)
    if SIGNED:
        scaled = raw.to(tl.float32) * s
        absolute = tl.abs(scaled)
        sign = tl.where(scaled >= 0.0, 1.0, -1.0)
        power = libdevice.pow(tl.maximum(absolute, quant_eps), 1.0 / expand)
        value = tl.where(absolute > 0.0, sign * power * sqrtmm, 0.0)
    else:
        scaled = tl.maximum(raw.to(tl.float32) * s, 0.0)
        power = libdevice.pow(tl.maximum(scaled, quant_eps), 1.0 / expand)
        value = tl.where(scaled > 0.0, power * sqrtmm, 0.0)
    tl.store(out_ptr + offs, value, mask=mask)


@triton.jit
def _quantize_kernel(value_ptr, data_ptr, scale_ptr, expand_ptr, sqrt_minmax_ptr, fp8_max, expand_min, numel,
                     SIGNED: tl.constexpr, GROUP: tl.constexpr):
    quant_eps = 1e-30
    large_val = tl.full((GROUP,), 3.4028235e38, tl.float32)
    pid = tl.program_id(0)
    offs = pid * GROUP + tl.arange(0, GROUP)
    mask = offs < numel
    m = tl.load(value_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    abs_m = tl.abs(m)
    valid_m = mask & (abs_m > 0.0)
    absmax_m = tl.max(tl.where(mask, abs_m, 0.0), axis=0)
    absmin_m = tl.min(tl.where(valid_m, abs_m, large_val), axis=0)
    nonzero_m = tl.sum(valid_m.to(tl.int32), axis=0)
    absmax_m = tl.maximum(absmax_m, quant_eps)
    absmin_m = tl.where(nonzero_m > 0, tl.maximum(absmin_m, quant_eps), quant_eps)

    ratio_m = absmax_m / absmin_m
    ratio_m = tl.maximum(ratio_m, 1.0 + quant_eps)
    ratio_upper_m = fp8_max * fp8_max / 2.0
    log_ratio_m = tl.log2(ratio_m)
    safe_log_ratio_m = tl.where(log_ratio_m > quant_eps, log_ratio_m, 1.0)
    raw_expand_m = tl.floor((tl.log2(ratio_upper_m) / safe_log_ratio_m) * expand_min) / expand_min
    min_expand = 1.0 / expand_min
    new_expand_m = tl.where(ratio_m <= 1.0 + quant_eps, 1.0, tl.maximum(raw_expand_m, min_expand))

    sqrt_minmax_m = tl.sqrt(absmax_m) * tl.sqrt(absmin_m)
    sqrt_minmax_m = tl.maximum(sqrt_minmax_m, quant_eps)
    norm_base_m = tl.maximum(abs_m / sqrt_minmax_m, quant_eps)
    norm_pow_m = libdevice.pow(norm_base_m, new_expand_m)
    if SIGNED:
        norm_sign_m = tl.where(m >= 0.0, 1.0, -1.0)
        normalized_m = tl.where(mask & (abs_m > 0.0), norm_sign_m * norm_pow_m, 0.0)
    else:
        normalized_m = tl.where(mask & (abs_m > 0.0), norm_pow_m, 0.0)

    scale_base_m = tl.maximum(absmax_m / sqrt_minmax_m, quant_eps)
    new_scale_m = libdevice.pow(scale_base_m, new_expand_m) / fp8_max
    new_scale_m = tl.maximum(new_scale_m, quant_eps)
    q_m = (normalized_m / new_scale_m).to(tl.float32)

    tl.store(data_ptr + offs, q_m, mask=mask)
    tl.store(scale_ptr + pid, new_scale_m.to(tl.float32))
    tl.store(expand_ptr + pid, new_expand_m.to(tl.float32))
    tl.store(sqrt_minmax_ptr + pid, sqrt_minmax_m.to(tl.float32))


def dequantize(state, name, signed):
    data = state[name]
    out = torch.empty(data.shape, device=data.device, dtype=torch.float32)
    _dequantize_kernel[(triton.cdiv(data.numel(), GROUP_SIZE),)](
        data.view(-1), out.view(-1), state[f"scale_{name}"], state[f"expand_{name}"], state[f"sqrt_minmax_{name}"],
        data.numel(), SIGNED=signed, GROUP=GROUP_SIZE)
    return out


def quantize_(state, name, value, signed, dtype, expand_min):
    """Store the FP32 `value` as state[name] in COAT's format, allocating the storage on first use."""
    groups = triton.cdiv(value.numel(), GROUP_SIZE)
    if state.get(name) is None or state[name].dtype != dtype:
        state[name] = torch.empty(value.shape, device=value.device, dtype=dtype)
        for key in ("scale", "expand", "sqrt_minmax"):
            state[f"{key}_{name}"] = torch.empty(groups, device=value.device, dtype=torch.float32)
    data = state[name]
    _quantize_kernel[(groups,)](
        value.contiguous().view(-1), data.view(-1), state[f"scale_{name}"], state[f"expand_{name}"],
        state[f"sqrt_minmax_{name}"], float(torch.finfo(dtype).max), expand_min, value.numel(),
        SIGNED=signed, GROUP=GROUP_SIZE)


class CoatStateMonarchMuon(MonarchMuonOptimizer):
    def __init__(self, *args, qargs, **kwargs):
        assert qargs.qgroup_size == GROUP_SIZE, "COAT's optimizer kernels use groups of 128"
        assert qargs.first_order_expansion in ("expand", "true") and qargs.second_order_expansion in ("expand", "true")
        self.dtypes = {"momentum_buffer": FP8[qargs.first_order_bit], "exp_avg": FP8[qargs.first_order_bit],
                       "exp_avg_sq": FP8[qargs.second_order_bit]}
        self.expand_min = qargs.expand_min
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        active = [self.state[p] for group in self.param_groups for p in group["params"] if p.grad is not None]
        for state in active:
            for name, signed in STATES.items():
                if name in state:
                    state[name] = dequantize(state, name, signed)
        loss = super().step(closure)
        for state in active:
            for name, signed in STATES.items():
                if name in state:
                    value = state.pop(name)
                    quantize_(state, name, value, signed, self.dtypes[name], self.expand_min)
        return loss

    def load_state_dict(self, state_dict):
        # torch.optim casts floating-point state to the parameter dtype; put the FP8 payloads back
        # (FP8 values are exact in FP32, so the round trip is lossless).
        saved = {key: {name: value.dtype for name, value in entry.items() if name in STATES}
                 for key, entry in state_dict["state"].items()}
        super().load_state_dict(state_dict)
        params = [p for group in self.param_groups for p in group["params"]]
        for key, dtypes in saved.items():
            state = self.state[params[key]]
            for name, dtype in dtypes.items():
                state[name] = state[name].to(dtype)
