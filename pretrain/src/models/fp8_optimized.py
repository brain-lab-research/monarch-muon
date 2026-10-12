from types import MethodType
import weakref

import torch
import triton
import triton.language as tl

from .fp8_batched import matmul


@triton.jit
def _offset(i, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr,
            S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr):
    return (i // (D1 * D2 * D3)) * S0 + (i // (D2 * D3) % D1) * S1 + (i // D3 % D2) * S2 + (i % D3) * S3


@triton.jit
def _amax(X, PART, NUM: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr,
          S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr, BLOCK: tl.constexpr,
          PHYSICAL: tl.constexpr = False):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offset = i if PHYSICAL else _offset(i, D1, D2, D3, S0, S1, S2, S3)
    value = tl.load(X + offset, i < NUM, 0).to(tl.bfloat16).to(tl.float32)
    tl.store(PART + tl.program_id(0), tl.max(tl.abs(value), 0))


@triton.jit
def _scale(PART, SCALE, NUM: tl.constexpr, LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    maximum = tl.max(tl.load(PART + i, i < NUM, 0), 0)
    # Match ATen's CPU-scalar division: multiply by an FP32 reciprocal.
    reciprocal = tl.full((), 1.0 / LIMIT, tl.float32)
    tl.store(SCALE, tl.where(maximum > 0, maximum * reciprocal, 1.0))


@triton.jit
def _cast_transpose(X, Q, QT, SCALE, M: tl.constexpr, N: tl.constexpr,
                    D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr,
                    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
                    LIMIT: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr):
    row = tl.program_id(0) * TM + tl.arange(0, TM)
    col = tl.program_id(1) * TN + tl.arange(0, TN)
    batch = tl.program_id(2)
    offset = batch * S0 + row[:, None] * S1 + (col[None, :] // D3) * S2 + (col[None, :] % D3) * S3
    value = tl.load(X + offset, (row[:, None] < M) & (col[None, :] < N), 0)
    value = value.to(tl.bfloat16)
    value = value.to(tl.float32)
    value = tl.minimum(tl.maximum(tl.div_rn(value, tl.load(SCALE)), -LIMIT), LIMIT).to(Q.dtype.element_ty)
    mask = (row[:, None] < M) & (col[None, :] < N)
    tl.store(Q + (batch * M + row[:, None]) * N + col[None, :], value, mask)
    tl.store(QT + (batch * N + col[None, :]) * M + row[:, None], value, mask)


def quantize(tensor, dtype):
    # Four logical dimensions allow the Monarch permutation to fuse into the cast.
    if tensor.ndim == 2:
        tensor = tensor[None, :, :, None]
    elif tensor.ndim == 3:
        tensor = tensor[..., None]
    b, m, d2, d3 = tensor.shape
    n = d2 * d3
    reduction_block = 4096
    parts = triton.cdiv(tensor.numel(), reduction_block)
    partial = torch.empty(parts, device=tensor.device, dtype=torch.float32)
    scale = torch.empty((), device=tensor.device, dtype=torch.float32)
    q = torch.empty((b, m, n), device=tensor.device, dtype=dtype)
    qt = torch.empty((b, n, m), device=tensor.device, dtype=dtype)
    shape = (m, d2, d3, *tensor.stride())
    limit = torch.finfo(dtype).max
    # The amax may read the storage linearly only when the view covers it densely.
    physical, span = True, 1
    for stride, size in sorted((s, d) for s, d in zip(tensor.stride(), tensor.shape) if d > 1):
        physical = physical and stride == span
        span *= size
    _amax[(parts,)](tensor, partial, tensor.numel(), *shape, reduction_block, physical, num_warps=4)
    _scale[(1,)](partial, scale, parts, limit, triton.next_power_of_2(parts))
    tile = 64
    _cast_transpose[(triton.cdiv(m, tile), triton.cdiv(n, tile), b)](
        tensor, q, qt, scale, m, n, *shape, limit, tile, tile, num_warps=8)
    return q, qt, scale


class Monarch(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, w1, w2, cached, input_quantized):
        ctx.input_shape = x.shape
        k, q, p = w1.shape
        l, s, r = w2.shape
        xq, xt, sx = input_quantized if input_quantized is not None else quantize(x.reshape(-1, k, p).transpose(0, 1), torch.float8_e4m3fn)
        (w1q, w1t, s1), (w2q, w2t, s2) = cached
        h = matmul(xq, w1q.mT, sx, s1)
        h = h.view(k, -1, q // l, l).permute(3, 1, 0, 2)
        hq, ht, sh = quantize(h, torch.float8_e4m3fn)
        out = matmul(hq, w2q.mT, sh, s2)
        ctx.save_for_backward(xt, w1t, w2t, ht, sx, s1, s2, sh)
        return out.permute(1, 2, 0).reshape(*x.shape[:-1], s * l)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad):
        xt, w1t, w2t, ht, sx, s1, s2, sh = ctx.saved_tensors
        k, p, q = w1t.shape
        l, r, s = w2t.shape
        grad = grad.reshape(-1, s, l).permute(2, 0, 1)
        gq, gt, sg = quantize(grad, torch.float8_e5m2)
        dw2 = matmul(gt, ht.mT, sg, sh) if ctx.needs_input_grad[2] else None
        dx, dw1 = None, None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            dh = matmul(gq, w2t.mT, sg, s2)
            dh = dh.view(l, -1, k, q // l).permute(2, 1, 3, 0)
            dhq, dht, sdh = quantize(dh, torch.float8_e5m2)
            if ctx.needs_input_grad[0]:
                dx = matmul(dhq, w1t.mT, sdh, s1, interleaved=True).transpose(0, 1).reshape(ctx.input_shape)
            if ctx.needs_input_grad[1]:
                dw1 = matmul(dht, xt.mT, sdh, sx)
        return dx, dw1, dw2, None, None


def _shared_input(module, x):
    state = getattr(module, "_fp8_shared_input", None)
    if state is None:
        return None
    source = state.get("source")
    if source is None or source() is not x or state["version"] != x._version:
        k, _, p = module.blkdiag1.shape
        state.update(source=weakref.ref(x), version=x._version,
                     quantized=quantize(x.reshape(-1, k, p).transpose(0, 1), torch.float8_e4m3fn))
    result = state["quantized"]
    if module._fp8_shared_last:
        state.clear()
    return result


def _monarch_forward(self, x):
    if self._fp8_cache is None:
        self._fp8_cache = tuple(quantize(w, torch.float8_e4m3fn) for w in (self.blkdiag1, self.blkdiag2))
    x = self.preprocess(x)
    return self.postprocess(Monarch.apply(x, self.blkdiag1, self.blkdiag2, self._fp8_cache, _shared_input(self, x)))


def apply_fp8_optimized(model):
    """FP8 GEMMs for every MonarchLinear of `model` (layer benchmark).

    The FP8 weights are cached on the first call and reused afterwards; q/k/v and gate/up
    projections that read the same input share one quantization of it.
    """
    from .monarch.monarch_linear import MonarchLinear

    for name, module in model.named_modules():
        if "lm_head" in name or not isinstance(module, MonarchLinear):
            continue
        module.forward_matmul = MethodType(_monarch_forward, module)
        module._fp8_cache = None
        module._fp8_shared_input = None

    for parent in model.modules():
        for names in (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")):
            if all(name in parent._modules for name in names):
                state = {}
                for name in names:
                    module = parent._modules[name]
                    module._fp8_shared_input = state
                    module._fp8_shared_last = name == names[-1]
