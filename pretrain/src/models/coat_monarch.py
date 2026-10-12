"""COAT FP8 activation recipe for Monarch blocks (``--fp8``).

The precision flow is COAT's FP8 transformer block: the residual stream
enters every RMSNorm as an FP8 copy with 1x16 group scales, RMSNorm/SiLU/mul
run on FP8 inputs and keep FP8 tensors for backward, gate/up outputs are
re-quantized 1x16 before SiLU*mul, and each projection consumes a per-tensor
FP8 input. COAT's dense GEMMs are replaced by the two block GEMMs of a Monarch
projection, computed exactly as ``fp8_optimized.Monarch`` does with batched
cuBLASLt (E4M3 forward, E5M2 gradients, per-tensor current scaling, no fast
accumulation).

Gradients between projections are not re-quantized by COAT's add kernels:
the full-precision residual gradient feeds the Monarch backward, which
quantizes it per tensor to E5M2 itself.
"""
import torch

from third_party.coat.activation.real_quantization import (
    fp8_add_Ifp_Ifp_Ofp_Og16,
    fp8_mul_backward,
    fp8_mul_forward,
    fp8_quantize,
    fp8_rmsnorm_backward,
    fp8_rmsnorm_forward,
    fp8_silu_backward,
    fp8_silu_forward,
)

from .fp8_batched import matmul
from .fp8_optimized import quantize

E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2


def weights(module):
    if module._coat_cache is None:
        module._coat_cache = tuple(quantize(w, E4M3) for w in (module.blkdiag1, module.blkdiag2))
    return module._coat_cache


def coat_input(x, x_t, scale, module):
    """Blocked operands from a COAT per-tensor FP8 tensor and its 2-D transpose."""
    k, _, p = module.blkdiag1.shape
    n = x_t.shape[-1]
    xq = x.reshape(n, k, p).transpose(0, 1).contiguous()
    return xq, x_t.view(k, p, n), scale.float().reshape(())


def monarch_forward(xq, xt, sx, cache):
    (w1q, _, s1), (w2q, _, s2) = cache
    k, q, _ = w1q.shape
    l, s, _ = w2q.shape
    h = matmul(xq, w1q.mT, sx, s1)
    hq, ht, sh = quantize(h.view(k, -1, q // l, l).permute(3, 1, 0, 2), E4M3)
    out = matmul(hq, w2q.mT, sh, s2)
    return out.permute(1, 2, 0).reshape(-1, s * l), (xt, sx, ht, sh, cache)


def monarch_backward(grad, state):
    xt, sx, ht, sh, ((_, w1t, s1), (_, w2t, s2)) = state
    k, p, q = w1t.shape
    l, _, s = w2t.shape
    gq, gt, sg = quantize(grad.reshape(-1, s, l).permute(2, 0, 1), E5M2)
    dw2 = matmul(gt, ht.mT, sg, sh)
    dh = matmul(gq, w2t.mT, sg, s2).view(l, -1, k, q // l).permute(2, 1, 3, 0)
    dhq, dht, sdh = quantize(dh, E5M2)
    dx = matmul(dhq, w1t.mT, sdh, s1, interleaved=True).transpose(0, 1).reshape(-1, k * p)
    dw1 = matmul(dht, xt.mT, sdh, sx)
    return dx, dw1, dw2


class BeforeAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, re_x, in_x, in_s, norm_weight, eps, group_size, modules, *params):
        ctx.set_materialize_grads(False)
        in_x = in_x.view(E4M3)
        ln_x, ln_s, ln_x_t, (_, rstd, num_warps) = fp8_rmsnorm_forward(
            in_x, in_s, norm_weight, group_size, eps, transpose_output_2d=True)
        xq, xt, sx = coat_input(ln_x, ln_x_t, ln_s, modules[0])
        outputs, ctx.states = [], []
        for module in modules:
            y, state = monarch_forward(xq, xt, sx, weights(module))
            outputs.append(y.view(*re_x.shape[:-1], -1))
            ctx.states.append(state)
        ctx.save_for_backward(in_x, in_s, norm_weight, rstd)
        ctx.group_size, ctx.num_warps = group_size, num_warps
        return (re_x, *outputs)

    @staticmethod
    def backward(ctx, fp_grad, *grads):
        in_x, in_s, norm_weight, rstd = ctx.saved_tensors
        # States live on ctx, which autograd keeps after backward; drop them like saved tensors.
        states, ctx.states = ctx.states, None
        dx, param_grads = 0, []
        for grad, state in zip(grads, states):
            g, dw1, dw2 = monarch_backward(grad, state)
            dx = dx + g
            param_grads += [dw1, dw2]
        in_g, norm_g = fp8_rmsnorm_backward(in_x, in_s, dx.view(in_x.shape), norm_weight, rstd,
                                            ctx.group_size, ctx.num_warps)
        return (fp_grad + in_g, None, None, norm_g, None, None, None, *param_grads)


class AfterAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, re_x, attn, group_size, module, w1, w2):
        ctx.set_materialize_grads(False)
        k, _, p = w1.shape
        xq, xt, sx = quantize(attn.reshape(-1, k, p).transpose(0, 1), E4M3)
        y, ctx.state = monarch_forward(xq, xt, sx, weights(module))
        fp_x, (out_x, out_s) = fp8_add_Ifp_Ifp_Ofp_Og16(re_x, y.view(re_x.shape), E4M3, group_size)
        ctx.attn_shape = attn.shape
        return fp_x, out_x, out_s

    @staticmethod
    def backward(ctx, fp_grad, out_g, out_gs):
        state, ctx.state = ctx.state, None
        dx, dw1, dw2 = monarch_backward(fp_grad, state)
        return fp_grad, dx.view(ctx.attn_shape), None, None, dw1, dw2


class MLPResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, re_x, in_x, in_s, norm_weight, eps, group_size, modules, *params):
        ctx.set_materialize_grads(False)
        gate, up, down = modules
        in_x = in_x.view(E4M3)
        ln_x, ln_s, ln_x_t, (_, rstd, num_warps) = fp8_rmsnorm_forward(
            in_x, in_s, norm_weight, group_size, eps, transpose_output_2d=True)
        xq, xt, sx = coat_input(ln_x, ln_x_t, ln_s, gate)
        gate_y, gate_state = monarch_forward(xq, xt, sx, weights(gate))
        up_y, up_state = monarch_forward(xq, xt, sx, weights(up))
        gate_x, gate_s = fp8_quantize(gate_y, group_size, E4M3)
        up_x, up_s = fp8_quantize(up_y, group_size, E4M3)
        silu_x, silu_s = fp8_silu_forward(gate_x, gate_s, group_size)
        mul_x, mul_s, mul_x_t = fp8_mul_forward(silu_x, silu_s, up_x, up_s, group_size, transpose_output_2d=True)
        xq, xt, sx = coat_input(mul_x, mul_x_t, mul_s, down)
        down_y, down_state = monarch_forward(xq, xt, sx, weights(down))
        fp_x, (out_x, out_s) = fp8_add_Ifp_Ifp_Ofp_Og16(re_x, down_y.view(re_x.shape), E4M3, group_size)
        ctx.save_for_backward(in_x, in_s, norm_weight, rstd, gate_x, gate_s, up_x, up_s, silu_x, silu_s)
        ctx.states = gate_state, up_state, down_state
        ctx.group_size, ctx.num_warps = group_size, num_warps
        return fp_x, out_x, out_s

    @staticmethod
    def backward(ctx, fp_grad, out_g, out_gs):
        in_x, in_s, norm_weight, rstd, gate_x, gate_s, up_x, up_s, silu_x, silu_s = ctx.saved_tensors
        (gate_state, up_state, down_state), ctx.states = ctx.states, None
        group_size = ctx.group_size
        mul_g, down_dw1, down_dw2 = monarch_backward(fp_grad, down_state)
        silu_g, (up_g, _) = fp8_mul_backward(silu_x, silu_s, up_x, up_s, mul_g, group_size, "E5M2")
        gate_g, _ = fp8_silu_backward(gate_x, gate_s, silu_g, group_size, "E5M2")
        gate_dx, gate_dw1, gate_dw2 = monarch_backward(gate_g, gate_state)
        up_dx, up_dw1, up_dw2 = monarch_backward(up_g, up_state)
        in_g, norm_g = fp8_rmsnorm_backward(in_x, in_s, (gate_dx + up_dx).view(in_x.shape), norm_weight, rstd,
                                            group_size, ctx.num_warps)
        return (fp_grad + in_g, None, None, norm_g, None, None, None,
                gate_dw1, gate_dw2, up_dw1, up_dw2, down_dw1, down_dw2)


def attach(model, optimizer):
    """Give every Monarch projection an FP8 weight cache cleared after each optimizer step."""
    from .monarch.monarch_linear import MonarchLinear

    modules = [m for m in model.modules() if isinstance(m, MonarchLinear)]
    for module in modules:
        assert module.bias is None
        assert module.in_features == module.in_features_extended
        assert module.out_features == module.out_features_extended
        module._coat_cache = None

    def clear_cache(*_):
        for module in modules:
            module._coat_cache = None

    optimizer.register_step_post_hook(clear_cache)
    model.register_load_state_dict_post_hook(clear_cache)
    return clear_cache
