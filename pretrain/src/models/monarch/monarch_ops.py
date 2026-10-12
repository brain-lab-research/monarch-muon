"""Opaque custom op for the Monarch butterfly multiply (MonarchLinear.forward_matmul).

Why: torch.compile/Inductor decomposes the eager butterfly's permute chain
into worse copies and generates a slow backward (measured +48% on fwd+bwd).
Registering the butterfly as a custom op makes it an atomic node: Inductor
fuses the surrounding glue (RMSNorm/RoPE/SwiGLU/residual) and leaves the
butterfly alone.

The op returns out2 in its NATURAL bmm layout (B, l, s) — no interleave copy
inside. The feature-order permute happens in the python wrapper (compiled
region), where Inductor fuses it into pointwise consumers; the backward
receives a blocked grad, so the dout gather-copy becomes a strided VIEW for
the cuBLAS bmms. The riffle between the two factors uses the Triton kernels of
monarch_riffle.py for 2 or 4 blocks, else an eager copy.
"""
from __future__ import annotations

import torch

from . import monarch_riffle


def _butterfly_fwd_blocked(x2, w1_bfly, w2_bfly):
    """x2 (B, n) -> out_blocked (B, l, s) contiguous, out1 (l, B, r) contiguous."""
    B = x2.shape[0]
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    x_r = x2.reshape(B, k, p).transpose(0, 1)
    out1 = torch.empty(B, k, q, device=x2.device, dtype=x2.dtype).transpose(0, 1)
    out1 = torch.bmm(x_r, w1_bfly.transpose(-1, -2), out=out1)
    if l == k and l in (2, 4) and q % l == 0:
        out1 = monarch_riffle.riffle_triton(out1.transpose(0, 1), B, l, q)
    else:
        out1 = (out1.transpose(0, 1).reshape(B, r, l)
                .permute(2, 0, 1).contiguous())      # riffle (eager fallback)
    out2 = torch.empty(B, l, s, device=x2.device, dtype=x2.dtype).transpose(0, 1)
    out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
    return out2.transpose(0, 1), out1                # (B, l, s) contiguous view


@torch.library.custom_op("monarch::butterfly_blk", mutates_args=())
def _butterfly_blk_op(x2: torch.Tensor, w1_bfly: torch.Tensor,
                      w2_bfly: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    out_blk, out1 = _butterfly_fwd_blocked(x2, w1_bfly, w2_bfly)
    return out_blk.contiguous(), out1


@_butterfly_blk_op.register_fake
def _(x2, w1_bfly, w2_bfly):
    B = x2.shape[0]
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    return x2.new_empty(B, l, s), x2.new_empty(l, B, r)


@torch.library.custom_op("monarch::butterfly_blk_bwd", mutates_args=())
def _butterfly_blk_bwd_op(x2: torch.Tensor, w1_bfly: torch.Tensor,
                          w2_bfly: torch.Tensor, out1: torch.Tensor,
                          dout_blk: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = x2.shape[0]
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    # blocked grad -> (l, B, s) strided VIEW: no gather copy needed
    dout_reshaped = dout_blk.permute(1, 0, 2)
    dw2 = torch.bmm(dout_reshaped.transpose(-1, -2), out1)
    if l == k and l in (2, 4) and q % l == 0:
        dt = torch.empty(l, B, r, device=x2.device, dtype=x2.dtype)
        torch.bmm(dout_reshaped, w2_bfly, out=dt)
        dout1 = monarch_riffle.unriffle_triton(dt, B, k, q).transpose(0, 1)
    else:
        dout1 = torch.empty(B, l, r, device=x2.device, dtype=x2.dtype).transpose(0, 1)
        dout1 = torch.bmm(dout_reshaped, w2_bfly, out=dout1)
        dout1 = (dout1.transpose(0, 1).transpose(-1, -2).contiguous()
                 .reshape(B, k, q).transpose(0, 1))  # inverse riffle (eager)
    dx = torch.empty(B, k, p, device=x2.device, dtype=x2.dtype)
    dx = (torch.bmm(dout1, w1_bfly, out=dx.transpose(0, 1))
          .transpose(0, 1).reshape(B, k * p))
    x_r = x2.reshape(B, k, p).transpose(0, 1)
    dw1 = torch.bmm(dout1.transpose(-1, -2), x_r)
    return dx, dw1, dw2


@_butterfly_blk_bwd_op.register_fake
def _(x2, w1_bfly, w2_bfly, out1, dout_blk):
    return (x2.new_empty(x2.shape), w1_bfly.new_empty(w1_bfly.shape),
            w2_bfly.new_empty(w2_bfly.shape))


def _blk_setup_context(ctx, inputs, output):
    x2, w1_bfly, w2_bfly = inputs
    ctx.save_for_backward(x2, w1_bfly, w2_bfly, output[1])


def _blk_backward(ctx, dout_blk, dout1_unused):
    x2, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
    return _butterfly_blk_bwd_op(x2, w1_bfly, w2_bfly, out1,
                                 dout_blk.contiguous())


_butterfly_blk_op.register_autograd(_blk_backward, setup_context=_blk_setup_context)


def butterfly_blk(x, w1_bfly, w2_bfly):
    """Blocked-output butterfly; interleave done here (Inductor fuses it)."""
    if torch.is_autocast_enabled():
        dt = torch.get_autocast_dtype("cuda")
        x, w1_bfly, w2_bfly = x.to(dt), w1_bfly.to(dt), w2_bfly.to(dt)
    batch_shape, n = x.shape[:-1], x.shape[-1]
    out_blk, _ = _butterfly_blk_op(x.reshape(-1, n), w1_bfly, w2_bfly)
    B, l, s = out_blk.shape
    return out_blk.transpose(1, 2).reshape(*batch_shape, s * l)

