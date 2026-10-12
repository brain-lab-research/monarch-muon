"""RMSNorm + SwiGLU MLP block, forward+backward, dense vs Monarch, at MoE-layer shapes.

The routed experts of each MoE shape are replaced by one MLP with the same GEMM work (width top-k x expert width).
The BF16 baseline and native FP8 are Transformer Engine: a dense projection is te.Linear,
a Monarch projection is two te.GroupedLinear (one group per block) with the butterfly permutation between them.
Prints MONARCH_TE json lines per shape:
  dense/te_bf16              dense TE RMSNorm + TE GEMMs (cuBLAS), the same-FLOPs reference
  m<blocks>/te_bf16          Monarch with 2 or 4 blocks, TE RMSNorm + TE grouped GEMMs
  m<blocks>/te_fp8_<recipe>_<cache>
                             the same modules under TE FP8 autocast (blockwise, tensorwise, delayed 1024/max)
  m<blocks>/ours_fp8_warm    TE RMSNorm + our cuBLASLt FP8 Monarch GEMMs (fp8_optimized, shared input quantization)
`cold` requantizes the weights (first micro-batch after an optimizer step), `warm` reuses them; the TE `warm`
arms run after their `cold` arm on the same modules.
Every Monarch variant shares one set of weights; out_rel_err / dx_rel_err are against te_bf16.
"""
import json
import sys
import time

import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe

from models.fp8_optimized import apply_fp8_optimized
from models.monarch import MonarchLinear

EPS = 1e-5
# name: hidden, expert width, top-k, tokens. MLP width = top-k x expert width.
SHAPES = {
    "stage3_moe_1b": (1024, 256, 8, 26624),
    "qwen3_moe_30b_like": (2048, 768, 8, 16384),
    "wide_experts_2048x2048": (2048, 2048, 8, 8192),
    "deepseek_v3_like_64_of_256_experts": (7168, 2048, 8, 8192),
}
RECIPES = {
    "blockwise": lambda: te_recipe.Float8BlockScaling(fp8_format=te_recipe.Format.E4M3),
    "tensorwise": lambda: te_recipe.Float8CurrentScaling(fp8_format=te_recipe.Format.HYBRID),
    "delayed1024": lambda: te_recipe.DelayedScaling(fp8_format=te_recipe.Format.HYBRID, amax_history_len=1024,
                                                    amax_compute_algo="max"),
}
_autocast = getattr(te, "autocast", None) or te.fp8_autocast


def autocast(recipe):
    try:
        return _autocast(enabled=True, recipe=recipe)
    except TypeError:
        return _autocast(enabled=True, fp8_recipe=recipe)


def report(**values):
    print("MONARCH_TE " + json.dumps(values), flush=True)


def timed(fn, repeat=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return round(sorted(times)[len(times) // 2], 3)


def measure(fn):
    torch.cuda.reset_peak_memory_stats()
    try:
        return {"ms": timed(fn), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    except Exception as error:
        return {"error": repr(error)[:300]}
    finally:
        torch.cuda.empty_cache()


class TEMonarch(torch.nn.Module):
    """A MonarchLinear's two block-diagonal factors as TE grouped GEMMs, same arithmetic as monarch_forward."""

    def __init__(self, monarch):
        super().__init__()
        k, q, p = monarch.blkdiag1.shape
        l, s, r = monarch.blkdiag2.shape
        self.shape = k, q, p, l, s, r
        self.g1 = te.GroupedLinear(k, p, q, bias=False, params_dtype=torch.bfloat16)
        self.g2 = te.GroupedLinear(l, r, s, bias=False, params_dtype=torch.bfloat16)
        with torch.no_grad():
            for i in range(k):
                getattr(self.g1, f"weight{i}").copy_(monarch.blkdiag1[i])
            for i in range(l):
                getattr(self.g2, f"weight{i}").copy_(monarch.blkdiag2[i])

    def forward(self, x, first=None):
        k, q, p, l, s, r = self.shape
        n = x.shape[0]
        h = self.g1(x.view(n, k, p).transpose(0, 1).reshape(k * n, p), [n] * k, is_first_microbatch=first)
        h = h.view(k, n, q // l, l).permute(3, 1, 0, 2).reshape(l * n, r)
        y = self.g2(h, [n] * l, is_first_microbatch=first)
        return y.view(l, n, s).permute(1, 2, 0).reshape(n, s * l)


class TEDense(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.linear = te.Linear(n_in, n_out, bias=False, params_dtype=torch.bfloat16)

    def forward(self, x, first=None):
        return self.linear(x, is_first_microbatch=first)


class Projections(torch.nn.Module):
    """Names fp8_optimized looks for to share the gate/up input quantization."""

    def __init__(self, hidden, width, blocks):
        super().__init__()
        self.gate_proj = MonarchLinear(hidden, width, bias=False, nblocks=blocks)
        self.up_proj = MonarchLinear(hidden, width, bias=False, nblocks=blocks)
        self.down_proj = MonarchLinear(width, hidden, bias=False, nblocks=blocks)


def bench(name, hidden, expert, topk, tokens):
    width = topk * expert
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=generator).requires_grad_(True)
    grad = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=generator)
    te_norm = te.RMSNorm(hidden, eps=EPS, params_dtype=torch.bfloat16)
    reference = {}

    def mlp(projections, first=None):
        gate_proj, up_proj, down_proj = projections
        h = te_norm(x)
        return down_proj(torch.nn.functional.silu(gate_proj(h, first)) * up_proj(h, first), first)

    def run(label, step, flops):
        values = {"shape": name, "arm": label, "hidden": hidden, "mlp_width": width, "tokens": tokens,
                  "gemm_tflop": round(flops / 1e12, 3)}
        try:
            x.grad = None
            out = step().detach().float()
        except Exception as error:
            report(**values, error=repr(error)[:300])
            torch.cuda.empty_cache()
            return
        values.update(measure(step))
        if "ms" in values:
            if label.endswith("te_bf16"):
                reference[label.split("/")[0]] = out, x.grad.float().clone()
            else:
                ref_out, ref_grad = reference[label.split("/")[0]]
                values["out_rel_err"] = float((out - ref_out).norm() / ref_out.norm())
                values["dx_rel_err"] = float((x.grad.float() - ref_grad).norm() / ref_grad.norm())
        if "ms" in values:
            values["tflops"] = round(flops / values["ms"] / 1e9, 1)
        report(**values)

    # ---- dense: TE BF16, the same-FLOPs reference of the MoE benchmark
    flops = 3 * 2 * tokens * 3 * hidden * width
    dense = [TEDense(hidden, width), TEDense(hidden, width), TEDense(width, hidden)]

    def te_step(projections, recipe=None, first=None):
        def step():
            if recipe is None:
                y = mlp(projections)
            else:
                with autocast(recipe):
                    y = mlp(projections, first)
            y.backward(grad)
            return y
        return step

    run("dense/te_bf16", te_step(dense), flops)
    del dense
    torch.cuda.empty_cache()

    for blocks in (2, 4):
        key = f"m{blocks}"
        torch.manual_seed(blocks)
        ours = Projections(hidden, width, blocks).cuda()
        monarchs = (ours.gate_proj, ours.up_proj, ours.down_proj)
        flops = 3 * 2 * tokens * sum(m.blkdiag1.numel() + m.blkdiag2.numel() for m in monarchs)

        run(f"{key}/te_bf16", te_step([TEMonarch(m) for m in monarchs]), flops)
        for label, recipe in RECIPES.items():
            r = recipe()
            projections = [TEMonarch(m) for m in monarchs]
            run(f"{key}/te_fp8_{label}_cold", te_step(projections, r, True), flops)
            run(f"{key}/te_fp8_{label}_warm", te_step(projections, r, False), flops)
            del projections
            torch.cuda.empty_cache()

        fp8 = Projections(hidden, width, blocks).cuda()
        fp8.load_state_dict(ours.state_dict())
        apply_fp8_optimized(fp8)

        def ours_fp8():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = te_norm(x)
                y = fp8.down_proj(torch.nn.functional.silu(fp8.gate_proj(h)) * fp8.up_proj(h))
            y.backward(grad)
            return y
        run(f"{key}/ours_fp8_warm", ours_fp8, flops)
        del fp8, ours, monarchs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    for shape in sys.argv[1:] or SHAPES:
        try:
            bench(shape, *SHAPES[shape])
        except Exception as error:
            report(shape=shape, error=repr(error)[:400])
        torch.cuda.empty_cache()
