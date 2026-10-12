import json
import unittest

import torch

from arm_build import build
from models.coat_monarch import coat_input, monarch_backward, monarch_forward
from models.fp8_optimized import Monarch, quantize
from models.monarch import MonarchLinear

E4M3 = torch.float8_e4m3fn


def exact(a, b):
    torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0)


def relative(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@unittest.skipUnless(torch.cuda.is_available(), "H100 required")
class CoatMonarchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_monarch_functions_match_fp8_optimized(self):
        for blocks in (2, 4):
            for n_in, n_out in ((1024, 1024), (1024, 2816), (2816, 1024)):
                with self.subTest(blocks=blocks, n_in=n_in, n_out=n_out):
                    layer = MonarchLinear(n_in, n_out, nblocks=blocks, bias=False).cuda()
                    cache = tuple(quantize(w, E4M3) for w in (layer.blkdiag1, layer.blkdiag2))
                    x = torch.randn(8192, n_in, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                    grad = torch.randn(8192, n_out, device="cuda", dtype=torch.bfloat16)
                    expected = Monarch.apply(x, layer.blkdiag1, layer.blkdiag2, cache, None)
                    expected.backward(grad)
                    k, _, p = layer.blkdiag1.shape
                    xq, xt, sx = quantize(x.detach().reshape(-1, k, p).transpose(0, 1), E4M3)
                    output, state = monarch_forward(xq, xt, sx, cache)
                    dx, dw1, dw2 = monarch_backward(grad, state)
                    exact(output, expected)
                    exact(dx, x.grad)
                    exact(dw1, layer.blkdiag1.grad)
                    exact(dw2, layer.blkdiag2.grad)

    def test_coat_operands_match_blocked_quantization(self):
        for blocks in (2, 4):
            for n in (1024, 2816):
                with self.subTest(blocks=blocks, n=n):
                    layer = MonarchLinear(n, 1024, nblocks=blocks, bias=False).cuda()
                    x = torch.randn(4, 1024, n, device="cuda", dtype=torch.bfloat16)
                    flat, flat_t, scale = quantize(x.reshape(-1, n), E4M3)
                    xq, xt, sx = coat_input(flat[0].view(x.shape), flat_t[0], scale, layer)
                    k, _, p = layer.blkdiag1.shape
                    eq, et, es = quantize(x.reshape(-1, k, p).transpose(0, 1), E4M3)
                    exact(xq, eq)
                    exact(xt, et)
                    exact(sx, es)

    def test_coat_arms_learn_like_bf16_and_evaluate_in_bf16(self):
        tokens = torch.randint(50304, (2, 4, 1024), device="cuda")
        arms = {"monarch-muon-nb2": "MonarchMuonOptimizer", "monarch-muon-nb2-fp8gemm": "MonarchMuonOptimizer",
                "monarch-adamw-nb2": "AdamW", "monarch-adamw-nb2-fp8gemm": "AdamW"}
        final = {}
        for arm, optimizer_name in arms.items():
            with self.subTest(arm=arm):
                model, optimizer, _ = build(arm)
                self.assertEqual(type(optimizer).__name__, optimizer_name)
                if "adamw" in arm:
                    # Without warmup, AdamW at the recipe's 1e-3 spikes on a fixed batch even in BF16.
                    for group in optimizer.param_groups:
                        group["lr"] = 2e-4
                monarch = [m for m in model.modules() if isinstance(m, MonarchLinear)]
                self.assertEqual(len(monarch), 84)
                losses = []
                for _ in range(15):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(tokens[0], targets=tokens[1])["loss"]
                        loss.backward()
                    losses.append(loss.item())
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if "fp8gemm" in arm:
                        self.assertTrue(all(m._coat_cache is None for m in monarch))
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    train_mode = model(tokens[0], targets=tokens[1])["loss"].item()
                    model.eval()
                    eval_mode = model(tokens[0], targets=tokens[1])["loss"].item()
                print("COAT_LEARN=" + json.dumps({"arm": arm, "losses": losses, "train_mode": train_mode,
                                                  "eval_mode": eval_mode}), flush=True)
                final[arm] = sum(losses[-5:]) / 5
                self.assertTrue(all(map(torch.isfinite, map(torch.tensor, losses))), losses)
                self.assertLess(losses[-1], losses[0] - 0.3, losses)
                self.assertLess(abs(train_mode - eval_mode) / eval_mode, 0.02)
                reference = arm.removesuffix("-fp8gemm")
                self.assertLess(abs(final[arm] - final[reference]) / final[reference], 0.03)
                del model, optimizer
                torch.cuda.empty_cache()

if __name__ == "__main__":
    unittest.main()
