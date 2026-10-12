import io
import json
import unittest

import torch

from arm_build import build
from models.monarch.coat_state_muon import dequantize, quantize_
from third_party.coat.optimizer.triton_fp8_adamw import TritonCoatAdamW
from third_party.coat.utils._fp8_quantization_config import QuantizationConfig

E4M3 = torch.float8_e4m3fn
NAMES = ("exp_avg", "scale_exp_avg", "expand_exp_avg", "sqrt_minmax_exp_avg")


def exact(a, b):
    torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "H100 required")
class CoatStateMuonTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_codec_is_the_coat_adamw_kernel(self):
        # numel not a multiple of 128 and values over many decades, so expansion and the tail group both act.
        p = torch.zeros(1000, 129, device="cuda")
        g = torch.randn_like(p) * torch.logspace(-6, 1, p.numel(), device="cuda").view_as(p)
        p.grad = g.clone()
        optimizer = TritonCoatAdamW([p], lr=0.0, betas=(0.0, 0.0), weight_decay=0.0, qargs=QuantizationConfig())
        optimizer.step()  # m = g, v = g * g, each stored by the kernel's quantizer
        upstream = optimizer.state[p]
        for name, value, signed in (("exp_avg", g, True), ("exp_avg_sq", g * g, False)):
            ours = {}
            quantize_(ours, name, value, signed, E4M3, 16)
            for key in (name, f"scale_{name}", f"expand_{name}", f"sqrt_minmax_{name}"):
                with self.subTest(quantize=key):
                    exact(ours[key], upstream[key])

        # g = 0 and beta = 0.5: the kernel dequantizes, halves exactly and requantizes.
        before = {name: (dequantize(upstream, name, signed), signed) for name, signed in (("exp_avg", True), ("exp_avg_sq", False))}
        for group in optimizer.param_groups:
            group["betas"] = (0.5, 0.5)
        p.grad.zero_()
        optimizer.step()
        for name, (value, signed) in before.items():
            ours = {}
            quantize_(ours, name, value * 0.5, signed, E4M3, 16)
            for key in (name, f"scale_{name}", f"expand_{name}", f"sqrt_minmax_{name}"):
                with self.subTest(dequantize=key):
                    exact(ours[key], optimizer.state[p][key])

    def test_coatopt_arms_learn_like_bf16_and_keep_fp8_states(self):
        tokens = torch.randint(50304, (2, 4, 1024), device="cuda")
        arms = {"monarch-muon-nb2": "MonarchMuonOptimizer", "monarch-muon-nb2-fp8states": "CoatStateMonarchMuon",
                "monarch-adamw-nb2": "AdamW", "monarch-adamw-nb2-fp8states": "TritonCoatAdamW"}
        final = {}
        for arm, optimizer_name in arms.items():
            with self.subTest(arm=arm):
                model, optimizer, config = build(arm)
                self.assertEqual(type(optimizer).__name__, optimizer_name)
                self.assertFalse(config.fp8)  # BF16 GEMMs
                if "adamw" in arm:
                    for group in optimizer.param_groups:
                        group["lr"] = 2e-4
                losses = []
                for _ in range(15):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(tokens[0], targets=tokens[1])["loss"]
                        loss.backward()
                    losses.append(loss.item())
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                print("COAT_STATE_LEARN=" + json.dumps({"arm": arm, "losses": losses}), flush=True)
                final[arm] = sum(losses[-5:]) / 5
                self.assertTrue(all(map(torch.isfinite, map(torch.tensor, losses))), losses)
                self.assertLess(losses[-1], losses[0] - 0.3, losses)
                reference = arm.removesuffix("-fp8states")
                self.assertLess(abs(final[arm] - final[reference]) / final[reference], 0.03)
                if arm == "monarch-muon-nb2-fp8states":
                    states = [s for s in optimizer.state.values()]
                    self.assertTrue(all(s["momentum_buffer"].dtype == E4M3 for s in states if "momentum_buffer" in s))
                    self.assertTrue(all(s["exp_avg"].dtype == E4M3 and s["exp_avg_sq"].dtype == E4M3
                                        for s in states if "exp_avg" in s))
                    self.assertTrue(any("momentum_buffer" in s for s in states) and any("exp_avg" in s for s in states))
                    buffer = io.BytesIO()
                    torch.save(optimizer.state_dict(), buffer)
                    buffer.seek(0)
                    saved = optimizer.state_dict()
                    _, restored, _ = build(arm)
                    restored.load_state_dict(torch.load(buffer, weights_only=False))
                    for key, entry in restored.state_dict()["state"].items():
                        for name, value in entry.items():
                            if torch.is_tensor(value):
                                self.assertEqual(value.dtype, saved["state"][key][name].dtype, name)
                                exact(value, saved["state"][key][name])
                    del restored
                del model, optimizer
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
