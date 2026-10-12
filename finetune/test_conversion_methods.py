import io
import unittest

import torch
from torch import nn

from conversion_methods import (PermutedMonarchLinear, RotatedMLPBlock, apply_factors,
                                cayley_step, first_adjoint, fit_als, search_permutation,
                                fit_procrustes, install_rotated_mlp)
from torch_projection import MonarchLinear, materialize, project_dense


class ConversionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260929)
        torch.set_num_threads(2)

    def test_forward_backward_and_adjoint(self):
        for m, n in [(8, 8), (8, 16), (16, 8)]:
            first = torch.randn(2, min(m, n) // 2, n // 2, dtype=torch.double, requires_grad=True)
            second = torch.randn(2, m // 2, min(m, n) // 2, dtype=torch.double, requires_grad=True)
            x = torch.randn(7, n, dtype=torch.double, requires_grad=True)
            y = torch.randn(7, m, dtype=torch.double)
            direct = apply_factors(x, first, second)
            dense = x @ materialize(first, second).T
            torch.testing.assert_close(direct, dense)
            actual = torch.autograd.grad((direct * y).sum(), (x, first, second), retain_graph=True)
            expected = torch.autograd.grad((dense * y).sum(), (x, first, second))
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)
            torch.testing.assert_close(first_adjoint(y, x, second), actual[1])

    def test_svd_independent_lower_bound(self):
        for m, n in [(8, 8), (8, 16), (16, 8)]:
            weight = torch.randn(m, n)
            first, second = project_dense(weight, 2)
            rank = min(m, n) // 4
            lower_bound = sum(torch.linalg.svdvals(weight[j::2, i * (n // 2):(i + 1) * (n // 2)])[rank:].square().sum()
                              for i in range(2) for j in range(2))
            torch.testing.assert_close((weight - materialize(first, second)).square().sum(), lower_bound)

    def test_permutations_and_round_trip(self):
        for m, n in [(8, 8), (8, 16), (16, 8)]:
            weight = torch.randn(m, n)
            pi, po = torch.randperm(n), torch.randperm(m)
            first, second = project_dense(weight[po][:, pi], 2)
            bias, x = torch.randn(m), torch.randn(2, 3, n)
            model = PermutedMonarchLinear(first, second, pi, po, bias)
            full = materialize(first, second)[torch.argsort(po)][:, torch.argsort(pi)]
            torch.testing.assert_close(model(x), x @ full.T + bias)
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            clone = PermutedMonarchLinear(first * 0, second * 0, torch.arange(n), torch.arange(m), bias * 0)
            clone.load_state_dict(torch.load(stream, weights_only=True))
            torch.testing.assert_close(model(x), clone(x))
            chosen_in, chosen_out, history = search_permutation(weight, steps=3)
            self.assertEqual(sorted(chosen_in.tolist()), list(range(n)))
            self.assertEqual(sorted(chosen_out.tolist()), list(range(m)))
            self.assertLessEqual(min(r["relative_squared_weight_error"] for r in history), history[0]["relative_squared_weight_error"])

    def test_als_objective_and_heldout_selection(self):
        for m, n in [(8, 8), (8, 16), (16, 8)]:
            weight = torch.randn(m, n)
            covariance = torch.randn(n, n)
            x, val = torch.randn(100, n) @ covariance, torch.randn(40, n) @ covariance
            first, second = project_dense(weight, 2)
            a, b, history = fit_als(weight, x, val, first, second, sweeps=3, cg_steps=100)
            objectives = [r["train_relative_squared_error"] for r in history]
            for old, new in zip(objectives, objectives[1:]):
                self.assertLessEqual(new, old + 2e-5)
            returned = ((apply_factors(val, a, b) - val @ weight.T).square().sum() / (val @ weight.T).square().sum()).item()
            self.assertAlmostEqual(returned, min(r["validation_relative_squared_error"] for r in history[::2]), places=5)
            self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())

    def test_rotated_residual_norm_bias_parity_and_groups(self):
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(8, 16, bias=True).double()
                self.up_proj = nn.Linear(8, 16, bias=True).double()
                self.down_proj = nn.Linear(16, 8, bias=True).double()

            def forward(self, x):
                return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

        norm = nn.RMSNorm(8, eps=1e-6).double()
        norm.weight.data.copy_(torch.randn(8, dtype=torch.double))
        mlp = MLP()
        hidden = torch.randn(2, 7, 8, dtype=torch.double)
        expected = hidden + mlp(norm(hidden))
        qi, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.double))
        qo, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.double))
        with torch.no_grad():
            for name in ("gate_proj", "up_proj"):
                projection = getattr(mlp, name)
                projection.weight.copy_(projection.weight @ qi)
            mlp.down_proj.weight.copy_(qo.T @ mlp.down_proj.weight)
            mlp.down_proj.bias.copy_(mlp.down_proj.bias @ qo)
        block = RotatedMLPBlock(norm, mlp, qi, qo)
        torch.testing.assert_close(block(hidden), expected, atol=1e-12, rtol=1e-12)
        self.assertFalse(any("rotation" in name for name, _ in block.named_parameters()))
        updated = cayley_step(qi, torch.randn_like(qi), 0.2)
        torch.testing.assert_close(updated.T @ updated, torch.eye(8, dtype=torch.double), atol=1e-12, rtol=1e-12)

    def test_qwen_rotations_only_and_backward(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from sft_optimizer import parameter_groups
        config = Qwen2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, vocab_size=64)
        model = Qwen2ForCausalLM(config).double().eval()
        ids = torch.randint(0, 64, (2, 12))
        for layer in model.model.layers:
            layer.post_attention_layernorm.weight.data.copy_(torch.rand(16, dtype=torch.double) + 0.5)
        expected = model(ids, use_cache=False).logits.detach()
        for layer in model.model.layers:
            qi, _ = torch.linalg.qr(torch.randn(16, 16, dtype=torch.double))
            qo, _ = torch.linalg.qr(torch.randn(16, 16, dtype=torch.double))
            install_rotated_mlp(layer, qi, qo)
        torch.testing.assert_close(model(ids, use_cache=False).logits, expected, atol=1e-7, rtol=1e-6)
        model(ids, labels=ids, use_cache=False).loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        groups = parameter_groups(model, "muon", 3e-4, {"muon_adamw_lr": 3e-5})
        parameters = {id(p) for group in groups for p in group["params"]}
        self.assertTrue(all(id(p) not in parameters for name, p in model.named_buffers() if "rotation" in name))

    def test_procrustes_finite_and_orthogonal(self):
        weights = [torch.randn(16, 8), torch.randn(16, 8), torch.randn(8, 16)]
        inputs = [torch.randn(64, w.shape[1]) for w in weights]
        validation = [torch.randn(32, w.shape[1]) for w in weights]
        qi, qo, factors, history = fit_procrustes(weights, inputs, validation,
                                                 frobenius_sweeps=1, outer_sweeps=1,
                                                 rotation_steps=1, als_sweeps=1, cg_steps=60)
        for rotation in (qi, qo):
            torch.testing.assert_close(rotation.T @ rotation, torch.eye(8), atol=3e-6, rtol=3e-6)
        self.assertTrue(all(torch.isfinite(p).all() for pair in factors for p in pair))
        self.assertTrue(all(torch.isfinite(torch.tensor(record["validation_sum_relative_squared_error"])) for record in history))

    def test_external_score_gradients(self):
        for m, n in [(8, 16), (16, 8)]:
            original, approximate = torch.randn(m, n, dtype=torch.double), torch.randn(m, n, dtype=torch.double)
            scores = torch.eye(min(m, n), dtype=torch.double, requires_grad=True)
            prediction = approximate @ scores if n <= m else scores.T @ approximate
            residual = prediction - original
            gradient, = torch.autograd.grad(residual.square().sum() / 2, scores)
            expected = approximate.T @ residual if n <= m else approximate @ residual.T
            torch.testing.assert_close(gradient, expected)

    def test_complete_selected_initialization_and_checkpoint_round_trip(self):
        import copy
        from pathlib import Path
        import tempfile
        from safetensors.torch import load_model
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from conversion_initialization import install_initialization, LAYERS, NAMES
        from sft_model import save_checkpoint
        config = Qwen2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=28,
                             num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                             use_sliding_window=False)
        base = Qwen2ForCausalLM(config).bfloat16().eval()
        ids = torch.randint(0, 64, (1, 8))
        for arm in "ABCD":
            tensors = {}
            for index in LAYERS:
                if arm == "D":
                    tensors[f"rotation.{index}.input"] = torch.linalg.qr(torch.randn(16, 16))[0]
                    tensors[f"rotation.{index}.output"] = torch.linalg.qr(torch.randn(16, 16))[0]
                for name in NAMES:
                    dense = getattr(base.model.layers[index].mlp, name)
                    a, b = project_dense(dense.weight, 2)
                    prefix = f"model.layers.{index}.mlp.{name}"
                    tensors[prefix + ".blkdiag1"], tensors[prefix + ".blkdiag2"] = a.bfloat16(), b.bfloat16()
                    if arm == "C":
                        tensors[prefix + ".input_order"] = torch.randperm(dense.in_features)
                        tensors[prefix + ".output_order"] = torch.randperm(dense.out_features)
            model = copy.deepcopy(base)
            install_initialization(model, arm, tensors)
            self.assertEqual(sum(isinstance(m, (MonarchLinear, PermutedMonarchLinear)) for m in model.modules()), 39)
            self.assertTrue(torch.equal(base.model.layers[14].self_attn.q_proj.weight, model.model.layers[14].self_attn.q_proj.weight))
            expected = model(ids, use_cache=False).logits.detach()
            restored = copy.deepcopy(base)
            install_initialization(restored, arm, tensors)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "complete.safetensors"
                save_checkpoint(model, path)
                load_model(restored, str(path), strict=True)
                torch.testing.assert_close(restored(ids, use_cache=False).logits, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
