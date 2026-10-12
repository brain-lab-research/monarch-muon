from pathlib import Path
import tempfile
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from sft_model import replace_projections, save_checkpoint
from sft_optimizer import SFTOptimizer, parameter_groups


class SFTTests(unittest.TestCase):
    def test_adamw_matches_fp32_master_reference(self):
        torch.manual_seed(7)
        p = torch.nn.Parameter(torch.randn(8, 12).bfloat16())
        reference = torch.nn.Parameter(p.detach().float().clone())
        actual = SFTOptimizer([p], lr=3e-5)
        expected = torch.optim.AdamW([reference], lr=3e-5, betas=(0.9, 0.95), weight_decay=0.1)
        for _ in range(5):
            g = torch.randn_like(p)
            p.grad, reference.grad = g, g.float()
            actual.step()
            expected.step()
            torch.testing.assert_close(p, reference.bfloat16(), atol=0, rtol=0)
        self.assertEqual(actual.state[p]["master"].dtype, torch.float32)

    def test_small_updates_accumulate_in_master(self):
        p = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
        optimizer = SFTOptimizer([p], lr=1e-4, weight_decay=0)
        for _ in range(30):
            p.grad = torch.ones_like(p)
            optimizer.step()
        self.assertLess(float(p.detach().min()), 1.0)

    def test_qwen_scope_reload_and_gradients(self):
        from safetensors.torch import load_model
        config = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, tie_word_embeddings=False)
        model = Qwen2ForCausalLM(config).eval()
        embed = model.model.embed_tokens.weight.detach().clone()
        head = model.lm_head.weight.detach().clone()
        records = replace_projections(model, 2, project=True)
        self.assertEqual(len(records), 14)
        torch.testing.assert_close(model.model.embed_tokens.weight, embed, rtol=0, atol=0)
        torch.testing.assert_close(model.lm_head.weight, head, rtol=0, atol=0)
        ids = torch.randint(0, 64, (1, 16))
        before = model(ids).logits.detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_checkpoint(model, path)
            restored = Qwen2ForCausalLM(config).eval()
            replace_projections(restored, 2, project=False)
            load_model(restored, str(path))
            torch.testing.assert_close(restored(ids).logits, before, rtol=0, atol=0)
        model(ids, labels=ids).loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        groups = parameter_groups(model, "muon", 1e-3, {"muon_adamw_lr": 3e-5})
        self.assertEqual(len(groups[0]["params"]), 28)
        self.assertTrue(all(p.ndim == 3 for p in groups[0]["params"]))


if __name__ == "__main__":
    unittest.main()
