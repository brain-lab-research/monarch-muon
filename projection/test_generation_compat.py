"""Generation (KV cache) with Moonlight's remote modeling code on transformers 4.57 -- hf_compat's generation shims (GSM8K).
CPU, seconds.

    python test_generation_compat.py

Moonlight's modeling_deepseek.py calls the Cache API of transformers 4.36-4.4x (get_usable_length, seen_tokens,
from/to_legacy_cache), and its prepare_inputs_for_generation crops the attention mask at the CACHE length on every decode
step (`max_cache_length = past_key_values.get_seq_length()` in one copy of the file: "Attention mask should be of size
(b, 1, 1, L+1), but is (b, 1, 1, L)"; the other copy reads `get_max_length()` there, which transformers 4.57 no longer
has). The function below is that copy's logic, transcribed, so that the test does not need the checkpoint.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from hf_compat import apply_cache_compat, patch_remote_generation, prepare_inputs_for_generation


def remote_prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
                                         **kwargs):
    """DeepseekV3ForCausalLM.prepare_inputs_for_generation of Moonlight's modeling_deepseek.py (the original, with its
    max_cache_length = get_seq_length())."""
    from transformers.cache_utils import Cache

    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_seq_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        if max_cache_length is not None and attention_mask is not None and cache_length + input_ids.shape[1] > max_cache_length:
            attention_mask = attention_mask[:, -max_cache_length:]
    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}
    model_inputs.update({"position_ids": position_ids, "past_key_values": past_key_values,
                         "use_cache": kwargs.get("use_cache"), "attention_mask": attention_mask})
    return model_inputs


remote_prepare_inputs_for_generation.__module__ = "transformers_modules.Moonlight.modeling_deepseek"


def filled_cache(n_tokens=5, layers=2):
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for layer in range(layers):
        cache.update(torch.randn(1, 2, n_tokens, 4), torch.randn(1, 2, n_tokens, 3), layer)     # MLA: K and V dims differ
    return cache


def test_missing_cache_methods_are_added():
    from transformers.cache_utils import Cache, DynamicCache

    apply_cache_compat()
    assert apply_cache_compat() == [], "idempotent"
    for cls, name in ((Cache, "get_usable_length"), (Cache, "seen_tokens"), (DynamicCache, "from_legacy_cache"),
                      (DynamicCache, "to_legacy_cache")):
        assert hasattr(cls, name), name
    cache = filled_cache()
    assert cache.get_usable_length(3) == 5 and cache.get_usable_length(1, 1) == 5 and cache.seen_tokens == 5
    legacy = cache.to_legacy_cache()
    assert len(legacy) == 2 and legacy[0][0].shape == (1, 2, 5, 4) and legacy[0][1].shape == (1, 2, 5, 3)
    back = DynamicCache.from_legacy_cache(legacy)
    assert back.get_seq_length() == 5 and torch.equal(back.to_legacy_cache()[1][1], legacy[1][1])
    assert DynamicCache.from_legacy_cache(None).get_seq_length() == 0


def test_replacement_keeps_the_mask_that_the_remote_function_crops():
    apply_cache_compat()

    class FakeRemoteModel:
        prepare_inputs_for_generation = remote_prepare_inputs_for_generation

    ids, mask = torch.arange(6)[None], torch.ones(1, 6, dtype=torch.long)
    cache = filled_cache(5)                                                   # 5 cached tokens, 1 new
    original = FakeRemoteModel().prepare_inputs_for_generation(ids, past_key_values=cache, attention_mask=mask)
    assert original["attention_mask"].shape[1] == 5, "the bug: the mask is cropped to the cache length (needs 6)"

    assert patch_remote_generation(FakeRemoteModel()) and not patch_remote_generation(FakeRemoteModel()), "idempotent"
    assert FakeRemoteModel.prepare_inputs_for_generation is not prepare_inputs_for_generation
    assert prepare_inputs_for_generation.__module__ == "hf_compat", "the module-level function is not mutated"
    out = FakeRemoteModel().prepare_inputs_for_generation(ids, past_key_values=cache, attention_mask=mask)
    assert out["attention_mask"].shape[1] == 6 and out["input_ids"].tolist() == [[5]] and out["position_ids"].tolist() == [[5]], out
    assert out["use_cache"] is None and set(out) == {"input_ids", "position_ids", "past_key_values", "use_cache", "attention_mask"}

    # left padding: positions of a padded row
    ids2, mask2 = torch.arange(12).reshape(2, 6), torch.ones(2, 6, dtype=torch.long)
    mask2[1, :2] = 0
    out = FakeRemoteModel().prepare_inputs_for_generation(ids2, past_key_values=filled_cache(5), attention_mask=mask2)
    assert out["input_ids"].shape == (2, 1) and out["position_ids"].tolist() == [[5], [3]], out["position_ids"]
    first = FakeRemoteModel().prepare_inputs_for_generation(ids2, past_key_values=None, attention_mask=mask2)    # prefill
    assert first["input_ids"].shape == (2, 6) and first["position_ids"].tolist()[1] == [1, 1, 0, 1, 2, 3]

    # transformers >= 5 passes full-length position_ids: only the new tokens' are kept
    full_pos = torch.arange(6)[None]
    out = FakeRemoteModel().prepare_inputs_for_generation(ids, past_key_values=cache, attention_mask=mask, position_ids=full_pos)
    assert out["position_ids"].tolist() == [[5]], "sliced to the new token"
    out = FakeRemoteModel().prepare_inputs_for_generation(ids, past_key_values=filled_cache(0), attention_mask=mask, position_ids=full_pos)
    assert out["position_ids"].shape == (1, 6), "prefill keeps them all"

    # a legacy tuple cache and inputs_embeds
    legacy = tuple((torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 3)) for _ in range(2))
    out = FakeRemoteModel().prepare_inputs_for_generation(ids, past_key_values=legacy, attention_mask=mask)
    assert out["input_ids"].tolist() == [[5]]
    emb = torch.randn(1, 6, 8)
    assert "inputs_embeds" in FakeRemoteModel().prepare_inputs_for_generation(ids, inputs_embeds=emb, attention_mask=mask)

    class Library:                                     # transformers' own prepare_inputs is never touched
        def prepare_inputs_for_generation(self, input_ids, **kw):
            return {}
    Library.prepare_inputs_for_generation.__module__ = "transformers.generation.utils"
    assert not patch_remote_generation(Library())


def test_load_model_turns_the_forward_cache_off():
    from model_loading import load_model
    from toy_model import make_toy_model, save_toy_checkpoint

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        path = save_toy_checkpoint(make_toy_model(seed=0), Path(d) / "ckpt")
        model = load_model(path, dtype="float32", device="cpu", trust_remote_code=False)
        assert model.config.use_cache is False, "plain forward passes (lm_eval loglikelihood) build no KV cache"
        ids = torch.randint(0, 128, (1, 8))
        out = model(input_ids=ids)
        assert getattr(out, "past_key_values", None) is None
        gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=3, do_sample=False,
                             use_cache=True, pad_token_id=0)
        assert gen.shape == (1, 11), "generate() still works and uses its own cache"


def test_generate_computes_the_head_for_the_last_position_only():
    """The remote forward runs lm_head over the whole prompt in fp32 (batch x prompt x 163840 x 4 bytes); generate() reads the
    last position only. The hook keeps one position while a generate() call runs -- and only then: loglikelihood-style forward
    passes still get every position. (Stub: transformers' own classes pass logits_to_keep and need no hook.)"""
    from hf_compat import limit_logits_during_generation

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(6, 11, bias=False)
            self.steps = []

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, hidden):                                    # what DeepseekV3ForCausalLM.forward does with the head
            return self.lm_head(hidden).float()

        def generate(self, hidden, fail=False, nested=False):
            out = self(hidden)
            self.steps.append(tuple(out.shape))
            if nested:
                self.steps.append(tuple(self(hidden).shape))
            if fail:
                raise RuntimeError("generation failed")
            return out[:, -1, :]

    model = Stub()
    h = torch.randn(3, 9, 6)
    with torch.no_grad():
        full = model(h)
        assert limit_logits_during_generation(model) is True and limit_logits_during_generation(model) is False, "idempotent"
        assert torch.equal(model(h), full) and model(h).shape == (3, 9, 11), "outside generate(): every position"
        last = model.generate(h)
        assert model.steps[-1] == (3, 1, 11) and torch.allclose(last, full[:, -1, :], atol=1e-6), "inside: one position, the same logits"
        model.generate(h, nested=True)
        assert model.steps[-2:] == [(3, 1, 11), (3, 1, 11)]
        try:
            model.generate(h, fail=True)
        except RuntimeError:
            pass
        assert model.lm_head(h).shape == (3, 9, 11) and model._generation_head_state["active"] == 0, "restored after an exception"
        assert model(torch.randn(3, 6)).shape == (3, 11), "2-D inputs are left alone"
        assert model(torch.randn(3, 1, 6)).shape == (3, 1, 11)
    assert list(model.state_dict()) == ["lm_head.weight"], "no module was replaced: the state dict and the head weight are as before"


def test_load_model_installs_the_generation_head_hook_and_generation_is_unchanged():
    from model_loading import load_model
    from toy_model import make_toy_model, save_toy_checkpoint

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        path = save_toy_checkpoint(make_toy_model(seed=0), Path(d) / "ckpt")
        model = load_model(path, dtype="float32", device="cpu", trust_remote_code=False)
        assert model._generation_head_state is not None
        ids = torch.randint(3, 128, (3, 11), generator=torch.Generator().manual_seed(2))
        mask = torch.ones_like(ids)
        mask[1, :4] = 0
        kw = dict(attention_mask=mask, max_new_tokens=5, do_sample=False, pad_token_id=0)
        with torch.no_grad():
            with_hook = model.generate(input_ids=ids, **kw)
            model._generation_head_state["active"] = 0
            plain = type(model).generate(model, input_ids=ids, **kw)          # the library's generate, without the wrapper
        assert torch.equal(with_hook, plain) and model.get_output_embeddings().weight is model.lm_head.weight


TESTS = [
    ("missing Cache methods are added", test_missing_cache_methods_are_added),
    ("replacement of the remote prepare_inputs_for_generation: no mask crop, position_ids, padding, legacy cache", test_replacement_keeps_the_mask_that_the_remote_function_crops),
    ("load_model: forward passes without KV cache, generate still works", test_load_model_turns_the_forward_cache_off),
    ("generate(): LM head on the last position only; forward passes untouched; same tokens and scores", test_generate_computes_the_head_for_the_last_position_only),
    ("load_model installs the hook; generation unchanged", test_load_model_installs_the_generation_head_hook_and_generation_is_unchanged),
]


def main():
    import transformers

    print(f"transformers {transformers.__version__}")
    for i, (title, fn) in enumerate(TESTS, 1):
        print(f"=== Test {i}: {title} ===")
        fn()
        print("PASS\n")
    print("ALL GENERATION COMPAT TESTS PASSED")


if __name__ == "__main__":
    main()
