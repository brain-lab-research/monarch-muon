"""Generation (GSM8K in lm_eval) with Moonlight's remote modeling code on transformers 4.57.

The KV cache: Moonlight's modeling_deepseek.py was written against the Cache API of transformers 4.36-4.4x and calls
`cache.get_usable_length(...)` (attention and model forward), `cache.seen_tokens` (prepare_inputs_for_generation),
`DynamicCache.from_legacy_cache` / `to_legacy_cache` (model forward). Newer versions removed them.
`apply_cache_compat()` adds the missing methods (never replaces one).

Separately, its `prepare_inputs_for_generation` crops the attention mask to `max_cache_length`, and two copies of the
file exist (they differ in this one line): one reads `past_key_values.get_max_length()` -- an AttributeError on
transformers 4.57 (the method is gone), None-for-unbounded before; the other reads `get_seq_length()`, the cache length,
so the mask is cropped at EVERY decode step, one column short of what the next layer needs:
"Attention mask should be of size (b, 1, 1, L+1), but is (b, 1, 1, L)". A dynamic cache has no capacity, so nothing should
ever be cropped: `patch_remote_generation(model)` replaces the remote class's own `prepare_inputs_for_generation` by the
same function without that crop (everything else -- which input ids are kept, position_ids, the returned dict -- is the
remote logic), which makes both copies work.

`limit_logits_during_generation(model)` makes generate() compute the LM head on the last position only (memory).
"""
from __future__ import annotations

import functools
import weakref


# ---------------------------------------------------------------------------
# KV cache: the transformers 4.36-4.4x API that modeling_deepseek.py calls
# ---------------------------------------------------------------------------


def _get_usable_length(self, new_seq_length: int, layer_idx: int | None = 0) -> int:
    """Old `Cache.get_usable_length`. Only dynamic (unbounded) caches are used here, so this is the cached length."""
    return self.get_seq_length(layer_idx or 0)


def _seen_tokens(self) -> int:
    return self.get_seq_length()


def _from_legacy_cache(cls, past_key_values=None):
    cache = cls()
    for layer_idx, (key, value) in enumerate(past_key_values or ()):
        cache.update(key, value, layer_idx)
    return cache


def _to_legacy_cache(self):
    return tuple((layer.keys, layer.values) for layer in self.layers)


def apply_cache_compat() -> list[str]:
    """Add the Cache methods the remote code calls and this transformers lacks (never replaces one). Returns what it
    added, as "Class.name"."""
    try:
        from transformers.cache_utils import Cache, DynamicCache
    except Exception:
        return []
    added = []
    for cls, name, value in ((Cache, "get_usable_length", _get_usable_length),
                             (Cache, "seen_tokens", property(_seen_tokens)),
                             (DynamicCache, "from_legacy_cache", classmethod(_from_legacy_cache)),
                             (DynamicCache, "to_legacy_cache", _to_legacy_cache)):
        if not hasattr(cls, name):
            setattr(cls, name, value)
            added.append(f"{cls.__name__}.{name}")
    return added


def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
    """The remote class's own function (Moonlight's modeling_deepseek.py) minus the attention-mask crop at the cache
    length, which is wrong for a cache that has no capacity (see the module docstring). Everything else is the remote
    logic: keep only the not yet processed ids, position_ids from the mask, the same returned dict."""
    if past_key_values is not None:
        if hasattr(past_key_values, "get_seq_length"):
            past_length = past_key_values.get_seq_length()
        else:                                                      # legacy tuple of (key, value) per layer
            past_length = past_key_values[0][0].shape[2]
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # else: input_ids holds only unprocessed tokens already

    position_ids = kwargs.get("position_ids", None)
    if position_ids is not None and past_key_values is not None and position_ids.shape[1] > input_ids.shape[1]:
        # transformers >= 5 hands over position_ids of the whole sequence; the remote code expects the new tokens' only
        position_ids = position_ids[:, -input_ids.shape[1]:]
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1                # positions of a left-padded batch
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    if inputs_embeds is not None and past_key_values is None:             # embeddings only for the first step
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}
    model_inputs.update({"position_ids": position_ids, "past_key_values": past_key_values,
                         "use_cache": kwargs.get("use_cache"), "attention_mask": attention_mask})
    return model_inputs


def patch_remote_generation(model) -> bool:
    """Replace the remote (trust_remote_code) class's own `prepare_inputs_for_generation` by the function above.
    Transformers' own classes are left alone. Idempotent. Returns True when it replaced something."""
    for klass in type(model).__mro__:
        fn = vars(klass).get("prepare_inputs_for_generation")
        if fn is None:
            continue
        if getattr(fn, "_crop_free", False) or not getattr(fn, "__module__", "").startswith("transformers_modules"):
            return False
        @functools.wraps(fn)                                       # (a fresh function: the module-level one stays as is)
        def replacement(self, input_ids, *args, **kwargs):
            return prepare_inputs_for_generation(self, input_ids, *args, **kwargs)
        replacement._crop_free = True
        setattr(klass, "prepare_inputs_for_generation", replacement)
        return True
    return False


# ---------------------------------------------------------------------------
# generate(): logits of the LAST position only
# ---------------------------------------------------------------------------


def limit_logits_during_generation(model) -> bool:
    """generate() only reads the last position's logits, but the remote forward computes lm_head over the whole prompt in
    fp32: batch x prompt_length x vocab (163840) x 4 bytes -- 2 GiB at batch 4 for a 4-shot GSM8K prompt, and the reason
    the generation batch could not grow on a shared GPU. A forward PRE-hook on the output embedding keeps only the last
    position of the hidden states while a generate() call is running (the module, its weight and the state dict are
    untouched; forward passes outside generate(), e.g. lm_eval's loglikelihood, still return every position).
    Idempotent. Returns True when it installed the hook."""
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    generate_fn = getattr(type(model), "generate", None)
    if head is None or generate_fn is None or getattr(model, "_generation_head_state", None) is not None:
        return False                                              # no head to slice / no generate() to wrap / done before
    state = {"active": 0}
    model_ref = weakref.ref(model)               # (a strong reference from the wrapper would make a cycle that keeps the
                                                 #  model -- and its mmapped weights, GPU memory -- alive until a gc pass)

    def keep_last_position(_module, args):
        if state["active"] and args and getattr(args[0], "dim", lambda: 0)() == 3 and args[0].shape[1] > 1:
            return (args[0][:, -1:, :],) + tuple(args[1:])
        return None

    head.register_forward_pre_hook(keep_last_position)

    @functools.wraps(generate_fn)
    def generate_with_last_position_logits(*args, **kwargs):
        state["active"] += 1
        try:
            return generate_fn(model_ref(), *args, **kwargs)
        finally:
            state["active"] -= 1

    model.generate = generate_with_last_position_logits
    model._generation_head_state = state
    return True
