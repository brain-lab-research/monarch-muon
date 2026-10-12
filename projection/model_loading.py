"""Loading HF causal LMs and the few model-facing helpers the metrics need.

Kept deliberately thin: everything architecture-specific lives in the registry
(names/shapes from the checkpoint). The only structural assumptions made about
a loaded model are the standard HF ones -- `model.base_model` returns
`last_hidden_state` (final-norm output) and `model.get_output_embeddings()` is
the (bias-free) LM head.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

# This code is torch-only; a stray TensorFlow install would otherwise be imported
# (and may crash) by `import transformers`.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
import torch.nn.functional as F

_DTYPES = {
    "float32": torch.float32, "fp32": torch.float32,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
}


def resolve_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(_DTYPES)}") from None


def load_model(
    model_dir: str | Path,
    *,
    dtype: str | torch.dtype = "bfloat16",
    device: str = "cuda",
    trust_remote_code: bool = True,
    accel: str = "none",
):
    """Load a HF causal LM in eval mode. trust_remote_code defaults to True
    because the Moonlight repos ship their own modeling code; pass False to use
    transformers' native DeepseekV3 implementation. accel="grouped_mm" runs
    the MoE experts as one grouped GEMM (moe_accel.py; checked at load time -- if
    the check fails the load stops with SystemExit, the run asked for those numerics)."""
    from transformers import AutoModelForCausalLM

    from hf_compat import apply_cache_compat, limit_logits_during_generation, patch_remote_generation

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=resolve_dtype(dtype),
                                                 trust_remote_code=trust_remote_code)
    if trust_remote_code:                   # the KV-cache API the remote code calls (generation; see hf_compat)
        added, wrapped = apply_cache_compat(), patch_remote_generation(model)
        if added or wrapped:
            print("model: KV cache made usable by the remote code for generation: "
                  + ", ".join(added + (["prepare_inputs_for_generation without the attention-mask crop"] if wrapped else [])))
    limit_logits_during_generation(model)   # generate() keeps the logits of the last position only (memory, batch size)
    # Plain forward passes never need the KV cache (our metrics pass use_cache=False anyway; lm_eval's loglikelihood
    # calls do not say): without it they skip building one -- GBs on long contexts. generate() asks for it itself.
    model.config.use_cache = False
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if accel != "none":
        from moe_accel import ACCEL_CHOICES, apply_moe_accel

        if accel not in ACCEL_CHOICES:
            raise ValueError(f"unknown accel {accel!r}; expected one of {ACCEL_CHOICES}")
        model.moe_accel_report = apply_moe_accel(model)
        if not model.moe_accel_report["active"]:
            # explicitly asked for, not satisfied: stop here rather than run for days on the slow path, or build caches
            # whose numerics are not the ones asked for
            raise SystemExit(
                f"--accel {accel} was requested but is OFF: {model.moe_accel_report['reason']}\n"
                "Nothing was run. Fix the cause (see the 'accel:' lines above), or run without it (ACCEL=none / --accel none).")
    return model


def free_model(model) -> None:
    """Drop a model and release GPU memory (one model at a time)."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def body_and_head(model):
    """(decoder body returning last_hidden_state, LM head module)."""
    head = model.get_output_embeddings()
    if head is None:
        raise ValueError("model has no output embeddings (LM head)")
    return model.base_model, head


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def final_hidden(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Final-norm hidden states (b, T, d) for token ids on the model's input device."""
    body, _ = body_and_head(model)
    return body(input_ids=input_ids.to(input_device(model)), use_cache=False).last_hidden_state


def head_weight_fp32(model) -> torch.Tensor:
    """LM-head weight as an fp32 copy on the head's device. Logits are computed
    in fp32 on purpose: bf16 logits at |z|~10-20 are quantized to ~0.06-0.125,
    which is the same order as the KL differences we are trying to resolve."""
    _, head = body_and_head(model)
    if getattr(head, "bias", None) is not None:
        raise NotImplementedError("LM head with bias is not supported")
    return head.weight.detach().float()


def logits_fp32(hidden: torch.Tensor, head_w: torch.Tensor) -> torch.Tensor:
    return F.linear(hidden.float(), head_w)
