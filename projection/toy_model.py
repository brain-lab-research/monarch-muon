"""Tiny randomly initialized DeepSeek-V3-style model (MLA + MoE) for tests and
smoke runs -- CPU, a few hundred thousand parameters, seconds to build.

Uses transformers' native DeepseekV3 classes (the same module/tensor names as
the Moonlight checkpoint: self_attn.{q_proj|q_a_proj,q_b_proj,kv_a_proj_with_mqa,
kv_b_proj,o_proj}, mlp.{gate_proj,...} / mlp.shared_experts.* / mlp.experts.N.*,
mlp.gate). Dimensions are deliberately small.
"""
from __future__ import annotations

import os
from pathlib import Path

# This code is torch-only; a stray TensorFlow install (built against another numpy ABI)
# would otherwise be imported -- and crash -- by `import transformers`.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch


def make_toy_config(
    *,
    q_lora_rank: int | None = None,
    num_hidden_layers: int = 3,
    first_k_dense_replace: int = 1,
    n_routed_experts: int = 4,
    vocab_size: int = 128,
    hidden_size: int = 64,
):
    from transformers import DeepseekV3Config

    return DeepseekV3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=96,
        moe_intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=n_routed_experts,
        routed_scaling_factor=1.0,
        kv_lora_rank=16,
        q_lora_rank=q_lora_rank,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=16,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=first_k_dense_replace,
        norm_topk_prob=True,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )


def make_toy_model(seed: int = 0, dtype: torch.dtype = torch.float32, **config_kwargs):
    """Randomly initialized toy DeepseekV3ForCausalLM in eval mode. The init is
    made less degenerate than the default 0.02 std (router + projections get
    unit-ish scale) so that perturbing a matrix visibly changes the logits."""
    from transformers import DeepseekV3ForCausalLM

    torch.manual_seed(seed)
    cfg = make_toy_config(**config_kwargs)
    model = DeepseekV3ForCausalLM(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and "embed" not in name:
                p.copy_(torch.randn_like(p) * (1.0 / p.shape[1] ** 0.5))
            elif p.ndim == 2:
                p.copy_(torch.randn_like(p) * 0.5)
    model = model.to(dtype)
    model.eval()
    return model


def save_toy_checkpoint(model, out_dir: str | Path) -> Path:
    """Write config.json + model.safetensors (the layout the registry reads)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    return out_dir


def random_tokens(n_sequences: int, seq_len: int, vocab_size: int = 128, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    # avoid the special ids 0..2 so padding/eos handling never interferes
    return torch.randint(3, vocab_size, (n_sequences, seq_len), generator=g, dtype=torch.int64)
