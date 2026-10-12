import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    return torch.stack((cos_freqs, sin_freqs), dim=-1)


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape[:-1] == (x.shape[1], x.shape[-2])
    shape = [
        1 if i != 1 and i != ndim - 2 else d for i, d in enumerate(x.shape[:-1])
    ] + [2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(q, k, freqs_cis):
    q_dtype = q.dtype
    k_dtype = k.dtype
    q = q.float().reshape(*q.shape[:-1], -1, 2)
    k = k.float().reshape(*k.shape[:-1], -1, 2)

    freqs_cis = _reshape_for_broadcast(freqs_cis, q)

    q_cos = q[..., 0] * freqs_cis[..., 0] - q[..., 1] * freqs_cis[..., 1]
    q_sin = q[..., 0] * freqs_cis[..., 1] + q[..., 1] * freqs_cis[..., 0]
    k_cos = k[..., 0] * freqs_cis[..., 0] - k[..., 1] * freqs_cis[..., 1]
    k_sin = k[..., 0] * freqs_cis[..., 1] + k[..., 1] * freqs_cis[..., 0]

    q_out = torch.stack((q_cos, q_sin), dim=-1).reshape(q.shape).flatten(3)
    k_out = torch.stack((k_cos, k_sin), dim=-1).reshape(k.shape).flatten(3)

    return q_out.to(dtype=q_dtype), k_out.to(dtype=k_dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight.to(dtype=output.dtype)


def _mlp_hidden_dim(config) -> int:
    hidden_dim = config.n_embd * 4
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = config.multiple_of * (
        (hidden_dim + config.multiple_of - 1) // config.multiple_of
    )
    return hidden_dim


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = _mlp_hidden_dim(config)
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)


class LlamaBlock(nn.Module):
    """Unified transformer block. BF16 or FP8 path is selected at runtime."""

    def __init__(self, config, qargs=None):
        super().__init__()
        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Canonical construction order — identical across BF16 and FP8:
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.attn = Attention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.mlp  = MLP(config)

        self.fp8 = qargs is not None
        self.qargs = qargs

    def _project_qkv(self, x):
        B, T, C = x.size()
        x_norm = self.ln_1(x)
        q = self.attn.q_proj(x_norm)
        k = self.attn.k_proj(x_norm)
        v = self.attn.v_proj(x_norm)
        return x_norm, q, k, v

    def _non_fp8_forward(self, x, freqs_cis):
        B, T, C = x.size()
        _, q, k, v = self._project_qkv(x)

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        x = x + self.attn.o_proj(y)

        h = self.ln_2(x)
        hidden = F.silu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h)
        x = x + self.mlp.down_proj(hidden)
        return x

    def _attend(self, q, k, v, freqs_cis):
        B, T, _ = q.size()
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        return y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)

    def _coat_monarch_forward(self, x, Qx, Sx, freqs_cis):
        from models.coat_monarch import AfterAttention, BeforeAttention, MLPResidual

        attn, mlp, gs = self.attn, self.mlp, self.qargs.group_size
        qkv = (attn.q_proj, attn.k_proj, attn.v_proj)
        residual, q, k, v = BeforeAttention.apply(
            x, Qx, Sx, self.ln_1.weight, self.ln_1.eps, gs, qkv,
            *(w for m in qkv for w in (m.blkdiag1, m.blkdiag2)))
        hidden, Qx, Sx = AfterAttention.apply(
            residual, self._attend(q, k, v, freqs_cis), gs, attn.o_proj, attn.o_proj.blkdiag1, attn.o_proj.blkdiag2)
        ffn = (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
        return MLPResidual.apply(
            hidden, Qx, Sx, self.ln_2.weight, self.ln_2.eps, gs, ffn,
            *(w for m in ffn for w in (m.blkdiag1, m.blkdiag2)))

    def forward(self, x, Qx, Sx, freqs_cis):
        if self.fp8 and self.training:
            return self._coat_monarch_forward(x, Qx, Sx, freqs_cis)
        return self._non_fp8_forward(x, freqs_cis), None, None


class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.fp8 = bool(getattr(config, "fp8", False))
        qargs = getattr(config, "qargs", None) if self.fp8 else None
        if self.fp8:
            assert qargs is not None, (
                "Llama with --fp8 requires a QuantizationConfig on args.qargs "
                "(built by src/main.py when --fp8 is passed)."
            )
        self.qargs = qargs

        self.head_dim = config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(0.0),
            h    = nn.ModuleList([
                LlamaBlock(config, qargs) for _ in range(config.n_layer)
            ]),
            ln_f = RMSNorm(config.n_embd, eps=config.rmsnorm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # The FP8-only modules below have no nn.Parameter children and are registered
        # after the init, so they cannot perturb RNG consumption.
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0,
                    std=self.config.init_std / math.sqrt(2 * config.n_layer),
                )

        if self.fp8:
            from third_party.coat.activation.real_quantization import (
                Coat_quantize_bgn, Coat_quantize_end,
            )
            self.quantize_input  = Coat_quantize_bgn(qargs)
            self.quantize_output = Coat_quantize_end(qargs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def get_num_params(self, non_embedding=True):
        return sum(p.numel() for p in self.parameters())

    def get_parameter_group_specs(self):
        """
        Separate parameters into weight-decayed and non-decayed groups.
        Linear and Monarch weights are decayed; norms, embeddings and lm_head are not.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (RMSNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn in ("blkdiag1", "blkdiag2"):
                    decay.add(fpn)
                elif pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        decay.discard("lm_head.weight")
        no_decay.add("lm_head.weight")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        return [
            {"params": sorted(list(decay)), "is_proj_params": True},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0, "is_proj_params": False},
        ]

    def forward(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.sequence_length, (
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.config.sequence_length}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        x = self.transformer.drop(self.transformer.wte(idx))
        freqs_cis = self.freqs_cis.to(x.device)[pos]

        if self.fp8:
            x, Qx, Sx = self.quantize_input(x)
        else:
            Qx, Sx = None, None

        for block in self.transformer.h:
            x, Qx, Sx = block(x, Qx, Sx, freqs_cis)

        if self.fp8:
            x = self.quantize_output(x, Qx, Sx)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return {
            "logits": logits if get_logits else None,
            "loss": loss,
        }
