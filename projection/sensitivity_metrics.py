"""KL / perplexity of a (patched) model against the cached reference of the original model.

    kl           KL(p_orig || p_compressed), mean over evaluation tokens, nats.
                 p_orig comes from the cached final hidden states pushed through
                 the LM head in fp32 (exact, see calibration_data.py); kl_sem is
                 the standard error over sequences.
    ppl_orig / ppl_comp / dppl_rel   perplexity of the original / compressed model
                 on the same tokens and the relative change ppl_comp/ppl_orig - 1.
    nll_delta    mean per-token NLL change (log-ppl change), + nll_delta_sem.
    top1_agree   fraction of positions where argmax is unchanged.
    final_rel_err  ||h_comp - h_orig||_F / ||h_orig||_F of the final-norm hidden
                 state: the compression error propagated through all remaining
                 blocks to the LM head input.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from calibration_data import ReferenceCache
from model_loading import final_hidden, head_weight_fp32, logits_fp32


@dataclass
class ForwardMetrics:
    kl: float
    kl_sem: float
    ppl_orig: float
    ppl_comp: float
    dppl_rel: float
    nll_delta: float
    nll_delta_sem: float
    top1_agree: float
    final_rel_err: float
    n_tokens: int
    n_sequences: int

    def as_row(self) -> dict:
        return dict(self.__dict__)


_REF_MEMO_BYTES = 2 * 1024 ** 3          # keep the reference final states in RAM up to this size


class ForwardEvaluator:
    """KL / perplexity of the (possibly patched) model, run by its own forward from the tokens, against the cached
    reference states on the first `n_sequences` evaluation sequences."""

    def __init__(
        self,
        model,
        eval_tokens: torch.Tensor,
        cache: ReferenceCache,
        *,
        batch_size: int = 4,
        token_chunk: int = 2048,
    ):
        cache.check_tokens(eval_tokens)
        self.model, self.tokens, self.cache = model, eval_tokens, cache
        self.batch_size, self.token_chunk = batch_size, token_chunk
        self.head_w = head_weight_fp32(model)
        self._refs: tuple[int, list] | None = None                      # (n, [(start, hidden)])

    def _ref_shards(self, n: int):
        if self._refs is not None and self._refs[0] == n:
            yield from self._refs[1]
            return
        shards, total = [], 0
        for start, hidden_ref, _ in self.cache.iter_eval_shards(n):
            shards.append((start, hidden_ref))
            total += hidden_ref.numel() * hidden_ref.element_size()
            yield start, hidden_ref
        if total <= _REF_MEMO_BYTES:
            self._refs = (n, shards)

    def _score_batch(self, h: torch.Tensor, hr: torch.Tensor, ids: torch.Tensor):
        dev = self.head_w.device
        b, T, d = h.shape
        Hc, Hr = h[:, :-1].reshape(-1, d), hr[:, :-1].reshape(-1, d)
        tgt = ids[:, 1:].reshape(-1).to(dev)
        kl_t, nllc_t, nllr_t, agree_t = [], [], [], []
        for c in range(0, Hc.shape[0], self.token_chunk):
            lc = logits_fp32(Hc[c:c + self.token_chunk].to(dev), self.head_w)
            lr = logits_fp32(Hr[c:c + self.token_chunk].to(dev), self.head_w)
            lpc, lpr = F.log_softmax(lc, -1), F.log_softmax(lr, -1)
            kl_t.append((lpr.exp() * (lpr - lpc)).sum(-1))
            t = tgt[c:c + self.token_chunk].unsqueeze(-1)
            nllc_t.append(-lpc.gather(-1, t).squeeze(-1))
            nllr_t.append(-lpr.gather(-1, t).squeeze(-1))
            agree_t.append((lc.argmax(-1) == lr.argmax(-1)).float())
        per_seq = lambda xs: torch.cat(xs).reshape(b, T - 1).mean(1).cpu()
        return per_seq(kl_t), per_seq(nllc_t), per_seq(nllr_t), per_seq(agree_t)

    @torch.no_grad()
    def evaluate(self, n_sequences: int | None = None) -> ForwardMetrics:
        n_use = min(n_sequences or self.cache.n_eval, self.cache.n_eval)
        seq_kl, seq_nll_c, seq_nll_r, seq_agree = [], [], [], []
        err_sq, ref_sq, n_tok = 0.0, 0.0, 0
        for start, hidden_ref in self._ref_shards(n_use):
            for b0 in range(0, hidden_ref.shape[0], self.batch_size):
                nb = min(self.batch_size, hidden_ref.shape[0] - b0)
                lo = start + b0
                ids = self.tokens[lo: lo + nb]
                h = final_hidden(self.model, ids)                    # (b, T, d) compressed
                hr = hidden_ref[b0: b0 + nb].to(h.device)
                err_sq += float((h.float() - hr.float()).pow(2).sum())
                ref_sq += float(hr.float().pow(2).sum())
                kl_s, nc_s, nr_s, ag_s = self._score_batch(h, hr, ids)
                seq_kl.append(kl_s); seq_nll_c.append(nc_s); seq_nll_r.append(nr_s); seq_agree.append(ag_s)
                n_tok += nb * (h.shape[1] - 1)

        kl, nc, nr, ag = (torch.cat(x).double() for x in (seq_kl, seq_nll_c, seq_nll_r, seq_agree))
        n = kl.numel()
        sem = lambda x: float(x.std(unbiased=True) / math.sqrt(n)) if n > 1 else float("nan")
        ppl_o, ppl_c = float(nr.mean().exp()), float(nc.mean().exp())
        return ForwardMetrics(
            kl=float(kl.mean()), kl_sem=sem(kl),
            ppl_orig=ppl_o, ppl_comp=ppl_c, dppl_rel=ppl_c / ppl_o - 1.0,
            nll_delta=float((nc - nr).mean()), nll_delta_sem=sem(nc - nr),
            top1_agree=float(ag.mean()),
            final_rel_err=math.sqrt(err_sq / ref_sq) if ref_sq > 0 else float("nan"),
            n_tokens=n_tok, n_sequences=n,
        )

    def baseline_check(self, n_sequences: int = 4, *, kl_tol: float = 5e-3) -> ForwardMetrics:
        """Evaluate the UNPATCHED model through the same code path as the study: KL must be ~0 and the
        recomputed reference NLL must match the cached one. Catches a wrong model/tokens/dtype before a
        long run, and measures the numerical noise floor (batch-size / kernel nondeterminism) that real
        points have to be compared with."""
        m = self.evaluate(n_sequences)
        cached = []
        for _start, _h, nll in self.cache.iter_eval_shards(n_sequences):
            cached.append(nll.mean(1))
        cached_nll = float(torch.cat(cached).double().mean())
        recomputed = math.log(m.ppl_orig)
        if abs(cached_nll - recomputed) > 5e-2 or m.kl > kl_tol:
            raise RuntimeError(
                f"baseline check failed: KL(orig||unpatched model)={m.kl:.3e} (tol {kl_tol:.0e}), "
                f"cached mean NLL {cached_nll:.4f} vs recomputed {recomputed:.4f}. The reference cache does "
                "not match this model/dtype/device -- rebuild it with prepare_calibration.py using exactly "
                "these settings."
            )
        return m
