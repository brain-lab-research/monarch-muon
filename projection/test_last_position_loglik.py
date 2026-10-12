"""run_downstream_eval.last_position_hflm: the loglikelihood that applies the LM head only at the continuation positions
must return what lm_eval's full-logits path returns -- (sum of continuation log-probs, greedy match) per request, in
request order -- for mixed lengths, multi-token continuations, shared inputs (multiple-choice) and left truncation.
CPU, toy model, lm_eval's HFLM replaced by a minimal base class (lm_eval is not installed here)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from toy_model import make_toy_model


class _FakeHFLM:
    """The attributes of lm_eval's HFLM that the subclass uses."""

    def __init__(self, pretrained, tokenizer=None, batch_size=1):
        self.model, self.batch_size = pretrained, batch_size
        self.backend, self.max_length, self.rank = "causal", 12, 0
        self.cache_hook = types.SimpleNamespace(add_partial=lambda *a: None)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        raise AssertionError("the fast path must not fall back for an integer batch size")


_STUBBED = ("lm_eval", "lm_eval.models", "lm_eval.models.huggingface")


@torch.no_grad()
def _reference(model, requests, max_length):
    """lm_eval's causal semantics, one request at a time, full logits."""
    out = []
    for _, ctx, cont in requests:
        inp = (ctx + cont)[-(max_length + 1):][:-1]
        logp = F.log_softmax(model(torch.tensor([inp])).logits[0].float(), dim=-1)
        sel = logp[len(inp) - len(cont):len(inp)]
        tgt = torch.tensor(cont)
        out.append((float(sel.gather(-1, tgt[:, None]).sum()), bool((sel.argmax(-1) == tgt).all())))
    return out


def test_matches_full_logits():
    import run_downstream_eval

    saved = {k: sys.modules.get(k) for k in _STUBBED}
    hf = types.ModuleType("lm_eval.models.huggingface")
    hf.HFLM = _FakeHFLM
    sys.modules.update({"lm_eval": types.ModuleType("lm_eval"), "lm_eval.models": types.ModuleType("lm_eval.models"),
                        "lm_eval.models.huggingface": hf})
    run_downstream_eval._LAST_POSITION_HFLM = None
    try:
        _check(run_downstream_eval.last_position_hflm())
    finally:
        run_downstream_eval._LAST_POSITION_HFLM = None
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _check(cls):
    model = make_toy_model(seed=0).eval()
    g = torch.Generator().manual_seed(1)
    tok = lambda n: torch.randint(0, 128, (n,), generator=g).tolist()
    ctx_a = tok(7)
    requests = [(("q", c), ctx_a, c) for c in ([5], [9], [17], [33])]          # four choices sharing one input
    requests += [(("x", "y"), tok(3), tok(2)), (("x", "z"), tok(10), tok(4))]    # multi-token, the second one truncated
    requests += [(("p", "r"), ctx_a[:-1], [ctx_a[-1], 5]), (("p", "s"), tok(1), tok(1))]  # 2-token cont., input = ctx_a
    for bs in (1, 3, 16):
        lm = cls(model, batch_size=bs)
        got = lm._loglikelihood_tokens(requests)
        ref = _reference(model, requests, lm.max_length)
        for (a, ga), (b, gb) in zip(got, ref, strict=True):
            assert abs(a - b) < 1e-4 and ga == gb, (bs, got, ref)


TESTS = [("last-position loglikelihood == full-logits lm_eval semantics", test_matches_full_logits)]


def main():
    for name, fn in TESTS:
        fn()
        print(f"ok   {name}")


if __name__ == "__main__":
    main()
