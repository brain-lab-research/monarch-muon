"""Tests for calibration_data.py and sensitivity_metrics.py on a tiny random MLA+MoE model.

    python test_calibration_and_metrics.py

Every number the pipeline reports is re-derived here by an independent route
(the model's own logits and hidden states), not by the code
under test.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from calibration_data import (
    ReferenceCache,
    compute_reference_cache,
    load_tokens,
    parse_source_spec,
    save_tokens,
    tokenize_corpus,
)
from matrix_patching import MatrixPatcher
from model_registry import build_registry, shapes_from_model
from monarch_blocks import fit_svd
from sensitivity_metrics import ForwardEvaluator
from toy_model import make_toy_model, random_tokens

N_CALIB, N_EVAL, T = 6, 6, 16

_STATE: dict = {}


def setup():
    if "model" in _STATE:
        return _STATE
    model = make_toy_model(seed=0)
    reg = build_registry(model.config.to_dict(), shapes_from_model(model))
    toks = random_tokens(N_CALIB + N_EVAL, T, seed=3)
    calib, ev = toks[:N_CALIB], toks[N_CALIB:]
    tmp = tempfile.mkdtemp()
    compute_reference_cache(model, ev, tmp, batch_size=2, shard_size=4, token_chunk=32)
    _STATE.update(model=model, reg=reg, calib=calib, ev=ev, dir=tmp, cache=ReferenceCache(tmp))
    return _STATE


def test_reference_cache_matches_the_models_own_outputs():
    s = setup()
    model, ev, cache = s["model"], s["ev"], s["cache"]
    assert cache.meta["n_eval"] == N_EVAL and cache.meta["d_model"] == model.config.hidden_size
    with torch.no_grad():
        out = model(ev, output_hidden_states=True)
    starts, hid, nll = [], [], []
    for start, h, l in cache.iter_eval_shards():
        starts.append(start); hid.append(h); nll.append(l)
    assert starts == [0, 4]                                     # shard_size=4 over 6 sequences
    hid, nll = torch.cat(hid), torch.cat(nll)
    # the final-norm hidden state is what the LM head consumes
    assert torch.allclose(model.lm_head(hid), out.logits, atol=1e-5)
    expect = F.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.shape[-1]), ev[:, 1:].reshape(-1), reduction="none")
    assert torch.allclose(nll.reshape(-1), expect, atol=1e-5)
    # truncation to the first n sequences
    assert sum(h.shape[0] for _, h, _ in cache.iter_eval_shards(5)) == 5
    # tokens must match the ones the cache was built from
    cache.check_tokens(ev)
    try:
        cache.check_tokens(ev.flip(0))
    except ValueError:
        pass
    else:
        raise AssertionError("a different token set must be rejected")


def test_baseline_check_and_kl_against_independent_computation():
    s = setup()
    model, reg, ev, cache = s["model"], s["reg"], s["ev"], s["cache"]
    evaluator = ForwardEvaluator(model, ev, cache, batch_size=2, token_chunk=32)

    base = evaluator.baseline_check(n_sequences=4)
    assert base.kl < 1e-9 and abs(base.dppl_rel) < 1e-9 and base.top1_agree == 1.0 and base.final_rel_err < 1e-9

    with torch.no_grad():
        logits_orig = model(ev).logits
    u = reg.unit(1, "attn.kv_b_proj")
    ref = u.refs[0]
    W_hat = fit_svd(ref.read(model)[None], 2).dense[0]
    patcher = MatrixPatcher(model)
    with patcher.patched([(ref, W_hat)]):
        m = evaluator.evaluate()
        with torch.no_grad():
            logits_new = model(ev).logits

    lp_o, lp_n = F.log_softmax(logits_orig[:, :-1], -1), F.log_softmax(logits_new[:, :-1], -1)
    kl_ref = (lp_o.exp() * (lp_o - lp_n)).sum(-1)                         # (n, T-1)
    nll_o = -lp_o.gather(-1, ev[:, 1:].unsqueeze(-1)).squeeze(-1)
    nll_n = -lp_n.gather(-1, ev[:, 1:].unsqueeze(-1)).squeeze(-1)
    assert m.n_sequences == N_EVAL and m.n_tokens == N_EVAL * (T - 1)
    assert abs(m.kl - float(kl_ref.mean())) < 1e-6, (m.kl, float(kl_ref.mean()))
    per_seq = kl_ref.mean(1)
    assert abs(m.kl_sem - float(per_seq.std() / math.sqrt(N_EVAL))) < 1e-7
    assert abs(m.ppl_orig - float(nll_o.mean().exp())) < 1e-4
    assert abs(m.ppl_comp - float(nll_n.mean().exp())) < 1e-4
    assert abs(m.nll_delta - float((nll_n - nll_o).mean())) < 1e-6
    agree = (logits_orig[:, :-1].argmax(-1) == logits_new[:, :-1].argmax(-1)).float().mean()
    assert abs(m.top1_agree - float(agree)) < 1e-6          # fp32 per-sequence means averaged in fp64
    assert m.kl > 0

    # original values back in => exactly zero; a destroyed matrix is worse than its Monarch fit
    with patcher.patched([(ref, ref.read(model))]):
        assert evaluator.evaluate().kl < 1e-9
    with patcher.patched([(ref, torch.zeros(u.shape))]):
        assert evaluator.evaluate().kl > m.kl
    # first-n evaluation uses only those sequences
    assert evaluator.evaluate(2).n_sequences == 2


def test_baseline_check_catches_a_mismatched_model():
    s = setup()
    other = make_toy_model(seed=99)
    evaluator = ForwardEvaluator(other, s["ev"], s["cache"], batch_size=2, token_chunk=32)
    try:
        evaluator.baseline_check(n_sequences=4)
    except RuntimeError as exc:
        assert "baseline check failed" in str(exc)
    else:
        raise AssertionError("a cache built from a different model must fail the baseline check")


def test_corpus_tokenization_packing_and_split():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, tag in (("a", 5), ("b", 7)):
            with open(d / f"{name}.jsonl", "w") as f:
                for i in range(200):
                    f.write(json.dumps({"text": f"{tag} " * (3 + i % 11)}) + "\n")
        srcs = [parse_source_spec(f"name=a,jsonl={d / 'a.jsonl'},weight=3"),
                parse_source_spec(f"name=b,jsonl={d / 'b.jsonl'},weight=1")]
        encode = lambda text: [int(t) for t in text.split()]
        EOS = 1
        arr = tokenize_corpus(encode, EOS, srcs, seq_len=16, n_sequences=8, seed=0)
        assert arr.shape == (8, 16) and arr.dtype == np.int32
        content = (arr == 5).sum() + (arr == 7).sum()
        assert content + (arr == EOS).sum() == arr.size
        n_a = int(((arr == 5).sum(1) > 0).sum())
        assert 5 <= n_a <= 7, "weights 3:1 over 8 sequences should give ~6 sequences from source a"
        assert np.array_equal(arr, tokenize_corpus(encode, EOS, srcs, seq_len=16, n_sequences=8, seed=0))
        assert not np.array_equal(arr, tokenize_corpus(encode, EOS, srcs, seq_len=16, n_sequences=8, seed=1))

        save_tokens(d / "tok", arr, n_calib=3, meta={"tokenizer": "toy"})
        calib, ev, meta = load_tokens(d / "tok")
        assert calib.shape == (3, 16) and ev.shape == (5, 16) and meta["n_eval"] == 5 and calib.dtype == torch.int64
        assert torch.equal(torch.cat([calib, ev]).to(torch.int32), torch.from_numpy(arr))

        try:
            tokenize_corpus(encode, EOS, srcs, seq_len=16, n_sequences=100000, seed=0)
        except ValueError as exc:
            assert "ran out of documents" in str(exc)
        else:
            raise AssertionError("a source that cannot fill its quota must raise")
    for bad in ("name=x", "hf=a,jsonl=b", "jsonl=a,bogus=1", "jsonl"):
        try:
            parse_source_spec(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad spec {bad!r} must raise")


def test_reference_perplexity_and_the_plausibility_guard():
    import prepare_calibration as pc

    st = setup()
    model, ev = st["model"], st["ev"]
    got = pc.reference_perplexity(st["cache"], len(ev))
    with torch.no_grad():
        logits = model(input_ids=ev).logits.double()
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ev[:, 1:].reshape(-1))
    assert abs(got - math.exp(float(nll))) / got < 1e-4, (got, math.exp(float(nll)))

    line = pc.check_reference_plausible(12.5, 163840)
    assert "12.50" in line and "163840" in line
    for bad in (163840.0, 40000.0, float("nan"), float("inf")):          # chance level, above 0.2 x vocab, broken numbers
        try:
            pc.check_reference_plausible(bad, 163840)
        except SystemExit as exc:
            assert "no better than chance" in str(exc) and "--no-sanity-check" in str(exc)
        else:
            raise AssertionError(f"perplexity {bad} must be rejected")
    pc.check_reference_plausible(163840.0, 163840, enabled=False)       # explicit override
    pc.check_reference_plausible(float(128), 128)                       # toy vocabularies are never judged


TESTS = [
    ("reference perplexity + the plausibility guard", test_reference_perplexity_and_the_plausibility_guard),
    ("reference cache == the model's own hidden states / NLL", test_reference_cache_matches_the_models_own_outputs),
    ("KL / PPL / top-1 / SEM vs an independent computation; zero-change = 0", test_baseline_check_and_kl_against_independent_computation),
    ("baseline check rejects a cache from another model", test_baseline_check_catches_a_mismatched_model),
    ("corpus tokenization: quotas, packing, determinism, split", test_corpus_tokenization_packing_and_split),
]


def main():
    for i, (title, fn) in enumerate(TESTS, 1):
        print(f"=== Test {i}: {title} ===")
        fn()
        print("PASS\n")
    print("ALL CALIBRATION / METRICS TESTS PASSED")


if __name__ == "__main__":
    main()
