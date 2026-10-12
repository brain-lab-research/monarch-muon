#!/usr/bin/env python
"""Build the calibration corpus and the per-model reference cache.

Stage `tokenize` (model-independent, run once; both Moonlight checkpoints share
a tokenizer): pack a mixed web / code / math corpus into fixed-length sequences
(~1M tokens by default) -> <tokens-dir>/tokens.npy + meta.json. The first
--n-calib-sequences sequences are the calibration part (not used by the study),
the rest the evaluation part (KL / perplexity).

Stage `reference` (once per model): one forward pass of the ORIGINAL model ->
<tokens-dir>/refs/<model-tag>/ (or --cache-dir) with the final hidden states + NLL
of the evaluation part (see calibration_data.py for the layout and why logits are
recomputed, not stored).
Disk: ~ n_eval_tokens x d_model x 2 bytes (~3 GB for 750k tokens at d_model=2048, bf16).
Re-running skips this part when a cache for the same tokens already exists (--force-reference).

Nothing is downloaded: the corpus is read from LOCAL files given as --source specs (or the
`source` list of --config), or a pre-tokenized array is passed with --tokens-npy. A source spec is
comma-separated key=value, exactly one of jsonl= | parquet= (a file or a directory of *.parquet) | txt=:
    name=web,parquet=/data/fineweb-edu,field=text,weight=0.5
    name=code,jsonl=/data/code.jsonl,field=content,weight=0.25
Documents are read in file order; each source contributes weight-proportional sequences.

Usage:
    python prepare_calibration.py --config configs/calibration.json --tokens-dir /data/calib \\
        --tokenizer /data/Moonlight-16B-A3B --stages tokenize
    python prepare_calibration.py --tokens-dir /data/calib \\
        --model-dir /data/Moonlight-16B-A3B --model-tag moonlight_muon --stages reference
    ... --tokens-npy my_tokens.npy      # skip tokenization: use a pre-tokenized int array [N, T]
"""
from __future__ import annotations

import argparse
import math
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from calibration_data import (
    ReferenceCache,
    compute_reference_cache,
    load_tokens,
    parse_source_spec,
    save_tokens,
    tokenize_corpus,
)
from config_utils import add_config_argument, parse_args_with_config
from model_loading import free_model, load_model
from model_registry import read_config
from moe_accel import add_accel_argument, check_cache_accel, effective_accel

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(p)
    p.add_argument("--tokens-dir", default=None, help="Shared output dir for tokens.npy / meta.json / refs/. Required.")
    p.add_argument("--stages", nargs="+", choices=["tokenize", "reference"], default=["tokenize", "reference"])

    g = p.add_argument_group("tokenize")
    g.add_argument("--tokenizer", default=None, help="HF tokenizer id/path (default: --model-dir).")
    g.add_argument("--source", action="append", default=None, help="Corpus source spec (repeatable); see above.")
    g.add_argument("--tokens-npy", default=None, help="Pre-tokenized int array [N, T] instead of tokenizing.")
    g.add_argument("--seq-len", type=int, default=2048)
    g.add_argument("--n-sequences", type=int, default=512, help="Total sequences (512 x 2048 ~= 1.05M tokens).")
    g.add_argument("--n-calib-sequences", type=int, default=128,
                   help="Leading sequences set aside; the rest are the evaluation part.")
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("reference")
    g.add_argument("--model-dir", default=None)
    g.add_argument("--model-tag", default=None, help="Cache goes to <tokens-dir>/refs/<model-tag>.")
    g.add_argument("--cache-dir", default=None, help="Override the cache directory.")
    g.add_argument("--dtype", default="bfloat16",
                   help="Use the SAME dtype/device for run_joint_blocks.py, or the baseline check will report noise.")
    g.add_argument("--device", default=None)
    add_accel_argument(g)
    g.add_argument("--batch-size", type=int, default=4)
    g.add_argument("--shard-size", type=int, default=32, help="Sequences per cache file.")
    g.add_argument("--token-chunk", type=int, default=2048)
    g.add_argument("--force-reference", action="store_true",
                   help="Recompute the reference cache even if one for these tokens already exists.")
    g.add_argument("--no-sanity-check", action="store_true",
                   help="Do not abort when the reference model's perplexity is no better than uniform guessing.")

    return parse_args_with_config(p, argv, required=("tokens_dir",))


def _tokenizer_encoder(path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    def encode(text: str):
        try:
            return tok.encode(text, add_special_tokens=False)
        except TypeError:                       # tokenizers whose encode() takes no such flag
            return tok.encode(text)

    return tok, encode


def run_tokenize(args) -> None:
    tokens_dir = Path(args.tokens_dir)
    if args.tokens_npy:
        arr = np.load(args.tokens_npy)
        meta = dict(tokenizer="pre-tokenized", sources=[args.tokens_npy], seed=args.seed)
        print(f"using pre-tokenized array {args.tokens_npy}: {arr.shape}")
    else:
        specs = args.source
        if not specs:
            raise SystemExit("no corpus given: pass local --source specs (see --help) or --tokens-npy")
        tok_path = args.tokenizer or args.model_dir
        if not tok_path:
            raise SystemExit("--tokenizer (or --model-dir) is required to tokenize")
        sources = [parse_source_spec(s) for s in specs]
        tok, encode = _tokenizer_encoder(tok_path)
        if tok.eos_token_id is None:
            raise SystemExit("the tokenizer has no eos_token_id (needed as the document separator)")
        print(f"tokenizing {args.n_sequences} x {args.seq_len} tokens from {[s.name for s in sources]} ...")
        arr = tokenize_corpus(encode, tok.eos_token_id, sources, seq_len=args.seq_len, n_sequences=args.n_sequences,
                              seed=args.seed)
        meta = dict(tokenizer=str(tok_path), vocab_size=len(tok), eos_id=tok.eos_token_id,
                    sources=[s.__dict__ for s in sources], seed=args.seed)
    save_tokens(tokens_dir, arr, n_calib=args.n_calib_sequences, meta=meta)
    n, t = arr.shape
    print(f"wrote {tokens_dir / 'tokens.npy'}: {n} sequences x {t} tokens = {n * t:,} tokens "
          f"({args.n_calib_sequences} calibration, {n - args.n_calib_sequences} evaluation)")


PLAUSIBLE_MAX_FRACTION_OF_VOCAB = 0.2         # a language model on text is far below this; noise is at 1.0 (or NaN)


def reference_perplexity(cache: ReferenceCache, n_sequences: int) -> float:
    """exp(mean next-token NLL) of the ORIGINAL model over the cached evaluation sequences."""
    total, count = 0.0, 0
    for _start, _hidden, nll in cache.iter_eval_shards(n_sequences):
        total += float(nll.double().sum())
        count += nll.numel()
    return math.exp(total / count) if count else float("nan")


def check_reference_plausible(ppl: float, vocab_size: int, *, enabled: bool = True) -> str:
    """The log line; raises SystemExit when the model predicts the evaluation tokens no better than chance, which
    means broken outputs, not a weak model (real vocabularies only: toy models have tiny ones)."""
    line = f"reference perplexity on the evaluation tokens: {ppl:.2f} (uniform guessing would give {vocab_size})"
    if enabled and vocab_size >= 10_000 and not ppl <= PLAUSIBLE_MAX_FRACTION_OF_VOCAB * vocab_size:
        raise SystemExit(
            line + "\nThat is no better than chance: the model's outputs are noise. Likely causes: the checkpoint was "
            "loaded wrongly (transformers must be 4.57.x), the tokens come from a different tokenizer than the model's, "
            "or the dtype is wrong. Rebuild the cache after fixing it with --force-reference; --no-sanity-check overrides "
            "this check.")
    return line


def run_reference(args) -> None:
    if not (args.model_dir and args.model_tag):
        raise SystemExit("--model-dir and --model-tag are required for the reference stage")
    out_dir = Path(args.cache_dir or Path(args.tokens_dir) / "refs" / args.model_tag)
    calib, ev, tmeta = load_tokens(args.tokens_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # nothing to do? (do not load a multi-GB model just to find out)
    if (out_dir / "meta.json").is_file() and not args.force_reference:
        try:
            ReferenceCache(out_dir).check_tokens(ev)
        except ValueError as exc:
            raise SystemExit(f"{exc}\n(pass --force-reference to overwrite)")
        check_cache_accel(ReferenceCache(out_dir).meta, args.accel, what=f"the existing reference cache {out_dir}")
        print(f"reference cache already present for these tokens: {out_dir} (--force-reference to recompute)")
        print(check_reference_plausible(reference_perplexity(ReferenceCache(out_dir), len(ev)),
                                        read_config(args.model_dir)["vocab_size"], enabled=not args.no_sanity_check))
        return
    print(f"loading {args.model_dir} ({args.dtype}) on {device} ...")
    model = load_model(args.model_dir, dtype=args.dtype, device=device, accel=args.accel)
    accel = effective_accel(model)                  # what the cache is built with (a failed load-time check -> 'none')
    vocab = model.get_output_embeddings().weight.shape[0]
    top = int(max(calib.max(), ev.max()))
    if top >= vocab:
        raise SystemExit(f"token id {top} >= model vocab size {vocab}: wrong tokenizer for this model")

    compute_reference_cache(
        model, ev, out_dir, batch_size=args.batch_size, shard_size=args.shard_size, token_chunk=args.token_chunk,
        extra_meta=dict(model_dir=str(args.model_dir), model_tag=args.model_tag, dtype=args.dtype, device=device,
                        batch_size=args.batch_size, torch=torch.__version__, python=platform.python_version(),
                        accel=accel, accel_report=getattr(model, "moe_accel_report", None), tokens_meta=tmeta),
    )
    print(f"wrote reference cache -> {out_dir}")
    print(check_reference_plausible(reference_perplexity(ReferenceCache(out_dir), len(ev)), vocab,
                                    enabled=not args.no_sanity_check))
    free_model(model)


def main(argv=None):
    args = parse_args(argv)
    if "tokenize" in args.stages:
        run_tokenize(args)
    if "reference" in args.stages:
        run_reference(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
