"""Calibration corpus and the per-model reference cache.

Two stages, matching prepare_calibration.py:

1. tokenize_corpus -> tokens.npy (int32, [n_sequences, seq_len]) + meta.json.
   Model-independent (Moonlight and Moonlight-A share a tokenizer), so it is
   produced once and shared. Sequences are split in two disjoint parts: the
   first `n_calib` (the calibration part, not used by the study) and the rest,
   the *evaluation* part (KL / perplexity).

2. compute_reference_cache -> one directory per model:
     ref/shard_XXXX.pt   {"hidden": (n, T, d) final-norm hidden states in the model's
                         dtype, "nll": (n, T-1) fp32 next-token NLL} of the ORIGINAL
                         model on the evaluation part, `shard_size` sequences each.
     meta.json           provenance (model, dtype, shapes, sha256 of the eval tokens).

   Full reference logits are not stored: 1M tokens x 163,840 vocab is hundreds of
   GB. The cached final hidden state is the exact input of the (never compressed)
   LM head, so reference logits are recomputed on the fly, in fp32, bit-for-bit what
   the original model produces -- KL(p_orig || p_compressed) is exact, not a top-k
   approximation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from model_loading import final_hidden, head_weight_fp32, logits_fp32

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_kw):
        return it


# ---------------------------------------------------------------------------
# Corpus tokenization
# ---------------------------------------------------------------------------


@dataclass
class CorpusSource:
    name: str
    kind: str                  # jsonl | parquet | txt
    location: str
    field: str = "text"
    weight: float = 1.0


def parse_source_spec(spec: str) -> CorpusSource:
    """'name=web,parquet=/data/fineweb-edu,field=text,weight=0.5' or 'name=code,jsonl=/data/code.jsonl,weight=0.2'.
    Exactly one of jsonl/parquet/txt (local files only)."""
    kv = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"bad source spec fragment {part!r} in {spec!r} (expected key=value)")
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip()
    kinds = [k for k in ("jsonl", "parquet", "txt") if k in kv]
    if len(kinds) != 1:
        raise ValueError(f"source spec {spec!r} needs exactly one of jsonl=/parquet=/txt=")
    unknown = set(kv) - {"name", "jsonl", "parquet", "txt", "field", "weight"}
    if unknown:
        raise ValueError(f"unknown keys {sorted(unknown)} in source spec {spec!r}")
    kind = kinds[0]
    return CorpusSource(
        name=kv.get("name", kv[kind]), kind=kind, location=kv[kind], field=kv.get("field", "text"),
        weight=float(kv.get("weight", 1.0)),
    )


def iter_documents(src: CorpusSource) -> Iterator[str]:
    if src.kind == "jsonl":
        with open(src.location, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)[src.field]
    elif src.kind == "parquet":
        import pyarrow.parquet as pq

        loc = Path(src.location)
        files = sorted(loc.glob("*.parquet")) if loc.is_dir() else [loc]
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(columns=[src.field]):
                yield from batch.column(0).to_pylist()
    elif src.kind == "txt":
        yield Path(src.location).read_text(encoding="utf-8")
    else:
        raise ValueError(f"unknown source kind {src.kind!r}")


def _quotas(weights: Sequence[float], total: int) -> list[int]:
    """Largest-remainder split of `total` sequences proportional to the weights."""
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("source weights must sum to a positive number")
    raw = [w / s * total for w in weights]
    out = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[: total - sum(out)]:
        out[i] += 1
    return out


def tokenize_corpus(
    encode: Callable[[str], Sequence[int]],
    eos_id: int,
    sources: Sequence[CorpusSource],
    *,
    seq_len: int,
    n_sequences: int,
    seed: int = 0,
) -> np.ndarray:
    """Pack documents (each followed by EOS) into fixed-length sequences, drawing
    a weight-proportional number of sequences from every source, then shuffle the
    sequences deterministically."""
    quotas = _quotas([s.weight for s in sources], n_sequences)
    seqs: list[list[int]] = []
    for src, quota in zip(sources, quotas):
        got, buf = 0, []
        for doc in iter_documents(src):
            ids = list(encode(doc))
            if not ids:
                continue
            buf.extend(ids)
            buf.append(eos_id)
            while len(buf) >= seq_len and got < quota:
                seqs.append(buf[:seq_len])
                buf = buf[seq_len:]
                got += 1
            if got >= quota:
                break
        if got < quota:
            raise ValueError(f"source {src.name!r} ran out of documents: built {got} of {quota} sequences")
    arr = np.asarray(seqs, dtype=np.int32)
    return arr[np.random.default_rng(seed).permutation(len(arr))]


def save_tokens(out_dir: str | Path, tokens: np.ndarray, *, n_calib: int, meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n, t = tokens.shape
    if not 0 < n_calib < n:
        raise ValueError(f"n_calib must be in (0, {n}), got {n_calib}")
    np.save(out_dir / "tokens.npy", tokens.astype(np.int32))
    full = dict(meta, n_sequences=int(n), seq_len=int(t), n_calib=int(n_calib), n_eval=int(n - n_calib))
    (out_dir / "meta.json").write_text(json.dumps(full, indent=2))


def load_tokens(tokens_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """(calibration tokens, evaluation tokens, meta) as int64 tensors."""
    tokens_dir = Path(tokens_dir)
    meta = json.loads((tokens_dir / "meta.json").read_text())
    arr = np.load(tokens_dir / "tokens.npy")
    t = torch.from_numpy(arr.astype(np.int64))
    return t[: meta["n_calib"]], t[meta["n_calib"]:], meta


def tokens_sha256(tokens: torch.Tensor) -> str:
    return hashlib.sha256(tokens.to(torch.int32).cpu().numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Reference cache
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_reference_cache(
    model,
    eval_tokens: torch.Tensor,
    out_dir: str | Path,
    *,
    batch_size: int = 4,
    shard_size: int = 32,
    token_chunk: int = 2048,
    extra_meta: dict | None = None,
) -> Path:
    """Final hidden states + next-token NLL of the ORIGINAL model on the evaluation part."""
    out_dir = Path(out_dir)
    (out_dir / "ref").mkdir(parents=True, exist_ok=True)
    head_w = head_weight_fp32(model)
    n_eval, T = eval_tokens.shape
    d_model = None
    for shard_idx, start in enumerate(tqdm(range(0, n_eval, shard_size), desc="reference (eval part)")):
        ids = eval_tokens[start:start + shard_size]
        hs = [final_hidden(model, ids[i:i + batch_size]).cpu() for i in range(0, len(ids), batch_size)]
        hidden = torch.cat(hs)
        d_model = hidden.shape[-1]
        nll = token_nll(hidden, ids, head_w, token_chunk=token_chunk)
        torch.save({"hidden": hidden, "nll": nll, "start": start}, out_dir / "ref" / f"shard_{shard_idx:04d}.pt")

    meta = dict(
        n_eval=int(n_eval), seq_len=int(T), d_model=int(d_model), shard_size=int(shard_size),
        hidden_dtype=str(hidden.dtype).replace("torch.", ""),
        eval_tokens_sha256=tokens_sha256(eval_tokens),
        **(extra_meta or {}),
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


@torch.no_grad()
def token_nll(hidden: torch.Tensor, ids: torch.Tensor, head_w: torch.Tensor, *, token_chunk: int = 2048) -> torch.Tensor:
    """Next-token NLL (n, T-1), fp32, from final hidden states (n, T, d)."""
    n, T, d = hidden.shape
    dev = head_w.device
    h = hidden[:, :-1].reshape(-1, d)
    tgt = ids[:, 1:].reshape(-1).to(dev)
    out = torch.empty(h.shape[0], dtype=torch.float32)
    for c in range(0, h.shape[0], token_chunk):
        logits = logits_fp32(h[c:c + token_chunk].to(dev), head_w)
        out[c:c + token_chunk] = F.cross_entropy(logits, tgt[c:c + token_chunk], reduction="none").cpu()
    return out.reshape(n, T - 1)


class ReferenceCache:
    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        if not (self.dir / "meta.json").is_file():
            raise FileNotFoundError(f"{self.dir} is not a reference cache (no meta.json); run prepare_calibration.py first")
        self.meta = json.loads((self.dir / "meta.json").read_text())

    @property
    def n_eval(self) -> int:
        return self.meta["n_eval"]

    def check_tokens(self, eval_tokens: torch.Tensor) -> None:
        if tokens_sha256(eval_tokens) != self.meta["eval_tokens_sha256"]:
            raise ValueError(
                f"the evaluation tokens do not match the ones this cache ({self.dir}) was built from; "
                "re-run prepare_calibration.py for this tokens directory"
            )

    def iter_eval_shards(self, n_sequences: int | None = None) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
        """(start, hidden (n, T, d), nll (n, T-1)) shards in evaluation order,
        truncated to the first n_sequences sequences overall."""
        remaining = self.n_eval if n_sequences is None else min(n_sequences, self.n_eval)
        for path in sorted((self.dir / "ref").glob("shard_*.pt")):
            if remaining <= 0:
                return
            shard = torch.load(path, weights_only=True)
            n = min(shard["hidden"].shape[0], remaining)
            yield shard["start"], shard["hidden"][:n], shard["nll"][:n]
            remaining -= n
