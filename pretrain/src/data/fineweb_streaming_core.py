"""Deterministic FineWeb parquet streaming with exact resume state."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Callable, Iterator, Sequence

import pyarrow.parquet as pq


@dataclass(frozen=True)
class SchemaField:
    """A normalized parquet schema field."""

    name: str
    type: str
    nullable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "nullable": self.nullable,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SchemaField":
        return cls(
            name=str(payload["name"]),
            type=str(payload["type"]),
            nullable=bool(payload["nullable"]),
        )


@dataclass(frozen=True)
class ShardManifest:
    """Manifest entry for one parquet shard."""

    relative_path: str
    size_bytes: int
    schema: tuple[SchemaField, ...]
    num_rows: int
    row_group_rows: tuple[int, ...]

    @property
    def num_row_groups(self) -> int:
        return len(self.row_group_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "schema": [field.to_dict() for field in self.schema],
            "num_rows": self.num_rows,
            "row_group_rows": list(self.row_group_rows),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShardManifest":
        return cls(
            relative_path=str(payload["relative_path"]),
            size_bytes=int(payload["size_bytes"]),
            schema=tuple(SchemaField.from_dict(field) for field in payload["schema"]),
            num_rows=int(payload["num_rows"]),
            row_group_rows=tuple(int(value) for value in payload["row_group_rows"]),
        )


@dataclass(frozen=True)
class Manifest:
    """Dataset manifest built from parquet footers only."""

    dataset_root: str
    shards: tuple[ShardManifest, ...]
    fingerprint: str = field(init=False)
    row_group_offsets: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fingerprint", self._compute_fingerprint())

        offsets: list[int] = []
        running_total = 0
        for shard in self.shards:
            offsets.append(running_total)
            running_total += shard.num_row_groups
        object.__setattr__(self, "row_group_offsets", tuple(offsets))

    @property
    def dataset_path(self) -> Path:
        return Path(self.dataset_root)

    @property
    def total_row_groups(self) -> int:
        return sum(shard.num_row_groups for shard in self.shards)

    def global_row_group_index(self, file_index: int, row_group_index: int) -> int:
        return self.row_group_offsets[file_index] + row_group_index

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "fingerprint": self.fingerprint,
            "shards": [shard.to_dict() for shard in self.shards],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Manifest":
        manifest = cls(
            dataset_root=str(payload["dataset_root"]),
            shards=tuple(ShardManifest.from_dict(shard) for shard in payload["shards"]),
        )
        expected = str(payload.get("fingerprint", manifest.fingerprint))
        if manifest.fingerprint != expected:
            raise ValueError(
                "Manifest payload fingerprint does not match the shard metadata."
            )
        return manifest

    def _compute_fingerprint(self) -> str:
        payload = {"shards": [shard.to_dict() for shard in self.shards]}
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RowGroupRef:
    """Stable reference to one parquet row group."""

    relative_path: str
    row_group_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "row_group_index": self.row_group_index,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RowGroupRef":
        return cls(
            relative_path=str(payload["relative_path"]),
            row_group_index=int(payload["row_group_index"]),
        )


@dataclass(frozen=True)
class SplitPlan:
    """Deterministic split and shuffle order over manifest row groups."""

    manifest_fingerprint: str
    val_fraction: float
    split_seed: int
    shuffle_seed: int
    train_row_groups: tuple[RowGroupRef, ...]
    val_row_groups: tuple[RowGroupRef, ...]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fingerprint", self._compute_fingerprint())

    def row_groups_for_split(self, split: str) -> tuple[RowGroupRef, ...]:
        if split == "train":
            return self.train_row_groups
        if split == "val":
            return self.val_row_groups
        raise ValueError(f"Unsupported split {split!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_fingerprint": self.manifest_fingerprint,
            "fingerprint": self.fingerprint,
            "val_fraction": self.val_fraction,
            "split_seed": self.split_seed,
            "shuffle_seed": self.shuffle_seed,
            "train_row_groups": [
                row_group.to_dict() for row_group in self.train_row_groups
            ],
            "val_row_groups": [row_group.to_dict() for row_group in self.val_row_groups],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SplitPlan":
        split_plan = cls(
            manifest_fingerprint=str(payload["manifest_fingerprint"]),
            val_fraction=float(payload["val_fraction"]),
            split_seed=int(payload["split_seed"]),
            shuffle_seed=int(payload["shuffle_seed"]),
            train_row_groups=tuple(
                RowGroupRef.from_dict(row_group)
                for row_group in payload["train_row_groups"]
            ),
            val_row_groups=tuple(
                RowGroupRef.from_dict(row_group)
                for row_group in payload["val_row_groups"]
            ),
        )
        expected = str(payload.get("fingerprint", split_plan.fingerprint))
        if split_plan.fingerprint != expected:
            raise ValueError(
                "Split plan payload fingerprint does not match the row-group ordering."
            )
        return split_plan

    def _compute_fingerprint(self) -> str:
        payload = {
            "manifest_fingerprint": self.manifest_fingerprint,
            "val_fraction": self.val_fraction,
            "split_seed": self.split_seed,
            "shuffle_seed": self.shuffle_seed,
            "train_row_groups": [
                row_group.to_dict() for row_group in self.train_row_groups
            ],
            "val_row_groups": [row_group.to_dict() for row_group in self.val_row_groups],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SplitPlanWithValBlocks:
    plan: SplitPlan
    val_blocks: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _SourceCursor:
    assigned_row_group_index: int
    row_in_group: int

    def to_dict(self) -> dict[str, int]:
        return {
            "assigned_row_group_index": self.assigned_row_group_index,
            "row_in_group": self.row_in_group,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "_SourceCursor":
        return cls(
            assigned_row_group_index=int(payload["assigned_row_group_index"]),
            row_in_group=int(payload["row_in_group"]),
        )


@dataclass
class _PendingBatch:
    future: Future[list[list[int]]]
    end_cursor: _SourceCursor


_validated_dataset_fingerprints: dict[str, str] = {}
_validation_lock = threading.Lock()


class LocalParquetSource:
    kind = "local"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    @property
    def source_id(self) -> str:
        return f"local:{self.root.resolve()}"

    @property
    def manifest_root(self) -> str:
        return str(self.root)

    def list_parquet_files(self) -> tuple[str, ...]:
        return tuple(
            path.relative_to(self.root).as_posix()
            for path in sorted(self.root.glob("*.parquet"))
            if path.is_file()
        )

    def size_bytes(self, relative_path: str) -> int:
        return (self.root / relative_path).stat().st_size

    def open_parquet_file(self, relative_path: str):
        return pq.ParquetFile(self.root / relative_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dataset_root": str(self.root),
        }


def build_manifest_from_source(source) -> Manifest:
    """Build a manifest from parquet files exposed by a source."""

    parquet_paths = source.list_parquet_files()
    if not parquet_paths:
        raise FileNotFoundError(
            f"No parquet files found for FineWeb source {source.source_id}."
        )

    shards: list[ShardManifest] = []
    for relative_path in parquet_paths:
        parquet_file = source.open_parquet_file(relative_path)
        try:
            metadata = parquet_file.metadata
            schema = tuple(
                SchemaField(
                    name=field.name,
                    type=str(field.type),
                    nullable=field.nullable,
                )
                for field in parquet_file.schema_arrow
            )
            row_group_rows = tuple(
                int(metadata.row_group(row_group_index).num_rows)
                for row_group_index in range(metadata.num_row_groups)
            )
            if sum(row_group_rows) != metadata.num_rows:
                raise ValueError(
                    f"Row-group row counts do not add up for {relative_path}."
                )
            shards.append(
                ShardManifest(
                    relative_path=relative_path,
                    size_bytes=source.size_bytes(relative_path),
                    schema=schema,
                    num_rows=int(metadata.num_rows),
                    row_group_rows=row_group_rows,
                )
            )
        finally:
            parquet_file.close()

    manifest = Manifest(dataset_root=source.manifest_root, shards=tuple(shards))
    with _validation_lock:
        _validated_dataset_fingerprints[source.source_id] = manifest.fingerprint
    return manifest


def _validate_manifest_against_source(manifest: Manifest, source) -> None:
    root_key = source.source_id
    with _validation_lock:
        cached = _validated_dataset_fingerprints.get(root_key)
    if cached == manifest.fingerprint:
        return
    current = build_manifest_from_source(source)
    if current.fingerprint != manifest.fingerprint:
        raise ValueError(
            "Dataset files do not match the provided manifest fingerprint."
        )


def _stable_hash_hex(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _all_row_groups(manifest: Manifest) -> tuple[RowGroupRef, ...]:
    return tuple(
        RowGroupRef(relative_path=shard.relative_path, row_group_index=row_group_index)
        for shard in manifest.shards
        for row_group_index in range(shard.num_row_groups)
    )


def _ordered_row_groups_for_membership(
    manifest: Manifest,
    *,
    split_seed: int,
) -> tuple[RowGroupRef, ...]:
    return tuple(
        sorted(
            _all_row_groups(manifest),
            key=lambda row_group: (
                _stable_hash_hex(
                    "split",
                    split_seed,
                    row_group.relative_path,
                    row_group.row_group_index,
                ),
                row_group.relative_path,
                row_group.row_group_index,
            ),
        )
    )


def _shuffle_row_groups(
    split: str,
    row_groups: Sequence[RowGroupRef],
    *,
    shuffle_seed: int,
) -> tuple[RowGroupRef, ...]:
    return tuple(
        sorted(
            row_groups,
            key=lambda row_group: (
                _stable_hash_hex(
                    "shuffle",
                    shuffle_seed,
                    split,
                    row_group.relative_path,
                    row_group.row_group_index,
                ),
                row_group.relative_path,
                row_group.row_group_index,
            ),
        )
    )


def build_snapshot_split_plan_with_val_blocks(
    manifest: Manifest,
    tokenizer_factory: Callable[[], Any],
    *,
    block_tokens: int,
    val_sequences: int,
    split_seed: int = 0,
    shuffle_seed: int = 0,
    parquet_source=None,
) -> SplitPlanWithValBlocks:
    """Reserve val row groups and pack their tokens into the val blocks."""

    if block_tokens <= 0:
        raise ValueError("block_tokens must be a positive integer.")
    if val_sequences < 0:
        raise ValueError("val_sequences must be >= 0.")

    split_order = _ordered_row_groups_for_membership(manifest, split_seed=split_seed)
    val_blocks: list[tuple[int, ...]] = []
    if val_sequences == 0:
        val_row_groups: tuple[RowGroupRef, ...] = tuple()
    else:
        parquet_source = parquet_source or LocalParquetSource(manifest.dataset_root)
        tokenizer = tokenizer_factory()
        eos_token_id = _resolve_eos_token_id(tokenizer)
        file_index_by_relative_path = {
            shard.relative_path: index for index, shard in enumerate(manifest.shards)
        }
        parquet_files: dict[int, Any] = {}
        token_buffer: list[int] = []
        selected: list[RowGroupRef] = []

        try:
            for row_group in split_order:
                selected.append(row_group)
                file_index = file_index_by_relative_path[row_group.relative_path]
                parquet_file = parquet_files.get(file_index)
                if parquet_file is None:
                    parquet_file = parquet_source.open_parquet_file(
                        row_group.relative_path
                    )
                    parquet_files[file_index] = parquet_file

                table = parquet_file.read_row_group(
                    row_group.row_group_index,
                    columns=["text"],
                )
                texts = table.column("text").to_pylist()
                for text in texts:
                    token_buffer.extend(_tokenize_text(tokenizer, text))
                    token_buffer.append(eos_token_id)
                    while (
                        len(token_buffer) >= block_tokens
                        and len(val_blocks) < val_sequences
                    ):
                        val_blocks.append(tuple(token_buffer[:block_tokens]))
                        del token_buffer[:block_tokens]
                    if len(val_blocks) >= val_sequences:
                        break
                if len(val_blocks) >= val_sequences:
                    break
        finally:
            for parquet_file in parquet_files.values():
                parquet_file.close()

        if len(val_blocks) < val_sequences:
            raise ValueError(
                "Validation reservation does not cover the requested packed sequence budget."
            )
        val_row_groups = tuple(selected)

    val_membership = set(val_row_groups)
    all_row_groups = _all_row_groups(manifest)
    train_row_groups = _shuffle_row_groups(
        "train",
        tuple(row_group for row_group in all_row_groups if row_group not in val_membership),
        shuffle_seed=shuffle_seed,
    )
    val_fraction = (
        len(val_row_groups) / max(1, len(all_row_groups)) if all_row_groups else 0.0
    )

    plan = SplitPlan(
        manifest_fingerprint=manifest.fingerprint,
        val_fraction=val_fraction,
        split_seed=split_seed,
        shuffle_seed=shuffle_seed,
        train_row_groups=train_row_groups,
        val_row_groups=val_row_groups,
    )
    return SplitPlanWithValBlocks(plan=plan, val_blocks=tuple(val_blocks))


def _normalize_tokens(encoded: Any) -> list[int]:
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded.get("input_ids")

    if not isinstance(encoded, list):
        raise TypeError(
            "Tokenizer must return a list of token ids or an object exposing input_ids."
        )
    if encoded and isinstance(encoded[0], list):
        raise TypeError("Tokenizer must return one sequence per document, not a batch.")
    if not all(isinstance(token, int) for token in encoded):
        raise TypeError("Tokenizer output must contain only integer token ids.")
    return list(encoded)


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    if not isinstance(text, str):
        raise TypeError(f"Expected a string document, got {type(text)!r}.")

    if hasattr(tokenizer, "encode"):
        try:
            return _normalize_tokens(tokenizer.encode(text, add_special_tokens=False))
        except TypeError as exc:
            raise TypeError(
                "Tokenizer.encode must accept add_special_tokens=False."
            ) from exc

    if callable(tokenizer):
        try:
            return _normalize_tokens(tokenizer(text, add_special_tokens=False))
        except TypeError as exc:
            raise TypeError(
                "Callable tokenizer must accept add_special_tokens=False."
            ) from exc

    raise TypeError(
        "Tokenizer must provide encode(text, add_special_tokens=False) or be callable."
    )


def _resolve_eos_token_id(tokenizer: Any) -> int:
    for attribute_name in ("eos_token_id", "eot_token_id"):
        token_id = getattr(tokenizer, attribute_name, None)
        if token_id is not None:
            return int(token_id)
    raise ValueError("Tokenizer must define eos_token_id or eot_token_id.")


class FineWebEduStream(Iterator[list[int]]):
    """Iterate fixed-size token blocks from FineWeb parquet shards."""

    def __init__(
        self,
        manifest: Manifest,
        tokenizer_factory: Callable[[], Any],
        block_tokens: int,
        *,
        split_plan: SplitPlan,
        split: str = "train",
        rank: int = 0,
        world_size: int = 1,
        worker_id: int = 0,
        num_data_workers: int = 1,
        num_token_workers: int = 0,
        doc_batch_size: int = 32,
        prefetch_batches: int = 2,
        parquet_source=None,
    ) -> None:
        if block_tokens <= 0:
            raise ValueError("block_tokens must be a positive integer.")
        if split not in {"train", "val"}:
            raise ValueError("split must be either 'train' or 'val'.")
        if world_size <= 0:
            raise ValueError("world_size must be a positive integer.")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in the range [0, world_size).")
        if num_data_workers <= 0:
            raise ValueError("num_data_workers must be a positive integer.")
        if worker_id < 0 or worker_id >= num_data_workers:
            raise ValueError("worker_id must be in the range [0, num_data_workers).")
        if num_token_workers < 0:
            raise ValueError("num_token_workers must be >= 0.")
        if doc_batch_size <= 0:
            raise ValueError("doc_batch_size must be a positive integer.")
        if prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be a positive integer.")

        self.manifest = manifest
        self.split_plan = split_plan
        self.split = split
        self.block_tokens = block_tokens
        self.rank = rank
        self.world_size = world_size
        self.worker_id = worker_id
        self.num_data_workers = num_data_workers
        self.num_token_workers = num_token_workers
        self.doc_batch_size = doc_batch_size
        self.prefetch_batches = prefetch_batches
        self.parquet_source = parquet_source or LocalParquetSource(manifest.dataset_root)

        _validate_manifest_against_source(self.manifest, self.parquet_source)
        self._file_index_by_relative_path = {
            shard.relative_path: index for index, shard in enumerate(self.manifest.shards)
        }
        self._validate_split_plan_against_manifest()
        self._split_row_groups = self._resolve_split_row_groups()
        self._consumer_index = self.rank * self.num_data_workers + self.worker_id
        self._num_consumers = self.world_size * self.num_data_workers
        self._assigned_row_groups = self._split_row_groups[
            self._consumer_index : : self._num_consumers
        ]

        self._tokenizer_factory = tokenizer_factory
        self._main_tokenizer = tokenizer_factory()
        self.eos_token_id = _resolve_eos_token_id(self._main_tokenizer)
        self._thread_local = threading.local()
        self._executor = (
            ThreadPoolExecutor(max_workers=num_token_workers)
            if num_token_workers > 0
            else None
        )

        self._pending_batches: deque[_PendingBatch] = deque()
        self._source_cursor = _SourceCursor(0, 0)
        self._committed_cursor = _SourceCursor(0, 0)

        self._token_buffer: list[int] = []
        self._token_buffer_start = 0
        self._emitted_block_count = 0

        self._open_parquet_file = None
        self._open_file_index: int | None = None
        self._loaded_row_group_key: tuple[int, int] | None = None
        self._loaded_row_group_texts: list[str] | None = None

        self._closed = False

    def __iter__(self) -> "FineWebEduStream":
        return self

    def __next__(self) -> list[int]:
        self._ensure_open()

        while self._available_tokens() < self.block_tokens:
            self._maybe_submit_batches()
            if self._pending_batches:
                self._consume_next_pending_batch()
                continue
            if self._is_eof(self._source_cursor):
                raise StopIteration

        return self._consume_block()

    def close(self) -> None:
        if self._closed:
            return

        self._discard_pending_batches()

        if self._open_parquet_file is not None:
            self._open_parquet_file.close()
        self._open_parquet_file = None
        self._open_file_index = None
        self._loaded_row_group_key = None
        self._loaded_row_group_texts = None

        if self._executor is not None:
            self._executor.shutdown(wait=True)

        self._closed = True

    def barrier_and_snapshot(self) -> dict[str, Any]:
        self._ensure_open()
        self._discard_pending_batches()
        self._reset_source_to_committed()
        return self._snapshot_state()

    def state_dict(self) -> dict[str, Any]:
        return self.barrier_and_snapshot()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._ensure_open()
        self._discard_pending_batches()

        manifest_fingerprint = str(state["manifest_fingerprint"])
        if manifest_fingerprint != self.manifest.fingerprint:
            raise ValueError("Checkpoint manifest fingerprint does not match this stream.")

        split_plan_fingerprint = str(state["split_plan_fingerprint"])
        if split_plan_fingerprint != self.split_plan.fingerprint:
            raise ValueError("Checkpoint split plan fingerprint does not match this stream.")

        split = str(state["split"])
        if split != self.split:
            raise ValueError("Checkpoint split does not match this stream.")

        rank = int(state["rank"])
        world_size = int(state["world_size"])
        if rank != self.rank or world_size != self.world_size:
            raise ValueError("Checkpoint rank/world_size do not match this stream.")

        worker_id = int(state["worker_id"])
        num_data_workers = int(state["num_data_workers"])
        if worker_id != self.worker_id or num_data_workers != self.num_data_workers:
            raise ValueError("Checkpoint worker_id/num_data_workers do not match this stream.")

        committed_cursor = _SourceCursor.from_dict(state["committed_cursor"])
        self._validate_cursor(committed_cursor)

        token_buffer = [int(token) for token in state["token_buffer"]]
        emitted_block_count = int(state["emitted_block_count"])

        self._committed_cursor = committed_cursor
        self._source_cursor = committed_cursor
        self._token_buffer = token_buffer
        self._token_buffer_start = 0
        self._emitted_block_count = emitted_block_count

        self._clear_row_group_cache()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("This FineWebEduStream instance is closed.")

    def _snapshot_state(self) -> dict[str, Any]:
        unread_tokens = self._unread_tokens()
        return {
            "manifest_fingerprint": self.manifest.fingerprint,
            "split_plan_fingerprint": self.split_plan.fingerprint,
            "split": self.split,
            "rank": self.rank,
            "world_size": self.world_size,
            "worker_id": self.worker_id,
            "num_data_workers": self.num_data_workers,
            "committed_cursor": self._committed_cursor.to_dict(),
            "token_buffer": unread_tokens,
            "emitted_block_count": self._emitted_block_count,
        }

    def _discard_pending_batches(self) -> None:
        while self._pending_batches:
            pending = self._pending_batches.popleft()
            pending.future.result()
        self._reset_source_to_committed()

    def _reset_source_to_committed(self) -> None:
        self._source_cursor = self._committed_cursor
        self._clear_row_group_cache()

    def _clear_row_group_cache(self) -> None:
        self._loaded_row_group_key = None
        self._loaded_row_group_texts = None

    def _available_tokens(self) -> int:
        return len(self._token_buffer) - self._token_buffer_start

    def _unread_tokens(self) -> list[int]:
        if self._token_buffer_start == 0:
            return list(self._token_buffer)
        return list(self._token_buffer[self._token_buffer_start :])

    def _consume_block(self) -> list[int]:
        block = self._token_buffer[
            self._token_buffer_start : self._token_buffer_start + self.block_tokens
        ]
        self._token_buffer_start += self.block_tokens
        self._emitted_block_count += 1
        self._compact_token_buffer()
        return block

    def _compact_token_buffer(self) -> None:
        if self._token_buffer_start == 0:
            return
        if self._token_buffer_start < 4096 and self._token_buffer_start * 2 < len(
            self._token_buffer
        ):
            return
        self._token_buffer = self._token_buffer[self._token_buffer_start :]
        self._token_buffer_start = 0

    def _maybe_submit_batches(self) -> None:
        while (
            len(self._pending_batches) < self.prefetch_batches
            and not self._is_eof(self._source_cursor)
        ):
            texts, end_cursor = self._take_next_document_batch(self.doc_batch_size)
            if not texts:
                break
            future = self._submit_tokenize_batch(texts)
            self._pending_batches.append(
                _PendingBatch(future=future, end_cursor=end_cursor)
            )

    def _submit_tokenize_batch(self, texts: Sequence[str]) -> Future[list[list[int]]]:
        if self._executor is None:
            future: Future[list[list[int]]] = Future()
            future.set_result(self._tokenize_batch(texts))
            return future
        return self._executor.submit(self._tokenize_batch, list(texts))

    def _tokenize_batch(self, texts: Sequence[str]) -> list[list[int]]:
        tokenizer = self._main_tokenizer
        if self._executor is not None:
            tokenizer = getattr(self._thread_local, "tokenizer", None)
            if tokenizer is None:
                tokenizer = self._tokenizer_factory()
                self._thread_local.tokenizer = tokenizer
        return [_tokenize_text(tokenizer, text) for text in texts]

    def _consume_next_pending_batch(self) -> None:
        pending = self._pending_batches.popleft()
        tokenized_documents = pending.future.result()
        for tokens in tokenized_documents:
            self._token_buffer.extend(tokens)
            self._token_buffer.append(self.eos_token_id)
        self._committed_cursor = pending.end_cursor

    def _take_next_document_batch(
        self,
        limit: int,
    ) -> tuple[list[str], _SourceCursor]:
        texts: list[str] = []
        self._source_cursor = self._normalize_cursor(self._source_cursor)

        while len(texts) < limit and not self._is_eof(self._source_cursor):
            assigned_row_group_index = self._source_cursor.assigned_row_group_index
            file_index, row_group_index = self._assigned_row_groups[assigned_row_group_index]
            row_group_texts = self._load_row_group_texts(file_index, row_group_index)
            row_index = self._source_cursor.row_in_group
            while row_index < len(row_group_texts) and len(texts) < limit:
                text = row_group_texts[row_index]
                if not isinstance(text, str):
                    raise TypeError(
                        "Parquet text column must contain strings for streaming."
                    )
                texts.append(text)
                row_index += 1
                self._source_cursor = self._normalize_cursor(
                    _SourceCursor(
                        assigned_row_group_index=assigned_row_group_index,
                        row_in_group=row_index,
                    )
                )
                if self._is_eof(self._source_cursor):
                    break

        return texts, self._source_cursor

    def _load_row_group_texts(self, file_index: int, row_group_index: int) -> list[str]:
        key = (file_index, row_group_index)
        if self._loaded_row_group_key == key and self._loaded_row_group_texts is not None:
            return self._loaded_row_group_texts

        parquet_file = self._get_parquet_file(file_index)
        table = parquet_file.read_row_group(row_group_index, columns=["text"])
        texts = table.column("text").to_pylist()

        expected_rows = self.manifest.shards[file_index].row_group_rows[row_group_index]
        if len(texts) != expected_rows:
            raise ValueError(
                "Row-group row count changed since the manifest was built."
            )

        self._loaded_row_group_key = key
        self._loaded_row_group_texts = texts
        return texts

    def _get_parquet_file(self, file_index: int):
        if self._open_file_index == file_index and self._open_parquet_file is not None:
            return self._open_parquet_file

        if self._open_parquet_file is not None:
            self._open_parquet_file.close()

        shard = self.manifest.shards[file_index]
        self._open_parquet_file = self.parquet_source.open_parquet_file(
            shard.relative_path
        )
        self._open_file_index = file_index
        return self._open_parquet_file

    def _normalize_cursor(self, cursor: _SourceCursor) -> _SourceCursor:
        assigned_row_group_index = cursor.assigned_row_group_index
        row_in_group = cursor.row_in_group

        while assigned_row_group_index < len(self._assigned_row_groups):
            file_index, row_group_index = self._assigned_row_groups[assigned_row_group_index]
            row_group_rows = self.manifest.shards[file_index].row_group_rows[row_group_index]
            if row_in_group >= row_group_rows:
                assigned_row_group_index += 1
                row_in_group = 0
                continue
            return _SourceCursor(assigned_row_group_index, row_in_group)

        return _SourceCursor(len(self._assigned_row_groups), 0)

    def _is_eof(self, cursor: _SourceCursor) -> bool:
        return cursor.assigned_row_group_index >= len(self._assigned_row_groups)

    def _validate_cursor(self, cursor: _SourceCursor) -> None:
        normalized = self._normalize_cursor(cursor)
        if normalized != cursor:
            raise ValueError("Checkpoint cursor does not point to a valid document boundary.")
        if self._is_eof(cursor):
            return

    def _validate_split_plan_against_manifest(self) -> None:
        if self.split_plan.manifest_fingerprint != self.manifest.fingerprint:
            raise ValueError(
                "Split plan manifest fingerprint does not match the provided manifest."
            )

        seen: set[RowGroupRef] = set()
        for row_groups in (
            self.split_plan.train_row_groups,
            self.split_plan.val_row_groups,
        ):
            for row_group in row_groups:
                if row_group in seen:
                    raise ValueError(
                        "Split plan must assign each row group exactly once across splits."
                    )
                seen.add(row_group)
                self._resolve_row_group_ref(row_group)

        if len(seen) != self.manifest.total_row_groups:
            raise ValueError(
                "Split plan must cover every manifest row group exactly once."
            )

    def _resolve_split_row_groups(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            self._resolve_row_group_ref(row_group)
            for row_group in self.split_plan.row_groups_for_split(self.split)
        )

    def _resolve_row_group_ref(self, row_group: RowGroupRef) -> tuple[int, int]:
        file_index = self._file_index_by_relative_path.get(row_group.relative_path)
        if file_index is None:
            raise ValueError(
                f"Split plan references unknown parquet shard {row_group.relative_path!r}."
            )

        shard = self.manifest.shards[file_index]
        if row_group.row_group_index < 0 or row_group.row_group_index >= shard.num_row_groups:
            raise ValueError(
                f"Split plan references invalid row group {row_group.row_group_index} "
                f"for shard {row_group.relative_path!r}."
            )

        return file_index, row_group.row_group_index
