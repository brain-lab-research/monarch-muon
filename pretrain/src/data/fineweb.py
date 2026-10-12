from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from .fineweb_streaming_core import (
    FineWebEduStream,
    LocalParquetSource,
    Manifest,
    SplitPlan,
    build_manifest_from_source,
    build_snapshot_split_plan_with_val_blocks,
)


DEFAULT_FINEWEB_SPLIT_SEED = 2357
DEFAULT_DOC_BATCH_SIZE = 64
DEFAULT_PREFETCH_BATCHES = 4


def _direct_child_parquet_paths(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.glob("*.parquet") if path.is_file())


def _find_nested_parquet_dirs(dataset_root: Path) -> list[Path]:
    parquet_dirs = {path.parent for path in dataset_root.rglob("*.parquet") if path.is_file()}
    return sorted(parquet_dirs)


def _resolve_dataset_root(datasets_dir: str) -> Path:
    dataset_root = Path(datasets_dir).expanduser()
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"FineWeb dataset path does not exist or is not a directory: {dataset_root}"
        )
    if _direct_child_parquet_paths(dataset_root):
        return dataset_root

    parquet_dirs = _find_nested_parquet_dirs(dataset_root)
    if len(parquet_dirs) == 1:
        resolved_root = parquet_dirs[0]
        print(
            "FineWeb resolved --datasets-dir to nested parquet shard directory: "
            f"{resolved_root}"
        )
        return resolved_root

    if parquet_dirs:
        candidates = "\n".join(f"  - {path}" for path in parquet_dirs[:10])
        if len(parquet_dirs) > 10:
            candidates += f"\n  ... and {len(parquet_dirs) - 10} more"
        raise ValueError(
            "FineWeb found multiple nested parquet shard directories under "
            f"{dataset_root}. Set FINEWEB_DIR or --datasets-dir to the exact one:\n"
            f"{candidates}"
        )

    raise FileNotFoundError(
        "FineWeb requires --datasets-dir to point at a local parquet shard "
        f"directory, or a parent containing exactly one such directory. No parquet "
        f"shards found under {dataset_root}."
    )


def _resolve_parquet_source(args):
    return LocalParquetSource(_resolve_dataset_root(args.datasets_dir))


class FineWebValReader:
    def __init__(self, blocks: torch.Tensor, batch_size: int, sequence_length: int):
        if blocks.ndim != 2:
            raise ValueError("Validation blocks must be a 2-D tensor.")
        if blocks.shape[0] % batch_size != 0:
            raise ValueError("Validation blocks must divide exactly into full batches.")
        if blocks.shape[1] != sequence_length + 1:
            raise ValueError("Validation block width must equal sequence_length + 1.")

        self.blocks = blocks.contiguous()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.step = 0
        self._num_batches = self.blocks.shape[0] // batch_size

    def set_step(self, step: int):
        if step < 0 or step > self._num_batches:
            raise ValueError("Validation step is out of range.")
        self.step = step

    def num_batches(self):
        return self._num_batches

    def sample_batch(self):
        if self.step >= self._num_batches:
            raise RuntimeError("FineWeb validation reader exhausted")

        start = self.step * self.batch_size
        end = start + self.batch_size
        chunk = self.blocks[start:end]
        self.step += 1
        return chunk[:, :-1], chunk[:, 1:]


class FineWebTrainReader:
    requires_checkpoint_state = True

    def __init__(
        self,
        manifest: Manifest,
        split_plan: SplitPlan,
        tokenizer_factory: Callable[[], Any],
        *,
        parquet_source,
        tokenizer_name: str,
        batch_size: int,
        sequence_length: int,
        rank: int,
        world_size: int,
        num_token_workers: int,
        doc_batch_size: int = DEFAULT_DOC_BATCH_SIZE,
        prefetch_batches: int = DEFAULT_PREFETCH_BATCHES,
    ):
        self.manifest = manifest
        self.split_plan = split_plan
        self.tokenizer_factory = tokenizer_factory
        self.parquet_source = parquet_source
        self.source_metadata = parquet_source.to_dict()
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.rank = rank
        self.world_size = world_size
        self.num_token_workers = num_token_workers
        self.doc_batch_size = doc_batch_size
        self.prefetch_batches = prefetch_batches
        self.block_tokens = sequence_length + 1
        self.step = 0
        self._stream = self._make_stream()
        self._initial_stream_state = self._stream.state_dict()

    def _make_stream(self) -> FineWebEduStream:
        return FineWebEduStream(
            self.manifest,
            self.tokenizer_factory,
            block_tokens=self.block_tokens,
            split_plan=self.split_plan,
            split="train",
            rank=self.rank,
            world_size=self.world_size,
            worker_id=0,
            num_data_workers=1,
            num_token_workers=self.num_token_workers,
            doc_batch_size=self.doc_batch_size,
            prefetch_batches=self.prefetch_batches,
            parquet_source=self.parquet_source,
        )

    def _replace_stream(self, stream_state: dict[str, Any] | None = None):
        self._stream.close()
        self._stream = self._make_stream()
        if stream_state is not None:
            self._stream.load_state_dict(stream_state)

    def _wrap_stream(self):
        end_state = self._stream.state_dict()
        tail_tokens = list(end_state["token_buffer"])
        self._replace_stream()
        wrapped_state = self._stream.state_dict()
        wrapped_state["token_buffer"] = tail_tokens
        wrapped_state["committed_cursor"] = dict(
            self._initial_stream_state["committed_cursor"]
        )
        self._stream.load_state_dict(wrapped_state)

    def _next_block(self) -> list[int]:
        while True:
            try:
                return next(self._stream)
            except StopIteration:
                self._wrap_stream()

    def set_step(self, step: int):
        if step == self.step:
            return
        if step == 0:
            self.step = 0
            self._replace_stream(self._initial_stream_state)
            return
        raise RuntimeError(
            "FineWeb train reader cannot seek by step. Use checkpoint resume state instead."
        )

    def sample_batch(self):
        blocks = [self._next_block() for _ in range(self.batch_size)]
        batch = torch.tensor(blocks, dtype=torch.long)
        self.step += 1
        return batch[:, :-1], batch[:, 1:]

    def state_dict(self) -> dict[str, Any]:
        return {
            "reader_type": "fineweb_train_reader_v1",
            "manifest_fingerprint": self.manifest.fingerprint,
            "source_metadata": self.source_metadata,
            "tokenizer_name": self.tokenizer_name,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "step": self.step,
            "stream_state": self._stream.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]):
        if state.get("reader_type") != "fineweb_train_reader_v1":
            raise RuntimeError("Unsupported FineWeb train reader checkpoint format.")
        if (
            "manifest_fingerprint" in state
            and str(state["manifest_fingerprint"]) != self.manifest.fingerprint
        ):
            raise ValueError("Checkpoint manifest does not match this FineWeb reader.")
        state_source_metadata = state.get("source_metadata")
        if isinstance(state_source_metadata, dict):
            state_kind = str(state_source_metadata.get("kind", ""))
            current_kind = str(self.source_metadata.get("kind", ""))
            if state_kind != current_kind:
                raise ValueError("Checkpoint FineWeb source kind does not match.")
        if str(state["tokenizer_name"]) != self.tokenizer_name:
            raise ValueError("Checkpoint tokenizer does not match this FineWeb reader.")
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("Checkpoint batch_size does not match this FineWeb reader.")
        if int(state["sequence_length"]) != self.sequence_length:
            raise ValueError(
                "Checkpoint sequence_length does not match this FineWeb reader."
            )

        self.step = int(state["step"])
        self._replace_stream(state["stream_state"])


def _broadcast_split_plan(
    split_plan: SplitPlan | None,
    *,
    rank: int,
    world_size: int,
) -> SplitPlan:
    if world_size == 1:
        return split_plan
    assert dist.is_initialized(), (
        f"FineWeb split-plan broadcast requires torch.distributed to be initialized "
        f"(world_size={world_size})."
    )
    payload = [split_plan.to_dict()] if rank == 0 else [None]
    dist.broadcast_object_list(payload, src=0)
    return split_plan if rank == 0 else SplitPlan.from_dict(payload[0])


def _broadcast_val_blocks(
    val_blocks: torch.Tensor | None,
    *,
    rank: int,
    world_size: int,
    device: str | None,
) -> torch.Tensor:
    if world_size == 1:
        return val_blocks
    assert dist.is_initialized(), (
        f"FineWeb val broadcast requires torch.distributed to be initialized "
        f"(world_size={world_size})."
    )
    assert isinstance(device, str) and device.startswith("cuda"), (
        f"FineWeb val broadcast requires a CUDA device (got {device!r}); "
        "DDP uses NCCL, which does not support CPU tensors."
    )
    shape_payload = [list(val_blocks.shape)] if rank == 0 else [None]
    dist.broadcast_object_list(shape_payload, src=0)
    shape = shape_payload[0]
    staging = (
        val_blocks.to(device=device, dtype=torch.long)
        if rank == 0
        else torch.empty(shape, dtype=torch.long, device=device)
    )
    dist.broadcast(staging, src=0)
    return staging.to("cpu")


def build_fineweb_readers(
    args,
    *,
    tokenizer: Any,
    tokenizer_factory: Callable[[], Any],
    verbose: bool = True,
):
    block_tokens = args.sequence_length + 1
    val_sequences = args.eval_batches * args.eval_batch_size

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    parquet_source = _resolve_parquet_source(args)
    manifest = build_manifest_from_source(parquet_source)
    tokenizer_name = str(
        getattr(tokenizer, "name_or_path", None) or getattr(args, "tokenizer", "tokenizer")
    )
    num_token_workers = max(0, args.workers)

    if rank == 0:
        plan_with_val = build_snapshot_split_plan_with_val_blocks(
            manifest,
            tokenizer_factory,
            block_tokens=block_tokens,
            val_sequences=val_sequences,
            split_seed=DEFAULT_FINEWEB_SPLIT_SEED,
            shuffle_seed=args.data_seed,
            parquet_source=parquet_source,
        )
        split_plan = plan_with_val.plan
        val_blocks = (
            torch.tensor(plan_with_val.val_blocks, dtype=torch.long)
            if plan_with_val.val_blocks
            else torch.empty((0, block_tokens), dtype=torch.long)
        )
    else:
        split_plan = None
        val_blocks = None
    split_plan = _broadcast_split_plan(split_plan, rank=rank, world_size=world_size)

    train_reader = FineWebTrainReader(
        manifest,
        split_plan,
        tokenizer_factory,
        parquet_source=parquet_source,
        tokenizer_name=tokenizer_name,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        rank=rank,
        world_size=world_size,
        num_token_workers=num_token_workers,
        doc_batch_size=DEFAULT_DOC_BATCH_SIZE,
        prefetch_batches=DEFAULT_PREFETCH_BATCHES,
    )

    val_blocks = _broadcast_val_blocks(
        val_blocks,
        rank=rank,
        world_size=world_size,
        device=args.device if world_size > 1 else None,
    )
    val_reader = FineWebValReader(
        val_blocks,
        batch_size=args.eval_batch_size,
        sequence_length=args.sequence_length,
    )

    if verbose and rank == 0:
        print(f"Using FineWeb parquet source: {parquet_source.source_id}")
        print(
            f"FineWeb manifest: {len(manifest.shards)} shards, "
            f"{manifest.total_row_groups} row groups"
        )
        print(
            f"FineWeb split: {len(split_plan.train_row_groups)} train row groups, "
            f"{len(split_plan.val_row_groups)} val row groups"
        )
        print(
            f"FineWeb reader: world_size={world_size}, rank={rank}, "
            f"tokenizer_threads={num_token_workers}"
        )
        print(
            f"FineWeb val snapshot: {val_reader.num_batches()} batches x "
            f"{args.eval_batch_size} examples"
        )

    return {"train": train_reader, "val": val_reader}
