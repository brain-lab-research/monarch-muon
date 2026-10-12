import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import random
import time

from safetensors.torch import load_model
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.optim import ZeroRedundancyOptimizer
from transformers import AutoTokenizer

from sft_data import read_split
from sft_model import load_arm, prepare_projection, save_checkpoint, sha256_file
from sft_optimizer import SFTOptimizer, parameter_groups


def initialize():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.backends.cuda.matmul.allow_tf32 = True
    return rank, world, device


def batch_row(row, length, pad, device):
    n = len(row["input_ids"])
    return {"input_ids": torch.tensor([row["input_ids"] + [pad] * (length - n)], device=device),
            "attention_mask": torch.tensor([[1] * n + [0] * (length - n)], device=device),
            "labels": torch.tensor([row["labels"] + [-100] * (length - n)], device=device)}


def summed_loss(model, batch):
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   use_cache=False)
    logits, labels = output.logits[:, :-1], batch["labels"][:, 1:]
    selected = labels != -100
    return torch.nn.functional.cross_entropy(logits[selected].float(), labels[selected], reduction="sum")


@torch.no_grad()
def evaluate(model, rows, config, tokenizer, rank, world, device):
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    for row in rows[rank::world]:
        batch = batch_row(row, config["sequence_length"], tokenizer.pad_token_id, device)
        totals[0] += summed_loss(model, batch).double()
        totals[1] += (batch["labels"][:, 1:] != -100).sum()
    dist.all_reduce(totals)
    model.train()
    return {"nll": (totals[0] / totals[1]).item(), "tokens": int(totals[1].item())}


def train_trial(model, rows, validation, config, tokenizer, optimizer_name, lr,
                rank, world, device, output, run_name, smoke):
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ddp = DistributedDataParallel(model, device_ids=[device.index], gradient_as_bucket_view=True,
                                  broadcast_buffers=False)
    groups = parameter_groups(model, optimizer_name, lr, config)
    optimizer = ZeroRedundancyOptimizer(groups, optimizer_class=SFTOptimizer,
                                       momentum=config["momentum"], betas=tuple(config["adamw_betas"]),
                                       weight_decay=config["weight_decay"], eps=config["adamw_eps"])
    initial_lrs = [g["lr"] for g in optimizer.param_groups]
    order = list(range(len(rows)))
    random.Random(config["seed"]).shuffle(order)
    batch_size = config["global_batch_size"]
    steps = math.ceil(len(order) / batch_size)
    if smoke:
        steps = min(steps, 2)
    warmup = max(1, math.ceil(steps * config["warmup_fraction"]))
    baseline = evaluate(model, validation, config, tokenizer, rank, world, device)
    history = []
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()
    training_started = time.monotonic()
    # Let DDP establish gradient buckets before optimizer states consume memory.
    with torch.random.fork_rng(devices=[device.index]):
        row = rows[order[rank]]
        warmup_batch = batch_row(row, config["sequence_length"], tokenizer.pad_token_id, device)
        (summed_loss(ddp, warmup_batch) / row["supervised_tokens"]).backward()
    optimizer.zero_grad(set_to_none=True)
    del warmup_batch
    for step in range(steps):
        indices = order[step * batch_size:(step + 1) * batch_size]
        # Pad the index list with masked dummy examples, without repeating any training targets.
        per_rank = math.ceil(len(indices) / world)
        padded = indices + [None] * (per_rank * world - len(indices))
        local_indices = padded[rank::world]
        token_count = sum(rows[i]["supervised_tokens"] for i in indices)
        multiplier = ((step + 1) / warmup if step < warmup
                      else max(0.0, (steps - step) / max(1, steps - warmup)))
        for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
            group["lr"] = initial_lr * multiplier
        # Keep DDP bucket views during no_sync accumulation instead of allocating duplicate gradients.
        optimizer.zero_grad(set_to_none=False)
        torch.cuda.synchronize()
        started = time.monotonic()
        total_loss = torch.zeros((), dtype=torch.float64, device=device)
        for micro, index in enumerate(local_indices):
            row = rows[index if index is not None else indices[0]]
            batch = batch_row(row, config["sequence_length"], tokenizer.pad_token_id, device)
            context = ddp.no_sync() if micro < len(local_indices) - 1 else nullcontext()
            with context:
                loss_sum = summed_loss(ddp, batch)
                if index is None:
                    loss_sum = loss_sum * 0.0
                (loss_sum * world / token_count).backward()
            total_loss += loss_sum.detach().double()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"],
                                                   error_if_nonfinite=True)
        torch.cuda.synchronize()
        optimizer_started = time.monotonic()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = torch.tensor([time.monotonic() - started,
                                time.monotonic() - optimizer_started], device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(total_loss)
        record = {"step": step + 1, "train/nll": total_loss.item() / token_count,
                  "train/target_tokens": token_count, "train/grad_norm": grad_norm.item(),
                  "time/step_seconds": elapsed[0].item(),
                  "time/optimizer_seconds": elapsed[1].item(),
                  "train/lr": optimizer.param_groups[0]["lr"],
                  "throughput/input_tokens_per_second": len(indices) * config["sequence_length"] / elapsed[0].item(),
                  "throughput/target_tokens_per_second": token_count / elapsed[0].item()}
        history.append(record)
        if rank == 0:
            with (output / f"{run_name}.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if step == 0 or (step + 1) % 10 == 0:
                print(json.dumps({"run": run_name, **record}), flush=True)
    dist.barrier()
    training_seconds = time.monotonic() - training_started
    validation_score = evaluate(model, validation, config, tokenizer, rank, world, device)
    memory = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                          dtype=torch.float64, device=device)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    result = {"run": run_name, "optimizer": optimizer_name, "lr": lr,
              "baseline": baseline, "validation": validation_score,
              "training_seconds": training_seconds, "steps": steps,
              "target_tokens": sum(r["train/target_tokens"] for r in history),
              "max_allocated_bytes": int(memory[0]), "max_reserved_bytes": int(memory[1]),
              "parameters": sum(p.numel() for p in model.parameters())}
    if rank == 0:
        (output / f"{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
        if smoke:
            print("SFT_SMOKE_METRICS=" + json.dumps(result), flush=True)
    del optimizer, ddp
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sft_config.json")
    parser.add_argument("--model-dir", required=True, help="Local Qwen2.5-7B directory.")
    parser.add_argument("--data", required=True, help="Directory written by sft_data.py.")
    parser.add_argument("--root", required=True, help="Output directory for projection, runs and checkpoints.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--trial-index", type=int, choices=range(12))
    mode.add_argument("--evaluate-index", type=int, choices=range(4))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["epochs"] != 1 or config["micro_batch_size"] != 1:
        raise ValueError("This runner implements one epoch with one example per microbatch")
    rank, world, device = initialize()
    expected_world = 1 if args.evaluate_index is not None else config["gpus"]
    gpu = torch.cuda.get_device_name(device)
    if world != expected_world or "H100" not in gpu:
        raise RuntimeError(f"The paper setup uses {expected_world} H100 80GB GPU(s) for this mode; "
                           f"got {world} process(es) on {gpu}")
    root = Path(args.root)
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    if (manifest["config"] | {"gpus": config["gpus"]}) != config:
        raise ValueError("Data manifest differs from the experiment configuration")
    for split, record in manifest["splits"].items():
        if sha256_file(data / f"{split}.jsonl") != record["sha256"]:
            raise ValueError("Dataset checksum mismatch")
    arms = [(s, a) for s in ("dense", "monarch") for a in ("adamw", "muon")]
    grid = [(s, a, lr) for s, a in arms for lr in config[f"{a}_lrs"]]
    name_of = lambda s, a, lr: f"{s}-{a}-lr{lr:g}-g{config['gpus']}"
    smoke_output = root / f"smoke-g{config['gpus']}"
    runs_output = root / f"runs-g{config['gpus']}"
    checkpoint_root = root / "checkpoints"
    projection = root / "monarch-initial.safetensors"
    projection_sha = [None]
    if rank == 0:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        if args.smoke:
            if not projection.with_suffix(".json").exists():
                prepare_projection(config, args.model_dir, projection, device)
        elif not projection.with_suffix(".json").exists():
            raise RuntimeError("Run --smoke first: it creates the shared Monarch projection")
        initialization = json.loads(projection.with_suffix(".json").read_text())
        if (initialization["config"] | {"gpus": config["gpus"]}) != config:
            raise ValueError("Initialization configuration mismatch")
        projection_sha[0] = sha256_file(projection)
        if projection_sha[0] != initialization["sha256"]:
            raise ValueError("Initialization checksum mismatch")
        print("SFT_INITIALIZATION_VERIFIED " + projection_sha[0], flush=True)
    dist.broadcast_object_list(projection_sha, src=0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    identity = {"world_size": world, "gpu": gpu,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "config": config, "projection_sha256": projection_sha[0],
                "data_manifest_sha256": sha256_file(data / "manifest.json")}

    if args.evaluate_index is not None:
        structure, algorithm = arms[args.evaluate_index]
        candidates = []
        for lr in config[f"{algorithm}_lrs"]:
            name = name_of(structure, algorithm, lr)
            result = json.loads((runs_output / name / "complete.json").read_text())
            if result["identity"]["projection_sha256"] != projection_sha[0]:
                raise ValueError("Candidates used different initializations")
            candidates.append(result)
        best = min(candidates, key=lambda r: r["validation"]["nll"])
        arm_name = f"{structure}-{algorithm}-g{config['gpus']}"
        output = root / f"evaluation-g{config['gpus']}" / arm_name
        output.mkdir(parents=True, exist_ok=False)
        checkpoint = checkpoint_root / f"{best['run']}.safetensors"
        if sha256_file(checkpoint) != best["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint checksum mismatch")
        model = load_arm(config, args.model_dir, structure, projection, device)
        load_model(model, str(checkpoint), device=str(device), strict=True)
        test = read_split(data / "test.jsonl")
        best["test"] = evaluate(model, test, config, tokenizer, rank, world, device)
        (output / f"{arm_name}-selected.json").write_text(json.dumps(best, indent=2) + "\n")
        (output / "complete.json").write_text(json.dumps(best, indent=2) + "\n")
        print("SFT_SELECTED_EVALUATION_COMPLETE " + str(args.evaluate_index), flush=True)
        dist.destroy_process_group()
        return

    rows, validation = read_split(data / "train.jsonl"), read_split(data / "validation.jsonl")
    if args.smoke:
        rows, validation = rows[:64], validation[:8]
        trials = [(s, a, config[f"{a}_lrs"][0]) for s, a in arms]
        output = smoke_output
    else:
        trials = [grid[args.trial_index]]
        output = runs_output / name_of(*trials[0])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    dist.barrier()
    results = []
    for structure, algorithm, lr in trials:
        trial_started = time.monotonic()
        model = load_arm(config, args.model_dir, structure, projection, device)
        name = name_of(structure, algorithm, lr)
        result = train_trial(model, rows, validation, config, tokenizer, algorithm,
                             lr, rank, world, device, output, name, args.smoke)
        result.update(structure=structure, identity=identity)
        if rank == 0 and not args.smoke:
            checkpoint = checkpoint_root / f"{name}.safetensors"
            result["checkpoint_sha256"] = save_checkpoint(model, checkpoint)
            result["wall_seconds_load_train_validate_save"] = time.monotonic() - trial_started
            (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    if rank == 0:
        (output / "complete.json").write_text(json.dumps(results if args.smoke else results[0], indent=2) + "\n")
        print("SFT_SMOKE_PASS" if args.smoke else f"SFT_TRIAL_COMPLETE {args.trial_index}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
