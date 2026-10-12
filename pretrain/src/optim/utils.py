from pathlib import Path
import random
import numpy as np
import torch
from contextlib import nullcontext
import torch.distributed as dist
import math

from tqdm.auto import trange


def get_batch(datareader, device="cpu"):
    x, y = datareader.sample_batch()
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def wsd_schedule(
    n_iterations,
    final_lr_factor=0.0,
    n_warmup=1000,
    init_div_factor=100,
    fract_decay=0.1,
):
    """Warmup, hold, and cosine decay schedule.
    Args:
        n_iterations: total number of iterations
        final_lr_factor: factor by which to reduce max_lr at the end
        n_warmup: number of warmup iterations
        init_div_factor: initial division factor for warmup
        fract_decay: fraction of iterations used for decay
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    n_anneal_steps = int(fract_decay * n_iterations)
    n_hold = n_iterations - n_anneal_steps

    def schedule(step):
        if step < n_warmup:
            return (step / n_warmup) + (1 - step / n_warmup) / init_div_factor
        elif step < n_hold:
            return 1.0
        elif step < n_iterations:
            return (
                final_lr_factor
                + (1 - final_lr_factor)
                * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                * 0.5
            )
        else:
            return final_lr_factor

    return schedule


def build_scheduler(opt, args):
    assert args.warmup_steps < args.iterations, "Warmup steps must be < total scheduler steps."
    lambda_schedule = wsd_schedule(
        n_iterations=args.iterations,
        n_warmup=args.warmup_steps,
        fract_decay=args.wsd_fract_decay,
        init_div_factor=1e2,
        final_lr_factor=args.wsd_final_lr_scale,
    )
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)


@torch.no_grad()
def eval(
    model,
    reader,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    loss_list_val, acc_list = [], []

    for idx in trange(max_num_batches):
        x, y = get_batch(reader, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True)
        val_loss = outputs["loss"]

        loss_list_val.append(val_loss)
        acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828**val_loss

    return val_acc, val_loss, val_perplexity


def save_checkpoint(model, opt, scheduler, itr, ckpt_dir: Path):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "itr": itr,
    }
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(checkpoint, ckpt_dir / "main.pt")


def load_checkpoint(
    model,
    opt,
    scheduler,
    ckpt_path,
    device,
):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    itr = ckpt["itr"]
    return itr


def save_worker_state(ckpt_dir: Path, train_reader=None):
    # Dataloader, rng states
    worker_state = {
        "rng_torch_cpu": torch.random.get_rng_state(),
        "rng_torch_gpu": torch.cuda.get_rng_state(),
        "rng_np": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    if train_reader is not None and hasattr(train_reader, "state_dict"):
        worker_state["train_reader_state"] = train_reader.state_dict()
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(worker_state, ckpt_dir / f"worker_{rank}.pt")


def load_worker_state(ckpt_dir: Path, train_reader=None):
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    worker_state = torch.load(ckpt_dir / f"worker_{rank}.pt", weights_only=False)
    torch.random.set_rng_state(worker_state["rng_torch_cpu"])
    torch.cuda.set_rng_state(worker_state["rng_torch_gpu"])
    np.random.set_state(worker_state["rng_np"])
    random.setstate(worker_state["rng_python"])

    if train_reader is not None and getattr(train_reader, "requires_checkpoint_state", False):
        if "train_reader_state" not in worker_state:
            raise RuntimeError(
                "Checkpoint is missing FineWeb train reader state. "
                "Old FineWeb streaming checkpoints are unsupported."
            )

    if train_reader is not None and "train_reader_state" in worker_state:
        if not hasattr(train_reader, "load_state_dict"):
            raise RuntimeError(
                "Checkpoint contains train reader state but the active reader "
                "does not support load_state_dict()."
            )
        train_reader.load_state_dict(worker_state["train_reader_state"])
        return True

    return False
