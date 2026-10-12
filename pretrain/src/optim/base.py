from contextlib import nullcontext
from pathlib import Path
import time

import torch
import wandb
from tqdm import tqdm

from .utils import (
    eval,
    get_batch,
    load_checkpoint,
    load_worker_state,
    save_checkpoint,
    save_worker_state,
)


def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        type_ctx = nullcontext()

    train_reader, val_reader = datareaders["train"], datareaders["val"]
    loaded_train_reader_state = False

    if cfg.resume_from:
        # Full resume: model weights, optimizer, scheduler, dataloader state and random state.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
        )
        loaded_train_reader_state = load_worker_state(
            ckpt_dir,
            train_reader=train_reader,
        )
    else:
        curr_iter = 0

    substep = curr_iter * cfg.acc_steps
    if not loaded_train_reader_state:
        train_reader.set_step(substep)
    stats = {
        "train_loss": [],
        "val_loss": [],
        "val_pp": [],
        "val_acc": [],
    }
    model.train()

    # Initialize the progress bar
    if distributed_backend.is_master_process():
        pbar = tqdm(total=cfg.iterations, desc="Training Progress", position=curr_iter)
    else:
        pbar = None

    last_iter = min(cfg.iterations, cfg.stop_after or cfg.iterations)
    while curr_iter <= last_iter:
        # Save temporary checkpoint for resuming training
        if curr_iter > 0 and cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == last_iter:
                ckpt_dir = exp_dir / "ckpts" / "latest"
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir, train_reader=train_reader)

        ws = distributed_backend.get_world_size()
        if curr_iter % cfg.eval_interval == 0 or curr_iter == last_iter:
            eval_result = eval_and_log(
                curr_iter,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
            )
            if eval_result is not None:
                val_loss, val_pp, val_acc = eval_result
                stats["val_loss"].append(val_loss)
                stats["val_pp"].append(val_pp)
                stats["val_acc"].append(val_acc)

        if curr_iter == last_iter:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # Train model
        t_start = time.perf_counter_ns()
        if "cuda" in cfg.device:
            torch.cuda.reset_peak_memory_stats()

        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = get_batch(train_reader, device=cfg.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    outputs = model(x, targets=y)

            loss = outputs["loss"] / cfg.acc_steps
            with type_ctx:
                loss.backward()
            substep += 1

        if cfg.grad_clip != 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        dt = (time.perf_counter_ns() - t_start) / 1e9

        curr_iter += 1
        if distributed_backend.is_master_process():
            pbar.update(1)

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        ):
            train_loss = loss.detach().cpu().item() * cfg.acc_steps
            stats["train_loss"].append(train_loss)
            consumed_tokens = curr_iter * ws * cfg.acc_steps * cfg.batch_size * cfg.sequence_length

            current_lrs = [param_group["lr"] for param_group in opt.param_groups]

            peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if "cuda" in cfg.device else 0.0
            reserved_mem_gb = torch.cuda.memory_reserved() / 1e9 if "cuda" in cfg.device else 0.0

            print(
                f"Train: Iter={curr_iter} "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"lr={current_lrs[0]:.2e} "
                f"peak_mem={peak_mem_gb:.2f}GB "
                f"consumed_tokens={consumed_tokens}"
            )

            if cfg.wandb:
                wandb.log(
                    {
                        "iter": curr_iter,
                        "train/loss": train_loss,
                        "train/perplexity": 2.71828**train_loss,
                        "lr": current_lrs[0],
                        "iter_dt": dt,
                        "consumed_tokens": consumed_tokens,
                        "tok_gpu_sec": cfg.sequence_length * cfg.batch_size * cfg.acc_steps / dt,
                        "grad_norm": grad_norm,
                        "memory/peak_allocated_gb": peak_mem_gb,
                        "memory/reserved_gb": reserved_mem_gb,
                    }
                )

    return stats


def eval_and_log(
    curr_iter,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return None

    model.eval()

    if curr_iter == cfg.iterations:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
    )

    print(
        f">Eval: Iter={curr_iter} "
        f"consumed_tokens={curr_iter * distributed_backend.get_world_size() * cfg.acc_steps * cfg.batch_size * cfg.sequence_length} "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations:
            logs = {
                "iter": curr_iter,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
                "consumed_tokens": curr_iter * distributed_backend.get_world_size() * cfg.acc_steps * cfg.batch_size * cfg.sequence_length,
            }
        else:
            logs = {
                "iter": curr_iter,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
                "consumed_tokens": curr_iter * distributed_backend.get_world_size() * cfg.acc_steps * cfg.batch_size * cfg.sequence_length,
            }

        wandb.log(logs)
    model.train()
    return val_loss, val_perplexity, val_acc
