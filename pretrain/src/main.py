import argparse
import json
import sys
import os
from pathlib import Path
import random

import numpy as np
import torch
import wandb

# Add src/ to path so all modules resolve correctly when running from project root
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in [_SRC, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.base import parse_args
from data.utils import get_tokenizer
from data.fineweb import build_fineweb_readers
import distributed
from models.llama import Llama
from optim.base import train
from optim.utils import build_scheduler


def define_wandb_metrics():
    wandb.define_metric("iter")
    wandb.define_metric("train/*", step_metric="iter")
    wandb.define_metric("val/*", step_metric="iter")
    wandb.define_metric("final-val/*", step_metric="iter")
    wandb.define_metric("memory/*", step_metric="iter")
    wandb.define_metric("lr", step_metric="iter")
    wandb.define_metric("iter_dt", step_metric="iter")
    wandb.define_metric("tok_gpu_sec", step_metric="iter")
    wandb.define_metric("grad_norm", step_metric="iter")
    wandb.define_metric("consumed_tokens", step_metric="iter")


def main(args):
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()

    if not 0 <= args.stop_after <= args.iterations:
        raise ValueError("--stop-after must be between zero and --iterations.")
    monarch = args.opt == "monarch_muon" or args.apply_monarch
    if args.fp8 and not monarch:
        raise ValueError("--fp8 requires a Monarch model (--apply-monarch or --opt monarch_muon).")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))

    # ── FP8 setup ────────────────────────────────────────────────────────────
    args.qargs = None
    if args.fp8 or args.fp8_optim:
        from third_party.coat.utils._fp8_quantization_config import QuantizationConfig
        args.qargs = QuantizationConfig()
        if distributed_backend.is_master_process():
            print(f"\nFP8 QuantizationConfig:\n{args.qargs}\n")

    # ── Experiment naming / WandB ─────────────────────────────────────────
    exp_name = args.experiment_name
    wandb_group = get_wandb_group(args)
    args.wandb_group = wandb_group
    exp_dir = Path(args.results_base_folder) / wandb_group / exp_name
    print(f"Starting Experiment: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Config:\n{vars(args)}\n")

    datareaders = get_data_readers(args)

    if distributed_backend.is_master_process() and args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=exp_name,
            group=wandb_group,
            config=vars(args),
        )
        define_wandb_metrics()

    model = Llama(args).to(args.device)
    print(f"\nModel:\n{model}")

    if monarch:
        from models.monarch import apply_monarch

        apply_monarch(model, nblocks=args.monarch_nblocks)

    model = distributed_backend.transform_model(model)

    # ── Parameter groups ───────────────────────────────────────────────
    if args.opt == "monarch_muon":
        group_specs = None
        optimized_params_cnt = sum(
            p.numel()
            for p in distributed_backend.get_raw_model(model).parameters()
            if p.requires_grad
        )
    else:
        group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
        param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
        optimized_params_cnt = 0
        for g in group_specs:
            params = []
            for p_name in g["params"]:
                translated_p_names = (
                    distributed_backend.translate_model_parameter_name_for_node(p_name)
                )
                params += [param_name_mapping[p_name] for p_name in translated_p_names]
            g["params"] = params
            optimized_params_cnt += sum([p.numel() for p in g["params"]])

    params_cnt = distributed_backend.get_raw_model(model).get_num_params()
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
    if args.wandb and distributed_backend.is_master_process():
        wandb.log({"parameters": params_cnt, "optimized_parameters": optimized_params_cnt})

    # ── Optimiser ─────────────────────────────────────────────────────────
    if args.opt == "triton_coat_adamw":
        from third_party.coat.optimizer.triton_fp8_adamw import TritonCoatAdamW
        if args.qargs is None:
            raise ValueError("triton_coat_adamw requires --fp8-optim (which builds qargs).")
        opt = TritonCoatAdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            qargs=args.qargs,
        )
    elif args.opt == "muon":
        from optim.muon import Muon
        muon_params, adamw_params = [], []
        for name, p in distributed_backend.get_raw_model(model).named_parameters():
            if p.ndim == 2 and not any(k in name for k in ("wte", "lm_head", "embed")):
                muon_params.append(p)
            else:
                adamw_params.append(p)
        opt = Muon(muon_params, adamw_params, lr=args.lr, weight_decay=args.weight_decay,
                   momentum=args.momentum, adamw_betas=(args.beta1, args.beta2))
    elif args.opt == "monarch_muon":
        if args.fp8_optim:
            from functools import partial
            from models.monarch.coat_state_muon import CoatStateMonarchMuon
            optimizer_class = partial(CoatStateMonarchMuon, qargs=args.qargs)
        else:
            from models.monarch import MonarchMuonOptimizer as optimizer_class

        raw_model = distributed_backend.get_raw_model(model)
        muon_params = []
        adamw_params = []
        for name, parameter in raw_model.named_parameters():
            if parameter.ndim >= 2 and not any(
                excluded in name for excluded in ("wte", "wpe", "lm_head", "embed")
            ):
                muon_params.append(parameter)
            else:
                adamw_params.append(parameter)
        opt = optimizer_class(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=args.lr,
            momentum=args.momentum,
            adamw_betas=(args.beta1, args.beta2),
            adamw_weight_decay=args.weight_decay,
            adamw_eps=args.eps,
        )
    elif args.opt == "adamw":
        opt = torch.optim.AdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    else:
        from optim.memory_efficient.frugal import GaloreAdamW
        opt = GaloreAdamW(
            group_specs,
            update_gap=args.update_gap,
            density=args.density,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            betas=(args.beta1, args.beta2),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.eps,
        )

    if args.fp8:
        from models.coat_monarch import attach
        attach(distributed_backend.get_raw_model(model), opt)

    print(f"\nOptimizer:\n{opt}")

    # ── LR scheduler ──────────────────────────────────────────────────────
    scheduler = build_scheduler(opt, args)

    # ── Auto-resume ───────────────────────────────────────────────────────
    args.resume_from = None
    if (exp_dir / "ckpts" / "latest" / "main.pt").exists():
        args.resume_from = str(exp_dir / "ckpts" / "latest")
    elif distributed_backend.is_master_process():
        exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────
    stats = train(
        model=model,
        opt=opt,
        datareaders=datareaders,
        scheduler=scheduler,
        exp_dir=exp_dir,
        distributed_backend=distributed_backend,
        cfg=args,
    )

    # We don't need to save such complex structure as it's fields are already in args
    del args.qargs
    stats["args"] = vars(args)
    if distributed_backend.is_master_process():
        with open(exp_dir / "summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


def get_wandb_group(args):
    """Returns the wandb group: explicit --wandb-group or auto-generated from model/data/iterations."""
    if args.wandb_group is not None:
        return args.wandb_group
    return f"llama-{args.n_layer}L{args.n_head}H_fineweb_{args.iterations // 1000}k"


def get_data_readers(args):
    """FineWeb parquet shards, streamed and tokenized on the fly."""
    return build_fineweb_readers(
        args,
        tokenizer=get_tokenizer(args),
        tokenizer_factory=lambda: get_tokenizer(args, verbose=False),
    )


if __name__ == "__main__":
    main(parse_args(argparse.ArgumentParser(allow_abbrev=False), None, None))
