"""Model and optimizer of a train.py arm, built exactly as src/main.py builds them (single GPU, no data)."""
import argparse

import torch

from config.base import parse_args
from models.llama import Llama
from models.monarch import apply_monarch
from train import arm_args


def build(arm):
    config = parse_args(argparse.ArgumentParser(), ["--experiment-name", "test", *arm_args(arm)], None)
    config.qargs = None
    if config.fp8 or config.fp8_optim:
        from third_party.coat.utils._fp8_quantization_config import QuantizationConfig
        config.qargs = QuantizationConfig()
    torch.manual_seed(config.seed)
    model = Llama(config).cuda().train()
    monarch = config.opt == "monarch_muon" or config.apply_monarch
    if monarch:
        apply_monarch(model, nblocks=config.monarch_nblocks, verbose=False)
    if config.opt == "monarch_muon":
        muon, adamw = [], []
        for name, parameter in model.named_parameters():
            selected = parameter.ndim >= 2 and not any(x in name for x in ("wte", "wpe", "lm_head", "embed"))
            (muon if selected else adamw).append(parameter)
        if config.fp8_optim:
            from models.monarch.coat_state_muon import CoatStateMonarchMuon as MonarchMuonOptimizer
            extra = {"qargs": config.qargs}
        else:
            from models.monarch import MonarchMuonOptimizer
            extra = {}
        optimizer = MonarchMuonOptimizer(muon_params=muon, adamw_params=adamw, lr=config.lr, **extra,
            momentum=config.momentum, adamw_betas=(config.beta1, config.beta2),
            adamw_weight_decay=config.weight_decay, adamw_eps=config.eps)
    else:
        named = dict(model.named_parameters())
        groups = [dict(g, params=[named[n] for n in g["params"]]) for g in model.get_parameter_group_specs()]
        if config.opt == "adamw":
            optimizer = torch.optim.AdamW(groups, lr=config.lr, betas=(config.beta1, config.beta2),
                                          weight_decay=config.weight_decay)
        else:
            from third_party.coat.optimizer.triton_fp8_adamw import TritonCoatAdamW
            optimizer = TritonCoatAdamW(groups, lr=config.lr, betas=(config.beta1, config.beta2),
                                        weight_decay=config.weight_decay, qargs=config.qargs)
    if config.fp8 and monarch:
        from models.coat_monarch import attach
        attach(model, optimizer)
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    return model, optimizer, config
