"""
MonarchMuon optimizer — Muon with batched Newton-Schulz for 3-D block-diagonal
parameters (blkdiag1 / blkdiag2 from MonarchLinear) plus an embedded AdamW for
embeddings, biases, layer-norm weights, and the lm_head.

Design: a single torch.optim.Optimizer with two param groups so that a
standard LR scheduler can wrap it and update both groups via param_groups.

  Group 0 (update_type="muon"):  3-D block-diagonal Monarch factors
  Group 1 (update_type="adamw"): 1-D params, embeddings, lm_head

Usage (from main.py):
    from models.monarch import MonarchMuonOptimizer
    opt = MonarchMuonOptimizer(
        muon_params=muon_params,
        adamw_params=adamw_params,
        lr=args.lr,
        momentum=args.momentum,
        adamw_betas=(args.beta1, args.beta2),
        adamw_weight_decay=args.weight_decay,
        adamw_eps=args.eps,
    )
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch


# ---------------------------------------------------------------------------
# Newton-Schulz helpers
# ---------------------------------------------------------------------------

def _newton_schulz_batched(B: torch.Tensor) -> torch.Tensor:
    """Batched Newton-Schulz for a stack of matrices: (nblocks, d1, d2)."""
    norms = B.norm(p="fro", dim=(-2, -1), keepdim=True)
    B = B / (norms + 1e-7)
    if B.shape[-2] > B.shape[-1]:
        B = B.transpose(-2, -1)
        do_T = True
    else:
        do_T = False
    for _ in range(5):
        BBt = B @ B.transpose(-2, -1)
        B = 3.4445 * B - 4.7750 * BBt @ B + 2.0315 * (BBt @ BBt) @ B
    if do_T:
        B = B.transpose(-2, -1)
    return B


# ---------------------------------------------------------------------------
# Combined optimizer
# ---------------------------------------------------------------------------

class MonarchMuonOptimizer(torch.optim.Optimizer):
    """Muon (with 3-D batching) + AdamW in a single Optimizer object.

    Parameters
    ----------
    muon_params : list[Tensor]
        Parameters to update with the Muon rule (3-D Monarch factors).
    adamw_params : list[Tensor]
        Parameters to update with AdamW (1-D, embeddings, lm_head).
    lr : float
        Base learning rate, shared by both groups (scheduler will scale it).
    momentum : float
        Momentum for the Muon group.
    adamw_betas : tuple[float, float]
        (beta1, beta2) for the AdamW group.
    adamw_weight_decay : float
        Weight decay for the AdamW group.
    adamw_eps : float
        Epsilon for the AdamW group.
    """

    def __init__(
        self,
        muon_params,
        adamw_params,
        *,
        lr: float = 5e-3,
        momentum: float = 0.95,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_weight_decay: float = 0.1,
        adamw_eps: float = 1e-8,
    ):
        muon_params = list(muon_params)
        adamw_params = list(adamw_params)

        defaults = dict(lr=lr)
        param_groups = [
            {
                "params": muon_params,
                "update_type": "muon",
                "momentum": momentum,
            },
            {
                "params": adamw_params,
                "update_type": "adamw",
                "betas": adamw_betas,
                "weight_decay": adamw_weight_decay,
                "eps": adamw_eps,
            },
        ]
        super().__init__(param_groups, defaults)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            if group["update_type"] == "muon":
                self._muon_step(group, lr)
            else:
                self._adamw_step(group, lr)

        return loss

    # ------------------------------------------------------------------
    # Private: Muon update (3-D params, batched per block shape)
    # ------------------------------------------------------------------

    def _get_momentum_buffer(self, param: torch.Tensor) -> torch.Tensor:
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(param)
        return state["momentum_buffer"]

    def _muon_step(self, group: dict, lr: float) -> None:
        momentum = group["momentum"]

        params = [p for p in group["params"] if p.grad is not None]
        bufs = [self._get_momentum_buffer(p) for p in params]
        grads = [p.grad for p in params]

        # momentum update: buf = momentum * buf + grad
        for buf, grad in zip(bufs, grads):
            buf.mul_(momentum).add_(grad)

        # Accumulate 3-D params by (device, dtype, a, b) for batched NS
        grouped_3d: dict = defaultdict(list)
        for param, buf in zip(params, bufs):
            key = (buf.device, buf.dtype, buf.shape[1], buf.shape[2])
            grouped_3d[key].append((param, buf))

        # Batched NS for same-shaped 3-D blocks
        for (_, _, a, b), items in grouped_3d.items():
            merged = torch.cat([buf for _, buf in items], dim=0)
            ortho_merged = _newton_schulz_batched(merged)
            scale = (a / b) ** 0.5
            offset = 0
            for param, buf in items:
                n = buf.shape[0]
                param.data.add_(ortho_merged[offset:offset + n], alpha=-lr * scale)
                offset += n

    # ------------------------------------------------------------------
    # Private: AdamW update
    # ------------------------------------------------------------------

    def _adamw_step(self, group: dict, lr: float) -> None:
        beta1, beta2 = group["betas"]
        weight_decay = group["weight_decay"]
        eps = group["eps"]

        for param in group["params"]:
            if param.grad is None:
                continue
            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)

            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            grad = param.grad

            if weight_decay > 0:
                param.data.mul_(1.0 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            step_size = lr / bc1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
            param.data.addcdiv_(exp_avg, denom, value=-step_size)
