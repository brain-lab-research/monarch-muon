"""
MonarchMuon optimizer — Muon with batched Newton-Schulz for 3-D block-diagonal
parameters (blkdiag1 / blkdiag2 from MonarchLinear) plus an embedded AdamW for
embeddings, biases, layer-norm weights, and the lm_head.

Design: a single torch.optim.Optimizer with two param groups so that a
standard LR scheduler can wrap it and update both groups via param_groups.

  Group 0 (update_type="muon"):  3-D block-diagonal Monarch factors
  Group 1 (update_type="adamw"): 1-D params, embeddings, lm_head

Usage:
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
        3-D Monarch factors to update with the Muon rule; Newton-Schulz runs
        in BF16.
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
    # Private: Muon update (3-D params)
    # ------------------------------------------------------------------

    def _get_momentum_buffer(self, param: torch.Tensor) -> torch.Tensor:
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(param)
        return state["momentum_buffer"]

    def _muon_step(self, group: dict, lr: float) -> None:
        momentum = group["momentum"]

        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        bufs = [self._get_momentum_buffer(p) for p in params]
        grads = [p.grad for p in params]

        # momentum update: buf = momentum * buf + grad
        torch._foreach_mul_(bufs, momentum)
        torch._foreach_add_(bufs, grads)

        # Accumulate 3-D params by (device, dtype, a, b) for batched NS
        grouped_3d: dict = defaultdict(list)
        for param, buf in zip(params, bufs):
            key = (buf.device, buf.dtype, buf.shape[1], buf.shape[2])
            grouped_3d[key].append((param, buf))

        # Batched NS for same-shaped 3-D blocks
        for (_, _, a, b), items in grouped_3d.items():
            merged = torch.cat([buf for _, buf in items], dim=0)
            if merged.dtype != torch.bfloat16:
                merged = merged.to(torch.bfloat16)
            ortho_merged = _newton_schulz_batched(merged)
            if ortho_merged.dtype != items[0][1].dtype:
                ortho_merged = ortho_merged.to(items[0][1].dtype)
            scale = (a / b) ** 0.5
            updates = list(ortho_merged.split([buf.shape[0] for _, buf in items], dim=0))
            torch._foreach_add_([p.data for p, _ in items], updates,
                                alpha=-lr * scale)

    # ------------------------------------------------------------------
    # Private: AdamW update
    # ------------------------------------------------------------------

    def _adamw_step(self, group: dict, lr: float) -> None:
        beta1, beta2 = group["betas"]
        weight_decay = group["weight_decay"]
        eps = group["eps"]

        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        for param in params:
            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
            state["step"] += 1
        # Every parameter in the group steps together, so one step count serves all.
        step = self.state[params[0]]["step"]
        grads = [p.grad for p in params]
        exp_avgs = [self.state[p]["exp_avg"] for p in params]
        exp_avg_sqs = [self.state[p]["exp_avg_sq"] for p in params]
        if weight_decay > 0:
            torch._foreach_mul_([p.data for p in params], 1.0 - lr * weight_decay)
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        denom = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(denom, math.sqrt(bc2))
        torch._foreach_add_(denom, eps)
        torch._foreach_addcdiv_([p.data for p in params], exp_avgs, denom,
                                value=-lr / bc1)
