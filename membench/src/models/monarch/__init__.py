"""Monarch-Muon: butterfly linears + batched-Newton-Schulz Muon.

The configuration used by the benchmark is:

    apply_monarch(model, nblocks=4)   # swap nn.Linear -> MonarchLinear
    patch_monarch_linear()            # custom op, blocked layout, Triton riffle
    MonarchMuonOptimizer(...)         # BF16 batched Newton-Schulz, foreach updates
"""
from .monarch_linear import MonarchLinear, apply_monarch
from .monarch_muon import MonarchMuonOptimizer
from .monarch_ops import patch_monarch_linear

__all__ = [
    "MonarchLinear",
    "apply_monarch",
    "MonarchMuonOptimizer",
    "patch_monarch_linear",
]
