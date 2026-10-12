"""Monarch linears (block-diagonal factors + butterfly permutations) and Monarch-Muon.

    apply_monarch(model, nblocks=2)          # swap nn.Linear -> MonarchLinear (lm_head excluded)

MonarchLinear runs the butterfly multiply as an opaque custom op in the blocked
layout, with Triton riffle/unriffle kernels for the permutation between the factors.
"""
from .monarch_linear import MonarchLinear, apply_monarch
from .monarch_muon import MonarchMuonOptimizer

__all__ = [
    "MonarchLinear",
    "apply_monarch",
    "MonarchMuonOptimizer",
]
