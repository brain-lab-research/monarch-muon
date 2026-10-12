"""In-place patching of a loaded model.

MatrixPatcher overwrites the target weight values with the dense reconstruction
of the Monarch approximation. The forward pass is the model's own (fused kernels,
expert loops, ...), so it is functionally identical to running the factored
layer. The original values are saved verbatim (same dtype, no arithmetic) and
restored LIFO, so `restore()` leaves every parameter bit-identical, with
unchanged identity / data_ptr.

Patched values are written in the parameter's dtype: for a bf16 model the dense
reconstruction is rounded to bf16 (relative 2^-9 per entry), exactly as it
would be stored.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import torch

from model_registry import MatrixRef


class MatrixPatcher:
    def __init__(self, model: torch.nn.Module, *, save_on_cpu: bool = False):
        """save_on_cpu: keep the saved original values in host memory instead of on the
        weights' device -- for patching many units at once. Restore is still exact: the
        saved values are verbatim copies."""
        self.model = model
        self.save_on_cpu = save_on_cpu
        self._stack: list[tuple[MatrixRef, torch.Tensor]] = []          # LIFO stack of (ref, saved values)

    @property
    def active(self) -> bool:
        return bool(self._stack)

    def read(self, ref: MatrixRef) -> torch.Tensor:
        return ref.read(self.model)

    @torch.no_grad()
    def apply(self, assignments: Sequence[tuple[MatrixRef, torch.Tensor]]) -> None:
        """Overwrite each ref's matrix with the given dense tensor (same shape)."""
        for ref, W in assignments:
            expected = ref.shape(self.model)
            if tuple(W.shape) != expected:
                raise ValueError(f"{ref.param_name}: patch has shape {tuple(W.shape)}, matrix is {expected}")
            p = ref._param(self.model).detach()
            self._stack.append((ref, p.to("cpu", copy=True) if self.save_on_cpu else p.clone()))
            ref.write(self.model, W)

    @torch.no_grad()
    def _restore_one(self) -> None:
        ref, saved = self._stack.pop()
        ref._param(self.model).copy_(saved)

    def restore(self) -> None:
        """Undo every apply() since construction, newest first."""
        while self._stack:
            self._restore_one()

    @contextmanager
    def patched(self, assignments: Sequence[tuple[MatrixRef, torch.Tensor]]) -> Iterator[None]:
        """Patch on entry, restore on exit (also when the body raises)."""
        mark = len(self._stack)
        try:
            self.apply(assignments)
            yield
        finally:
            while len(self._stack) > mark:
                self._restore_one()
