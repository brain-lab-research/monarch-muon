"""
Monarch structured linear layer (block-diagonal butterfly decomposition).
Adapted from: https://github.com/HazyResearch/monarch

Public API
----------
MonarchLinear  – drop-in replacement for nn.Linear with butterfly decomposition
apply_monarch  – patch any model in-place: replace nn.Linear → MonarchLinear
"""
from __future__ import annotations

import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .monarch_ops import butterfly_blk


class StructuredLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = out_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def reset_parameters_bias(self):
        if self.bias is not None:
            fan_in = self.bias.shape[-1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def preprocess(self, x):
        in_features = x.shape[-1]
        if in_features < self.in_features_extended:
            x = F.pad(x, (0, self.in_features_extended - in_features))
        return x

    def postprocess(self, output):
        out_features_extended = output.shape[-1]
        if out_features_extended > self.out_features:
            output = output[..., :self.out_features]
        return output

    def forward_matmul(self, x):
        raise NotImplementedError

    def forward(self, x):
        output = self.forward_matmul(x)
        return (output + self.bias.to(dtype=output.dtype)) if self.bias is not None else output


class MonarchLinear(StructuredLinear):

    def __init__(self, *args, nblocks=4, **kwargs):
        super().__init__(*args, **kwargs)
        device = kwargs.get('device', None)
        dtype = kwargs.get('dtype', None)
        factory_kwargs = {'device': device, 'dtype': dtype}

        in_blksz = int(math.ceil(self.in_features / nblocks))
        out_blksz = int(math.ceil(self.out_features / nblocks))
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks

        if self.in_features_extended < self.out_features_extended:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, in_blksz, in_blksz, **factory_kwargs))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz, **factory_kwargs))
        else:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz, **factory_kwargs))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, out_blksz, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for blkdiag in [self.blkdiag1, self.blkdiag2]:
            fan_in = blkdiag.shape[-1]
            gain = init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std
            with torch.no_grad():
                blkdiag.uniform_(-bound, bound)
        self.reset_parameters_bias()

    def forward_matmul(self, x):
        output = butterfly_blk(self.preprocess(x), self.blkdiag1, self.blkdiag2)
        return self.postprocess(output)


# ---------------------------------------------------------------------------
# Model surgery helper
# ---------------------------------------------------------------------------

_DEFAULT_EXCLUDE = ("lm_head",)


def apply_monarch(model: nn.Module, nblocks: int = 4, exclude: tuple = _DEFAULT_EXCLUDE,
                  verbose: bool = True) -> None:
    """Replace nn.Linear layers with MonarchLinear in-place.

    Must be called **before** DDP wrapping.

    Args:
        model:   The model to modify (Llama, or any nn.Module).
        nblocks: Number of butterfly blocks per MonarchLinear layer.
        exclude: Substrings — any layer whose full name contains one of these
                 will be skipped (default: skip lm_head).
        verbose: Print each replacement.
    """
    before = sum(p.numel() for p in model.parameters())

    replacements = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(ex in full_name for ex in exclude):
                continue
            new_mod = MonarchLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
                nblocks=nblocks,
            )
            replacements.append((parent, child_name, child, new_mod, full_name))

    for parent, child_name, old, new, full_name in replacements:
        setattr(parent, child_name, new)
        if verbose:
            print(f"  Monarch: {full_name}  {old.weight.shape} → blkdiag {new.blkdiag1.shape}")
        del old

    gc.collect()

    after = sum(p.numel() for p in model.parameters())
    print(
        f"\nMonarch applied: nblocks={nblocks}, replaced {len(replacements)} layers\n"
        f"  Params before : {before / 1e6:.2f}M\n"
        f"  Params after  : {after / 1e6:.2f}M  ({after / before:.2%})\n"
    )
