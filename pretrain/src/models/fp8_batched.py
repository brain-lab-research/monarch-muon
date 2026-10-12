from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def extension():
    return load(name="monarch_fp8_batched", sources=[str(Path(__file__).with_suffix(".cpp"))],
                extra_cflags=["-O2"], extra_ldflags=["-lcublasLt"], with_cuda=True)


_workspace = None


def workspace(device):
    global _workspace
    if _workspace is None or _workspace.device != device:
        _workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    return _workspace


def matmul(a, b, sa, sb, interleaved=False):
    scratch = workspace(a.device)
    return extension().matmul(a, b, sa, sb, scratch, interleaved)
