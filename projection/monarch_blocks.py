"""Monarch factorization in the form of the training code's `MonarchLinear(nblocks=B)`.

The layer is the Monarch layer used for training (adapted from HazyResearch/monarch). It has two parameter stacks and nothing else:

    w1 = blkdiag1   (B, q, p)    B blocks, each q x p    (acts on the input, which is cut into B chunks of p)
    w2 = blkdiag2   (B, s, r)    B blocks, each s x r    (acts after a fixed stride permutation)

    in_blk = ceil(n_in / B), out_blk = ceil(n_out / B), in_ext = in_blk * B, out_ext = out_blk * B   (zero padding)
    in_ext <  out_ext:  w1 = (B, in_blk,  in_blk),  w2 = (B, out_blk, in_blk)
    otherwise:          w1 = (B, out_blk, in_blk),  w2 = (B, out_blk, out_blk)

`monarch_dense` is its forward pass applied to the identity: the dense matrix the layer represents, batched over a
leading dimension.

Nothing forces a rank when such a layer is TRAINED: it is just w1 and w2. A rank appears as a consequence of the
shapes (verified against the training code): rows with the same residue mod B and one of the B contiguous column
chunks form a block of the dense matrix, and each such block is a sum of q / B outer products, i.e. has rank <= q / B,
q = min(in_blk, out_blk). For B = sqrt(n) that is rank 1: the definition of Dao et al. (2022).

`fit_svd` gives w1, w2 (full-rank blocks) for a given dense W (out x in) in closed form: per-block truncated SVD of
rank q / B written into w1, w2. The blocks of the dense matrix are independent, so this is the global minimum of
||W - M(w1, w2)||_F^2. It needs B to divide q, which holds for every Moonlight matrix at B = 4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlockShape:
    n_out: int
    n_in: int
    nblocks: int

    @property
    def in_blk(self) -> int:
        return -(-self.n_in // self.nblocks)

    @property
    def out_blk(self) -> int:
        return -(-self.n_out // self.nblocks)

    @property
    def in_ext(self) -> int:
        return self.in_blk * self.nblocks

    @property
    def out_ext(self) -> int:
        return self.out_blk * self.nblocks

    @property
    def w1_shape(self) -> tuple[int, int, int]:
        if self.in_ext < self.out_ext:
            return (self.nblocks, self.in_blk, self.in_blk)
        return (self.nblocks, self.out_blk, self.in_blk)

    @property
    def w2_shape(self) -> tuple[int, int, int]:
        if self.in_ext < self.out_ext:
            return (self.nblocks, self.out_blk, self.in_blk)
        return (self.nblocks, self.out_blk, self.out_blk)

    @property
    def n_params(self) -> int:
        return math.prod(self.w1_shape) + math.prod(self.w2_shape)

    @property
    def n_dense(self) -> int:
        return self.n_out * self.n_in

    @property
    def fraction(self) -> float:
        """Parameters of the layer / parameters of the dense matrix (the training code's `MonarchLinear.saving`)."""
        return self.n_params / self.n_dense

    @property
    def inner(self) -> int:
        """q: the output size of a w1 block = min(in_blk, out_blk)."""
        return self.w1_shape[1]

    @property
    def terms(self) -> int:
        """Outer products summed in every block of the dense matrix (its rank bound), q // B (>= 1)."""
        return max(1, self.inner // self.nblocks)

    @property
    def exact_ranks(self) -> bool:
        """True when B divides q, so every block has exactly `terms` outer products (needed by fit_svd)."""
        return self.inner % self.nblocks == 0


def monarch_dense(w1: torch.Tensor, w2: torch.Tensor, n_out: int, n_in: int) -> torch.Tensor:
    """Dense (..., n_out, n_in) matrix of the Monarch layer with factors w1 (..., B, q, p), w2 (..., B, s, r).

    The forward pass of MonarchLinear on the identity, batched over the leading dimensions (what
    `blockdiag_butterfly_multiply` does for one matrix): block-diagonal multiply, riffle (reshape (r, l) and
    transpose), block-diagonal multiply, output index s*l + lambda."""
    lead = w1.shape[:-3]
    k, q, p = w1.shape[-3:]
    l, s, r = w2.shape[-3:]
    if l * r != k * q:
        raise ValueError(f"inconsistent factors: w1 {tuple(w1.shape[-3:])}, w2 {tuple(w2.shape[-3:])}")
    in_ext = k * p
    w1f, w2f = w1.reshape(-1, k, q, p), w2.reshape(-1, l, s, r)
    e = w1f.shape[0]
    x = torch.eye(in_ext, dtype=w1.dtype, device=w1.device).reshape(in_ext, k, p)           # batch = the basis vectors
    out1 = torch.einsum("bkp,ekqp->ebkq", x, w1f).reshape(e, in_ext, k * q)                 # first block-diagonal factor
    out1 = out1.reshape(e, in_ext, r, l).transpose(-1, -2)                                  # riffle: (l, r) groups
    out2 = torch.einsum("eblr,elsr->ebsl", out1, w2f).reshape(e, in_ext, s * l)             # second factor, out = s*l + lam
    return out2.transpose(1, 2)[:, :n_out, :n_in].reshape(*lead, n_out, n_in)


@dataclass
class FitResult:
    w1: torch.Tensor            # (E, B, q, p)
    w2: torch.Tensor            # (E, B, s, r)
    dense: torch.Tensor         # (E, n_out, n_in) what the layer represents (fp32)
    mse: torch.Tensor           # (E,) ||W - dense||_F^2 / (n_out * n_in)
    rel_err: torch.Tensor       # (E,) ||W - dense||_F / ||W||_F
    w_rms: torch.Tensor         # (E,) ||W||_F / sqrt(n_out * n_in)
    method: str


def _stats(W: torch.Tensor, dense: torch.Tensor):
    err = (W - dense).pow(2).sum((-2, -1))
    norm = W.pow(2).sum((-2, -1))
    n = W.shape[-2] * W.shape[-1]
    return err / n, (err / norm.clamp_min(1e-30)).sqrt(), (norm / n).sqrt()


@torch.no_grad()
def fit_svd(W: torch.Tensor, nblocks: int) -> FitResult:
    """Closed-form optimum: per-block truncated SVD, rank q/B, written into w1/w2. W: (E, n_out, n_in)."""
    W = W.float()
    e, n_out, n_in = W.shape
    sh = BlockShape(n_out, n_in, nblocks)
    if not sh.exact_ranks:
        raise ValueError(f"fit_svd needs nblocks ({nblocks}) to divide min(in_blk, out_blk) = {sh.inner} "
                         f"for a {n_out}x{n_in} matrix")
    B, ib, ob, rho = nblocks, sh.in_blk, sh.out_blk, sh.terms
    Wp = torch.zeros(e, sh.out_ext, sh.in_ext, dtype=W.dtype, device=W.device)
    Wp[:, :n_out, :n_in] = W
    # rows sigma*B + lam -> (sigma, lam); columns a*ib + i -> (a, i); block (lam, a) is (sigma, i)
    blocks = Wp.reshape(e, ob, B, B, ib).permute(0, 2, 3, 1, 4)                  # (E, lam, a, sigma, i)
    U, S, Vh = torch.linalg.svd(blocks, full_matrices=False)
    US = U[..., :rho] * S[..., :rho].unsqueeze(-2)                                # (E, lam, a, sigma, t)
    w2 = torch.zeros(e, *sh.w2_shape, dtype=W.dtype, device=W.device)
    w1 = torch.zeros(e, *sh.w1_shape, dtype=W.dtype, device=W.device)
    w2[:, :, :ob, :B * rho] = US.permute(0, 1, 3, 2, 4).reshape(e, B, ob, B * rho)          # w2[lam][sigma, a*rho + t]
    w1[:, :, :B * rho, :ib] = Vh[..., :rho, :].permute(0, 2, 3, 1, 4).reshape(e, B, rho * B, ib)  # w1[a][t*B + lam, i]
    dense = monarch_dense(w1, w2, n_out, n_in)
    mse, rel, rms = _stats(W, dense)
    return FitResult(w1, w2, dense, mse, rel, rms, "svd")
