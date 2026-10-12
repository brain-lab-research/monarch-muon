"""Fast riffle for the butterfly forward: out1 (B,k,q) -> t (l,B,r).

The eager riffle is a stride-4 gather (~1 TB/s). The Triton kernel loads
contiguous (BB, Q) rows, de-interleaves them in registers via reshape+split and
stores four contiguous l-planes, so it is coalesced both ways. It is
bitwise-identical to the eager riffle (only the order in which finished values
are moved changes). The unriffle kernel is its inverse, for the backward.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _riffle_kernel(O1, T, B, Q: tl.constexpr, QPAD: tl.constexpr,
                   BB: tl.constexpr):
    """O1 (B, k, Q) contiguous storage; T (4, B, Q). One CTA: (BB rows, one k)."""
    pid_b = tl.program_id(0)
    k = tl.program_id(1)
    NBK: tl.constexpr = 4
    QSUB: tl.constexpr = Q // NBK
    JPAD: tl.constexpr = QPAD // NBK
    K: tl.constexpr = 4  # nb blocks in stage 1

    offs_b = pid_b * BB + tl.arange(0, BB)
    bm = offs_b < B
    offs_q = tl.arange(0, QPAD)
    qm = offs_q < Q
    src = O1 + offs_b[:, None] * (K * Q) + k * Q + offs_q[None, :]
    x = tl.load(src, mask=bm[:, None] & qm[None, :], other=0.0)

    x = tl.reshape(x, (BB, JPAD, 2, 2))                # q = j*4 + (hi*2+lo)
    lo0, lo1 = tl.split(x)                             # (BB, JPAD, 2) each
    p0, p2 = tl.split(lo0)                             # l = 0, 2
    p1, p3 = tl.split(lo1)                             # l = 1, 3

    offs_j = tl.arange(0, JPAD)
    jm = offs_j < QSUB
    base = offs_b[:, None] * Q + k * QSUB + offs_j[None, :]
    mask = bm[:, None] & jm[None, :]
    tl.store(T + 0 * (B * Q) + base, p0, mask=mask)
    tl.store(T + 1 * (B * Q) + base, p1, mask=mask)
    tl.store(T + 2 * (B * Q) + base, p2, mask=mask)
    tl.store(T + 3 * (B * Q) + base, p3, mask=mask)


def riffle_triton(out1_storage, B, nb, q):
    """out1_storage: (B, nb, q) contiguous. Returns t (nb, B, q) blocked."""
    t = torch.empty(nb, B, q, device=out1_storage.device, dtype=out1_storage.dtype)
    BB = 32
    qpad = triton.next_power_of_2(q)
    _riffle_kernel[(triton.cdiv(B, BB), nb)](out1_storage, t, B, q, qpad, BB,
                                  num_warps=4, num_stages=2)
    return t


@triton.jit
def _unriffle_kernel(DT, DO1, B, Q: tl.constexpr, QPAD: tl.constexpr,
                     BB: tl.constexpr):
    """Inverse riffle: dout1[b, k, j*4+l] = dt[l, b, k*QSUB+j].

    DT (4, B, Q) blocked planes -> DO1 (B, K, Q) contiguous storage.
    Loads four contiguous plane segments, joins in registers, one
    contiguous store."""
    pid_b = tl.program_id(0)
    k = tl.program_id(1)
    NBK: tl.constexpr = 4
    QSUB: tl.constexpr = Q // NBK
    JPAD: tl.constexpr = QPAD // NBK
    K: tl.constexpr = 4

    offs_b = pid_b * BB + tl.arange(0, BB)
    bm = offs_b < B
    offs_j = tl.arange(0, JPAD)
    jm = offs_j < QSUB
    src = offs_b[:, None] * Q + k * QSUB + offs_j[None, :]
    mask = bm[:, None] & jm[None, :]
    p0 = tl.load(DT + 0 * (B * Q) + src, mask=mask, other=0.0)
    p1 = tl.load(DT + 1 * (B * Q) + src, mask=mask, other=0.0)
    p2 = tl.load(DT + 2 * (B * Q) + src, mask=mask, other=0.0)
    p3 = tl.load(DT + 3 * (B * Q) + src, mask=mask, other=0.0)

    j01 = tl.join(p0, p1)                 # (BB, JPAD, 2)  last axis c1 = l&1
    j23 = tl.join(p2, p3)
    jj = tl.join(j01, j23)                # (BB, JPAD, 2, 2) [.., c1, c2] = p_{2*c2+c1}
    jj = tl.permute(jj, (0, 1, 3, 2))     # -> [.., c2, c1]: flat l = 2*c2+c1 holds p_l
    x = tl.reshape(jj, (BB, JPAD * 4))    # q = j*4 + l

    offs_q = tl.arange(0, QPAD)
    qm = offs_q < Q
    dst = DO1 + offs_b[:, None] * (K * Q) + k * Q + offs_q[None, :]
    tl.store(dst, x, mask=bm[:, None] & qm[None, :])


def unriffle_triton(dt, B, nb, q):
    """dt (nb, B, q) blocked -> dout1 storage (B, nb, q) contiguous."""
    do1 = torch.empty(B, nb, q, device=dt.device, dtype=dt.dtype)
    BB = 32
    qpad = triton.next_power_of_2(q)
    _unriffle_kernel[(triton.cdiv(B, BB), nb)](dt, do1, B, q, qpad, BB,
                                    num_warps=4, num_stages=2)
    return do1

