"""Fast riffle for the butterfly forward: out1 (B,k,q) -> t (l,B,r), and its inverse.

The eager riffle is a stride-4 gather (~1 TB/s). The Triton kernel loads
contiguous (BB, Q) rows, de-interleaves in registers via reshape+split and
stores the contiguous l-planes, coalesced both ways. It is bitwise-identical
to the eager riffle (only the order in which finished values are moved changes).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _riffle_kernel(O1, T, B, Q: tl.constexpr, QPAD: tl.constexpr,
                   NBK: tl.constexpr, BB: tl.constexpr):
    """O1 (B, k, Q) contiguous storage; T (NBK, B, Q)."""
    pid_b = tl.program_id(0)
    k = tl.program_id(1)
    QSUB: tl.constexpr = Q // NBK
    JPAD: tl.constexpr = QPAD // NBK

    offs_b = pid_b * BB + tl.arange(0, BB)
    bm = offs_b < B
    offs_q = tl.arange(0, QPAD)
    qm = offs_q < Q
    src = O1 + offs_b[:, None] * (NBK * Q) + k * Q + offs_q[None, :]
    x = tl.load(src, mask=bm[:, None] & qm[None, :], other=0.0)

    offs_j = tl.arange(0, JPAD)
    jm = offs_j < QSUB
    base = offs_b[:, None] * Q + k * QSUB + offs_j[None, :]
    mask = bm[:, None] & jm[None, :]
    if NBK == 2:
        p0, p1 = tl.split(tl.reshape(x, (BB, JPAD, 2)))
        tl.store(T + 0 * (B * Q) + base, p0, mask=mask)
        tl.store(T + 1 * (B * Q) + base, p1, mask=mask)
    else:
        x = tl.reshape(x, (BB, JPAD, 2, 2))
        lo0, lo1 = tl.split(x)
        p0, p2 = tl.split(lo0)
        p1, p3 = tl.split(lo1)
        tl.store(T + 0 * (B * Q) + base, p0, mask=mask)
        tl.store(T + 1 * (B * Q) + base, p1, mask=mask)
        tl.store(T + 2 * (B * Q) + base, p2, mask=mask)
        tl.store(T + 3 * (B * Q) + base, p3, mask=mask)


def riffle_triton(out1_storage, B, nb, q):
    """out1_storage: (B, nb, q) contiguous. Returns t (nb, B, q) blocked."""
    if nb not in (2, 4):
        raise ValueError(f"fast riffle supports 2 or 4 blocks, got {nb}")
    t = torch.empty(nb, B, q, device=out1_storage.device, dtype=out1_storage.dtype)
    BB = 32
    qpad = triton.next_power_of_2(q)
    _riffle_kernel[(triton.cdiv(B, BB), nb)](out1_storage, t, B, q, qpad, nb, BB,
                                  num_warps=4, num_stages=2)
    return t


@triton.jit
def _unriffle_kernel(DT, DO1, B, Q: tl.constexpr, QPAD: tl.constexpr,
                     NBK: tl.constexpr, BB: tl.constexpr):
    """Inverse riffle: dout1[b, k, j*NBK+l] = dt[l, b, k*QSUB+j].

    DT (NBK, B, Q) blocked planes -> DO1 (B, NBK, Q) contiguous storage.
    Loads contiguous plane segments, joins in registers, one
    contiguous store."""
    pid_b = tl.program_id(0)
    k = tl.program_id(1)
    QSUB: tl.constexpr = Q // NBK
    JPAD: tl.constexpr = QPAD // NBK

    offs_b = pid_b * BB + tl.arange(0, BB)
    bm = offs_b < B
    offs_j = tl.arange(0, JPAD)
    jm = offs_j < QSUB
    src = offs_b[:, None] * Q + k * QSUB + offs_j[None, :]
    mask = bm[:, None] & jm[None, :]
    p0 = tl.load(DT + 0 * (B * Q) + src, mask=mask, other=0.0)
    p1 = tl.load(DT + 1 * (B * Q) + src, mask=mask, other=0.0)
    if NBK == 2:
        x = tl.reshape(tl.join(p0, p1), (BB, JPAD * 2))
    else:
        p2 = tl.load(DT + 2 * (B * Q) + src, mask=mask, other=0.0)
        p3 = tl.load(DT + 3 * (B * Q) + src, mask=mask, other=0.0)
        j01 = tl.join(p0, p1)
        j23 = tl.join(p2, p3)
        jj = tl.join(j01, j23)
        jj = tl.permute(jj, (0, 1, 3, 2))
        x = tl.reshape(jj, (BB, JPAD * 4))

    offs_q = tl.arange(0, QPAD)
    qm = offs_q < Q
    dst = DO1 + offs_b[:, None] * (NBK * Q) + k * Q + offs_q[None, :]
    tl.store(dst, x, mask=bm[:, None] & qm[None, :])


def unriffle_triton(dt, B, nb, q):
    """dt (nb, B, q) blocked -> dout1 storage (B, nb, q) contiguous."""
    if nb not in (2, 4):
        raise ValueError(f"fast unriffle supports 2 or 4 blocks, got {nb}")
    do1 = torch.empty(B, nb, q, device=dt.device, dtype=dt.dtype)
    BB = 32
    qpad = triton.next_power_of_2(q)
    _unriffle_kernel[(triton.cdiv(B, BB), nb)](dt, do1, B, q, qpad, nb, BB,
                                    num_warps=4, num_stages=2)
    return do1

