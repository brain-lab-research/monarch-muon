"""Tests for monarch_blocks.py: the training code's MonarchLinear(nblocks=B) layer and its closed-form fit to a dense matrix.

    python test_monarch_blocks.py

Small matrices only (<= 128 x 256), a few seconds in total. The reference forward below is the math of
`BlockdiagButterflyMultiply.forward` of the training code's Monarch layer (adapted from HazyResearch/monarch),
transcribed with plain bmm; the real module was compared with monarch_blocks
separately (shapes, parameter counts, dense matrix, loading fitted factors): identical, also for padded shapes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from monarch_blocks import BlockShape, fit_svd, monarch_dense


def repo_forward(x, w1, w2):
    """x (batch, in_ext) -> (batch, out_ext); w1 (k, q, p), w2 (l, s, r)."""
    batch = x.shape[0]
    k, q, p = w1.shape
    l, s, r = w2.shape
    out1 = torch.bmm(x.reshape(batch, k, p).transpose(0, 1), w1.transpose(-1, -2))                  # (k, batch, q)
    out1 = out1.transpose(0, 1).reshape(batch, r, l).transpose(-1, -2).contiguous().transpose(0, 1)  # riffle -> (l, batch, r)
    out2 = torch.bmm(out1, w2.transpose(-1, -2))                                                    # (l, batch, s)
    return out2.permute(1, 2, 0).reshape(batch, s * l)


def random_factors(sh: BlockShape, e: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(e, *sh.w1_shape, generator=g), torch.randn(e, *sh.w2_shape, generator=g)


def num_rank(M, tol=1e-5):
    s = torch.linalg.svdvals(M.double())
    return int((s > tol * s[0]).sum())


def test_shapes_parameters_and_the_moonlight_matrices():
    assert BlockShape(64, 64, 4).w1_shape == (4, 16, 16) and BlockShape(64, 64, 4).w2_shape == (4, 16, 16)
    assert BlockShape(64, 64, 4).fraction == 0.5 and BlockShape(64, 64, 8).fraction == 0.25
    assert BlockShape(96, 48, 4).fraction == 0.375 and BlockShape(48, 96, 4).fraction == 0.375
    padded = BlockShape(50, 70, 4)                                     # ceil blocks + zero padding, like the repo
    assert (padded.in_blk, padded.out_blk, padded.in_ext, padded.out_ext) == (18, 13, 72, 52)
    assert padded.w1_shape == (4, 13, 18) and padded.w2_shape == (4, 13, 13) and padded.n_params == 936 + 676
    # physical matrices of Moonlight at nblocks = 4 (in_ext < out_ext -> w1 (B,in,in), w2 (B,out,in))
    q_proj, kv_a = BlockShape(3072, 2048, 4), BlockShape(576, 2048, 4)
    assert q_proj.w1_shape == (4, 512, 512) and q_proj.w2_shape == (4, 768, 512) and abs(q_proj.fraction - 2621440 / 6291456) < 1e-12
    assert kv_a.w1_shape == (4, 144, 512) and kv_a.w2_shape == (4, 144, 144) and abs(kv_a.fraction - 0.3203125) < 1e-12
    for sh in (BlockShape(2048, 2048, 4), BlockShape(1408, 2048, 4), BlockShape(2816, 2048, 8), BlockShape(4096, 512, 4)):
        assert sh.exact_ranks, sh                                      # B divides min(block dims): fit_svd applies
    assert not BlockShape(1408, 2048, 16).exact_ranks                  # 88 / 16: fit_svd does not apply


def test_dense_matches_the_repo_forward_also_when_padded():
    for n_out, n_in, B in [(64, 64, 4), (96, 48, 4), (48, 96, 4), (50, 70, 4), (70, 50, 8), (33, 130, 8)]:
        sh = BlockShape(n_out, n_in, B)
        w1, w2 = random_factors(sh, e=2, seed=n_out)
        got = monarch_dense(w1, w2, n_out, n_in)
        assert got.shape == (2, n_out, n_in)
        for i in range(2):
            x = torch.eye(sh.in_ext)
            want = repo_forward(x, w1[i], w2[i]).T[:n_out, :n_in]
            assert torch.allclose(got[i], want, atol=1e-5), (n_out, n_in, B)
            v = torch.randn(3, sh.in_ext)                              # and as a matrix-vector product
            v[:, n_in:] = 0.0
            assert torch.allclose(repo_forward(v, w1[i], w2[i])[:, :n_out], v[:, :n_in] @ got[i].T, atol=1e-4)


def test_rank_structure_of_the_dense_matrix():
    """No rank is imposed anywhere, yet: rows with the same residue mod B and one of the B contiguous column chunks
    form a block of rank <= q/B (rank 1 at B = sqrt(n): the paper's definition)."""
    for n_out, n_in, B in [(64, 64, 2), (64, 64, 4), (64, 64, 8), (128, 32, 4), (32, 128, 4), (96, 48, 4)]:
        sh = BlockShape(n_out, n_in, B)
        w1, w2 = random_factors(sh, seed=B)
        M = monarch_dense(w1, w2, n_out, n_in)[0]
        ranks = {num_rank(M[lam::B, a * sh.in_blk:(a + 1) * sh.in_blk]) for lam in range(B) for a in range(B)}
        assert ranks == {sh.terms}, (n_out, n_in, B, ranks, sh.terms)
    assert BlockShape(64, 64, 8).terms == 1


def test_svd_fit_is_the_per_block_truncated_svd_and_loadable():
    torch.manual_seed(0)
    for n_out, n_in, B in [(64, 64, 4), (128, 32, 4), (32, 128, 4)]:
        sh = BlockShape(n_out, n_in, B)
        W = torch.randn(2, n_out, n_in)
        r = fit_svd(W, B)
        want = torch.zeros_like(W)                                     # independent construction, direct slicing
        for e in range(2):
            for lam in range(B):
                for a in range(B):
                    blk = W[e, lam::B, a * sh.in_blk:(a + 1) * sh.in_blk]
                    U, S, Vh = torch.linalg.svd(blk, full_matrices=False)
                    want[e, lam::B, a * sh.in_blk:(a + 1) * sh.in_blk] = (U[:, :sh.terms] * S[:sh.terms]) @ Vh[:sh.terms]
        assert torch.allclose(r.dense, want, atol=1e-5), (n_out, n_in, B)
        assert torch.allclose(monarch_dense(r.w1, r.w2, n_out, n_in), r.dense, atol=1e-5), "the factors ARE the matrix"
        assert tuple(r.w1.shape[1:]) == sh.w1_shape and tuple(r.w2.shape[1:]) == sh.w2_shape
        for e in range(2):                                             # the factor blocks themselves are full rank
            assert all(num_rank(r.w1[e, a]) == min(sh.w1_shape[1:]) for a in range(B)), "w1 blocks"
            assert all(num_rank(r.w2[e, lam]) == min(sh.w2_shape[1:]) for lam in range(B)), "w2 blocks"
        assert torch.allclose(r.rel_err, (W - want).flatten(1).norm(dim=1) / W.flatten(1).norm(dim=1), atol=1e-5)
    try:
        fit_svd(torch.randn(1, 50, 70), 4)
    except ValueError as exc:
        assert "divide" in str(exc)
    else:
        raise AssertionError("fit_svd must refuse shapes whose block ranks are not exact")


TESTS = [
    ("shapes, parameter counts, Moonlight's matrices", test_shapes_parameters_and_the_moonlight_matrices),
    ("dense matrix == the repo forward (also padded)", test_dense_matches_the_repo_forward_also_when_padded),
    ("rank structure that follows from the shapes", test_rank_structure_of_the_dense_matrix),
    ("fit_svd == per-block truncated SVD, loadable", test_svd_fit_is_the_per_block_truncated_svd_and_loadable),
]


def main():
    for i, (title, fn) in enumerate(TESTS, 1):
        print(f"=== Test {i}: {title} ===")
        fn()
        print("PASS\n")
    print("ALL MONARCH BLOCK TESTS PASSED")


if __name__ == "__main__":
    main()
