"""Tests of moe_accel.py (grouped-GEMM MoE experts), CPU only, seconds.

    python test_moe_accel.py

What is checked: the offsets and the grouped GEMM (loop path -- the real torch._grouped_mm kernel needs a GPU and is
verified at load time on the GPU by the self-check, not here); that the accelerated block reproduces the original
expert loop of the remote-code style (`moe_infer`, host-synchronised loop) and of the native class (`moe`), including
experts that get no token; that the experts' weights are views of the stacks, so MatrixPatcher patches and restores what
the grouped GEMM reads; the fallback when the kernel is wrong or raises; load_model(accel=...); the reference-cache
guard.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

import moe_accel
from matrix_patching import MatrixPatcher
from model_loading import final_hidden, load_model
from model_registry import build_registry, read_checkpoint_shapes, read_config
from moe_accel import (accel_note, apply_moe_accel, check_cache_accel, effective_accel, grouped_mm, grouped_mm_loop, moe_modules)
from toy_model import make_toy_model, save_toy_checkpoint


class Expert(torch.nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate_proj = torch.nn.Linear(h, i, bias=False)
        self.up_proj = torch.nn.Linear(h, i, bias=False)
        self.down_proj = torch.nn.Linear(i, h, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RemoteStyleMoE(torch.nn.Module):
    """moe_infer as in the checkpoint's modeling_deepseek.py: sort the tokens by expert, loop over the experts in Python
    with the counts copied to the host, un-sort, weight by the router, sum over the top-k."""

    def __init__(self, h=16, i=24, n_experts=8, k=2):
        super().__init__()
        self.experts = torch.nn.ModuleList([Expert(h, i) for _ in range(n_experts)])
        self.gate = torch.nn.Linear(h, n_experts, bias=False)
        self.k, self.ep_size = k, 1

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs, start_idx = [], 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            outputs.append(self.experts[i](sorted_tokens[start_idx:end_idx]))
            start_idx = end_idx
        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        return (new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype))

    def route(self, x):
        w, ids = torch.softmax(self.gate(x).float(), -1).topk(self.k, dim=-1)
        return ids, w / w.sum(-1, keepdim=True)

    def forward(self, x):
        ids, w = self.route(x)
        return self.moe_infer(x, ids, w)


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def test_offsets_and_loop_kernel():
    ids = torch.tensor([3, 0, 3, 3, 1, 1, 7])
    offs = moe_accel._offsets(ids, 8)
    assert offs.dtype == torch.int32 and offs.tolist() == [1, 3, 3, 6, 6, 6, 6, 7], offs.tolist()   # experts 2, 4, 5, 6 get none
    torch.manual_seed(0)
    x, w = torch.randn(7, 5), torch.randn(8, 4, 5)
    out = grouped_mm(x, w, offs)                                   # CPU: the loop path
    want = torch.cat([F.linear(x[a:b], w[e]) for e, (a, b) in enumerate(zip([0] + offs.tolist(), offs.tolist())) if b > a])
    assert torch.equal(out, want) and torch.equal(out, grouped_mm_loop(x, w, offs))
    assert not moe_accel.kernel_available(torch.device("cpu"), torch.bfloat16)


def test_remote_style_block_matches_the_original_loop():
    torch.manual_seed(0)
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 3e-2)):
        block = RemoteStyleMoE().to(dtype)
        ref = RemoteStyleMoE().to(dtype)
        ref.load_state_dict(block.state_dict())
        assert [(n, m is block, meth) for n, m, meth in moe_modules(block)] == [("", True, "moe_infer")]
        report = apply_moe_accel(block, check=False, log=lambda *_: None)        # a bare MoE module works like a model
        assert report["active"] and report["layers"] == 1, report
        for t in (1, 5, 37):                                       # 1 token: most experts get nothing
            x = torch.randn(t, 16).to(dtype)
            assert rel(block(x), ref(x)) <= tol, (dtype, t)
        block._grouped_moe.on = False                              # the switch restores the original loop exactly
        x = torch.randn(9, 16).to(dtype)
        assert torch.equal(block(x), ref(x))
        block._grouped_moe.on = True
        assert apply_moe_accel(block, check=False, log=lambda *_: None)["active"], "applying twice keeps the stacks"
        assert block._grouped_moe.gate_up.shape == (8, 48, 16)


def toy_pair(dtype=torch.float32, seed=0):
    a, b = make_toy_model(seed=seed, dtype=dtype), make_toy_model(seed=seed, dtype=dtype)
    return a, b                                                    # identical weights; `a` gets accelerated


def test_toy_model_native_class_views_patching_and_restore():
    model, ref = toy_pair()
    ids = torch.randint(0, 128, (2, 24), generator=torch.Generator().manual_seed(1))
    before = final_hidden(model, ids).clone()
    report = apply_moe_accel(model, log=lambda *_: None)
    assert report["active"] and report["layers"] == 2 and not report["kernel"] and report["reason"] == "", report
    assert report["selfcheck_rel_diff"] <= 1e-4
    assert len(report["layer_checks"]) == 2 and all(c["ok"] and c["err_grouped"] <= 1e-5 for c in report["layer_checks"]), report
    model.moe_accel_report = report
    assert effective_accel(model) == "grouped_mm" and effective_accel(ref) == "none"
    assert rel(final_hidden(model, ids), before) <= 1e-4 and rel(final_hidden(model, ids), final_hidden(ref, ids)) <= 1e-4

    mlp = model.model.layers[1].mlp                                # views of the stacks, not copies
    st = mlp._grouped_moe
    for e in (0, 3):
        for name, stack in (("gate_proj", st.gate_up), ("up_proj", st.gate_up), ("down_proj", st.down)):
            w = getattr(mlp.experts[e], name).weight
            assert w.untyped_storage().data_ptr() == stack.untyped_storage().data_ptr() and not w.requires_grad
    assert torch.equal(st.gate_up[2, :32], mlp.experts[2].gate_proj.weight)
    assert torch.equal(st.gate_up[2, 32:], mlp.experts[2].up_proj.weight)

    # patch a routed expert through the study's MatrixPatcher: the grouped GEMM must see it, restore must undo it exactly
    tmp = Path(tempfile.mkdtemp())
    try:
        path = save_toy_checkpoint(ref, tmp / "ckpt")
        registry = build_registry(read_config(path), read_checkpoint_shapes(path))
        unit = [u for u in registry.select(layers=[1], groups=("moe_routed",)) if u.mtype.endswith("down_proj")][0]
        gen = torch.Generator().manual_seed(5)
        W = torch.randn(len(unit.refs), *unit.shape, generator=gen)
        for target in (model, ref):                                # same patch on the accelerated and the original model
            patcher = MatrixPatcher(target)
            patcher.apply([(r, W[i]) for i, r in enumerate(unit.refs)])
            target._patcher = patcher
        patched_acc, patched_ref = final_hidden(model, ids), final_hidden(ref, ids)
        assert rel(patched_acc, before) > 1e-2, "the patch must be visible"
        assert rel(patched_acc, patched_ref) <= 1e-4, "the grouped GEMM reads the patched weights"
        for target in (model, ref):
            target._patcher.restore()
        assert torch.equal(final_hidden(model, ids), before) or rel(final_hidden(model, ids), before) <= 1e-6
        assert torch.equal(mlp.experts[1].down_proj.weight, st.down[1]), "restore went into the stack"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fallback_when_the_kernel_is_wrong_or_raises():
    original = moe_accel.grouped_mm
    try:
        for label, bad in (("wrong", lambda x, w, offs: grouped_mm_loop(x, w, offs) * 2.0),
                           ("raises", lambda x, w, offs: (_ for _ in ()).throw(RuntimeError("no grouped_mm on this device")))):
            model, ref = toy_pair()
            ids = torch.randint(0, 128, (1, 20), generator=torch.Generator().manual_seed(2))
            moe_accel.grouped_mm = bad
            lines = []
            report = apply_moe_accel(model, log=lines.append)
            moe_accel.grouped_mm = original
            assert not report["active"] and report["reason"], (label, report)
            assert any("OFF" in l for l in lines), lines
            model.moe_accel_report = report
            assert effective_accel(model) == "none" and "grouped_mm was requested but is OFF" in accel_note(model)
            assert torch.equal(final_hidden(model, ids), final_hidden(ref, ids)), f"{label}: the original path runs unchanged"
    finally:
        moe_accel.grouped_mm = original


def test_selfcheck_judges_the_kernel_by_its_error_against_float32_not_by_model_level_noise():
    """On the real checkpoints the final hidden states of two paths that are equal in exact arithmetic differed by 4-5%
    (bf16 noise through ~27 layers); that must not switch a correct kernel off, while a kernel that is noticeably less
    accurate than the loop -- or wrong -- must be."""
    original = moe_accel.grouped_mm

    def noisy(level):
        def kernel(x, w, offs):
            out = grouped_mm_loop(x, w, offs)
            return out * (1 + level * torch.randn(out.shape, generator=torch.Generator().manual_seed(7)).to(out.dtype))
        return kernel

    try:
        for level, want_active in ((0.0, True), (1e-3, True), (6e-2, False)):
            model, ref = toy_pair(torch.bfloat16)
            moe_accel.grouped_mm = noisy(level)
            report = apply_moe_accel(model, log=lambda *_: None)
            moe_accel.grouped_mm = original
            checks = report["layer_checks"]
            assert checks and report["active"] == want_active, (level, report["reason"], checks)
            assert all(c["err_loop"] > 1e-4 for c in checks), "bf16: the loop itself is ~1e-3..1e-2 off float32"
            if want_active:
                assert all(c["ok"] and c["err_grouped"] <= 2 * c["err_loop"] + 3e-3 for c in checks), checks
            else:
                assert "differ from the original loop by more than the loop's own bf16 error" in report["reason"], report["reason"]
        # the model-level difference is informational: a huge one is only a cap for gross errors
        model, _ = toy_pair(torch.bfloat16)
        assert apply_moe_accel(model, log=lambda *_: None)["selfcheck_rel_diff"] is not None
    finally:
        moe_accel.grouped_mm = original


def test_a_kernel_error_midway_falls_back_to_the_loop_once():
    """torch._grouped_mm raising on some shape must not end a run of hours: one warning, then the loop."""
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("strides should be multiples of 16 bytes\nsecond line")

    saved_kernel, saved_fn = moe_accel.kernel_available, getattr(torch, "_grouped_mm", None)
    moe_accel.kernel_available, torch._grouped_mm = (lambda device, dtype: True), boom
    moe_accel.KERNEL_FAILURES.clear()
    try:
        x, w = torch.randn(7, 5), torch.randn(8, 4, 5)
        offs = moe_accel._offsets(torch.tensor([3, 0, 3, 3, 1, 1, 7]), 8)
        assert torch.equal(grouped_mm(x, w, offs), grouped_mm_loop(x, w, offs)) and len(calls) == 1
        assert moe_accel.KERNEL_FAILURES == ["strides should be multiples of 16 bytes"]
        assert torch.equal(grouped_mm(x, w, offs), grouped_mm_loop(x, w, offs)) and len(calls) == 1, "the kernel is not retried"
        moe_accel.KERNEL_FAILURES.clear()
        model, ref = toy_pair()
        lines = []
        report = apply_moe_accel(model, log=lines.append)            # the self-check runs through the failing kernel
        assert report["active"] and report["kernel"] is False and any("the kernel failed: strides" in l for l in lines), lines
    finally:
        moe_accel.kernel_available = saved_kernel
        if saved_fn is None:
            del torch._grouped_mm
        else:
            torch._grouped_mm = saved_fn
        moe_accel.KERNEL_FAILURES.clear()


def test_models_without_a_replaceable_experts_loop_are_left_alone():
    class Dense(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)
    lines = []
    report = apply_moe_accel(Dense(), log=lines.append)
    assert not report["active"] and "no MoE block" in report["reason"] and report["layers"] == 0 and lines


def test_load_model_accel_option():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:       # (Windows keeps the loaded safetensors mapped)
        path = save_toy_checkpoint(make_toy_model(seed=3), Path(d) / "ckpt")
        plain = load_model(path, dtype="float32", device="cpu", trust_remote_code=False)
        fast = load_model(path, dtype="float32", device="cpu", trust_remote_code=False, accel="grouped_mm")
        assert effective_accel(plain) == "none" and effective_accel(fast) == "grouped_mm"
        ids = torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(4))
        assert rel(final_hidden(fast, ids), final_hidden(plain, ids)) <= 1e-4
        assert all(not p.requires_grad for p in fast.parameters())
        try:
            load_model(path, dtype="float32", device="cpu", trust_remote_code=False, accel="bogus")
        except ValueError as exc:
            assert "bogus" in str(exc)
        else:
            raise AssertionError("an unknown accel must be rejected")
        # asked for, but the check fails: the load stops (no silent slow run, no cache with the wrong numerics)
        original = moe_accel.grouped_mm
        moe_accel.grouped_mm = lambda x, w, offs: grouped_mm_loop(x, w, offs) * 2.0
        try:
            load_model(path, dtype="float32", device="cpu", trust_remote_code=False, accel="grouped_mm")
        except SystemExit as exc:
            assert "was requested but is OFF" in str(exc) and "ACCEL=none" in str(exc) and "differ from the original loop" in str(exc)
        else:
            raise AssertionError("a failed self-check must stop a run that asked for grouped_mm")
        finally:
            moe_accel.grouped_mm = original


def test_cache_guards():
    check_cache_accel({}, "none")                                  # caches written before the option existed are 'none'
    check_cache_accel({"accel": "grouped_mm"}, "grouped_mm")
    for meta, now in (({}, "grouped_mm"), ({"accel": "grouped_mm"}, "none")):
        try:
            check_cache_accel(meta, now, note=" (fallback)")
        except SystemExit as exc:
            msg = str(exc)
            assert "accel=" in msg and "Rebuild" in msg and "(fallback)" in msg and "--cache-dir" in msg
        else:
            raise AssertionError(f"{meta} vs {now} must be refused")


TESTS = [
    ("offsets and the loop kernel", test_offsets_and_loop_kernel),
    ("remote-style moe_infer block: grouped == original loop, incl. empty experts; switch off restores it", test_remote_style_block_matches_the_original_loop),
    ("toy model (native moe): same outputs, weights are views, patch + restore through MatrixPatcher", test_toy_model_native_class_views_patching_and_restore),
    ("fallback when the kernel is wrong or raises", test_fallback_when_the_kernel_is_wrong_or_raises),
    ("self-check: per-layer error against float32, not model-level bf16 noise", test_selfcheck_judges_the_kernel_by_its_error_against_float32_not_by_model_level_noise),
    ("a kernel error midway: one warning, then the loop", test_a_kernel_error_midway_falls_back_to_the_loop_once),
    ("no replaceable MoE block", test_models_without_a_replaceable_experts_loop_are_left_alone),
    ("load_model(accel=...)", test_load_model_accel_option),
    ("cache guards", test_cache_guards),
]


def main():
    for i, (title, fn) in enumerate(TESTS, 1):
        print(f"=== Test {i}: {title} ===")
        fn()
        print("PASS\n")
    print("ALL MOE ACCEL TESTS PASSED")


if __name__ == "__main__":
    main()
