"""Tests for model_registry.py and matrix_patching.py on a tiny random MLA+MoE model.

    python test_registry_and_patching.py

Also discoverable by pytest. Covers the registry against the checkpoint and the
architecture, and that patching + restore leaves the model bit-identical.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from matrix_patching import MatrixPatcher
from model_registry import build_registry, read_checkpoint_shapes, read_config, shapes_from_model
from toy_model import make_toy_model, save_toy_checkpoint

_CACHE: dict = {}


def toy(q_lora_rank=None):
    if q_lora_rank not in _CACHE:
        model = make_toy_model(seed=0, q_lora_rank=q_lora_rank)
        with tempfile.TemporaryDirectory() as d:
            save_toy_checkpoint(model, d)
            cfg, shapes = read_config(d), read_checkpoint_shapes(d)
        _CACHE[q_lora_rank] = (model, cfg, shapes, build_registry(cfg, shapes))
    return _CACHE[q_lora_rank]


def snapshot(model):
    return {k: (v.detach().clone(), v.data_ptr()) for k, v in model.state_dict().items()}


def assert_bit_identical(model, snap):
    for k, v in model.state_dict().items():
        assert torch.equal(v, snap[k][0]), f"{k} changed"
        assert v.data_ptr() == snap[k][1], f"{k} was reallocated"


def test_registry_matches_checkpoint_and_architecture():
    for qr in (None, 24):
        model, cfg, shapes, reg = toy(qr)
        assert shapes == shapes_from_model(model), "safetensors header shapes differ from the live model"
        assert reg.layer_prefix == "model.layers"
        assert reg.n_layers == 3 and reg.dense_layers == [0] and reg.moe_layers == [1, 2]
        assert not reg.warnings, reg.warnings

        types = {u.mtype for u in reg.units}
        expected_attn = {"attn.kv_a_proj_with_mqa", "attn.kv_b_proj", "attn.o_proj"}
        assert expected_attn <= types
        assert ("attn.q_a_proj" in types) == (qr is not None)
        assert ("attn.q_b_proj" in types) == (qr is not None) and ("attn.q_proj" in types) == (qr is None)

        # the dense layer has dense-FFN units and no routed ones; MoE layers the reverse
        assert {u.mtype for u in reg.units if u.layer == 0 and u.group in ("ffn", "moe_routed", "moe_shared")} == \
            {"ffn.gate_proj", "ffn.up_proj", "ffn.down_proj"}
        for layer in (1, 2):
            groups = {u.group for u in reg.units if u.layer == layer}
            assert "ffn" not in groups and {"moe_routed", "moe_shared"} <= groups
        routed = reg.unit(1, "moe_routed.gate_proj")
        assert len(routed.refs) == cfg["n_routed_experts"] and routed.expert_ids == tuple(range(cfg["n_routed_experts"]))
        assert routed.shape == (cfg["moe_intermediate_size"], cfg["hidden_size"])
        shared = reg.unit(1, "moe_shared.down_proj")
        assert shared.shape == (cfg["hidden_size"], cfg["moe_intermediate_size"] * cfg["n_shared_experts"])

        # every unit's refs address real parameters with the declared shape
        for u in reg.units:
            for ref in u.refs:
                assert ref.shape(model) == u.shape, (u.uid, ref.shape(model), u.shape)

        # router and norms are never factorized; every factorizable tensor is covered by a unit
        roles = {r["name"]: r for r in reg.table}
        assert roles["model.layers.1.mlp.gate.weight"]["factorizable"] == "no"
        assert roles["model.layers.1.mlp.gate.weight"]["role"] == "router"
        assert all(not r["units"] for r in reg.table if r["factorizable"] == "no")
        assert all(r["units"] for r in reg.table if r["factorizable"] == "yes")
        # every matrix is covered by exactly one unit
        assert roles["model.layers.0.self_attn.kv_b_proj.weight"]["units"] == "L0/attn.kv_b_proj"

        # selection filters
        assert {u.layer for u in reg.select(layers=[0])} == {0}
        assert {u.mtype for u in reg.select(layers=[2], groups=("moe_routed",))} == \
            {"moe_routed.gate_proj", "moe_routed.up_proj", "moe_routed.down_proj"}


def test_registry_warnings_and_skipped_layers():
    _, cfg, shapes, _ = toy(None)
    reg = build_registry(dict(cfg, first_k_dense_replace=2), shapes)
    assert any("first_k_dense_replace" in w for w in reg.warnings)
    reg = build_registry(dict(cfg, num_hidden_layers=2), shapes)           # layer 2 treated as an extra (MTP-like) layer
    assert reg.n_layers == 2 and all(u.layer < 2 for u in reg.units)
    assert any(r["role"] == "skipped" for r in reg.table)
    # a fused-experts layout is reported, not silently ignored
    fused = dict(shapes)
    fused["model.layers.1.mlp.experts.gate_up_proj"] = (4, 64, 64)
    reg = build_registry(cfg, fused)
    assert any("gate_up_proj" in w for w in reg.warnings)


def test_patch_restore_is_bit_identical():
    model, _, _, reg = toy(None)
    snap = snapshot(model)
    patcher = MatrixPatcher(model)
    g = torch.Generator().manual_seed(0)
    # attention matrices and an aggregated routed unit
    units = [reg.unit(1, "attn.o_proj"), reg.unit(1, "moe_routed.up_proj"), reg.unit(1, "attn.kv_b_proj")]
    assignments = []
    for u in units:
        for ref in u.refs:
            assignments.append((ref, torch.randn(u.shape, generator=g)))
    patcher.apply(assignments)
    assert patcher.active
    assert any(not torch.equal(v, snap[k][0]) for k, v in model.state_dict().items()), "patch had no effect"
    patcher.restore()
    assert not patcher.active
    assert_bit_identical(model, snap)

    # context manager restores on exit AND when the body raises
    with patcher.patched(assignments):
        assert not torch.equal(model.model.layers[1].self_attn.o_proj.weight, snap["model.layers.1.self_attn.o_proj.weight"][0])
    assert_bit_identical(model, snap)
    try:
        with patcher.patched(assignments):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert_bit_identical(model, snap)


def test_patch_restore_is_bit_identical_in_bf16():
    model = make_toy_model(seed=1, dtype=torch.bfloat16)
    cfg = model.config.to_dict()
    reg = build_registry(cfg, shapes_from_model(model))
    snap = snapshot(model)
    patcher = MatrixPatcher(model)
    u = reg.unit(2, "attn.q_proj")
    ref = u.refs[0]
    with patcher.patched([(ref, torch.randn(u.shape))]):                  # fp32 patch written into a bf16 param
        assert model.get_parameter(ref.param_name).dtype == torch.bfloat16
    assert_bit_identical(model, snap)


def test_patch_rejects_wrong_shape():
    model, _, _, reg = toy(None)
    ref = reg.unit(1, "attn.o_proj").refs[0]
    try:
        MatrixPatcher(model).apply([(ref, torch.zeros(3, 3))])
    except ValueError as exc:
        assert "patch has shape" in str(exc)
    else:
        raise AssertionError("expected ValueError")


TESTS = [
    ("registry matches the checkpoint and the architecture (q-LoRA on/off)", test_registry_matches_checkpoint_and_architecture),
    ("registry warnings: first_k_dense mismatch, skipped layers, fused experts", test_registry_warnings_and_skipped_layers),
    ("patch + restore is bit-identical", test_patch_restore_is_bit_identical),
    ("patch + restore is bit-identical in bf16", test_patch_restore_is_bit_identical_in_bf16),
    ("patch rejects a wrong-shaped matrix", test_patch_rejects_wrong_shape),
]


def main():
    for i, (title, fn) in enumerate(TESTS, 1):
        print(f"=== Test {i}: {title} ===")
        fn()
        print("PASS\n")
    print("ALL REGISTRY / PATCHING TESTS PASSED")


if __name__ == "__main__":
    main()
