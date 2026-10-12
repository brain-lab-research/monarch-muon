import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.monarch_benchmark.benchmark_train_step import (
    OPTIMIZER_BACKENDS,
    force_projector_resample,
    make_config,
    moment_state,
    projector_state,
    tensor_bytes,
    tensor_dtypes,
)
from scripts.monarch_benchmark.common import (
    HARNESS_REVISION,
    MODEL_SPECS,
    VARIANTS,
    model_geometry,
    percentile,
    result_is_complete,
    result_is_recorded,
    result_matches_request,
    summarize,
)
from scripts.monarch_benchmark.run_sweep import result_path
from models.llama import Llama
from models.monarch import apply_monarch


def controls(microbatch=4, **overrides):
    values = {
        "harness_revision": HARNESS_REVISION,
        "storage_dtype": "bfloat16",
        "autocast_dtype": "bfloat16",
        "optimizer_moment_dtype": "bfloat16",
        "sequence_length": 1024,
        "microbatch": microbatch,
        "accumulation_steps": 16384 // (microbatch * 1024),
        "tokens_per_step": 16384,
        "warmup_steps": 1,
        "measured_steps": 2,
        "compile_model": False,
        "density": 0.25,
        "update_proj_gap": 200,
        "resample_steps": 3,
        "lr": 1e-4,
        "momentum": 0.95,
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "eps": 1e-7,
        "seed": 0,
    }
    values.update(overrides)
    return values


def complete_result(model, variant, microbatch):
    return {
        "status": "complete",
        "model": {"name": model["name"]},
        "variant": {"name": variant},
        "benchmark": controls(microbatch),
        "samples": [{}, {}],
    }


def oom_result(model, variant, microbatch):
    return {
        "status": "oom",
        "model_size": model["name"],
        "variant": variant,
        "requested_controls": controls(microbatch),
    }


class BenchmarkMatrixTest(unittest.TestCase):
    def test_model_ramp_and_verified_parameter_counts(self):
        dense = [spec["dense_params_expected"] for spec in MODEL_SPECS]
        self.assertEqual(dense, sorted(dense))
        self.assertGreater(dense[-1], 6_800_000_000)
        self.assertLess(dense[-1], 7_000_000_000)
        monarch = [spec["monarch_params_expected"] for spec in MODEL_SPECS]
        self.assertEqual(monarch, sorted(monarch))
        self.assertTrue(all(m < d for m, d in zip(monarch, dense)))
        for spec in MODEL_SPECS:
            self.assertEqual(
                model_geometry(spec, "monarch_muon")["params_expected"],
                spec["monarch_params_expected"],
            )

    def test_meta_models_match_declared_parameter_counts(self):
        with torch.device("meta"):
            for spec in MODEL_SPECS:
                config = make_config(spec, sequence_length=256)
                with self.subTest(model=spec["name"]):
                    dense = Llama(config)
                    self.assertEqual(
                        sum(parameter.numel() for parameter in dense.parameters()),
                        spec["dense_params_expected"],
                    )
                with self.subTest(model=spec["name"], monarch=True):
                    monarch = Llama(config)
                    with contextlib.redirect_stdout(io.StringIO()):
                        apply_monarch(monarch, nblocks=4, verbose=False)
                    self.assertEqual(
                        sum(parameter.numel() for parameter in monarch.parameters()),
                        spec["monarch_params_expected"],
                    )

    def test_every_variant_has_a_declared_backend(self):
        self.assertEqual(
            sorted(variant["name"] for variant in VARIANTS), sorted(OPTIMIZER_BACKENDS)
        )
        families = {variant["family"] for variant in VARIANTS}
        self.assertTrue(families.issubset({"dense", "monarch", "memory_efficient"}))

    def test_statistics_are_deterministic(self):
        values = [1.0, 2.0, 3.0, 4.0]
        stats = summarize(values)
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["median"], 2.5)
        self.assertAlmostEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(percentile(values, 0.1), 1.3)
        self.assertAlmostEqual(percentile(values, 0.9), 3.7)


class ResumeTest(unittest.TestCase):
    def test_complete_result_requires_all_samples(self):
        payload = {
            "status": "complete",
            "benchmark": {"measured_steps": 2},
            "samples": [{}, {}],
        }
        self.assertTrue(result_is_complete(payload))
        payload["samples"].pop()
        self.assertFalse(result_is_complete(payload))

    def test_out_of_memory_counts_as_a_recorded_point(self):
        payload = oom_result(MODEL_SPECS[-1], "dense_adamw", 4)
        self.assertFalse(result_is_complete(payload))
        self.assertTrue(result_is_recorded(payload))
        self.assertTrue(
            result_matches_request(
                payload,
                model_name=MODEL_SPECS[-1]["name"],
                variant_name="dense_adamw",
                controls=controls(4),
            )
        )
        self.assertFalse(
            result_matches_request(
                payload,
                model_name=MODEL_SPECS[-1]["name"],
                variant_name="dense_adamw",
                controls=controls(4, sequence_length=512),
            )
        )

    def test_resume_requires_exact_identity_and_controls(self):
        payload = complete_result(MODEL_SPECS[0], "dense_adamw", 4)
        kwargs = {
            "model_name": MODEL_SPECS[0]["name"],
            "variant_name": "dense_adamw",
            "controls": controls(4),
        }
        self.assertTrue(result_matches_request(payload, **kwargs))
        payload["benchmark"]["sequence_length"] = 128
        self.assertFalse(result_matches_request(payload, **kwargs))

    def test_result_path_names_model_variant_and_batch(self):
        self.assertEqual(
            result_path(Path("/out"), "257m", "galore", 4),
            Path("/out/runs/257m-galore-bs4.json"),
        )


class ProjectorAccountingTest(unittest.TestCase):
    """GaLore/APOLLO/Fira keep the projection in an attribute of a plain object,
    so a walk over tensors and containers alone reports it as zero bytes."""

    class Projector:
        def __init__(self, matrix):
            self.ortho_matrix = matrix
            self.update_proj_gap = 200
            self.rank = 4

    def optimizer_with(self, extra):
        parameter = torch.zeros(4, 4, dtype=torch.bfloat16)
        state = {parameter: {"step": 3, "exp_avg": torch.zeros(4, 4, dtype=torch.bfloat16), **extra}}
        return SimpleNamespace(state=state, param_groups=[{"update_proj_gap": 200, "params": [parameter]}])

    def test_projection_matrix_is_counted(self):
        matrix = torch.zeros(4, 8, dtype=torch.bfloat16)
        optimizer = self.optimizer_with({"projector": self.Projector(matrix)})
        moments = tensor_bytes(moment_state(optimizer))
        self.assertEqual(moments, 4 * 4 * 2)
        self.assertEqual(tensor_bytes(projector_state(optimizer)), 4 * 8 * 2)
        self.assertEqual(tensor_bytes(optimizer.state), moments + 4 * 8 * 2)

    def test_projector_dtypes_stay_out_of_the_moment_assertion(self):
        class IndexProjector:
            def __init__(self):
                self.indices = torch.zeros(6, dtype=torch.int64)

        optimizer = self.optimizer_with({"projector": IndexProjector()})
        self.assertIn("int64", tensor_dtypes(optimizer.state))
        self.assertEqual(
            sorted(tensor_dtypes(moment_state(optimizer), nonscalar_only=True)), ["bfloat16"]
        )

    def test_scalars_are_excluded_from_the_moment_assertion(self):
        optimizer = self.optimizer_with({"scaling_grad": torch.tensor(1.5, dtype=torch.bfloat16)})
        self.assertEqual(
            sorted(tensor_dtypes(moment_state(optimizer), nonscalar_only=True)), ["bfloat16"]
        )

    def test_forcing_a_resample_reaches_group_and_projector(self):
        optimizer = self.optimizer_with({"projector": self.Projector(torch.zeros(2, 2))})
        force_projector_resample(optimizer)
        self.assertEqual(optimizer.param_groups[0]["update_proj_gap"], 1)
        self.assertEqual(projector_state(optimizer)[0].update_proj_gap, 1)


if __name__ == "__main__":
    unittest.main()
