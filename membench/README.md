# Memory and time benchmark

This directory produces the training-cost measurements in the section
"Training efficiency improves with width" (the table of peak memory,
optimizer-state memory, full training-step time and optimizer-step time) and
the step-time and optimizer-state breakdown figure in the Introduction. The
methods are Dense Muon, Monarch-Muon (4 blocks), GaLore, FRUGAL, APOLLO, Fira
and Dense AdamW, on LLaMA-style models from 257M to 6.9B parameters, on a
single H100.

Every point runs complete training steps: forward, backward and optimizer
step. Each step processes 16,384 tokens: sequence length 1024, microbatch 4,
4 accumulation steps. Inputs are random token batches that are allocated once
before timing, so data loading is not part of the measurement. Models run in
eager mode with BF16 parameters, gradients and optimizer moments (scalar
counters may stay FP32). Matmuls run under BF16 autocast, and Newton–Schulz
also runs in BF16.

## Layout

```
scripts/monarch_benchmark/
  benchmark_train_step.py  one (model, variant) point -> one JSON file
  run_sweep.py             runs the grid, one subprocess per point; resumable
  common.py                model sizes, variant list, shared controls and helpers
src/models/llama.py        LLaMA model
src/models/monarch/        MonarchLinear, the custom-op butterfly with Triton riffle kernels, MonarchMuonOptimizer
src/optim/memory_efficient/{apollo,fira,frugal}/   vendored baseline optimizers
src/optim/sota_opt/dion/   vendored Muon (used for Dense Muon)
tests/test_monarch_large_benchmark.py
```

## Variants

| `--variant` | Method | Optimizer | Model |
|---|---|---|---|
| `dense_muon` | Dense Muon | `dion.Muon`: Nesterov momentum 0.95, 5 Newton–Schulz steps (BF16) on hidden linear weights, learning rate scaled by 0.2·sqrt(max(A, B)); AdamW at the base learning rate for embeddings, LM head, norms | dense |
| `monarch_muon` | Monarch-Muon | `MonarchMuonOptimizer` (`src/models/monarch/monarch_muon.py`) | Monarch, 4 blocks |
| `galore` | GaLore | `fira.GaLoreAdamW`, SVD projection | dense |
| `frugal` | FRUGAL | `frugal.BlockAdamW`, blockwise AdamW state + SignSGD elsewhere | dense |
| `apollo` | APOLLO | `apollo.APOLLOAdamW`, random projection, channel-wise scaling | dense |
| `fira` | Fira | `fira.FiraAdamW`, SVD projection + norm-based scaling | dense |
| `dense_adamw` | Dense AdamW | `torch.optim.AdamW(fused=True)` | dense |

Monarch-Muon is the `MonarchMuonOptimizer` in this directory's
`src/models/monarch/monarch_muon.py`. It does batched BF16 Newton–Schulz with
foreach updates on the block-diagonal Monarch factors, and uses AdamW for
embeddings, the LM head and 1-D parameters. Every linear layer except the LM
head is replaced by a 4-block `MonarchLinear`, which runs through the blocked
custom-op butterfly with the Triton riffle kernels (`patch_monarch_linear()`).

GaLore, FRUGAL, APOLLO and Fira run on the dense model. They project only the
decayed `nn.Linear` weights (the `is_proj_params` group from
`model.get_parameter_group_specs()`). Embeddings and the LM head keep full
AdamW state. The projection rank is `--density` × hidden size, with density
0.25 by default. FRUGAL uses the same density for its active-block fraction.

## Model sizes

Vocabulary 50,304, sequence length 1024, head dimension 128, untied LM head.

| `--model-size` | Size | Layers | Hidden | Heads | Dense params | Monarch params (4 blocks) |
|---|---|---|---|---|---|---|
| `257m` | 257M | 12 | 1024 | 8 | 257,188,864 | 163,603,456 |
| `834m` | 834M | 24 | 1536 | 12 | 834,086,400 | 423,568,896 |
| `1p4b` | 1.4B | 24 | 2048 | 16 | 1,439,270,912 | 690,587,648 |
| `3p5b` | 3.5B | 28 | 3072 | 24 | 3,480,136,704 | 1,564,388,352 |
| `6p9b` | 6.9B | 32 | 4096 | 32 | 6,888,361,984 | 2,970,882,048 |

Before timing anything, each run checks that the model it built has exactly
the parameter count listed here.

## Requirements

- A CUDA GPU (the measurements used one H100 80GB, with the job holding the
  GPU to itself).
- PyTorch ≥ 2.4 with Triton. The Linux CUDA wheels include Triton.
- `numpy`.

```bash
pip install -r requirements.txt
```

Run every command from this directory.

## One point

```bash
PYTHONPATH=.:src python -m scripts.monarch_benchmark.benchmark_train_step \
  --variant monarch_muon --model-size 1p4b --output out.json
```

Swap `--variant` and `--model-size` for any other cell. The defaults are the
benchmark settings: microbatch 4, 4 accumulation steps, 3 warm-up steps, 12
measured steps, 3 rebuild steps, `--update-proj-gap 200`, `--density 0.25`,
lr 1e-4, momentum 0.95, betas (0.9, 0.95), weight decay 0.1, eps 1e-7, seed 0.

## Full sweep

```bash
PYTHONPATH=.:src python -m scripts.monarch_benchmark.run_sweep \
  --models 257m,834m,1p4b,3p5b,6p9b \
  --variants dense_muon,monarch_muon,galore,frugal,apollo,fira,dense_adamw
```

Each point runs in its own subprocess on the visible GPU, at microbatch 4 with
4 accumulation steps. The sweep writes one file per point to
`benchmark_results/monarch-muon-large/runs/<model>-<variant>-bs4.json` and
rewrites `benchmark_results/monarch-muon-large/results.json` after every
point. The sweep is resumable: a point that already exists is reused if its
recorded controls match the request. `--rerun` forces every point to run
again. An out-of-memory error is recorded as a result (`"status": "oom"`) and
does not stop the sweep.

## Output fields (per-point JSON)

| Table column | JSON field |
|---|---|
| Full training-step time (ms) | `summary.host_total_ms.median` |
| Optimizer-step time (ms) | `summary.optimizer_ms.median` |
| Peak memory (GB) | `memory.peak_allocated_bytes` / 1e9 |
| Optimizer state (GB) | `memory.optimizer_state_bytes` / 1e9 |

The breakdown figure uses the `6p9b` points: the bars are
`summary.forward_ms.median`, `summary.backward_ms.median` and
`summary.optimizer_ms.median`, the labels are `summary.host_total_ms.median`,
and the state panel is `memory.optimizer_state_bytes` / 1e9.

- `summary.host_total_ms`: wall-clock time for one full step: all
  accumulation microsteps plus the optimizer step plus `zero_grad`. A
  device-wide synchronization closes each step. The value reported is the
  median over the 12 measured steps. `summary.*` also holds mean, std, min,
  p10, p90 and max, and there are matching summaries for `gpu_total_ms`,
  `forward_ms`, `backward_ms` and `tokens_per_second`. Raw per-step values are
  in `samples`.
- `summary.optimizer_ms`: CUDA-event time of `optimizer.step()` alone.
- `summary.forward_ms`, `summary.backward_ms`: CUDA-event time of the
  forward and backward passes, summed over the accumulation microsteps.
- `memory.peak_allocated_bytes`: `torch.cuda.max_memory_allocated` over the
  measured window. The counter is reset after warm-up, so it covers steady
  state, not the projection rebuild.
- `memory.optimizer_state_bytes`: every tensor reachable from
  `optimizer.state`, **including the projection matrices**.
  `memory.optimizer_moment_bytes` (moments only) and
  `memory.optimizer_projector_bytes` (projectors only) break it down. Do not
  add the projector bytes to `optimizer_state_bytes` again.
- `memory.model_bytes`: parameter storage.

**Projection rebuild ("resample").** Projections are built during warm-up and
rebuilt every `--update-proj-gap` = 200 steps. That gap is longer than
warm-up plus the measured window, so the steady-state numbers above contain
no rebuild. For GaLore, Fira (SVD), APOLLO (random projection) and FRUGAL
(random block re-selection), the harness then sets the gap to 1 and measures
3 rebuild steps in a separate window, with its own peak-memory reset. Those
results are in `resample_summary` (for example
`resample_summary.host_total_ms.median`) and
`resample_memory.peak_allocated_bytes`.

`benchmark` records every control setting. `model` records the geometry and
the parameter count that was verified. `variant` records the variant name.
`gpu` and `environment` record the device name, memory, GPU UUID and the
torch/CUDA versions.

## Unit tests

The tests need PyTorch but no GPU: model parameter counts are checked on the
`meta` device.

```bash
PYTHONPATH=.:src python -m unittest tests.test_monarch_large_benchmark
```

## Vendored code

`src/optim/memory_efficient/{apollo,fira,frugal}` and `src/optim/sota_opt/dion`
are trimmed copies of the public reference implementations of APOLLO, Fira
(which also provides the GaLore AdamW used here), FRUGAL and Dion/Muon, cut
down to the code paths this benchmark runs.
`src/models/monarch/monarch_linear.py` is adapted from the HazyResearch Monarch
code. Upstream copyright headers are kept.
APOLLO is licensed under CC BY-NC 4.0 (see
`src/optim/memory_efficient/apollo/LICENSE`); Fira is Apache-2.0, Dion is MIT and
HazyResearch Monarch is Apache-2.0.
