# Fine-tuning: Qwen2.5-7B SFT with Monarch-Muon

Code for the Qwen2.5-7B fine-tuning experiments of the section "Initialization for dense-to-Monarch transfer"
and the appendix "Fine-tuning": the selective-conversion table (selective 2-block conversion followed by SFT) and
the fixed-factor-shape table (initializations A-D of the MLPs of layers 14-26).
Qwen2.5-7B (28 layers) is fine-tuned for one epoch on 10,000 examples of the Tulu-3 SFT mixture,
either dense or after replacing some or all of its linear projections by 2-block Monarch factors.
Dense weights and non-matrix parameters are trained with AdamW or Muon; Monarch factors are trained
with AdamW or Monarch-Muon (Muon applied blockwise to the 3-D Monarch factors).
The Qwen2.5-7B weight-projection error figure of the same section is produced in [`../projection/`](../projection/).

## Files

| File | Purpose |
|---|---|
| `sft_config.json` | All fixed hyperparameters (sizes, seed, batch, schedule, optimizer constants, full-SFT LR grids). |
| `sft_data.py` | Builds `train/validation/test(/calibration/calibration_validation).jsonl` + `manifest.json` from local parquet shards. |
| `torch_projection.py` | Closed-form (block-SVD) projection of a dense matrix onto Monarch factors, `MonarchLinear`. |
| `projection.py` | NumPy reference projection, used only as a test oracle. |
| `sft_model.py` | Loads the base model, replaces all 196 projections by Monarch factors, saves checkpoints. |
| `sft_optimizer.py` | `SFTOptimizer`: FP32 master weights; Muon for every `ndim >= 2` parameter except embeddings / LM head, AdamW for the rest. |
| `train_sft.py` | Dense vs. all-196 Monarch SFT (rows "None (dense)" and "All projections" of the selective-conversion table), LR grid, validation-based selection, test NLL. |
| `sft_selective.py` | SFT with only a selected subset of projections converted (remaining rows of the selective-conversion table). |
| `conversion_methods.py` | Initializations B (ALS), C (permutation search), D (Procrustes rotations). |
| `conversion_initialization.py` | Builds and measures the four initializations A-D of the MLPs of layers 14-26. |
| `conversion_training.py`, `conversion_campaign.py` | SFT from initializations A-D and the dense control (fixed-factor-shape table). |
| `run.sh` | All commands below, with the selection and LR presets. |
| `test_*.py` | Unit tests. |

## Requirements

`pip install -r requirements.txt`. `transformers==4.51.3` is required exactly: initialization D replaces
the Qwen2 decoder-layer `forward`, whose signature depends on the version.

Hardware used in the paper:

* SFT (`train_sft.py`, `sft_selective.py`, `conversion_campaign.py train`): one node with **2x H100 80GB**,
  DDP with ZeRO-1 (`ZeroRedundancyOptimizer`). `train_sft.py` refuses other GPU counts or non-H100 devices,
  because the reported numbers come from exactly that setup. Peak memory in a 2-step smoke was 73.5 GiB per GPU for
  dense AdamW, 61 GiB dense Muon, 51 GiB Monarch AdamW, 43 GiB Monarch-Muon.
* Test evaluation of `train_sft.py` (`--evaluate-index`): 1x H100.
* Conversion A-D (`conversion_campaign.py convert`): 1 GPU with **>= 75 GiB** (measurement holds two FP32
  copies of the model).
* Disk: every SFT trial of `train_sft.py` saves a full BF16 checkpoint (~14 GiB, 12 trials); `conversion_campaign.py
  train` keeps only the current best checkpoint of its grid.

## Inputs (local only; nothing is downloaded)

* `MODEL_DIR`: a local copy of **Qwen2.5-7B** (base) in Hugging Face format (config, tokenizer, safetensors).
  The paper used `Qwen/Qwen2.5-7B` at revision `d149729398750b98c0af14eb82c78cfe92750796`.
* `TULU_DIR`: a local copy of the **Tulu-3 SFT mixture** with its six shards at
  `TULU_DIR/data/train-0000{0..5}-of-00006.parquet` (the layout of the Hugging Face dataset repository).
  The paper used `allenai/tulu-3-sft-mixture` at revision `b14afda60f1bbebe55d5d2fa1e4df5042f97f8be`.

These identifiers are recorded in `sft_config.json` as provenance only. Use absolute paths.

## 1. Data

```bash
MODEL_DIR=/abs/Qwen2.5-7B TULU_DIR=/abs/tulu-3-sft-mixture OUT=/abs/out ./run.sh data
# = python sft_data.py --model-dir $MODEL_DIR --tulu-dir $TULU_DIR --output $OUT/data \
#       --calibration-examples 128 --calibration-validation-examples 32
```

A reservoir sample (seed 20260929, candidate pool 3x the needed size) over all six shards, shuffled, deduplicated
by conversation, rendered with the native Qwen chat template, assistant-only labels (including `<|im_end|>`),
right truncation to 1024 tokens, no packing. Splits: 10,000 train / 512 validation / 512 test examples, followed by
128 calibration + 32 calibration-validation examples used only by initializations B-D. The calibration splits are
taken after the first three, so the train/validation/test splits are identical with or without them; one data
directory serves all experiments. With the inputs above, the manifest should report 3,561,198 supervised training
tokens (313 optimizer steps of 32 examples), 168,277 validation tokens and 184,841 test tokens.

## 2. Dense vs. all-196 Monarch SFT (`train_sft.py`, rows "None (dense)" and "All projections")

Arms: {dense, monarch (all 196 projections, 2 blocks)} x {AdamW, Muon}, LRs AdamW {1e-5, 3e-5, 1e-4},
Muon {3e-4, 1e-3, 3e-3} (auxiliary AdamW LR 3e-5 for embeddings, LM head, norms and biases).

```bash
./run.sh full-smoke          # torchrun --standalone --nproc_per_node=2 train_sft.py --model-dir ... --data ... --root $OUT/full --smoke
./run.sh full-trial          # trial indices 0..11, or e.g. `./run.sh full-trial 3 4`
./run.sh full-eval           # arm indices 0..3 (dense-adamw, dense-muon, monarch-adamw, monarch-muon), 1 GPU each
```

`--smoke` computes the closed-form projection of all 196 matrices once (`$OUT/full/monarch-initial.safetensors`,
with its SHA-256 and per-matrix projection errors) and runs two optimizer steps per arm; every Monarch trial starts
from that file. Each trial (`--trial-index`) trains one (arm, LR), writes per-step logs
(`runs-g2/<run>/<run>.jsonl`), the final validation NLL (`<run>.json`) and a checkpoint
(`checkpoints/<run>.safetensors`). It never reads the test split. `--evaluate-index` selects, per arm, the LR with
the lowest final validation NLL, loads its checkpoint and writes the test NLL to
`evaluation-g2/<arm>/<arm>-selected.json`.

Training details shared by every script: global batch 32 = 2 GPUs x 16 micro-batches of 1 example, loss summed over
assistant tokens and divided by their global count per step, gradient clipping 1.0, 3% linear warmup then linear
decay to zero, AdamW betas (0.9, 0.95), eps 1e-8, weight decay 0.1 (none on Muon matrices), Muon heavy-ball
momentum 0.95 with 5 Newton-Schulz steps and scale sqrt(rows/cols) per block, BF16 weights and compute with FP32
master weights and optimizer states, gradient checkpointing, one seed.

**Metric.** NLL = summed cross-entropy over assistant (response) tokens / number of such tokens, in nats; validation
selects the LR, test is reported for the selected LR only.

## 3. Selective conversion (`sft_selective.py`, remaining rows of the selective-conversion table)

Only the selected projections are converted (closed-form projection, computed in-process); everything else follows
section 2 above. Each trial reports the final validation and test NLL in `<out>/<label>-<opt>-lr<lr>-g2.json`; select the
LR by validation NLL and report the test NLL of that trial. No checkpoints are saved.

```bash
./run.sh selective <selection> <lr preset>
# = torchrun --standalone --nproc_per_node=2 sft_selective.py --model-dir ... --data ... --out ... \
#       --label mlp14 --select mlp@14-26 --trials muon:0.0003 muon:0.001 muon:0.003
```

| Row | Selection preset (`--select`) | AdamW preset | Muon preset |
|---|---|---|---|
| MLP, layers 14-26 | `mlp14` (`mlp@14-26`) | `adamw3` {1e-5, 3e-5, 1e-4} | `muon3` {3e-4, 1e-3, 3e-3} |
| MLP, layers 8-26 | `mlp8` (`mlp@8-26`) | `adamwlo` {3e-6, 1e-5, 3e-5} | `muon3` |
| gate+up, layers 14-26 | `gu14` | `adamwlo` | `muonlo` {1e-4, 3e-4} |
| gate+up, layers 8-26 | `gu8` | `adamwlo` | `muonlo` |
| gate+up, layers 2-26 | `gu2` | `adamwlo` | `muonlo` |

Layers are 0-indexed; attention projections, the first layers and the last layer (27) stay dense in all
selective rows. The dense and all-projection rows come from section 2 above.

## 4. Initializations A-D (`conversion_campaign.py`, fixed-factor-shape table)

The same Monarch factor shapes (2 blocks) for gate, up and down projections of the MLPs of layers 14-26
(39 matrices), initialized in four ways:

* **A** closed-form block-SVD projection (as in sections 2-3 above).
* **B** A, then activation-output least squares on 128 calibration examples (up to 256 positions each):
  4 alternating passes, the left factor by an exact least-squares solve, the right factor by 30 conjugate-gradient
  steps on its normal equations; the pass (or A itself) with the lowest output error on 32 held-out calibration
  examples is kept.
* **C** a permutation of the shorter external axis found by 16 iterations of a straight-through projection /
  linear-assignment search (exact Hungarian solver), then B in the permuted layout. Permutations are fixed buffers.
* **D** orthogonal rotations `Qi` (shared by the gate/up inputs) and `Qo` (down output), alternating with B's
  fitting: two Frobenius-Procrustes initialization sweeps, then two sweeps of Cayley-step rotation updates
  (3 steps, backtracking) and refitting; selected by held-out output error. The block computes
  `h + mlp(norm(h) @ Qi) @ Qo.T`, so the residual stream and RMSNorm stay in the original coordinates; rotations
  are FP32 buffers excluded from both optimizers.

```bash
./run.sh convert             # python conversion_campaign.py convert --root $OUT/conversion --data $OUT/data --model-dir $MODEL_DIR
./run.sh init-sft A muon     # repeat for arm in dense A B C D and optimizer in adamw muon
```

`convert` writes `initial-{a,b,c,d}.safetensors` (+ `.json` with per-matrix weight and held-out output errors),
per-layer logs, and then measures every initialization as a whole model on the validation split
(`initial-metrics-*.json`: NLL before SFT, KL to the dense model), after checking that the rotations alone
reproduce the dense model (`rotations-only-parity.json`). The dense model's NLL is in `initial-metrics-dense.json`.
The output error of the table is obtained from `initial-{a,b,c,d}.json`: square the
`teacher_calibration_validation_output_relative_error` of each matrix, sum over the three MLP matrices of a layer,
and average over layers 14-26. SFT requires `initialization-complete.json` from `convert`.

`init-sft <arm> <optimizer>` trains the whole grid AdamW {3e-6, 1e-5, 3e-5} or Muon {1e-4, 3e-4, 1e-3} in one
2-GPU job, keeps the checkpoint with the lowest final validation NLL, and writes its test NLL to
`sft-<arm>-<optimizer>/selected.json`. `dense` is the dense control with the same grid.

## Tests

From this directory:

```bash
python -m unittest test_projection test_torch_projection test_sft test_conversion_methods   # CPU
torchrun --standalone --nproc_per_node=2 test_sft_distributed.py                              # CPU, gloo
```

They check the projection against the NumPy oracle and its optimality bound, the Monarch forward/backward,
the FP32-master AdamW against `torch.optim.AdamW`, checkpoint round trips, the ALS / permutation / Procrustes
algebra, exact output parity of the rotated Qwen2 MLP block, and that ZeRO-1 + DDP gradient accumulation reproduces
serial updates for both optimizers.
