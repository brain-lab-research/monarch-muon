# Pretraining: Monarch-Muon and baselines

Code for the pretraining experiments:

- **Pretraining quality** (section "Pretraining quality at fixed structure", dense LLaMA-style model): dense AdamW,
  dense Muon, GaLore, Monarch-AdamW and Monarch-Muon with 2 and 4 blocks.
- **Low-precision training** (section "FP8 complements structured training", 2-block Monarch model): FP8 GEMM (COAT
  protocol) and FP8 optimizer states, for Monarch-Muon and Monarch-AdamW.
- **Layer benchmark** (same section; RMSNorm + SwiGLU Monarch MLP block, our FP8 vs Transformer Engine):
  `bench/bench_monarch_te.py`.

Monarch replaces every linear layer of the transformer (Q, K, V, O, gate, up, down) except `lm_head`; the token embedding
and `lm_head` are untied and stay dense. Monarch-Muon applies Muon blockwise: one batched Newton-Schulz orthogonalization
per block-diagonal factor; embeddings, norms and `lm_head` use the embedded AdamW.

## Requirements

```bash
pip install -r requirements.txt
```

- PyTorch >= 2.8 with Triton, one NVIDIA GPU (the reported runs used a single H100 each).
- FP8 arms: H100 (sm_90). The `*-fp8gemm` arms JIT-compile a small cuBLASLt extension (`src/models/fp8_batched.cpp`)
  on first use, which needs a CUDA toolkit (`nvcc`, cuBLASLt headers) and `ninja`.
- W&B logging is off by default (`--wandb` turns it on; `--wandb-project` sets the project, the standard `WANDB_*`
  environment variables set host and credentials).

## Data

Training reads FineWeb-Edu parquet shards from a local directory; nothing is downloaded. Pass the directory as
`DATA_DIR`: it contains the FineWeb-Edu `sample/100BT` parquet files directly (`000_00000.parquet`, ...; a parent
directory with exactly one nested shard directory also works). Each file needs a `text` column.

Documents are tokenized on the fly with the GPT-2 tokenizer, an EOS token is appended to every document, and tokens
are packed into 1,025-token blocks (1,024 inputs + 1 shifted target). Row groups are split deterministically into
train and held-out sets (split seed 2357, fixed in code); the validation set is the first 1,024 packed sequences
(32 batches x 32) of the held-out row groups and is never trained on. The training order is a seeded shuffle of the
remaining row groups (`--data-seed 1337`). With the unmodified `sample/100BT` files (dataset revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`) and one GPU this reproduces the data order of the reported runs.
The GPT-2 tokenizer is loaded with `transformers.AutoTokenizer.from_pretrained("gpt2")` (pinned revision in
`train.py`); on an offline machine place it in the Hugging Face cache and set `HF_HUB_OFFLINE=1`.

## Running

```bash
bash run.sh monarch-muon-nb2 /path/to/fineweb-edu/sample/100BT
# same as: python train.py monarch-muon-nb2 /path/to/fineweb-edu/sample/100BT
```

`run.sh`/`train.py` expand the arm into `torchrun --standalone --nproc_per_node=$NPROC src/main.py <flags>` and print the
full command. Environment: `NPROC` (default 1), `OUT_DIR` (default `./exps`). Extra arguments are appended to the
`src/main.py` flags (later flags win), e.g. `--wandb` or `--stop-after 100`.

Shared recipe (all arms): LLaMA-style decoder, 12 layers, width 1024, 8 heads, RoPE, RMSNorm, SwiGLU MLP (hidden 2816 =
`--multiple-of 256` rule), untied `lm_head`, vocabulary 50,304, sequence length 1,024; 39,250 steps of 32 x 4 = 128
sequences (gradient accumulation 4); peak LR 1e-3, weight decay 0.1, betas (0.9, 0.99), momentum 0.95,
gradient clipping 1.0; WSD schedule with 2,000 warmup steps and the last 10% decayed to 0 with a cosine; bf16 autocast
over fp32 weights; seed 0, data seed 1337; validation every 500 steps on the held-out set.

| Arm | Flags on top of the shared recipe |
|---|---|
| `dense-adamw` | `--opt adamw` |
| `dense-muon` | `--opt muon` |
| `galore` | `--opt galore_adamw --density 0.25 --update_gap 50 --beta2 0.999 --inactive_update_rule sign_sgd --inactive_lr_scale 1 --acc-steps 2` |
| `monarch-adamw-nb2` / `-nb4` | `--opt adamw --apply-monarch --monarch-nblocks 2` / `4` |
| `monarch-muon-nb2` / `-nb4` | `--opt monarch_muon --monarch-nblocks 2` / `4` |
| `monarch-muon-nb2-fp8gemm` | `monarch-muon-nb2` + `--fp8` |
| `monarch-adamw-nb2-fp8gemm` | `monarch-adamw-nb2` + `--fp8` |
| `monarch-muon-nb2-fp8states` | `monarch-muon-nb2` + `--fp8-optim` |
| `monarch-adamw-nb2-fp8states` | `--opt triton_coat_adamw --apply-monarch --monarch-nblocks 2 --fp8-optim` |

Results go to `$OUT_DIR/llama-12L8H_fineweb_39k/<arm>/`: `summary.json` (train/val loss curves), and
`ckpts/latest` (every 5,000 steps). Rerunning the same arm with the same `OUT_DIR` resumes from `ckpts/latest`.
The training log prints `>Eval: Iter=... val_loss=...` lines; the final validation loss is the metric of the
pretraining quality table, and the `Train:` lines carry the step time (`iter_dt`) and peak memory (`peak_mem`).

### Notes on the arms

- **Monarch-Muon** (`--opt monarch_muon`, `src/models/monarch/monarch_muon.py`): heavy-ball momentum 0.95 (no
  Nesterov), no weight decay on the Monarch factors, 5 Newton-Schulz steps batched over the blocks of each factor,
  per-block update scale `sqrt(rows/cols)`; the embedded AdamW (embeddings, norms, `lm_head`) uses betas (0.9, 0.99),
  eps 1e-7, weight decay 0.1.
- **Dense Muon** (`--opt muon`, `src/optim/muon.py`): Muon on every 2-D weight of the transformer blocks with momentum
  0.95 and a Nesterov-style look-ahead, 6 Newton-Schulz steps with the coefficients listed in the file, update scaled by
  `0.2 * sqrt(max(rows, cols))`, weight decay 0.1; embeddings, norms and `lm_head` use AdamW (betas 0.9/0.99, eps 1e-8,
  weight decay 0.1).
- **AdamW arms** use `torch.optim.AdamW` with eps 1e-8; weight decay applies to the transformer's linear and Monarch
  weights, not to embeddings, norms or `lm_head`.
- **GaLore** is the FRUGAL implementation's `GaloreAdamW` on the transformer's linear weights (`lm_head`, embeddings and
  norms use plain AdamW): rank = 0.25 x the smaller matrix dimension, SVD projector refreshed every 50 steps, AdamW
  (beta2 0.999, eps 1e-7) inside the subspace. The gradient component outside the subspace is applied with signSGD
  (`--inactive_update_rule sign_sgd`, `--inactive_lr_scale 1`). The arm uses gradient accumulation 2 (64 sequences per
  step, 39,250 steps).
- **FP8 GEMM** (`--fp8`, `src/models/coat_monarch.py`): the COAT FP8 activation flow (1x16-group FP8 inputs to RMSNorm,
  SiLU and the gate*up product, BF16 residual stream) with the two block GEMMs of every Monarch projection in FP8 via
  batched cuBLASLt (E4M3 forward, E5M2 gradients, per-tensor current scaling, no fast accumulation). Evaluation runs in
  BF16.
- **FP8 optimizer states** (`--fp8-optim`): BF16 GEMMs; Monarch-Muon momentum and the embedded AdamW moments
  (`src/models/monarch/coat_state_muon.py`), or the AdamW moments (`triton_coat_adamw`, a Triton port of COAT's FP8
  AdamW, eps 1e-8), are stored in E4M3 with groups of 128 and COAT's dynamic range expansion.
- With `NPROC > 1` the global batch stays 128 sequences (accumulation steps are divided across GPUs), but the data
  stream is sharded per rank, so the order differs from the single-GPU runs.

## Layer benchmark

```bash
PYTHONPATH=src:. python bench/bench_monarch_te.py [stage3_moe_1b qwen3_moe_30b_like wide_experts_2048x2048 deepseek_v3_like_64_of_256_experts]
```

Times forward+backward of an RMSNorm + SwiGLU MLP block (MLP width = top-k x expert width of each MoE shape) on one H100:
dense in Transformer Engine BF16 (the reference), 2/4-block Monarch in Transformer Engine BF16 and in its FP8 recipes
(blockwise, tensorwise, delayed scaling; `warm` arms reuse the quantized weights), and 2/4-block Monarch with our
batched FP8 GEMMs and one shared input quantization for gate/up (`ours_fp8_warm`). Prints one `MONARCH_TE {json}` line
per arm with time, peak memory and relative error vs TE BF16.
Needs Transformer Engine 2.x (`GroupedLinear`, `Float8BlockScaling`) and the FP8 extension prerequisites above.

## Tests

```bash
python -m pytest pretrain/tests
```

The tests check that the FP8 Monarch GEMM functions match the reference bit for bit, that the FP8 codec equals COAT's
AdamW kernel, and that the FP8 arms train like their BF16 counterparts for a few steps. They need an H100 and are
skipped without CUDA.

## Third-party code

`third_party/coat` is adapted from NVlabs/COAT (Apache-2.0, see `third_party/coat/LICENSE` and `NOTICE`).
`src/models/monarch/monarch_linear.py` is adapted from HazyResearch/monarch. `src/optim/memory_efficient/frugal` is the
FRUGAL/GaLore optimizer implementation.
