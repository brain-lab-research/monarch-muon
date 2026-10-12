# Monarch projection study

Code for the subsection "Initialization for dense-to-Monarch transfer" (the Moonlight and Qwen projection results) and
the appendix "Monarch projection of Moonlight".

Two models trained identically except for the optimizer, **Moonlight-16B-A3B (Muon)** and **Moonlight-A (AdamW)**, both
at step 42000, have selected weight matrices replaced by their closed-form Monarch projection (Dao et al. 2022, Alg. 1:
per-block truncated SVD in the `MonarchLinear(nblocks=B)` layout, B = 4). There is no training. We measure the damage
against the original model. The projection error relative to a Gaussian matrix is also computed for **Qwen2.5-7B** with
B = 2.

| paper item | command | output |
|---|---|---|
| Figure: projection error / Gaussian-matrix error, per layer and matrix type (Moonlight, B = 4) | `relerr_vs_gaussian.py` (both Moonlight models), then `plots/plot_relerr_strip.py --muon ... --adamw ...` | TSV |
| Figure: single-block FFN replacement, KL per layer (Muon vs AdamW) | `run_joint_blocks.py --plan sweep` (both Moonlight models), then `plots/plot_singles.py` | `configs.csv`, rows `kind=single`, `config=L<l>_ffn` |
| Joint replacement of the FFNs of layers 2-25 against the sum of their single-block KLs | `run_joint_blocks.py --plan sweep` (both Moonlight models) | `configs.csv`, row `config=ffn_2_25`, columns `kl` and `kl_sum_singles` |
| Table: Moonlight joint 4-block conversion (Saved, KL, MMLU 5-shot) | `run_joint_blocks.py --plan sweep` and `--plan arms` (both Moonlight models) | `configs.csv` (`kind=joint`), `mmlu.csv` |
| Figure: projection error / Gaussian-matrix error for Qwen2.5-7B (B = 2) | `relerr_vs_gaussian.py --nblocks 2`, then `plots/plot_relerr_strip.py --qwen ...` | TSV |
| Appendix "Monarch projection of Moonlight": MMLU and GSM8K of the original models | `run_joint_blocks.py --plan sweep` (MMLU) and `--plan arms` (GSM8K), both Moonlight models | `mmlu.csv` of the `sweep` run and `gsm8k.csv` of the `arms` run, rows `config=dense` |

KL is KL(original ‖ patched) in nats/token on the first 32 held-out evaluation sequences (2048 tokens each) of the
calibration corpus. Table rows map to `configs.csv` / `mmlu.csv` `config` values as follows. None is `dense` (MMLU only).
FFN layers 19-25, 14-25 and 8-25 are `ffn_19_25`, `ffn_14_25` and `ffn_8_25`. Routed experts 14-25 and 2-13 are
`routed_14_25` and `routed_2_13`. All blocks is `all`. `dense` and `all` come from the `sweep` run, the other rows from
the `arms` run. The Saved column is `params_saved` divided by the total parameter count of the model.
`kl_sum_singles` is the sum of the single-block KLs of a set's members; the `sweep` run measures single-block KLs for
the FFN blocks, so it applies to FFN sets such as `ffn_2_25`.

## Setup

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128   # torch >= 2.9: torch._grouped_mm
pip install -r requirements.txt
```

`transformers` must stay at 4.57.x.

## Local inputs

Nothing in this folder downloads models or data. You supply:

* **Model directories** in HF format: Moonlight (Muon) and Moonlight-A (AdamW) at step 42000, including their
  `modeling_deepseek.py` / `tokenization_moonshot.py` remote code, and Qwen2.5-7B (read only by `relerr_vs_gaussian.py`).
* **Calibration corpus**, either as text or as tokens:
  * *Texts*: local copies of FineWeb-Edu (sample-10BT, field `text`), codeparrot-clean (field `content`) and
    open-web-math (field `text`). Edit the paths in `configs/calibration.json`. Each source is a parquet file or
    directory, a `.jsonl` file or a `.txt` file. Sequences are drawn 0.5 / 0.25 / 0.25 from the three sources, giving
    512 x 2048 tokens: 128 sequences set aside, then 384 evaluation sequences.
  * *Tokens*: an int array `[512, 2048]` saved with `np.save`, passed with `--tokens-npy` (it skips tokenization).
* **MMLU / GSM8K** are loaded by `lm_eval` (`cais/mmlu`, `openai/gsm8k`) on first use. To run offline, populate the HF
  datasets cache once (`HF_DATASETS_CACHE`) and then set `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1`.

## Commands

Examples use `M=/data/Moonlight-16B-A3B`, `A=/data/Moonlight-A`, `Q=/data/Qwen2.5-7B`, run from this folder.

**1. Tokenize** (once; both Moonlight models share one tokenizer):

```bash
python prepare_calibration.py --config configs/calibration.json --tokens-dir calib/moonlight --tokenizer $M --stages tokenize
# or, with pre-tokenized data:  ... --tokens-npy tokens.npy --stages tokenize
```

**2. Reference cache** (one forward pass of the original model: final hidden states + NLL of the evaluation part):

```bash
for X in "muon $M" "adamw $A"; do set -- $X
  python prepare_calibration.py --config configs/calibration.json --tokens-dir calib/moonlight --model-dir $2 \
      --model-tag moonlight_$1 --stages reference --accel grouped_mm \
      --cache-dir calib/moonlight/refs/moonlight_$1_grouped_mm
done
```

**3. Single blocks and joint sets** (KL, MMLU limit 100/subject, GSM8K 400 problems):

```bash
for X in "muon $M" "adamw $A"; do set -- $X
  for PLAN in sweep arms; do
    python run_joint_blocks.py --model-dir $2 --model-tag moonlight_$1 --tokens-dir calib/moonlight \
        --cache-dir calib/moonlight/refs/moonlight_$1_grouped_mm --accel grouped_mm --nblocks 4 \
        --plan $PLAN --mmlu-limit 100 --gsm8k-limit 400 --out-dir out/joint/moonlight_$1/$PLAN
  done
done
```

The `sweep` plan replaces every FFN block alone, then `all` and `ffn_2_25`, and runs MMLU on `dense` and `all`. The
`arms` plan runs `ffn_19_25`, `ffn_14_25`, `ffn_8_25`, `routed_2_13` and `routed_14_25` with MMLU on `dense` and every
set, and GSM8K on `dense`, `ffn_19_25` and `ffn_14_25`. Outputs per run: `configs.csv` (one row per configuration: `kl`,
`nll_delta`, `top1_agree`, `kl_sum_singles`, `params_saved`, ...), `mmlu.csv`, `gsm8k.csv` (`arms` only), `matrices.csv`
(per-matrix error of the fit, `rel_err`, and `fraction` = Monarch parameters / dense parameters), `joint_sets.json`
(members of every set), `run_info.json` and `samples/` (per-document benchmark results). The Moonlight runs used one
80 GB GPU.

**4. Projection error relative to a Gaussian matrix** (weights only, read with `safetensors`, no model forward):

```bash
python relerr_vs_gaussian.py --model-dir $M --model-name muon  --nblocks 4 --out out/relerr/muon.tsv
python relerr_vs_gaussian.py --model-dir $A --model-name adamw --nblocks 4 --out out/relerr/adamw.tsv
python relerr_vs_gaussian.py --model-dir $Q --model-name qwen  --nblocks 2 --out out/relerr/qwen.tsv
```

There is one row per (layer, group, matrix type). `ratio = rel_err / gauss_rel_err`, where the Gaussian reference is
the mean over 3 seeds for that shape. Routed experts are pooled over the 64 experts of a layer as
sqrt(Σ‖W−M‖²/Σ‖W‖²). The Qwen mean over its 84 MLP matrices is the mean of `rel_err` and of `gauss_rel_err` over the
rows with group `dense FFN`.

**5. Figures**:

```bash
python plots/plot_singles.py --muon out/joint/moonlight_muon/sweep/configs.csv \
    --adamw out/joint/moonlight_adamw/sweep/configs.csv --out moonlight_single_block_kl
python plots/plot_relerr_strip.py --muon out/relerr/muon.tsv --adamw out/relerr/adamw.tsv --out relerr_strip
python plots/plot_relerr_strip.py --qwen out/relerr/qwen.tsv --out relerr_strip_qwen
```

## Files

| file | role |
|---|---|
| `prepare_calibration.py`, `calibration_data.py`, `config_utils.py`, `configs/calibration.json` | tokenized corpus (local sources) and per-model reference cache |
| `run_joint_blocks.py` | single-block and joint-replacement study (entry point) |
| `monarch_blocks.py` | `MonarchLinear(nblocks=B)` layout and its closed-form fit (`fit_svd`) |
| `run_downstream_eval.py` | lm_eval wrapper (last-position log-likelihood for multiple choice) |
| `model_registry.py`, `matrix_patching.py`, `model_loading.py`, `hf_compat.py`, `moe_accel.py` | matrix inventory, in-place patching/restore, loading Moonlight on transformers 4.57 (with the KV-cache shims for GSM8K generation), grouped-GEMM MoE experts |
| `sensitivity_metrics.py` | KL / NLL / top-1 against the reference cache |
| `relerr_vs_gaussian.py` | projection error vs i.i.d. Gaussian matrices |
| `plots/` | the paper figures |
| `test_*.py`, `toy_model.py` | CPU tests on tiny random MLA+MoE models |

## Tests

```bash
python -m pytest projection -q      # from the release root; CPU only, no checkpoints, lm_eval stubbed where needed
```
