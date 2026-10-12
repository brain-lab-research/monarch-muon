#!/bin/bash
# Commands of the fine-tuning experiments (see README.md). Run from any directory:
#   MODEL_DIR=/path/to/Qwen2.5-7B TULU_DIR=/path/to/tulu-3-sft-mixture OUT=/path/to/out ./run.sh <command> [args]
#
#   data                              build train/validation/test (+ calibration) splits from local parquet shards
#   full-smoke                        2-step smoke of dense/Monarch x AdamW/Muon; creates the 196-matrix projection
#   full-trial [i ...]                full SFT trials 0..11 (dense/monarch x adamw/muon x 3 LRs), default all
#   full-eval [i ...]                 pick LR by validation NLL and evaluate test NLL for arms 0..3, default all
#   selective <selection> <lrs>       selective conversion SFT, e.g. `selective mlp14 muon3`
#   convert                           build initializations A-D for the MLP of layers 14-26 (1 GPU) and measure them
#   init-sft <dense|A|B|C|D> <adamw|muon>   SFT from one initialization over its 3-LR grid, then test NLL
set -euo pipefail
cd "$(dirname "$0")"
: "${MODEL_DIR:?set MODEL_DIR to a local Qwen2.5-7B directory}"
OUT=${OUT:-outputs}
DATA=$OUT/data
TORCHRUN="torchrun --standalone --nproc_per_node=2"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
COMMAND=${1:?command}
shift
case "$COMMAND" in
    data)
        python sft_data.py --model-dir "$MODEL_DIR" --tulu-dir "${TULU_DIR:?set TULU_DIR}" --output "$DATA" \
            --calibration-examples 128 --calibration-validation-examples 32 ;;
    full-smoke)
        $TORCHRUN train_sft.py --model-dir "$MODEL_DIR" --data "$DATA" --root "$OUT/full" --smoke ;;
    full-trial)
        for i in ${@:-$(seq 0 11)}; do
            $TORCHRUN train_sft.py --model-dir "$MODEL_DIR" --data "$DATA" --root "$OUT/full" --trial-index "$i"
        done ;;
    full-eval)
        for i in ${@:-0 1 2 3}; do
            torchrun --standalone --nproc_per_node=1 train_sft.py --model-dir "$MODEL_DIR" --data "$DATA" \
                --root "$OUT/full" --evaluate-index "$i"
        done ;;
    selective)
        SEL=${1:?selection preset}; TRIALS=${2:?LR preset}
        case "$SEL" in
            mlp14)  SELECT="mlp@14-26" ;;         # attention, first and last layers stay dense
            mlp8)   SELECT="mlp@8-26" ;;
            gu14)   SELECT="mlp.gate_proj@14-26 mlp.up_proj@14-26" ;;   # down_proj stays dense
            gu8)    SELECT="mlp.gate_proj@8-26 mlp.up_proj@8-26" ;;
            gu2)    SELECT="mlp.gate_proj@2-26 mlp.up_proj@2-26" ;;
            *) echo "unknown selection preset $SEL" >&2; exit 1 ;;
        esac
        case "$TRIALS" in
            adamw3)  TR="adamw:1e-05 adamw:3e-05 adamw:0.0001" ;;
            adamwlo) TR="adamw:3e-06 adamw:1e-05 adamw:3e-05" ;;
            muon3)   TR="muon:0.0003 muon:0.001 muon:0.003" ;;
            muonlo)  TR="muon:0.0001 muon:0.0003" ;;
            *) echo "unknown LR preset $TRIALS" >&2; exit 1 ;;
        esac
        $TORCHRUN sft_selective.py --model-dir "$MODEL_DIR" --data "$DATA" --out "$OUT/selective/$SEL-$TRIALS" \
            --label "$SEL" --select $SELECT --trials $TR ;;
    convert)
        python conversion_campaign.py convert --root "$OUT/conversion" --data "$DATA" --model-dir "$MODEL_DIR" ;;
    init-sft)
        $TORCHRUN conversion_campaign.py train --arm "${1:?arm}" --optimizer "${2:?optimizer}" \
            --root "$OUT/conversion" --data "$DATA" --model-dir "$MODEL_DIR" ;;
    *) echo "unknown command $COMMAND" >&2; exit 1 ;;
esac
