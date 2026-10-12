"""Launch one pretraining arm: python train.py ARM DATA_DIR [extra src/main.py flags...]

DATA_DIR holds local FineWeb-Edu parquet shards.
Environment: NPROC (GPUs, default 1), OUT_DIR (results, default ./exps).
"""
import os
from pathlib import Path
import sys

# 12-layer LLaMA, width 1024, 8 heads, SwiGLU (hidden 2816), seq 1024, 39,250 steps of 32 x 4 sequences,
# WSD schedule (warmup 2000, last 10% cosine decay to 0), bf16 autocast over fp32 weights.
RECIPE = [
    "--n-layer", "12", "--n-embd", "1024", "--n-head", "8", "--multiple-of", "256",
    "--sequence-length", "1024", "--vocab-size", "50304", "--init-std", "0.02",
    "--batch-size", "32", "--acc-steps", "4", "--iterations", "39250",
    "--lr", "0.001", "--weight-decay", "0.1", "--momentum", "0.95", "--beta1", "0.9", "--beta2", "0.99",
    "--eps", "1e-07", "--grad-clip", "1",
    "--warmup-steps", "2000", "--wsd-fract-decay", "0.1", "--wsd-final-lr-scale", "0",
    "--seed", "0", "--data-seed", "1337",
    "--tokenizer-revision", "607a30d783dfa663caf39e06633721c8d4cfcd7e", "--workers", "8",
    "--eval-interval", "500", "--eval-batches", "32", "--eval-batch-size", "32",
    "--log-interval", "10", "--latest-ckpt-interval", "5000",
]

MONARCH_MUON = ["--opt", "monarch_muon"]
MONARCH_ADAMW = ["--opt", "adamw", "--apply-monarch"]

ARMS = {
    "dense-adamw": ["--opt", "adamw"],
    "dense-muon": ["--opt", "muon"],
    "galore": ["--opt", "galore_adamw", "--density", "0.25", "--update_gap", "50",
               "--beta2", "0.999", "--inactive_update_rule", "sign_sgd", "--inactive_lr_scale", "1",
               "--acc-steps", "2"],
    "monarch-adamw-nb2": MONARCH_ADAMW + ["--monarch-nblocks", "2"],
    "monarch-adamw-nb4": MONARCH_ADAMW + ["--monarch-nblocks", "4"],
    "monarch-muon-nb2": MONARCH_MUON + ["--monarch-nblocks", "2"],
    "monarch-muon-nb4": MONARCH_MUON + ["--monarch-nblocks", "4"],
    # Low precision, 2-block model (H100). fp8gemm: FP8 Monarch block GEMMs (cuBLASLt) inside the COAT FP8
    # activation flow. fp8states: BF16 GEMMs, optimizer states in FP8 (E4M3, groups of 128, dynamic range expansion).
    "monarch-muon-nb2-fp8gemm": MONARCH_MUON + ["--monarch-nblocks", "2", "--fp8"],
    "monarch-adamw-nb2-fp8gemm": MONARCH_ADAMW + ["--monarch-nblocks", "2", "--fp8"],
    "monarch-muon-nb2-fp8states": MONARCH_MUON + ["--monarch-nblocks", "2", "--fp8-optim"],
    "monarch-adamw-nb2-fp8states": ["--opt", "triton_coat_adamw", "--apply-monarch", "--monarch-nblocks", "2",
                                    "--fp8-optim"],
}


def arm_args(arm):
    """src/main.py flags of an arm (later flags override the shared recipe)."""
    return RECIPE + ARMS[arm]


def data_args(data_dir):
    return ["--datasets-dir", str(Path(data_dir).expanduser().resolve())]


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ARMS:
        sys.exit(f"usage: python train.py ARM DATA_DIR [extra flags]\narms: {', '.join(ARMS)}")
    arm, data_dir, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    here = Path(__file__).resolve().parent
    command = [
        "torchrun", "--standalone", f"--nproc_per_node={os.environ.get('NPROC', '1')}", str(here / "src/main.py"),
        "--experiment-name", arm, "--results-base-folder", str(Path(os.environ.get("OUT_DIR", "exps")).resolve()),
        "--distributed-backend", "nccl", *arm_args(arm), *data_args(data_dir), *extra,
    ]
    print(" ".join(command), flush=True)
    os.chdir(here)
    os.execvp(command[0], command)
