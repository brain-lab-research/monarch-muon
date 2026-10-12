# Monarch-Muon: Structure-Aware Muon for Fast and Memory-Efficient Pretraining

Code for the experiments of the paper. Each folder is self-contained, has its own `requirements.txt` and a
README with the exact commands. Nothing in this repository downloads models or datasets: every script takes
local paths to data and checkpoints that you already have.

| Folder | Experiments |
|---|---|
| [`pretrain/`](pretrain/) | Pretraining of the dense LLaMA-style model (final validation loss): dense AdamW, dense Muon, GaLore, Monarch-AdamW and Monarch-Muon with 2 and 4 blocks. Optional low-precision training of the 2-block model (FP8 GEMM following the COAT protocol, FP8 optimizer states) and the FP8 Monarch MLP layer benchmark. |
| [`membench/`](membench/) | Peak memory, optimizer state, full training-step and optimizer-step time of Dense Muon, Monarch-Muon, GaLore, FRUGAL, APOLLO, Fira and Dense AdamW at 257M-6.9B parameters. |
| [`projection/`](projection/) | Closed-form Monarch projection of pretrained models: single-block and joint replacement of Moonlight-16B-A3B (Muon) and Moonlight-A (AdamW) with KL, MMLU and GSM8K; projection error relative to an i.i.d. Gaussian matrix for every matrix of Moonlight and Qwen2.5-7B. |
| [`finetune/`](finetune/) | Supervised fine-tuning of Qwen2.5-7B on the Tulu 3 SFT mixture: dense, fully and selectively converted to Monarch form, and the four initializations of the Monarch factors (closed-form, output fit, permutation, rotations). |

The MoE pretraining experiments are not included.

## Where Monarch-Muon lives

- `pretrain/src/models/monarch/monarch_linear.py`: Monarch layer `P1 L P2 R` and its application to a model.
- `pretrain/src/models/monarch/monarch_muon.py`: Monarch-Muon used for the pretraining runs (heavy-ball momentum,
  per-block Newton-Schulz batched over all blocks of the same shape, per-block step scaled by `sqrt(rows/cols)`);
  selected with `--opt monarch_muon`.
- `finetune/sft_optimizer.py`: the same blockwise update for the 3-D Monarch factors of Qwen2.5-7B.

## Tests

Each folder has unit tests (see its README). The FP8 tests need an H100.

## License

MIT, see [`LICENSE`](LICENSE). Adapted third-party code keeps its upstream license: `pretrain/third_party/coat`
(NVlabs/COAT, Apache-2.0) and `membench/src/optim/memory_efficient/apollo` (APOLLO, CC BY-NC 4.0); see the folder
READMEs for the other adapted components.
