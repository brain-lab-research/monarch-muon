import distributed


def parse_args(base_parser, args, namespace):
    parser = base_parser
    # General training params
    parser.add_argument("--experiment-name", required=True, type=str,
        help="Short name for this run (e.g. 'monarch-muon-nb2'). Used for wandb and checkpoints.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--data-seed", default=1337, type=int)
    parser.add_argument("--eval-interval", default=200, type=int)
    parser.add_argument("--eval-batches", default=32, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument(
        "--distributed-backend",
        default=None,
        type=str,
        required=False,
        choices=distributed.registered_backends(),
    )
    parser.add_argument("--log-interval", default=50, type=int)

    # Checkpointing
    parser.add_argument("--results-base-folder", default="./exps", type=str)
    parser.add_argument("--latest-ckpt-interval", default=0, type=int)
    parser.add_argument("--stop-after", default=0, type=int,
                        help="Stop early without changing the scheduler horizon.")

    # Logging (WandB)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="monarch-muon", type=str)
    parser.add_argument("--wandb-group", default=None, type=str,
        help="WandB group. Auto-generated from model/data/iterations if not set.")

    # Schedule (WSD: linear warmup, constant, cosine decay)
    parser.add_argument("--iterations", default=15000, type=int)
    parser.add_argument("--warmup-steps", default=300, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--wsd-final-lr-scale", default=0.0, type=float)
    parser.add_argument("--wsd-fract-decay", default=0.1, type=float)

    # Optimisation
    parser.add_argument(
        "--opt",
        default="adamw",
        choices=["adamw", "galore_adamw", "muon", "monarch_muon", "triton_coat_adamw"],
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--eval-batch-size", default=32, type=int)
    parser.add_argument("--acc-steps", default=4, type=int)
    parser.add_argument("--weight-decay", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--grad-clip", default=1.0, type=float)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-7)

    # Dataset
    parser.add_argument("--datasets-dir", type=str, default="./datasets/")
    parser.add_argument("--vocab-size", default=50304, type=int)
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--workers", type=int, default=8, help="Number of tokenizer threads")

    # Model
    parser.add_argument("--monarch-nblocks", type=int, choices=[2, 4], default=2)
    parser.add_argument("--apply-monarch", action="store_true")
    parser.add_argument("--init-std", default=0.02, type=float)
    parser.add_argument("--n-head", default=12, type=int)
    parser.add_argument("--n-layer", default=24, type=int)
    parser.add_argument("--sequence-length", default=1024, type=int)
    parser.add_argument("--n-embd", default=768, type=int)
    parser.add_argument("--multiple-of", default=256, type=int)
    parser.add_argument("--rmsnorm-eps", default=1e-5, type=float)

    # ── FP8 training (COAT) ─────────────────────────────────────────────────
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="FP8 Monarch GEMMs inside the COAT FP8 activation flow.",
    )
    parser.add_argument(
        "--fp8-optim",
        action="store_true",
        help="Enable COAT FP8 optimizer states (E4M3, dynamic range expansion).",
    )

    # GaLore
    parser.add_argument("--update_gap", type=int, default=50)
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--inactive_update_rule", type=str, default="sign_sgd", choices=["no", "sgd", "sign_sgd"])
    parser.add_argument("--inactive_lr_scale", type=float, default=1.0)

    return parser.parse_args(args, namespace)
