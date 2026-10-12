import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["convert", "train"])
    parser.add_argument("--arm", choices=["dense", "A", "B", "C", "D"])
    parser.add_argument("--optimizer", choices=["adamw", "muon"])
    parser.add_argument("--root", type=Path, required=True, help="Output directory of the campaign.")
    parser.add_argument("--data", type=Path, required=True, help="sft_data.py output built with the calibration splits.")
    parser.add_argument("--model-dir", required=True, help="Local Qwen2.5-7B directory.")
    args = parser.parse_args()
    if args.mode == "train" and (args.arm is None or args.optimizer is None):
        parser.error("train requires --arm and --optimizer")
    args.root.mkdir(parents=True, exist_ok=True)
    if args.mode == "convert":
        from conversion_initialization import convert
        convert(args.root, args.data, args.model_dir)
    else:
        from conversion_training import train
        train(args.root, args.data, args.model_dir, args.arm, args.optimizer)


if __name__ == "__main__":
    main()
