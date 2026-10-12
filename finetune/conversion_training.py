import gc
import json
from pathlib import Path
import time

from safetensors.torch import load_file, load_model
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from conversion_initialization import install_initialization
from sft_data import read_split
from sft_model import load_base, save_checkpoint, sha256_file
from train_sft import evaluate, initialize, train_trial


def load_initial(model_dir, arm, root, device):
    model = load_base(model_dir, device)
    if arm != "dense":
        path = root / f"initial-{arm.lower()}.safetensors"
        tensors = load_file(str(path), device=str(device))
        install_initialization(model, arm, tensors)
    return model


def train(root, data, model_dir, arm, algorithm):
    rank, world, device = initialize()
    if world != 2:
        raise RuntimeError("Expected one two-GPU job (torchrun --nproc_per_node=2)")
    if not (root / "initialization-complete.json").exists():
        raise RuntimeError("Run the convert step first: four-arm initialization correctness "
                           "and model metrics are required before SFT")
    config = json.loads(Path("sft_config.json").read_text())
    config["adamw_lrs"], config["muon_lrs"] = [3e-6, 1e-5, 3e-5], [1e-4, 3e-4, 1e-3]
    data_manifest = json.loads((data / "manifest.json").read_text())
    for split, record in data_manifest["splits"].items():
        if sha256_file(data / f"{split}.jsonl") != record["sha256"]:
            raise RuntimeError("Data checksum mismatch")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rows = read_split(data / "train.jsonl")
    validation = read_split(data / "validation.jsonl")
    output = root / f"sft-{arm.lower()}-{algorithm}"
    checkpoints = root / "checkpoints"
    if rank == 0:
        output.mkdir(exist_ok=False)
        checkpoints.mkdir(exist_ok=True)
    dist.barrier()
    best = None
    best_path = None
    initial = None if arm == "dense" else sha256_file(root / f"initial-{arm.lower()}.safetensors")
    for lr in config[f"{algorithm}_lrs"]:
        name = f"{arm.lower()}-{algorithm}-lr{lr:g}-g2"
        started = time.monotonic()
        model = load_initial(model_dir, arm, root, device)
        result = train_trial(model, rows, validation, config, tokenizer, algorithm, lr,
                             rank, world, device, output, name, False)
        result.update(arm=arm, initial_sha256=initial,
                      data_manifest_sha256=sha256_file(data / "manifest.json"), seed=config["seed"])
        if best is None or result["validation"]["nll"] < best["validation"]["nll"]:
            candidate_path = checkpoints / f"{name}.safetensors"
            if rank == 0:
                result["checkpoint_sha256"] = save_checkpoint(model, candidate_path)
                if best_path is not None:
                    if sha256_file(best_path) != best["checkpoint_sha256"]:
                        raise RuntimeError("Previous owned candidate checksum mismatch")
                    best_path.unlink()
            dist.barrier()
            result["checkpoint_sha256"] = sha256_file(candidate_path)
            best, best_path = result, candidate_path
        result["wall_seconds_load_train_validate_save"] = time.monotonic() - started
        if rank == 0:
            (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
            print("CONVERSION_SFT_TRIAL_PASS=" + json.dumps({k: result[k] for k in ("run", "baseline", "validation", "steps", "initial_sha256")}), flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()
        dist.barrier()
    model = load_initial(model_dir, arm, root, device)
    load_model(model, str(best_path), device=str(device), strict=True)
    test = read_split(data / "test.jsonl")
    best["test"] = evaluate(model, test, config, tokenizer, rank, world, device)
    if rank == 0:
        (output / "selected.json").write_text(json.dumps(best, indent=2) + "\n")
        print("CONVERSION_SELECTED_SFT_PASS=" + json.dumps({k: best[k] for k in ("arm", "optimizer", "lr", "validation", "test")}), flush=True)
    dist.barrier()
    dist.destroy_process_group()
