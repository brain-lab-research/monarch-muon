import hashlib
import json
import time
from pathlib import Path

from safetensors.torch import load_model
import torch
from transformers import AutoModelForCausalLM

from torch_projection import MonarchLinear, materialize, project_dense


TARGETS = {"self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
           "mlp": ("gate_proj", "up_proj", "down_proj")}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_projections(model, blocks, project):
    records = []
    for index, layer in enumerate(model.model.layers):
        for parent_name, names in TARGETS.items():
            parent = getattr(layer, parent_name)
            for name in names:
                dense = getattr(parent, name)
                m, n = dense.weight.shape
                if min(m, n) % (blocks * blocks):
                    raise ValueError("This experiment requires uniform integer block ranks")
                q = min(m, n) // blocks
                if project:
                    with torch.no_grad():
                        first, second = project_dense(dense.weight, blocks)
                        error = ((materialize(first, second) - dense.weight.float()).norm()
                                 / dense.weight.float().norm()).item()
                else:
                    first = dense.weight.new_empty((blocks, q, n // blocks))
                    second = dense.weight.new_empty((blocks, m // blocks, q))
                    error = None
                structured = MonarchLinear(first.to(dense.weight.dtype),
                                           second.to(dense.weight.dtype), dense.bias)
                setattr(parent, name, structured)
                records.append({"name": f"model.layers.{index}.{parent_name}.{name}",
                                "shape": [m, n], "rank": q // blocks,
                                "factor_shapes": [list(first.shape), list(second.shape)],
                                "relative_projection_error": error})
                if project and name == "down_proj":
                    print(f"MONARCH_PROJECTED_LAYER {index}", flush=True)
    return records


def load_base(model_dir, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16,
        device_map={"": str(device)}, attn_implementation="sdpa")
    model.config.use_cache = False
    return model


def save_checkpoint(model, path):
    path = Path(path)
    temporary = path.with_suffix(".pending")
    cpu_state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    from safetensors.torch import save_file
    save_file(cpu_state, str(temporary))
    temporary.replace(path)
    return sha256_file(path)


def prepare_projection(config, model_dir, path, device):
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.with_suffix(".json").read_text())
        if (manifest["sha256"] != sha256_file(path)
                or (manifest["config"] | {"gpus": config["gpus"]}) != config):
            raise ValueError("Existing initialization does not match the experiment")
        return
    started = time.monotonic()
    model = load_base(model_dir, device)
    records = replace_projections(model, config["blocks"], project=True)
    manifest = {"config": config, "projections": records,
                "parameters": sum(p.numel() for p in model.parameters()),
                "sha256": save_checkpoint(model, path)}
    manifest["creation_seconds_load_project_save"] = time.monotonic() - started
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    del model
    torch.cuda.empty_cache()


def load_arm(config, model_dir, structure, projection_path, device):
    model = load_base(model_dir, device)
    if structure == "monarch":
        replace_projections(model, config["blocks"], project=False)
        load_model(model, str(projection_path), device=str(device), strict=True)
    return model
