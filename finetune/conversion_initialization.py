import gc
import json
import resource
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from conversion_methods import (PermutedMonarchLinear, fit_als, fit_procrustes, install_rotated_mlp,
                                search_permutation)
from sft_data import read_split
from sft_model import load_base, sha256_file
from torch_projection import MonarchLinear, materialize, project_dense
from train_sft import batch_row


NAMES = ("gate_proj", "up_proj", "down_proj")
LAYERS = range(14, 27)


def install_initialization(model, arm, tensors):
    for index in LAYERS:
        layer = model.model.layers[index]
        if arm == "D":
            qi, qo = tensors[f"rotation.{index}.input"], tensors[f"rotation.{index}.output"]
            install_rotated_mlp(layer, qi, qo, rotate_dense=False)
            parent = layer.mlp_block.mlp
        else:
            parent = layer.mlp
        for name in NAMES:
            prefix = f"model.layers.{index}.mlp.{name}"
            first, second = tensors[prefix + ".blkdiag1"], tensors[prefix + ".blkdiag2"]
            if arm == "C":
                module = PermutedMonarchLinear(first, second, tensors[prefix + ".input_order"],
                                               tensors[prefix + ".output_order"])
            else:
                module = MonarchLinear(first, second)
            setattr(parent, name, module)


@torch.no_grad()
def collect_inputs(teacher, layer_index, rows, tokenizer, device, limit=256):
    layer = teacher.model.layers[layer_index]
    collected = {"gate": [], "down": []}
    positions = None

    def capture(name):
        def hook(module, arguments):
            collected[name].append(arguments[0][0, positions].detach().cpu())
        return hook

    hooks = [layer.mlp.gate_proj.register_forward_pre_hook(capture("gate")),
             layer.mlp.down_proj.register_forward_pre_hook(capture("down"))]
    sample_positions = []
    try:
        for row in rows:
            length = len(row["input_ids"])
            positions = torch.linspace(0, length - 1, min(limit, length), device=device).long()
            sample_positions.append({"conversation_sha256": row["conversation_sha256"],
                                     "positions": positions.cpu().tolist()})
            ids = torch.tensor([row["input_ids"]], device=device)
            teacher(ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    gate = torch.cat(collected["gate"]).to(device, torch.float32)
    down = torch.cat(collected["down"]).to(device, torch.float32)
    return [gate, gate, down], sample_positions


@torch.no_grad()
def model_metrics(teacher, student, rows, tokenizer, device):
    totals = {"teacher_nll_sum": 0.0, "student_nll_sum": 0.0, "kl_sum": 0.0, "tokens": 0,
              "logit_error_squared": 0.0, "teacher_logit_squared": 0.0}
    teacher.eval()
    student.eval()
    started = time.monotonic()
    for row in rows:
        batch = batch_row(row, len(row["input_ids"]), tokenizer.pad_token_id, device)
        reference = teacher(input_ids=batch["input_ids"], use_cache=False).logits[:, :-1]
        prediction = student(input_ids=batch["input_ids"], use_cache=False).logits[:, :-1]
        labels = batch["labels"][:, 1:]
        selected = (labels[0] != -100).nonzero().flatten()
        for positions in selected.split(32):
            t = reference[0, positions].float()
            s = prediction[0, positions].float()
            target = labels[0, positions]
            totals["teacher_nll_sum"] += torch.nn.functional.cross_entropy(t, target, reduction="sum").item()
            totals["student_nll_sum"] += torch.nn.functional.cross_entropy(s, target, reduction="sum").item()
            lt, ls = t.log_softmax(-1), s.log_softmax(-1)
            totals["kl_sum"] += (lt.exp() * (lt - ls)).sum().item()
            totals["logit_error_squared"] += (s - t).square().sum().item()
            totals["teacher_logit_squared"] += t.square().sum().item()
            totals["tokens"] += positions.numel()
        del reference, prediction
    return {"teacher_nll": totals["teacher_nll_sum"] / totals["tokens"],
            "student_nll": totals["student_nll_sum"] / totals["tokens"],
            "full_vocab_teacher_student_kl_supervised_tokens": totals["kl_sum"] / totals["tokens"],
            "relative_logit_rms_error": (totals["logit_error_squared"] / totals["teacher_logit_squared"])**0.5,
            "supervised_tokens": totals["tokens"], "evaluation_seconds": time.monotonic() - started}


def convert(root, data, model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_properties(device).total_memory < 75 * 1024**3:
        raise RuntimeError("Conversion requires a GPU with >=75 GiB of memory")
    teacher = load_base(model_dir, device).eval()
    calibration = read_split(data / "calibration.jsonl")
    heldout = read_split(data / "calibration_validation.jsonl")
    states = {arm: {} for arm in "ABCD"}
    reports = {arm: [] for arm in "ABCD"}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for index in LAYERS:
        layer_started = time.monotonic()
        inputs, positions = collect_inputs(teacher, index, calibration, tokenizer, device)
        validation, val_positions = collect_inputs(teacher, index, heldout, tokenizer, device)
        weights = [getattr(teacher.model.layers[index].mlp, name).weight.detach().float() for name in NAMES]
        candidates = {arm: [] for arm in "ABCD"}
        costs = {arm: 0.0 for arm in "ABCD"}
        detail = {arm: [] for arm in "ABCD"}
        for weight, x, val in zip(weights, inputs, validation):
            stage = time.monotonic()
            first, second = project_dense(weight, 2)
            candidates["A"].append((first, second)); detail["A"].append([])
            svd_seconds = time.monotonic() - stage
            costs["A"] += svd_seconds
            stage = time.monotonic()
            a, b, history = fit_als(weight, x, val, first, second)
            candidates["B"].append((a, b)); detail["B"].append(history)
            costs["B"] += svd_seconds + time.monotonic() - stage
            stage = time.monotonic()
            pi, po, search = search_permutation(weight)
            layout, shuffled_x, shuffled_val = weight[po][:, pi], x[:, pi], val[:, pi]
            a, b, history = fit_als(layout, shuffled_x, shuffled_val, *project_dense(layout, 2))
            candidates["C"].append((a, b, pi, po)); detail["C"].append({"search": search, "als": history})
            costs["C"] += time.monotonic() - stage
        stage = time.monotonic()
        qi, qo, factors, history = fit_procrustes(weights, inputs, validation)
        candidates["D"], detail["D"] = factors, history
        costs["D"] = time.monotonic() - stage
        for arm in "ABCD":
            if arm == "D":
                states[arm][f"rotation.{index}.input"] = qi.contiguous().cpu()
                states[arm][f"rotation.{index}.output"] = qo.contiguous().cpu()
            for name, weight, val, pair in zip(NAMES, weights, validation, candidates[arm]):
                prefix = f"model.layers.{index}.mlp.{name}"
                first, second = [p.bfloat16() for p in pair[:2]]
                states[arm][prefix + ".blkdiag1"], states[arm][prefix + ".blkdiag2"] = first.cpu(), second.cpu()
                effective = materialize(first.float(), second.float())
                if arm == "C":
                    pi, po = pair[2:]
                    states[arm][prefix + ".input_order"], states[arm][prefix + ".output_order"] = pi.cpu(), po.cpu()
                    effective = effective[torch.argsort(po)][:, torch.argsort(pi)]
                if arm == "D":
                    effective = effective @ qi.T if name != "down_proj" else qo @ effective
                with torch.no_grad():
                    target = torch.nn.functional.linear(val.bfloat16(), weight.bfloat16()).float()
                    if arm == "C":
                        structured = PermutedMonarchLinear(first, second, pi, po)
                    else:
                        structured = MonarchLinear(first, second)
                    structured_input = (val @ qi).bfloat16() if arm == "D" and name != "down_proj" else val.bfloat16()
                    prediction = structured(structured_input).float()
                    if arm == "D" and name == "down_proj":
                        prediction = prediction @ qo.T
                report = {"name": prefix, "shape": list(weight.shape),
                          "factor_shapes": [list(first.shape), list(second.shape)],
                          "stored_factor_parameters": first.numel() + second.numel(),
                          "effective_weight_relative_error": ((effective - weight).norm() / weight.norm()).item(),
                          "teacher_calibration_validation_output_relative_error": ((prediction - target).norm() / target.norm()).item()}
                reports[arm].append(report)
        progress = {"layer": index, "seconds": time.monotonic() - layer_started,
                    "arm_conversion_seconds": costs, "calibration_positions": positions,
                    "calibration_validation_positions": val_positions, "method_details": detail,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        (root / f"conversion-layer-{index}.json").write_text(json.dumps(progress, indent=2) + "\n")
        print("CONVERSION_LAYER_PASS=" + json.dumps({k: progress[k] for k in ("layer", "seconds", "arm_conversion_seconds", "peak_cuda_allocated_bytes")}), flush=True)
        del inputs, validation, weights, candidates, qi, qo, factors, effective, first, second, a, b, x, val, target
        del layout, shuffled_x, shuffled_val, weight, pair
        del structured, structured_input, prediction
        gc.collect(); torch.cuda.empty_cache()
    artifacts = {}
    for arm in "ABCD":
        path = root / f"initial-{arm.lower()}.safetensors"
        save_file(states[arm], str(path))
        sha = sha256_file(path)
        reloaded = load_file(str(path))
        if any(not torch.equal(states[arm][key], reloaded[key]) for key in states[arm]):
            raise RuntimeError("Initialization round-trip changed tensors")
        metadata = {"arm": arm, "sha256": sha,
                    "projections": reports[arm], "conversion_total_shared_wall_seconds": time.monotonic() - started,
                    "data_manifest_sha256": sha256_file(data / "manifest.json")}
        (root / f"initial-{arm.lower()}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        artifacts[arm] = metadata
        del reloaded
    del teacher
    gc.collect(); torch.cuda.empty_cache()
    measure(root, data, model_dir, tokenizer, device, states, artifacts)


def measure(root, data, model_dir, tokenizer, device, states, artifacts):
    heldout = read_split(data / "calibration_validation.jsonl")
    rows = read_split(data / "validation.jsonl")
    teacher = load_base(model_dir, device).float().eval()
    rotated = load_base(model_dir, device).float().eval()
    for index in LAYERS:
        qi = states["D"][f"rotation.{index}.input"].to(device)
        qo = states["D"][f"rotation.{index}.output"].to(device)
        install_rotated_mlp(rotated.model.layers[index], qi, qo)
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    parity = model_metrics(teacher, rotated, heldout, tokenizer, device)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    (root / "rotations-only-parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    # FP32 rounding of 13 pairs of 3584-wide rotations measured 2.4e-3 relative logit error and KL 3e-6; a wiring
    # error (a wrong transpose or coordinate system) gives KL of order one.
    if parity["relative_logit_rms_error"] > 1e-2 or parity["full_vocab_teacher_student_kl_supervised_tokens"] > 1e-4:
        raise RuntimeError("Full pretrained FP32 rotations-only parity did not pass")
    print("ROTATIONS_ONLY_PRETRAINED_PASS=" + json.dumps(parity), flush=True)
    del rotated, teacher
    gc.collect(); torch.cuda.empty_cache()
    teacher = load_base(model_dir, device).eval()
    for arm in "ABCD":
        student = load_base(model_dir, device).eval()
        tensors = {key: value.to(device) for key, value in states[arm].items()}
        install_initialization(student, arm, tensors)
        metrics = model_metrics(teacher, student, rows, tokenizer, device)
        metrics.update(arm=arm, trainable_parameters=sum(p.numel() for p in student.parameters()),
                       stored_parameter_elements=sum(p.numel() for p in student.state_dict().values()),
                       initial_sha256=artifacts[arm]["sha256"])
        (root / f"initial-metrics-{arm.lower()}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print("INITIAL_MODEL_PASS=" + json.dumps({k: metrics[k] for k in ("arm", "student_nll", "full_vocab_teacher_student_kl_supervised_tokens", "trainable_parameters")}), flush=True)
        if arm == "A":
            (root / "initial-metrics-dense.json").write_text(json.dumps({"arm": "dense", "student_nll": metrics["teacher_nll"],
                                                                       "supervised_tokens": metrics["supervised_tokens"]}, indent=2) + "\n")
        del student, tensors
        gc.collect(); torch.cuda.empty_cache()
    (root / "initialization-complete.json").write_text(json.dumps({"artifacts": artifacts}, indent=2) + "\n")
    print("INITIALIZATION_CAMPAIGN_PASS", flush=True)
