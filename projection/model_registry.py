"""Matrix registry for DeepSeek-V3-style (MLA + MoE/SwiGLU) checkpoints.

Nothing here is hardcoded to Moonlight: the registry is built from the model's
config.json plus the checkpoint's tensor names and shapes (read from the
safetensors headers -- no weights are loaded). It produces

  * a per-tensor table (name, shape, layer, role, which unit covers it) so every
    assumption is visible and can be diffed against a different checkpoint;
  * a list of `Unit`s -- the (layer x matrix type) objects that are fitted and
    patched.

Matrix types, as the matrices physically exist in the checkpoint: attention
attn.{q_proj | q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj};
FFN (SwiGLU) ffn.{gate,up,down}_proj (dense layers, i.e. layers below
first_k_dense_replace), moe_shared.{...} (shared experts) and moe_routed.{...}
(all routed experts of the layer as ONE unit). The router (mlp.gate) is never a
target.

Whether a layer is dense or MoE is read off the tensor names (the state dict is
the ground truth); config.first_k_dense_replace is only cross-checked.
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

# ---------------------------------------------------------------------------
# References to concrete matrices inside a model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixRef:
    """One concrete 2-D matrix: a parameter of the model."""

    param_name: str

    def _param(self, model: torch.nn.Module) -> torch.Tensor:
        return model.get_parameter(self.param_name)

    def shape(self, model: torch.nn.Module) -> tuple[int, int]:
        return tuple(self._param(model).shape)

    def read(self, model: torch.nn.Module) -> torch.Tensor:
        """Detached copy of the matrix, in the parameter's dtype."""
        return self._param(model).detach().clone()

    @torch.no_grad()
    def write(self, model: torch.nn.Module, W: torch.Tensor) -> None:
        """In-place overwrite (values are cast to the parameter's dtype)."""
        p = self._param(model)
        p.copy_(W.to(device=p.device, dtype=p.dtype))


@dataclass(frozen=True)
class Unit:
    """One (layer x matrix type) unit; `refs` has several entries only for
    aggregated routed experts."""

    layer: int
    mtype: str
    group: str                 # attn | ffn | moe_shared | moe_routed
    refs: tuple[MatrixRef, ...]
    expert_ids: tuple[int | None, ...]
    shape: tuple[int, int]     # (d_out, d_in) of EVERY ref in the unit

    @property
    def uid(self) -> str:
        return f"L{self.layer}/{self.mtype}"

    @property
    def n_dense_params(self) -> int:
        return self.shape[0] * self.shape[1] * len(self.refs)


# ---------------------------------------------------------------------------
# Reading checkpoint metadata (no weights)
# ---------------------------------------------------------------------------


def read_safetensors_header(path: str | Path) -> dict[str, tuple[int, ...]]:
    """name -> shape from a .safetensors file's JSON header (8-byte length prefix)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}


def read_checkpoint_shapes(model_dir: str | Path) -> dict[str, tuple[int, ...]]:
    """Tensor name -> shape for a HF checkpoint directory (single or sharded
    safetensors). Only headers are read."""
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        paths = [model_dir / f for f in files]
    else:
        paths = sorted(model_dir.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(
            f"No *.safetensors files under {model_dir} (looked for model.safetensors.index.json "
            "and *.safetensors). The registry is built from safetensors headers."
        )
    shapes: dict[str, tuple[int, ...]] = {}
    for path in paths:
        shapes.update(read_safetensors_header(path))
    return shapes


def read_config(model_dir: str | Path) -> dict:
    return json.loads((Path(model_dir) / "config.json").read_text())


def shapes_from_model(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"^(?P<prefix>(?:\w+\.)*?layers)\.(?P<layer>\d+)\.(?P<rest>.+)$")
_ATTN_RE = re.compile(r"^self_attn\.(?P<leaf>q_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|k_proj|v_proj|o_proj)"
                      r"\.(?P<kind>weight|bias)$")
_FFN_RE = re.compile(r"^mlp\.(?P<leaf>gate_proj|up_proj|down_proj)\.(?P<kind>weight|bias)$")
_SHARED_RE = re.compile(r"^mlp\.shared_experts\.(?P<leaf>gate_proj|up_proj|down_proj)\.(?P<kind>weight|bias)$")
_ROUTED_RE = re.compile(r"^mlp\.experts\.(?P<expert>\d+)\.(?P<leaf>gate_proj|up_proj|down_proj)\.(?P<kind>weight|bias)$")
_ROUTER_RE = re.compile(r"^mlp\.gate\.(weight|e_score_correction_bias|bias)$")
_NORM_RE = re.compile(r"(layernorm|norm)\.weight$")

@dataclass
class Registry:
    units: list[Unit]
    table: list[dict]
    layer_prefix: str                      # e.g. "model.layers"
    n_layers: int
    dense_layers: list[int]
    moe_layers: list[int]
    warnings: list[str] = field(default_factory=list)

    def select(self, layers: Iterable[int] | None = None, groups: Sequence[str] | None = None) -> list[Unit]:
        """Units in the given layers and groups (None: all)."""
        layer_set = None if layers is None else set(layers)
        return [u for u in self.units
                if (layer_set is None or u.layer in layer_set) and (groups is None or u.group in groups)]

    def unit(self, layer: int, mtype: str) -> Unit:
        for u in self.units:
            if u.layer == layer and u.mtype == mtype:
                return u
        raise KeyError(f"no unit L{layer}/{mtype}")


def build_registry(config: Mapping, shapes: Mapping[str, Sequence[int]]) -> Registry:
    """Build the unit list and the per-tensor table from config.json (as a
    dict) and a {tensor name: shape} map. Pure function (no I/O, no weights)."""
    cfg = dict(config)
    warnings: list[str] = []
    n_cfg_layers = cfg.get("num_hidden_layers")

    # ---- classify every tensor -------------------------------------------------
    layer_prefix = None
    per_layer: dict[int, dict] = {}
    table: list[dict] = []

    def add_row(name, shape, layer, role, factorizable, note=""):
        row = dict(
            name=name, shape="x".join(map(str, shape)), numel=int(torch.Size(shape).numel()),
            layer="" if layer is None else layer, role=role,
            factorizable=factorizable, units="", note=note,
        )
        table.append(row)
        return row

    for name in sorted(shapes, key=lambda n: (_natural_key(n))):
        shape = tuple(shapes[name])
        m = _LAYER_RE.match(name)
        if m is None:
            if "embed" in name:
                add_row(name, shape, None, "embed", "no")
            elif name.endswith("lm_head.weight") or name.startswith("lm_head"):
                add_row(name, shape, None, "head", "no")
            elif _NORM_RE.search(name):
                add_row(name, shape, None, "norm", "no")
            else:
                add_row(name, shape, None, "other", "no", "not part of a decoder layer")
            continue

        layer, rest = int(m["layer"]), m["rest"]
        layer_prefix = layer_prefix or m["prefix"]
        if n_cfg_layers is not None and layer >= n_cfg_layers:
            add_row(name, shape, layer, "skipped", "no",
                    f"layer index >= num_hidden_layers ({n_cfg_layers}) -- e.g. a multi-token-prediction layer; ignored")
            continue
        info = per_layer.setdefault(layer, dict(attn={}, ffn={}, shared={}, routed={}))

        if (a := _ATTN_RE.match(rest)):
            if a["kind"] == "bias":
                add_row(name, shape, layer, "bias", "no", "bias vectors are never factorized")
                continue
            row = add_row(name, shape, layer, "attn", "yes")
            info["attn"][a["leaf"]] = (name, shape, row)
        elif (a := _FFN_RE.match(rest)):
            if a["kind"] == "bias":
                add_row(name, shape, layer, "bias", "no", "bias vectors are never factorized"); continue
            row = add_row(name, shape, layer, "ffn", "yes")
            info["ffn"][a["leaf"]] = (name, shape, row)
        elif (a := _SHARED_RE.match(rest)):
            if a["kind"] == "bias":
                add_row(name, shape, layer, "bias", "no", "bias vectors are never factorized"); continue
            row = add_row(name, shape, layer, "moe_shared", "yes")
            info["shared"][a["leaf"]] = (name, shape, row)
        elif (a := _ROUTED_RE.match(rest)):
            if a["kind"] == "bias":
                add_row(name, shape, layer, "bias", "no", "bias vectors are never factorized"); continue
            row = add_row(name, shape, layer, "moe_routed", "yes")
            info["routed"].setdefault(a["leaf"], {})[int(a["expert"])] = (name, shape, row)
        elif _ROUTER_RE.match(rest):
            add_row(name, shape, layer, "router", "no", "router is never factorized")
        elif _NORM_RE.search(rest):
            add_row(name, shape, layer, "norm", "no")
        else:
            add_row(name, shape, layer, "other", "no",
                    "UNRECOGNIZED tensor (e.g. a fused-expert layout); not covered by any unit")
            warnings.append(f"unrecognized tensor {name} {shape} -- not covered by any unit")

    if layer_prefix is None:
        raise ValueError("no tensors matching '*layers.<i>.*' found -- not a decoder-only layer-stacked checkpoint?")

    # ---- build units ------------------------------------------------------------
    units: list[Unit] = []

    def attach(unit: Unit, rows: Iterable[dict]):
        for row in rows:
            row["units"] = (row["units"] + ";" if row["units"] else "") + unit.uid

    for layer in sorted(per_layer):
        info = per_layer[layer]

        # attention
        for leaf, (name, shape, row) in info["attn"].items():
            if len(shape) != 2:
                raise ValueError(f"{name}: expected a 2-D weight, got shape {shape}")
            u = Unit(layer, f"attn.{leaf}", "attn", (MatrixRef(name),), (None,), tuple(shape))
            units.append(u); attach(u, [row])

        # dense / shared FFN
        for key, group, prefix in (("ffn", "ffn", "ffn"), ("shared", "moe_shared", "moe_shared")):
            for leaf, (name, shape, row) in info[key].items():
                u = Unit(layer, f"{prefix}.{leaf}", group, (MatrixRef(name),), (None,), tuple(shape))
                units.append(u); attach(u, [row])

        # routed experts: one aggregated unit per (layer, leaf)
        for leaf, experts in info["routed"].items():
            ids = sorted(experts)
            shp = {tuple(experts[e][1]) for e in ids}
            if len(shp) != 1:
                raise ValueError(f"layer {layer} routed {leaf}: experts have different shapes {shp}")
            expected = cfg.get("n_routed_experts")
            if expected is not None and len(ids) != expected:
                warnings.append(f"layer {layer} routed {leaf}: {len(ids)} experts in the state dict, config says {expected}")
            u = Unit(layer, f"moe_routed.{leaf}", "moe_routed",
                     tuple(MatrixRef(experts[e][0]) for e in ids), tuple(ids), next(iter(shp)))
            units.append(u); attach(u, [experts[e][2] for e in ids])

    dense_layers = sorted(l for l, i in per_layer.items() if not i["routed"])
    moe_layers = sorted(l for l, i in per_layer.items() if i["routed"])
    first_k = cfg.get("first_k_dense_replace")
    if first_k is not None and dense_layers != list(range(min(first_k, len(per_layer)))):
        warnings.append(
            f"layers without routed experts are {dense_layers}, but config.first_k_dense_replace={first_k}; "
            "using the state dict (the tensors actually present) as ground truth"
        )
    if n_cfg_layers is not None and len(per_layer) != n_cfg_layers:
        warnings.append(f"state dict has {len(per_layer)} decoder layers, config.num_hidden_layers={n_cfg_layers}")

    units.sort(key=lambda u: (u.layer, u.group != "attn", u.mtype))
    return Registry(
        units=units, table=table, layer_prefix=layer_prefix, n_layers=len(per_layer),
        dense_layers=dense_layers, moe_layers=moe_layers, warnings=warnings,
    )


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]
