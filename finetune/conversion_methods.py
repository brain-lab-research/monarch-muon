import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
import types

from torch_projection import MonarchLinear, materialize, project_dense


def first_hidden(x, first):
    blocks, q, p = first.shape
    batch = x.shape[0]
    h = torch.bmm(x.reshape(batch, blocks, p).transpose(0, 1), first.transpose(1, 2))
    return h.transpose(0, 1).reshape(batch, q, blocks).transpose(1, 2).transpose(0, 1)


def apply_factors(x, first, second):
    y = torch.bmm(first_hidden(x, first), second.transpose(1, 2))
    return y.permute(1, 2, 0).reshape(x.shape[0], -1)


def first_adjoint(y, x, second):
    blocks, s, q = second.shape
    batch = x.shape[0]
    g = torch.bmm(y.reshape(batch, s, blocks).permute(2, 0, 1), second)
    g = g.transpose(0, 1).transpose(1, 2).reshape(batch, blocks, q).transpose(0, 1)
    return torch.bmm(g.transpose(1, 2), x.reshape(batch, blocks, -1).transpose(0, 1))


@torch.no_grad()
def fit_als(weight, x, validation, first, second, sweeps=4, cg_steps=30, tolerance=1e-6):
    x, validation, weight = x.float(), validation.float(), weight.float()
    target, heldout = x @ weight.T, validation @ weight.T
    first, second = first.float().clone(), second.float().clone()
    records = []

    def record(stage):
        train = (apply_factors(x, first, second) - target).square().sum() / target.square().sum()
        val = (apply_factors(validation, first, second) - heldout).square().sum() / heldout.square().sum()
        if not torch.isfinite(train + val):
            raise RuntimeError("Nonfinite structured ALS objective")
        records.append({"stage": stage, "train_relative_squared_error": train.item(),
                        "validation_relative_squared_error": val.item(),
                        "first_norm": first.norm().item(), "second_norm": second.norm().item()})
        if len(records) > 1 and records[-1]["train_relative_squared_error"] > records[-2]["train_relative_squared_error"] + 1e-4:
            raise RuntimeError("Constrained ALS increased the calibration objective")
        return val.item()

    best_score = record("svd")
    best = first.clone(), second.clone()
    for sweep in range(sweeps):
        h = first_hidden(x, first)
        y = target.reshape(x.shape[0], second.shape[1], second.shape[0]).permute(2, 0, 1)
        driver = "gels" if x.is_cuda else "gelsd"
        second = torch.linalg.lstsq(h, y, driver=driver).solution.transpose(1, 2).contiguous()
        record(f"{sweep + 1}:L")
        rhs = first_adjoint(target, x, second)
        residual = rhs - first_adjoint(apply_factors(x, first, second), x, second)
        direction = residual.clone()
        rr = residual.square().sum()
        initial_rr = rr.clone()
        for _ in range(cg_steps):
            if rr <= tolerance**2 * initial_rr or rr == 0:
                break
            image = first_adjoint(apply_factors(x, direction, second), x, second)
            alpha = rr / (direction * image).sum()
            first.add_(direction * alpha)
            residual.sub_(image * alpha)
            next_rr = residual.square().sum()
            direction = residual + direction * (next_rr / rr)
            rr = next_rr
        score = record(f"{sweep + 1}:R")
        if score < best_score:
            best_score, best = score, (first.clone(), second.clone())
    return *best, records


class PermutedMonarchLinear(MonarchLinear):
    def __init__(self, first, second, input_order, output_order, bias=None):
        super().__init__(first, second, bias)
        self.register_buffer("input_order", input_order)
        self.register_buffer("output_inverse", torch.argsort(output_order))

    def forward(self, x):
        bias = self.bias
        # The inherited bias belongs to the original output coordinates.
        y = apply_factors(x.reshape(-1, self.in_features)[..., self.input_order],
                          self.blkdiag1, self.blkdiag2)
        y = y[..., self.output_inverse].reshape(*x.shape[:-1], self.out_features)
        return y + bias if bias is not None else y


@torch.no_grad()
def search_permutation(weight, blocks=2, steps=16):
    weight = weight.float()
    m, n = weight.shape
    on_input = n <= m
    size = min(m, n)
    scores = torch.eye(size, device=weight.device)
    order = torch.arange(size, device=weight.device)
    best_error, best_order = float("inf"), order.clone()
    history = []
    for iteration in range(steps + 1):
        layout = weight[:, order] if on_input else weight[order]
        first, second = project_dense(layout, blocks)
        approximate = materialize(first, second)
        error = (approximate - layout).square().sum().item()
        history.append({"iteration": iteration, "relative_squared_weight_error": error / weight.square().sum().item()})
        if error < best_error:
            best_error, best_order = error, order.clone()
        if iteration == steps:
            break
        restored = approximate[:, torch.argsort(order)] if on_input else approximate[torch.argsort(order)]
        residual = restored - weight
        gradient = approximate.T @ residual if on_input else approximate @ residual.T
        # A reproducible upper bound on the squared spectral norm in STEAM's step size.
        bound = restored.abs().sum(0).max() * restored.abs().sum(1).max()
        scores.sub_(gradient / (1.001 * bound))
        rows, cols = linear_sum_assignment(scores.cpu().numpy(), maximize=True)
        permutation = torch.empty(size, device=weight.device, dtype=torch.long)
        permutation[torch.as_tensor(rows, device=weight.device)] = torch.as_tensor(cols, device=weight.device)
        order = permutation
    input_order = best_order if on_input else torch.arange(n, device=weight.device)
    output_order = torch.arange(m, device=weight.device) if on_input else best_order
    return input_order, output_order, history


def cayley_step(rotation, gradient, step):
    tangent = gradient @ rotation.T - rotation @ gradient.T
    eye = torch.eye(rotation.shape[0], device=rotation.device, dtype=rotation.dtype)
    return torch.linalg.solve(eye + step * tangent / 2, (eye - step * tangent / 2) @ rotation)


def fit_procrustes(weights, inputs, validation, blocks=2, frobenius_sweeps=2,
                   outer_sweeps=2, rotation_steps=3, als_sweeps=4, cg_steps=30):
    weights = [w.float() for w in weights]
    inputs, validation = [x.float() for x in inputs], [x.float() for x in validation]
    width = weights[0].shape[1]
    qi = torch.eye(width, device=weights[0].device)
    qo = qi.clone()
    history = []

    def layouts():
        return [weights[0] @ qi, weights[1] @ qi, qo.T @ weights[2]]

    def approximations(factors):
        return [materialize(*pair).detach() for pair in factors]

    def objective(rotation, matrices, rows, incoming):
        if incoming:
            return sum(((rows[i] @ rotation @ matrices[i].T - rows[i] @ weights[i].T).square().sum()
                        / (rows[i] @ weights[i].T).square().sum()) for i in (0, 1))
        return ((rows[2] @ matrices[2].T @ rotation.T - rows[2] @ weights[2].T).square().sum()
                / (rows[2] @ weights[2].T).square().sum())

    with torch.no_grad():
        for _ in range(frobenius_sweeps):
            matrices = [materialize(*project_dense(w, blocks)) for w in layouts()]
            u, _, vh = torch.linalg.svd(weights[0].T @ matrices[0] + weights[1].T @ matrices[1])
            qi = u @ vh
            u, _, vh = torch.linalg.svd(weights[2] @ matrices[2].T)
            qo = u @ vh
    best_score, best = float("inf"), None
    for outer in range(outer_sweeps + 1):
        with torch.no_grad():
            rotated_inputs = [inputs[0] @ qi, inputs[1] @ qi, inputs[2]]
            rotated_validation = [validation[0] @ qi, validation[1] @ qi, validation[2]]
            factors, als_history = [], []
            for weight, x, val in zip(layouts(), rotated_inputs, rotated_validation):
                a, b, records = fit_als(weight, x, val, *project_dense(weight, blocks),
                                       sweeps=als_sweeps, cg_steps=cg_steps)
                factors.append((a, b))
                als_history.append(records)
            matrices = approximations(factors)
            score = (objective(qi, matrices, validation, True) + objective(qo, matrices, validation, False)).item()
            history.append({"outer": outer, "validation_sum_relative_squared_error": score,
                            "als": als_history})
            if score < best_score:
                best_score = score
                best = (qi.clone(), qo.clone(), [(a.clone(), b.clone()) for a, b in factors])
        if outer == outer_sweeps:
            break
        for incoming in (True, False):
            rotation = qi if incoming else qo
            for _ in range(rotation_steps):
                rotation = rotation.detach().requires_grad_()
                loss = objective(rotation, matrices, inputs, incoming)
                gradient, = torch.autograd.grad(loss, rotation)
                with torch.no_grad():
                    step = 1.0
                    for _ in range(8):
                        candidate = cayley_step(rotation, gradient, step)
                        if objective(candidate, matrices, inputs, incoming) <= loss:
                            rotation = candidate
                            break
                        step /= 2
                    rotation = rotation.detach()
            if incoming:
                qi = rotation
            else:
                qo = rotation
    return *best, history


class RotatedMLPBlock(nn.Module):
    def __init__(self, norm, mlp, input_rotation, output_rotation):
        super().__init__()
        self.norm, self.mlp = norm, mlp
        self.register_buffer("input_rotation", input_rotation)
        self.register_buffer("output_rotation", output_rotation)

    def forward(self, hidden):
        # The learned diagonal RMSNorm stays in the original coordinates and the
        # residual never leaves them, so only the MLP branch is rotated.
        dtype = hidden.dtype
        normalized = (self.norm(hidden).to(self.input_rotation.dtype) @ self.input_rotation).to(dtype)
        branch = self.mlp(normalized).to(self.output_rotation.dtype) @ self.output_rotation.T
        return hidden + branch.to(dtype)


def rotated_decoder_forward(self, hidden_states, attention_mask=None, position_ids=None,
                            past_key_value=None, output_attentions=False, use_cache=False,
                            cache_position=None, position_embeddings=None, **kwargs):
    residual = hidden_states
    hidden_states, attention = self.self_attn(
        hidden_states=self.input_layernorm(hidden_states), attention_mask=attention_mask,
        position_ids=position_ids, past_key_value=past_key_value,
        output_attentions=output_attentions, use_cache=use_cache,
        cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
    hidden_states = self.mlp_block(residual + hidden_states)
    return (hidden_states, attention) if output_attentions else (hidden_states,)


@torch.no_grad()
def install_rotated_mlp(layer, qi, qo, rotate_dense=True):
    if rotate_dense:
        for name in ("gate_proj", "up_proj"):
            dense = getattr(layer.mlp, name)
            dense.weight.copy_((dense.weight.to(qi.dtype) @ qi).to(dense.weight.dtype))
        dense = layer.mlp.down_proj
        dense.weight.copy_((qo.T @ dense.weight.to(qo.dtype)).to(dense.weight.dtype))
        if dense.bias is not None:
            dense.bias.copy_((dense.bias.to(qo.dtype) @ qo).to(dense.bias.dtype))
    layer.mlp_block = RotatedMLPBlock(layer.post_attention_layernorm, layer.mlp, qi, qo)
    del layer.post_attention_layernorm, layer.mlp
    layer.forward = types.MethodType(rotated_decoder_forward, layer)
