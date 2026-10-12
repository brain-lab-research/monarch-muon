"""Muon for the dense baseline (``--opt muon``).

Every 2-D weight of the transformer blocks gets Muon: an EMA momentum buffer with a Nesterov-style
look-ahead, six Newton-Schulz iterations and an update scaled by 0.2 * sqrt(max(rows, cols)).
The token embedding, the RMSNorm weights and lm_head get AdamW (eps added to sqrt(v) before bias
correction). Both use decoupled weight decay.
"""
import math

import torch

# Newton-Schulz coefficients, divided by 1.01, 1.01**3 and 1.01**5 for numerical safety.
NS_COEFFICIENTS = [
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for a, b, c in [
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    ]
]


@torch.compile(dynamic=False, fullgraph=True, backend="inductor")
def orthogonalize(G: torch.Tensor) -> torch.Tensor:
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-8)
    for a, b, c in NS_COEFFICIENTS:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, muon_params, adamw_params, *, lr, weight_decay, momentum=0.95,
                 adamw_betas=(0.9, 0.99), adamw_eps=1e-8):
        super().__init__(
            [{"params": list(muon_params), "use_muon": True},
             {"params": list(adamw_params), "use_muon": False}],
            dict(lr=lr, weight_decay=weight_decay, momentum=momentum, betas=adamw_betas, eps=adamw_eps),
        )

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if group["use_muon"]:
                    theta = group["momentum"]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(g)
                    state["momentum"] = state["momentum"] * theta + g * (1 - theta)
                    update = orthogonalize(state["momentum"] + g * (1 - theta) / theta)
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-0.2 * lr * math.sqrt(max(g.size(0), g.size(1))))
                else:
                    beta1, beta2 = group["betas"]
                    if "step" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(g)
                        state["moment2"] = torch.zeros_like(g)
                    state["step"] += 1
                    state["moment1"] = beta1 * state["moment1"] + (1 - beta1) * g
                    state["moment2"] = state["moment2"] * beta2 + (g ** 2) * (1 - beta2)
                    update = state["moment1"] / (state["moment2"].sqrt() + group["eps"])
                    scale = (1 - beta1 ** state["step"]) / (1 - beta2 ** state["step"]) ** 0.5
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr / scale)
