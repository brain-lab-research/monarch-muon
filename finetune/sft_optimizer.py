import torch
from torch.optim._functional import adamw


def orthogonalize(momentum):
    x = momentum.to(torch.bfloat16)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    for _ in range(5):
        gram = x @ x.transpose(-2, -1)
        x = 3.4445 * x - 4.7750 * gram @ x + 2.0315 * (gram @ gram) @ x
    return x.transpose(-2, -1) if transposed else x


class SFTOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-5, momentum=0.95, betas=(0.9, 0.95),
                 eps=1e-8, weight_decay=0.1):
        super().__init__(params, dict(lr=lr, momentum=momentum, betas=betas,
                                     eps=eps, weight_decay=weight_decay, algorithm="adamw"))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["master"] = p.detach().float().clone()
                    state["momentum"] = torch.zeros_like(state["master"])
                    state["step"] = 0
                    if group["algorithm"] == "adamw":
                        state["variance"] = torch.zeros_like(state["master"])
                        state["adam_step"] = torch.tensor(0.0, device=p.device)
                master, buffer = state["master"], state["momentum"]
                grad = p.grad.float()
                if group["algorithm"] == "muon":
                    state["step"] += 1
                    # Monarch-Muon update: heavy-ball momentum, five NS steps, sqrt(rows / columns);
                    # 3-D Monarch factors are orthogonalized block by block.
                    buffer.mul_(group["momentum"]).add_(grad)
                    update = orthogonalize(buffer).float()
                    scale = (p.shape[-2] / p.shape[-1]) ** 0.5
                    master.add_(update, alpha=-group["lr"] * scale)
                else:
                    beta1, beta2 = group["betas"]
                    adamw([master], [grad], [buffer], [state["variance"]], [],
                          [state["adam_step"]], fused=p.is_cuda, foreach=False,
                          amsgrad=False, beta1=beta1, beta2=beta2, lr=group["lr"],
                          weight_decay=group["weight_decay"], eps=group["eps"], maximize=False)
                p.copy_(master)
        return loss


def parameter_groups(model, optimizer, lr, config):
    matrix, auxiliary = [], []
    for name, p in model.named_parameters():
        if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name:
            matrix.append(p)
        else:
            auxiliary.append(p)
    if optimizer == "adamw":
        return [{"params": matrix + auxiliary, "algorithm": "adamw", "lr": lr}]
    return [{"params": matrix, "algorithm": "muon", "lr": lr, "weight_decay": 0.0},
            {"params": auxiliary, "algorithm": "adamw", "lr": config["muon_adamw_lr"]}]
