import torch
from torch import nn
from torch.optim import Optimizer
from typing import Callable, List, Dict, Iterable
from abc import abstractmethod


class ProjOptimizer(Optimizer):

    def __init__(
        self,

        params: Iterable[nn.parameter.Parameter],

        update_gap: int = 200,
        density=0.25,

    ):
        assert isinstance(params, List)
        assert 0.0 <= density <= 1.0

        proj_params_args_dict = {'density': density, 'update_gap': update_gap}
        for group in params:
            assert isinstance(group, Dict)
            if group.get("is_proj_params", False):
                for k, v in proj_params_args_dict.items():
                    group.setdefault(k, v)
                group["step"] = 0
        return params


    def is_proj_group(self, group):
        return "is_proj_params" in group and group["is_proj_params"]


    @torch.no_grad()
    @abstractmethod
    def _update_states(self, group):
        pass


    @torch.no_grad()
    @abstractmethod
    def _proj_params_update(self, grad, state, group):
        pass


    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if self.is_proj_group(group):
                if group["step"] % group["update_gap"] == 0:
                    self._update_states(group)
                group["step"] += 1
            for p in group["params"]:
                if p.grad is None:
                    continue
            
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ProjOptimizer does not support sparse gradients")
                
                state = self.state[p]
                
                if len(state) == 0:
                    self._init_state(example=p, state=state)

                p.mul_(1 - group["lr"] * group["weight_decay"])

                if not self.is_proj_group(group):
                    update = self._compute_update(grad, state, **group)
                else:
                    update = self._proj_params_update(grad, state, group)

                p.add_(update)

        return loss


class BlockOptimizer(ProjOptimizer):
    def __init__(
        self,

        params: Iterable[nn.parameter.Parameter],

        update_gap: int = 200,
        density=0.25,
    ):
        params = super().__init__(
            params=params,
            update_gap=update_gap,
            density=density,
        )
        for group in params:
            if self.is_proj_group(group):
                group["num_active_blocks"] = round(len(group["params"]) * group["density"])
        return params

    @torch.no_grad()
    def _activate_param(self, p):
        state = self.state[p]
        self._init_state(example=p, state=state)
        state["active"] = True

    @torch.no_grad()
    def _deactivate_param(self, p):
        state = self.state[p]
        state.clear()
        state["active"] = False

    @torch.no_grad()
    def _update_states(self, group):
        for p in group["params"]:
            # reset
            self._deactivate_param(p)
        current_blocks = torch.randperm(len(group["params"]))[:group["num_active_blocks"]]
        for idx in current_blocks:
            self._activate_param(group["params"][idx])

    @torch.no_grad()
    def _proj_params_update(self, grad, state, group):
        if state["active"]:
            return self._compute_update(grad, state, **group)
        return -group["lr"] * grad.sign()
