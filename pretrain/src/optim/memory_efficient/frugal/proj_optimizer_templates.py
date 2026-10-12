import torch
from torch import nn
from torch.optim import Optimizer
from typing import Callable, List, Dict, Iterable
from abc import abstractmethod

from .galore_projector import GaLoreProjector


class ProjOptimizer(Optimizer):

    def __init__(
        self,

        params: Iterable[nn.parameter.Parameter],

        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True, 
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,

    ):
        assert isinstance(params, List), "Pass parameter groups; proj groups are flagged with 'is_proj_params'."
        assert 0.0 <= density <= 1.0
        assert proj_params_lr_scale >= 0
        assert inactive_lr_scale >= 0

        
        proj_params_args_dict = {'density': density, 'update_gap': update_gap, 'proj_params_lr_scale': proj_params_lr_scale, 'reset_statistics': reset_statistics, 
                                 'inactive_lr_scale': inactive_lr_scale, 'inactive_update_rule': inactive_update_rule, '_example_state_init': _example_state_init}
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
    

class GaloreOptimizer(ProjOptimizer):
    def __init__(
        self,

        params: Iterable[nn.parameter.Parameter],

        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True,
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,
    ):
        params = super().__init__(
            params=params,
            proj_params_lr_scale=proj_params_lr_scale,
            update_gap=update_gap,
            density=density,
            reset_statistics=reset_statistics,
            inactive_update_rule=inactive_update_rule,
            inactive_lr_scale=inactive_lr_scale,
            _example_state_init=_example_state_init,
        )
        return params
    

    @torch.no_grad()
    def _update_states(self, group):
        for p in group["params"]:
            state = self.state[p]
            grad = p.grad
            if "projector" not in state:
                state["projector"] = GaLoreProjector(group["density"], grad_shape=grad.shape)
            state["projector"].update_proj(grad)
            if "step" not in state or group["reset_statistics"]:
                # reset
                if len(state) == 1 or group["_example_state_init"]:
                    self._init_state(example=state["projector"].project_down(grad), state=state)
                else:
                    self._init_state(state=state)

    
    @torch.no_grad()
    def _proj_params_update(self, grad, state, group):
        grad_down = state["projector"].project_down(grad)
        active_lr = group["lr"] * group["proj_params_lr_scale"]
        update = self._compute_update(grad_down, state, **{**group, 'lr': active_lr})
        update = state["projector"].project_up(update)
        if group["inactive_update_rule"] == "no":
            return update
        inactive_grad = grad - state["projector"].project_up(grad_down)
        inactive_lr = group["lr"] * group["proj_params_lr_scale"] * group["inactive_lr_scale"]
        if group["inactive_update_rule"] == "sgd":
            update.add_(-inactive_lr * inactive_grad)
        elif group["inactive_update_rule"] == "sign_sgd":
            update.add_(-inactive_lr * inactive_grad.sign())
        return update
