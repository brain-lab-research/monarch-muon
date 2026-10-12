import math
import torch
from itertools import chain
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Generator, List, Tuple, Dict
from . import newton_schulz_funcs

from .opt_utils import (
    AsyncTask,
    AsyncRuntime,
    to_local,
    create_param_batches,
    pad_batch,
)
from .scalar_opts import adamw_update_foreach


class Muon(Optimizer):
    """
    Single-GPU Muon optimizer.

    Muon groups use Nesterov momentum, 5 Newton-Schulz steps with Keller
    Jordan's coefficients in BF16, and a learning rate scaled by
    0.2 * sqrt(max(A, B)) for RMS-norm compatibility with AdamW. Groups with
    algorithm="adamw" use AdamW at the unscaled learning rate.

    Args:
        params: Parameters for the optimizer.
        lr: Base learning rate. For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        mu: Momentum factor for Muon algorithm.
        betas: Tuple of (beta1, beta2) for AdamW.
        weight_decay: Weight decay factor.
        epsilon: Small value to avoid division by zero.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")

        # Default arguments for each param group
        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            algorithm="muon",
            step=0,
            epsilon=epsilon,
        )
        super().__init__(params, defaults)

        self._newton_schulz_func = newton_schulz_funcs.zeropower_via_newtonschulz5_jordan

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "muon":
                muon_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        muon_tasks = self._create_muon_tasks(muon_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(muon_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=4)
        runtime.run()

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        if not state:
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
            state["momentum"] = torch.zeros_like(param)
        return state

    def _create_muon_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "muon",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of Muon matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Muon optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Wrap hyperparameters in tensors for torch.compile
            hyperparams = {
                'lr': torch.tensor(group["lr"]),
                'mu': torch.tensor(group["mu"]),
                'weight_decay': torch.tensor(group["weight_decay"]),
                'epsilon': torch.tensor(group["epsilon"]),
            }

            # One matrix per batch on a single GPU
            for params in create_param_batches(group_params, batch_size=1):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, algo_name) for p in params]

                yield AsyncTask(
                    muon_update_batch_async(
                        X=pad_batch(params, 1),
                        G=pad_batch(gradients, 1),
                        states=states,
                        hyperparams=hyperparams,
                        newton_schulz_func=self._newton_schulz_func,
                    )
                )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                )
            )


def muon_update_batch_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    states: List[dict],
    hyperparams: Dict,  # All hyperparameters
    newton_schulz_func,
) -> Generator[None, None, None]:
    """
    Muon update for a batch holding a single matrix.
    """

    assert len(X) == len(G)
    assert len(X) == len(states)
    assert len(X) == 1

    lr = hyperparams["lr"]
    weight_decay = hyperparams["weight_decay"]
    epsilon = hyperparams["epsilon"]

    # Update momentum and compute the input for orthogonalization
    U = muon_update_pre_orthogonalize(
        G=to_local(G),
        states=states,
        hyperparams=hyperparams,
    )

    single_matrix = U[0]
    U = [newton_schulz_func(single_matrix, epsilon=epsilon).reshape(single_matrix.shape)]

    # Compute scaled learning rate
    adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape)

    # Update model parameters with orthogonalized output
    muon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )
    # Keeps this a generator, as AsyncTask expects.
    yield


def adamw_update_foreach_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
) -> Generator[None, None, None]:
    """
    Async wrapper around foreach AdamW update.
    """
    adamw_update_foreach(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)
    yield


# @torch.compile(fullgraph=True)
def muon_update_pre_orthogonalize(
    G: List[Tensor],
    states: List[dict],
    hyperparams: Dict,
) -> List[Tensor]:
    """
    Update momentum with gradient and compute the input to orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """

    momentum = hyperparams['mu']

    M = [s["momentum"] for s in states]
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Update momentum with new gradient
    torch._foreach_mul_(M, momentum)
    torch._foreach_add_(M, G)

    # Nesterov momentum
    U = torch._foreach_mul(M, momentum)
    torch._foreach_add_(U, G)

    # Convert to bfloat16 before communication
    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U


@torch.compile(fullgraph=True)
def muon_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
):
    """
    Apply weight decay and weight update after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    # Apply weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Weight update
    U = torch._foreach_mul(U, adjusted_lr)
    torch._foreach_sub_(X, U)


def adjust_lr_rms_norm(lr, param_shape):
    # Adjust learning rate for constant element-wise RMS norm
    A, B = param_shape[:2]
    adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr
