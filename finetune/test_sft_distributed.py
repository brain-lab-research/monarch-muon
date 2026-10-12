import copy
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel

from sft_optimizer import SFTOptimizer


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    for algorithm in ("adamw", "muon"):
        torch.manual_seed(29)
        parameters = [torch.nn.Parameter(torch.randn(*shape))
                      for shape in [(8, 16), (16, 8), (2, 8, 16), (2, 16, 8)]]
        references = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
        distributed = ZeroRedundancyOptimizer(
            [{"params": parameters, "algorithm": algorithm}],
            optimizer_class=SFTOptimizer, lr=1e-3)
        serial = SFTOptimizer([{"params": references, "algorithm": algorithm}], lr=1e-3)
        for step in range(4):
            for p, reference in zip(parameters, references):
                p.grad = torch.randn_like(p)
                reference.grad = p.grad.clone()
            distributed.step()
            serial.step()
            for p, reference in zip(parameters, references):
                torch.testing.assert_close(p, reference, rtol=0, atol=0)
        owned = sum(p.numel() for group in distributed.optim.param_groups for p in group["params"])
        assert owned < sum(p.numel() for p in parameters)
        if rank == 0:
            print(f"DISTRIBUTED_{algorithm.upper()}_PARITY=PASS", flush=True)

    torch.manual_seed(17)
    model = torch.nn.Linear(4, 3)
    reference = copy.deepcopy(model)
    ddp = DistributedDataParallel(model, gradient_as_bucket_view=True)
    x = torch.randn(5, 4)
    labels = torch.tensor([0, 1, 2, 0, 1])
    torch.nn.functional.cross_entropy(reference(x), labels).backward()
    local = torch.arange(5)[rank::world]
    loss = torch.nn.functional.cross_entropy(ddp(x[local]), labels[local], reduction="sum")
    (loss * world / len(x)).backward()
    for p, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, expected.grad, atol=1e-7, rtol=1e-6)
    if rank == 0:
        print("UNEQUAL_TARGET_TOKEN_WEIGHTING=PASS", flush=True)

    for algorithm in ("adamw", "muon"):
        torch.manual_seed(31)
        model = torch.nn.Linear(4, 3, bias=False)
        reference = copy.deepcopy(model)
        ddp = DistributedDataParallel(model, gradient_as_bucket_view=True)
        optimizer = ZeroRedundancyOptimizer(
            [{"params": list(model.parameters()), "algorithm": algorithm}],
            optimizer_class=SFTOptimizer, lr=1e-3)
        serial = SFTOptimizer([{"params": list(reference.parameters()), "algorithm": algorithm}], lr=1e-3)
        x, target = torch.randn(12, 4), torch.randn(12, 3)
        with torch.random.fork_rng():
            torch.nn.functional.mse_loss(ddp(x[rank:rank + 1]), target[rank:rank + 1]).backward()
        optimizer.zero_grad(set_to_none=True)
        for step in range(3):
            pointers = [p.grad.data_ptr() if p.grad is not None else None for p in model.parameters()]
            optimizer.zero_grad(set_to_none=False)
            if step:
                assert pointers == [p.grad.data_ptr() for p in model.parameters()]
            serial.zero_grad(set_to_none=False)
            torch.nn.functional.mse_loss(reference(x), target).backward()
            chunks = torch.arange(12)[rank::world].chunk(2)
            for micro, indices in enumerate(chunks):
                with ddp.no_sync() if micro == 0 else nullcontext():
                    loss = torch.nn.functional.mse_loss(ddp(x[indices]), target[indices], reduction="sum")
                    (loss * world / target.numel()).backward()
            assert all(p.grad._base is not None for p in model.parameters())
            optimizer.step()
            serial.step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        if rank == 0:
            print(f"ACCUMULATION_BUCKET_VIEWS_{algorithm.upper()}=PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
