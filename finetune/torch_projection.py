import torch
from torch import nn


def project_dense(weight, nblocks):
    out_features, in_features = weight.shape
    if in_features % nblocks or out_features % nblocks:
        raise ValueError("Both dimensions must be divisible by nblocks")

    p = in_features // nblocks
    s = out_features // nblocks
    q = min(p, s)
    first = torch.zeros((nblocks, q, p), device=weight.device, dtype=torch.float32)
    second = torch.zeros((nblocks, s, q), device=weight.device, dtype=torch.float32)

    for input_block in range(nblocks):
        channels = torch.arange(input_block * q, (input_block + 1) * q,
                                device=weight.device)
        for output_block in range(nblocks):
            selected = channels[channels % nblocks == output_block]
            piece = weight[output_block::nblocks,
                           input_block * p:(input_block + 1) * p].float().contiguous()
            left, values, right = torch.linalg.svd(piece, full_matrices=False)
            rank = selected.numel()
            scales = values[:rank].sqrt()
            first[input_block].index_copy_(0, selected - input_block * q,
                                           scales[:, None] * right[:rank])
            second[output_block].index_copy_(1, selected // nblocks,
                                             left[:, :rank] * scales)

    return first, second


def materialize(first, second):
    nblocks, q, p = first.shape
    _, s, _ = second.shape
    weight = torch.empty((nblocks * s, nblocks * p),
                         device=first.device, dtype=first.dtype)
    for input_block in range(nblocks):
        channels = torch.arange(input_block * q, (input_block + 1) * q,
                                device=first.device)
        for output_block in range(nblocks):
            selected = channels[channels % nblocks == output_block]
            weight[output_block::nblocks,
                   input_block * p:(input_block + 1) * p] = (
                second[output_block].index_select(1, selected // nblocks)
                @ first[input_block].index_select(0, selected - input_block * q)
            )
    return weight


class MonarchLinear(nn.Module):
    def __init__(self, first, second, bias=None):
        super().__init__()
        self.blkdiag1 = nn.Parameter(first.contiguous())
        self.blkdiag2 = nn.Parameter(second.contiguous())
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None
        self.in_features = first.shape[0] * first.shape[2]
        self.out_features = second.shape[0] * second.shape[1]

    def forward(self, x):
        batch_shape = x.shape[:-1]
        batch = x.numel() // self.in_features
        nblocks, q, p = self.blkdiag1.shape
        _, s, _ = self.blkdiag2.shape
        hidden = torch.bmm(x.reshape(batch, nblocks, p).transpose(0, 1),
                           self.blkdiag1.transpose(1, 2))
        hidden = hidden.transpose(0, 1).reshape(batch, q, nblocks)
        hidden = hidden.transpose(1, 2).contiguous().transpose(0, 1)
        output = torch.bmm(hidden, self.blkdiag2.transpose(1, 2))
        output = output.permute(1, 2, 0).reshape(*batch_shape, self.out_features)
        return output + self.bias if self.bias is not None else output
