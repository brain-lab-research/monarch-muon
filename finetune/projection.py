import numpy as np


def project_dense(weight, nblocks):
    out_features, in_features = weight.shape
    assert in_features % nblocks == 0
    assert out_features % nblocks == 0

    p = in_features // nblocks
    s = out_features // nblocks
    q = min(p, s)
    first = np.zeros((nblocks, q, p), dtype=weight.dtype)
    second = np.zeros((nblocks, s, q), dtype=weight.dtype)

    for input_block in range(nblocks):
        for output_block in range(nblocks):
            channels = np.arange(input_block * q, (input_block + 1) * q)
            channels = channels[channels % nblocks == output_block]
            block = weight[output_block::nblocks,
                           input_block * p:(input_block + 1) * p]
            left, values, right = np.linalg.svd(block, full_matrices=False)
            rank = len(channels)
            scales = np.sqrt(values[:rank])
            first[input_block, channels - input_block * q] = scales[:, None] * right[:rank]
            second[output_block, :, channels // nblocks] = (left[:, :rank] * scales).T

    return first, second


def materialize(first, second):
    nblocks, q, p = first.shape
    _, s, _ = second.shape
    weight = np.zeros((nblocks * s, nblocks * p), dtype=first.dtype)
    for input_block in range(nblocks):
        for output_block in range(nblocks):
            channels = np.arange(input_block * q, (input_block + 1) * q)
            channels = channels[channels % nblocks == output_block]
            for channel in channels:
                weight[output_block::nblocks,
                       input_block * p:(input_block + 1) * p] += np.outer(
                    second[output_block, :, channel // nblocks],
                    first[input_block, channel - input_block * q],
                )
    return weight
