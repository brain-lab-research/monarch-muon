import unittest

import numpy as np

from projection import materialize, project_dense


class ProjectionTests(unittest.TestCase):
    def test_materialized_weight_matches_butterfly_forward(self):
        rng = np.random.default_rng(2109)
        for out_features, in_features, blocks in ((16, 16, 2), (24, 16, 2),
                                                   (16, 24, 2), (24, 16, 4)):
            p = in_features // blocks
            s = out_features // blocks
            q = min(p, s)
            first = rng.normal(size=(blocks, q, p))
            second = rng.normal(size=(blocks, s, q))
            inputs = rng.normal(size=(3, in_features))

            hidden = inputs.reshape(3, blocks, p).transpose(1, 0, 2)
            hidden = hidden @ first.transpose(0, 2, 1)
            hidden = hidden.transpose(1, 0, 2).reshape(3, q, blocks)
            hidden = hidden.transpose(0, 2, 1).transpose(1, 0, 2)
            outputs = hidden @ second.transpose(0, 2, 1)
            outputs = outputs.transpose(1, 2, 0).reshape(3, out_features)

            np.testing.assert_allclose(
                inputs @ materialize(first, second).T, outputs,
                rtol=1e-12, atol=1e-12,
            )

    def test_projection_attains_block_rank_bound(self):
        rng = np.random.default_rng(7)
        for out_features, in_features, blocks in ((16, 16, 2), (24, 16, 2),
                                                   (16, 24, 2), (24, 16, 4)):
            weight = rng.normal(size=(out_features, in_features))
            fitted = materialize(*project_dense(weight, blocks))
            p = in_features // blocks
            q = min(p, out_features // blocks)
            minimum_error = 0.0
            for input_block in range(blocks):
                for output_block in range(blocks):
                    channels = np.arange(input_block * q, (input_block + 1) * q)
                    rank = np.count_nonzero(channels % blocks == output_block)
                    piece = weight[output_block::blocks,
                                   input_block * p:(input_block + 1) * p]
                    singular_values = np.linalg.svd(piece, compute_uv=False)
                    minimum_error += np.sum(singular_values[rank:] ** 2)

            np.testing.assert_allclose(
                np.linalg.norm(weight - fitted) ** 2, minimum_error,
                rtol=1e-12, atol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
