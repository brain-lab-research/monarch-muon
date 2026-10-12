import unittest

import numpy as np
import torch

from projection import materialize as numpy_materialize
from projection import project_dense as numpy_project
from torch_projection import MonarchLinear, materialize, project_dense


class TorchProjectionTests(unittest.TestCase):
    def test_matches_numpy_oracle_and_forward(self):
        rng = np.random.default_rng(21)
        for out_features, in_features, blocks in ((16, 16, 2), (24, 16, 2),
                                                   (16, 24, 2), (24, 16, 4)):
            dense = rng.normal(size=(out_features, in_features))
            inputs = rng.normal(size=(2, 3, in_features))
            weight = torch.tensor(dense, dtype=torch.float64)
            first, second = project_dense(weight, blocks)
            reconstructed = materialize(first, second)
            numpy_reconstructed = numpy_materialize(*numpy_project(dense, blocks))
            np.testing.assert_allclose(reconstructed.numpy(), numpy_reconstructed,
                                       rtol=2e-5, atol=2e-5)

            layer = MonarchLinear(first, second)
            actual = layer(torch.tensor(inputs, dtype=torch.float32))
            expected = torch.nn.functional.linear(
                torch.tensor(inputs, dtype=torch.float32), reconstructed)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

            bf16_layer = MonarchLinear(first.bfloat16(), second.bfloat16())
            bf16_actual = bf16_layer(torch.tensor(inputs, dtype=torch.bfloat16))
            bf16_expected = torch.nn.functional.linear(
                torch.tensor(inputs, dtype=torch.bfloat16),
                materialize(first.float(), second.float()).bfloat16())
            torch.testing.assert_close(bf16_actual, bf16_expected,
                                       rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
