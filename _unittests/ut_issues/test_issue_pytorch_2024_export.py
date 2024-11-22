import unittest
import numpy as np
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    skipif_ci_windows,
)


class TestIssuesPytorch2024Export(ExtTestCase):

    @skipif_ci_windows("not working")
    @requires_torch("2.7")
    def test_index_put_none_decompositions(self):
        # see issue https://github.com/pytorch/pytorch/issues/141336
        import torch

        class UpdateModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.params = torch.zeros((4, 4, 10))

            def forward(self, update, index1, index2):
                copy = self.params.clone()
                copy[index1, torch.tensor([1, 2], dtype=torch.int64), index2] = update
                return copy

        model = UpdateModel()

        update = (torch.arange(2) + 10).reshape((2,)).to(torch.float32)
        index1 = torch.from_numpy(np.array([1, 2])).to(torch.int64)
        index2 = torch.from_numpy(np.array([7, 8])).to(torch.int64)
        model(update, index1, index2)

        ep = torch.export.export(model, (update, index1, index2))
        # print(ep.graph)
        ep.run_decompositions()  # Fails here


if __name__ == "__main__":
    unittest.main(verbosity=2)
