import unittest
from experimental_experiment.ext_test_case import ExtTestCase


class TestTorchRewritings(ExtTestCase):
    def test_diagnoal_mask(self):
        import torch

        def diagonal_mask(cu_len: torch.Tensor) -> torch.Tensor:
            mx = cu_len.max()
            indices = torch.arange(mx, dtype=cu_len.dtype, device=cu_len.device)
            dot = (cu_len.unsqueeze(1) <= indices.unsqueeze(0)).to(torch.int32)
            dot = dot.sum(dim=0)
            mask = dot.unsqueeze(1) @ dot.unsqueeze(0)
            mask = mask == dot**2
            return mask.to(torch.int32)

        cu_len = torch.tensor([0, 2, 3, 7, 8], dtype=torch.int64)
        mask = diagonal_mask(cu_len)
        expected = torch.tensor(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=torch.int32,
        )
        self.assertEqualArray(expected, mask)

        cu_len = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64)
        mask = diagonal_mask(cu_len)
        expected = torch.diag(torch.ones((9,))).to(torch.int32)
        self.assertEqualArray(expected, mask)


if __name__ == "__main__":
    unittest.main(verbosity=2)
