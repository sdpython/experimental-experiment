import unittest
import torch
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
)


class TestTorchExportExport(ExtTestCase):

    @requires_torch("2.7")
    def test_scaled_dot_product_attention(self):

        class DummyModel(torch.nn.Module):
            def __init__(self, enable_math: bool):
                super().__init__()
                self.enable_math = False

            def forward(self, query, key, value):
                res = torch.nn.functional.scaled_dot_product_attention(query, key, value)
                rest = res.transpose(0, 1)
                final = rest.view(8, 32, 128 * 64)
                return final

        model = DummyModel(False)
        device = "cpu"

        query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        expected = model(query, key, value)
        self.assertEqual(expected.dtype, torch.float16)
        self.assertEqual(expected.shape, (8, 32, 8192))

        cpl = torch.compile(model)
        new_output = cpl(query, key, value)
        self.assertEqual(new_output.dtype, torch.float16)
        self.assertEqual(new_output.shape, (8, 32, 8192))
        self.assertEqualArray(expected, new_output)

        export = torch.export.export(model, (query, key, value))
        # Fails here due to
        # Cannot view a tensor with shape torch.Size([8, 32, 128, 64]) and strides
        # (64, 512, 16384, 1) as a tensor with shape (8, 32, 8192)
        export.run_decompositions()


if __name__ == "__main__":
    unittest.main(verbosity=2)
