import unittest
from experimental_experiment.ext_test_case import ExtTestCase, requires_torch


class TestExportExport(ExtTestCase):
    @requires_torch("2.10.99")
    def test_export_dynamic_shapes_kwargs(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, **kwargs):
                return kwargs["x"] + kwargs["y"]

        x, y = torch.randn(2, 3), torch.randn(2, 3)
        Model()(x=x, y=y)
        ds = {
            "kwargs": {
                "x": {0: torch.export.Dim("batch")},
                "y": {0: torch.export.Dim("batch")},
            }
        }
        ep = torch.export.export(Model(), tuple(), kwargs={"x": x, "y": y}, dynamic_shapes=ds)
        self.assertNotEmpty(ep)


if __name__ == "__main__":
    unittest.main(verbosity=2)
