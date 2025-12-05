import unittest
from experimental_experiment.ext_test_case import ExtTestCase, requires_torch, requires_cuda
from experimental_experiment.torch_interpreter import to_onnx


class TestOnnxExportDevice(ExtTestCase):
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

    def test_export_devices(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        x, y = torch.randn(2, 3), torch.randn(2, 3)
        ds = ({0: "batch"}, {0: "batch"})
        Model()(x=x, y=y)
        _onx, builder = to_onnx(Model(), (x, y), dynamic_shapes=ds, return_builder=True)
        self.assertNotEmpty(builder._known_devices)
        self.assertEqual(set(builder._known_devices.values()), {-1})

    @requires_cuda()
    def test_export_devices_cuda(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        x, y = torch.randn(2, 3).cuda(), torch.randn(2, 3).cuda()
        ds = ({0: "batch"}, {0: "batch"})
        Model()(x=x, y=y)
        _onx, builder = to_onnx(Model(), (x, y), dynamic_shapes=ds, return_builder=True)
        self.assertNotEmpty(builder._known_devices)
        self.assertEqual(set(builder._known_devices.values()), {0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
