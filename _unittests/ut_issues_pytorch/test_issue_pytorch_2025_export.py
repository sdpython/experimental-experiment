import unittest
import onnx
from experimental_experiment.ext_test_case import ExtTestCase, requires_torch
from experimental_experiment.torch_interpreter import to_onnx


class TestIssuesPytorch2025Export(ExtTestCase):

    @requires_torch("2.6")
    def test_pads_with_constant_1(self):
        import torch

        def dummy_function(idx, x_len):
            # [1, 2, 3] becomes [1, 2, 3, x_len]
            return torch.nn.functional.pad(idx, (0, 1), value=x_len)

        class Model(torch.nn.Module):
            def forward(self, x, y):
                padded = dummy_function(x, y.shape[0])
                return torch.arange(padded.max())

        model = Model()
        inputs = (
            (torch.arange(3) + 1).to(torch.int64),
            torch.tensor([0, 5], dtype=torch.int64),
        )
        expected = model(*inputs)

        AUTO = torch.export.Dim.AUTO
        ep = torch.export.export(
            model, inputs, dynamic_shapes={"x": {0: AUTO}, "y": {0: AUTO}}
        )

        epo = torch.onnx.export(ep, dynamo=True)
        epo.optimize()
        epo.save("test_pads_with_constant_1.onnx")
        onx = to_onnx(ep)
        onnx.save(onx, "test_pads_with_constant_1.custom.onnx")
        onnx.checker.check_model(onx)

    def test_pads_with_constant_2(self):
        import torch

        def dummy_function(idx, x_len):
            # [1, 2, 3] becomes [1, 2, 3, x_len]
            return torch.cat(
                [idx, torch.tensor([x_len], dtype=torch.int64)],
                dim=0,
            )

        class Model(torch.nn.Module):
            def forward(self, x, y):
                padded = dummy_function(x, y.shape[0])
                return torch.arange(padded.max())

        model = Model()
        inputs = (
            (torch.arange(3) + 1).to(torch.int64),
            torch.tensor([0, 5], dtype=torch.int64),
        )
        expected = model(*inputs)

        AUTO = torch.export.Dim.AUTO
        ep = torch.export.export(
            model, inputs, dynamic_shapes={"x": {0: AUTO}, "y": {0: AUTO}}
        )

        epo = torch.onnx.export(ep, dynamo=True)
        epo.optimize()
        epo.save("test_pads_with_constant_2.onnx")
        onx = to_onnx(ep)
        onnx.save(onx, "test_pads_with_constant_2.custom.onnx")
        onnx.checker.check_model(onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
