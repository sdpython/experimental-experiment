import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_onnxscript,
    requires_torch,
)


class TestIssuePytorch_126856(ExtTestCase):

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_onnxscript("0.2")
    @requires_torch("2.6")
    def test_export_dynamo(self):
        import torch
        import onnxruntime as rt
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                mask = torch.ones((1, 3, 3), dtype=bool)
                x[mask] = 0
                return x

        model = Model()
        input_tensor = torch.randn((1, 3, 3))
        expected = model(input_tensor)
        onnx_program = torch.onnx.dynamo_export(model, input_tensor)
        session = rt.InferenceSession(
            onnx_program.model_proto.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        feeds = {
            onnx_program.model_proto.graph.input[0].name: input_tensor.cpu().numpy()
        }
        try:
            results = session.run(None, feeds)
        except ValueError as e:
            if "are missing from input feed" in str(e):
                raise unittest.SkipTest(f"bug in dynamo exporter: {e}")
        self.assertEqualArray(expected, results[0])

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_export_custom(self):
        import torch
        import onnxruntime as rt
        from torch import nn
        from experimental_experiment.torch_interpreter import to_onnx

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                mask = torch.ones((1, 3, 3), dtype=bool)
                x[mask] = 2
                return x

        model = Model()
        input_tensor = torch.randn((1, 3, 3))
        expected = model(input_tensor)
        onx = to_onnx(model, (input_tensor,), verbose=0, optimize=False)
        with open("debug.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        session = rt.InferenceSession(
            onx.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        results = session.run(None, {"x": input_tensor.cpu().numpy()})
        self.assertEqualArray(expected, results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
