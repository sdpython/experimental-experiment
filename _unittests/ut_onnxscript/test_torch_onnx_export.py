import unittest
from onnx.reference import ReferenceEvaluator
import torch
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    requires_torch,
    ignore_warnings,
    hide_stdout,
)


class TestTorchOnnxExport(ExtTestCase):

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    @requires_torch("2.5")
    @hide_stdout()
    def test_oxs_linear_regression_dynamic_derived_batch(self):

        class TorchLinearRegression(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(TorchLinearRegression, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return self.linear(x)

        # static
        model = TorchLinearRegression(3, 1)
        x = torch.randn(11, 3, dtype=torch.float32)
        expected = model(x)
        self.assertEqual(expected.shape, (x.shape[0], 1))

        onx = torch.onnx.export(model, (x,), dynamo=True).model_proto
        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in onx.graph.input[0].type.tensor_type.shape.dim
        )
        self.assertEqual(shape, (11, 3))
        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in onx.graph.output[0].type.tensor_type.shape.dim
        )
        self.assertEqual(shape, (11, 1))

        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.detach().cpu().numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

        # dynamic
        model = TorchLinearRegression(3, 1)
        x = torch.randn(10, 3, dtype=torch.float32)
        expected = model(x)
        self.assertEqual(expected.shape, (x.shape[0], 1))

        dynamic_shapes = {"x": {0: torch.export.Dim("batch") * 2}}
        onx = torch.onnx.export(
            model,
            (x,),
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
        ).model_proto

        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in onx.graph.input[0].type.tensor_type.shape.dim
        )
        self.assertEqual(("2*s1", 3), shape)
        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in onx.graph.output[0].type.tensor_type.shape.dim
        )
        self.assertEqual(("2*s1", 1), shape)

        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.detach().cpu().numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
