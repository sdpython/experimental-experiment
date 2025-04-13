import unittest
from experimental_experiment.ext_test_case import ExtTestCase, requires_torch
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.reference import ExtendedReferenceEvaluator


class TestOnnxExportFolding(ExtTestCase):
    @requires_torch("2.6", "owning module is None before that")
    def test_submodule_local_functions_simple(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = torch.nn.Parameter(torch.rand(1, 4, 5))

            def forward(self, x):
                wt = torch.transpose(self.weights, 2, 1)
                return torch.sigmoid(x @ wt + 1e-5)

        model = Model()
        inputs = (torch.randn(2, 3, 5),)
        expected = model(*inputs)

        onx = to_onnx(model, inputs, optimize=True, verbose=0)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": inputs[0].numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)
        self.assertNotIn("Transpose", [n.op_type for n in onx.graph.node])

        onx = to_onnx(
            model,
            inputs,
            optimize=True,
            verbose=0,
            options=OptimizationOptions(constant_folding={"Transpose"}),
        )
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": inputs[0].numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)
        self.assertNotIn("Transpose", [n.op_type for n in onx.graph.node])

        onx = to_onnx(
            model,
            inputs,
            optimize=True,
            verbose=0,
            options=OptimizationOptions(constant_folding={"Reshape"}),
        )
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": inputs[0].numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)
        self.assertIn("Transpose", [n.op_type for n in onx.graph.node])


if __name__ == "__main__":
    unittest.main(verbosity=2)
