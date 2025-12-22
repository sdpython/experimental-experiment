import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions, FunctionOptions


class TestOnnxExportTracing(ExtTestCase):
    def test_export_with_option_tracing(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return torch.relu(self.linear(x))

        model = Neuron(5, 3)
        x = torch.rand(2, 5)
        expected = model(x)
        onx = to_onnx(
            model,
            (x,),
            dynamic_shapes=({0: "batch"},),
            export_options=ExportOptions(tracing=True),
            verbose=0,
        )
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

    def test_export_with_option_tracing_2(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x, y):
                return torch.relu(self.linear(x)) + torch.relu(self.linear(y))

        model = Neuron(5, 3)
        x, y = torch.rand(2, 5), torch.rand(2, 5)
        expected = model(x, y)
        onx = to_onnx(
            model,
            (x, y),
            dynamic_shapes=({0: "batch"}, {0: "batch"}),
            export_options=ExportOptions(tracing=True),
            function_options=FunctionOptions(rename_allowed=True),
            verbose=0,
            as_function=True,
        )
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.numpy(), "y": y.numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
