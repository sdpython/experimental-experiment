import unittest
from onnx.checker import check_model
from experimental_experiment.ext_test_case import ExtTestCase, skipif_ci_windows
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.reference import ExtendedReferenceEvaluator


class TestOnnxExportSubModules(ExtTestCase):

    @skipif_ci_windows("not available on windows")
    def test_submodule_local_functions(self):
        import torch

        class SubNeuron2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z = self.linear(x)
                return torch.sigmoid(z)

        class Neuron2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.neuron = SubNeuron2(n_dims, n_targets)

            def forward(self, x):
                z = self.neuron(x)
                return torch.relu(z)

        model = Neuron2()
        inputs = (torch.randn(1, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 2)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
