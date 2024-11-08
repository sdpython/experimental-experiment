import unittest
from onnx.checker import check_model
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    requires_torch,
)
from experimental_experiment.xbuilder import FunctionOptions
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_test_helper import dummy_llm


class TestOnnxExportSubModules(ExtTestCase):

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    def test_submodule_local_functions_simple(self):
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

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    def test_submodule_local_functions_double(self):
        import torch

        class SubNeuron2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                z2 = self.linear2(x)
                return torch.sigmoid(z1) + torch.sigmoid(z2)

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
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 2)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    def test_submodule_local_functions_two_outputs(self):
        import torch

        class SubNeuron2Outputs(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()

            def forward(self, x):
                return (
                    torch.sigmoid(x),
                    torch.sigmoid(x * x),
                )

        class Neuron2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.neuron = SubNeuron2Outputs(n_dims, n_targets)

            def forward(self, x):
                z, z1 = self.neuron(x)
                return torch.relu(z) + z1

        model = Neuron2()
        inputs = (torch.randn(1, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        onx = to_onnx(
            model, inputs, export_modules_as_functions=True, optimize=False, verbose=0
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 1)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    def test_dummy_llm(self):
        model, inputs = dummy_llm()
        onx = to_onnx(
            model, inputs, export_modules_as_functions=True, optimize=False, verbose=0
        )
        node_names = [n.op_type for n in onx.graph.node]
        self.assertEqual(
            node_names, ["<locals>.Embedding", "<locals>.DecoderLayer", "Identity"]
        )
        node_names = [n.op_type for n in onx.functions[1].node]
        self.assertEqual(node_names, ["Embedding", "Embedding", "Add", "Identity"])
        p_names = set(name for name, _ in model.named_parameters())
        init_names = set(i.name for i in onx.graph.initializer)
        self.assertEqual(len(p_names & init_names), 12)
        check_model(onx)

        # from experimental_experiment.helpers import pretty_onnx
        # print(pretty_onnx(onx))


if __name__ == "__main__":
    unittest.main(verbosity=2)
