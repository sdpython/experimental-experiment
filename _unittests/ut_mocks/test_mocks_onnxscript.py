import contextlib
import unittest
from typing import Optional, Protocol, runtime_checkable
from onnx.reference import ReferenceEvaluator
from experimental_experiment.ext_test_case import ExtTestCase, with_path_append
from experimental_experiment.mocks import __mockpath__
from experimental_experiment.mocks.onnxscript.values import Opset
from experimental_experiment.mocks.onnxscript.onnx_opset import all_opsets
from experimental_experiment.mocks.onnxscript.onnx_function import OnnxFunction
from experimental_experiment.mocks.onnxscript.function_libs.torch_lib.graph_building import (
    TorchScriptTensor,
)
from experimental_experiment.mocks.onnxscript.function_libs.torch_lib._aten_functions import (
    aten_convolution,
)

try:
    import onnxscript  # noqa: F401

    path_to_add = None
except ImportError:
    path_to_add = __mockpath__


def return_module_cls():
    import torch
    from torch import nn

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 128, 5)

        def forward(self, x):
            return self.conv1(x)

    input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)
    return MyModel(), input_tensor


def return_module_cls_full():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 128, 5)
            self.conv2 = nn.Conv2d(128, 16, 5)
            self.fc1 = nn.Linear(13456, 1024)
            self.fc2 = nn.Linear(1024, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)
    return MyModel(), input_tensor


def export_utils(prefix, model, *args):
    import torch

    names = []
    name = f"{prefix}_script.onnx"
    torch.onnx.export(model, *args, name, input_names=["input"])
    names.append(name)

    name = f"{prefix}_dynamo.onnx"
    export_output = torch.onnx.dynamo_export(model, *args)
    export_output.save(name)
    names.append(name)
    return names


class TestMockExperimental(ExtTestCase):
    def test_install(self):
        self.assertIn(("", 18), all_opsets)
        self.assertIsInstance(all_opsets[("", 18)], Opset)

    @unittest.skipIf(path_to_add is None, reason="onnxscript available")
    def test_function(self):
        of = OnnxFunction(aten_convolution, domain="pkg.onnxscript.torch_lib")
        self.assertEqual(of.name, "aten_convolution")
        self.assertEqual(of.register_name, "aten::convolution")
        self.assertEqual(of.domain, "pkg.onnxscript.torch_lib")
        self.assertEqual(of.op_schema.name, "aten_convolution")
        param_schema = of.param_schemas()
        names = {p.name for p in param_schema if p.is_input}
        self.assertEqual(names, {"input", "weight", "bias"})
        names = {p.name for p in param_schema if not p.is_input}
        self.assertEqual(
            names,
            {"padding", "stride", "dilation", "groups", "transposed", "output_padding"},
        )

    def test_isinstance(self):
        import torch

        @runtime_checkable
        class TensorLike(Protocol):
            @property
            def dtype(self) -> Optional[torch.dtype]:
                ...

        t = TorchScriptTensor(torch.rand(1, 1, 128, 128), opset=18)
        self.assertNotEmpty(t.dtype)
        self.assertEqual(t.dtype, torch.float32)
        self.assertTrue(isinstance(t, TensorLike))

    @with_path_append(path_to_add)
    def test_match_function(self):
        import torch
        import torch.fx

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 128, 5)

            def forward(self, x):
                return self.conv1(x)

        m = MyModule()
        gm = torch.fx.symbolic_trace(m)
        # gm.graph.print_tabular()
        conv = None
        for node in gm.graph.nodes:
            if "conv" in str(node):
                conv = node
        self.assertNotEmpty(conv)

        from torch.onnx._internal.fx.onnxfunction_dispatcher import _OnnxSchemaChecker

        of = OnnxFunction(aten_convolution, domain="pkg.onnxscript.torch_lib")
        function_opschema = _OnnxSchemaChecker(of)
        args = (
            TorchScriptTensor(torch.rand(1, 1, 128, 128), opset=18),
            TorchScriptTensor(torch.rand(128, 1, 5, 5), opset=18),
            TorchScriptTensor(torch.rand(128), opset=18),
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        kwargs = {}

        class Diagnostic:
            def __init__(self):
                self.stdout = []

            def info(self, *args, **kwargs):
                self.stdout.append((args, kwargs))
                print("INFO", args, kwargs)

            @contextlib.contextmanager
            def log_section(self, level: int, message: str, *args, **kwargs):
                try:
                    yield
                finally:
                    pass

        match = function_opschema.perfect_match_inputs(Diagnostic(), args, kwargs)
        self.assertTrue(match)

    @unittest.skip("too many things to implement")
    @with_path_append(path_to_add)
    def test_mock_dynamo_export(self):
        model, input_tensor = return_module_cls()
        names = export_utils("test_dynamo_export", model, input_tensor)
        x = input_tensor.numpy()
        results = []
        for name in names:
            ref = ReferenceEvaluator(name)
            results.append(ref.run(None, {"input": x})[0])
        self.assertEqualArray(results[0], results[1])

    @unittest.skip("too many things to implement")
    @with_path_append(path_to_add)
    def test_mock_dynamo_export_full(self):
        model, input_tensor = return_module_cls_full()
        names = export_utils("test_dynamo_export_full", model, input_tensor)
        x = input_tensor.numpy()
        results = []
        for name in names:
            ref = ReferenceEvaluator(name)
            results.append(ref.run(None, {"input": x})[0])
        self.assertEqualArray(results[0], results[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
