import unittest
from collections import Counter
from typing import Any, Dict, List
import onnx
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    skipif_ci_windows,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, Dispatcher


class TestCustomCode(ExtTestCase):

    @classmethod
    def get_custom_model(cls):
        import torch

        class Twice(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x + x

            @staticmethod
            def symbolic(g: torch.Graph, x):
                # The conversion is wrong on purpose, to check, this function was used.
                return g.op("Add", x, g.op("Add", x, x))

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return Twice.apply(x) * x

        return DummyModel, torch.rand(5, 3)

    def test_custom_code_script(self):
        import torch

        cls, x = self.get_custom_model()
        model = cls()
        expected = model(x)
        self.assertNotEmpty(expected)
        filename = "test_custom_code_script.onnx"
        torch.onnx.export(model, (x,), filename, input_names=["x"], opset_version=18)
        with open(filename, "rb") as f:
            onx = onnx.load(f)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"Add": 2, "Mul": 1})
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.detach().numpy()})[0]
        expected = (x + x + x) * x
        self.assertEqualArray(expected, got)

    @skipif_ci_windows("dynamo not working on Windows")
    def test_custom_code_export_1(self):
        import torch

        cls, x = self.get_custom_model()
        model = cls()
        exported_program = torch.export.export(model, (x,))
        graph = exported_program.graph
        targets = [node.target for node in graph.nodes]
        names = [(t if isinstance(t, str) else t.name()) for t in targets]
        self.assertEqual(names, ["x", "aten::add.Tensor", "aten::mul.Tensor", "output"])

        # exported_program = torch.export.export(model, (x,),
        # preserve_module_call_signature=("DummyModel",))

    @skipif_ci_windows("dynamo not working on Windows")
    @unittest.skip("autograd.function not working with dynamo")
    def test_custom_code_dynamo(self):
        import torch

        cls, x = self.get_custom_model()
        model = cls()
        expected = model(x)
        self.assertNotEmpty(expected)
        filename = "test_custom_code_dynamo.onnx"
        torch.onnx.export(
            model,
            (x,),
            filename,
            input_names=["x"],
            opset_version=18,
            dynamo=True,
            fallback=False,
        )
        with open(filename, "rb") as f:
            onx = onnx.load(f)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"Add": 2, "Mul": 1})
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.detach().numpy()})[0]
        expected = (x + x + x) * x
        self.assertEqualArray(expected, got)

    @classmethod
    def get_custom_model_export(cls):
        import torch

        @torch.library.custom_op("testlib::twice", mutates_args=())
        def twice(x: torch.Tensor) -> torch.Tensor:
            return x + x

        @twice.register_fake
        def twice_meta(x):
            return torch.empty_like(x)

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.testlib.twice(x) * x

        return DummyModel, torch.rand(5, 3)

    @skipif_ci_windows("dynamo not working on Windows")
    def test_custom_code_export_2(self):
        import torch

        cls, x = self.get_custom_model_export()
        model = cls()
        expected = model(x)
        self.assertNotEmpty(expected)
        exported_program = torch.export.export(model, (x,))
        graph = exported_program.graph
        targets = [node.target for node in graph.nodes]
        names = [(t if isinstance(t, str) else t.name()) for t in targets]
        self.assertEqual(names, ["x", "testlib::twice", "aten::mul.Tensor", "output"])

    @classmethod
    def get_custom_model_export_fn(cls):
        import torch

        def twice_fn(x: torch.Tensor) -> torch.Tensor:
            return x + x

        def twice_meta(x):
            return torch.empty_like(x)

        schema_str = torch.library.infer_schema(twice_fn, mutates_args=())
        namespace, opname = "testlib", "twice"
        twice = torch.library.CustomOpDef(namespace, opname, schema_str, twice_fn)
        twice.register_kernel(None)(twice_fn)
        twice._abstract_fn = twice_meta

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.testlib.twice(x) * x

        return DummyModel, torch.rand(5, 3)

    @skipif_ci_windows("dynamo not working on Windows")
    @requires_torch("2.5", "module 'torch.library' has no attribute 'infer_schema'")
    def test_custom_code_export_2_fn(self):
        import torch

        cls, x = self.get_custom_model_export_fn()
        model = cls()
        expected = model(x)
        self.assertNotEmpty(expected)
        exported_program = torch.export.export(model, (x,))
        graph = exported_program.graph
        targets = [node.target for node in graph.nodes]
        names = [(t if isinstance(t, str) else t.name()) for t in targets]
        self.assertEqual(names, ["x", "testlib::twice", "aten::mul.Tensor", "output"])

    @unittest.skip(reason="custom_ops fails with torch.script")
    def test_custom_code_script_2_fn(self):
        import torch

        cls, x = self.get_custom_model_export_fn()
        model = cls()
        expected = model(x)
        self.assertNotEmpty(expected)
        filename = "test_custom_code_script_2_fn.onnx"
        torch.onnx.export(
            model,
            (x,),
            filename,
            input_names=["x"],
            opset_version=18,
            dynamo=False,
            fallback=False,
        )
        with open(filename, "rb") as f:
            onx = onnx.load(f)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"Add": 2, "Mul": 1})
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.detach().numpy()})[0]
        expected = (x + x + x) * x
        self.assertEqualArray(expected, got)

    @classmethod
    def get_custom_model_export_fn_onnxscript(cls):
        import torch
        import onnxscript

        def twice_fn(x: torch.Tensor) -> torch.Tensor:
            return x + x

        def twice_meta(x):
            return torch.empty_like(x)

        schema_str = torch.library.infer_schema(twice_fn, mutates_args=())
        namespace, opname = "testlib", "twice"
        twice = torch.library.CustomOpDef(namespace, opname, schema_str, twice_fn)
        twice.register_kernel(None)(twice_fn)
        twice._abstract_fn = twice_meta

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.testlib.twice(x) * x

        def twice_onnx_impl(x):
            return x + x + x

        opset = onnxscript.values.Opset("testlib", 1)
        onnxscript_twice = onnxscript.values.TracedOnnxFunction(opset, twice_onnx_impl)

        return DummyModel, torch.rand(5, 3), onnxscript_twice

    @unittest.skip("No ONNX function found for <OpOverload(op='testlib.twice')>")
    def test_custom_code_dynamo_2_fn_registry(self):
        import torch

        cls, x, onnxscript_twice = self.get_custom_model_export_fn_onnxscript()
        model = cls()
        expected = model(x)
        self.assertNotEmpty(expected)
        filename = "test_custom_code_dynamo_2_fn_registry.onnx"
        registry = torch.onnx.OnnxRegistry()
        registry.register_op(
            function=onnxscript_twice, namespace="testlib", op_name="twice_onnx"
        )

        torch.onnx.export(
            model,
            (x,),
            filename,
            input_names=["x"],
            opset_version=18,
            dynamo=True,
            fallback=False,
            registry=registry,
        )
        with open(filename, "rb") as f:
            onx = onnx.load(f)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"Add": 2, "Mul": 1})
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.detach().numpy()})[0]
        expected = (x + x + x) * x
        self.assertEqualArray(expected, got)

    @requires_torch("2.5", "module 'torch.library' has no attribute 'infer_schema'")
    def test_custom_code_custom(self):
        cls, x = self.get_custom_model_export_fn()
        model = cls()
        expected = model(x)
        self.assertNotEmpty(expected)
        T = str

        def testlib_twice(
            g: "GraphBuilder",  # noqa: F821
            sts: Dict[str, Any],
            outputs: List[str],
            x: T,
            name: str = "testlib_twice",
        ) -> T:
            return g.op.Add(x, g.op.Add(x, x, name=name), name=name, outputs=outputs)

        dispatcher = Dispatcher({"testlib::twice": testlib_twice})

        onx = to_onnx(model, (x,), dispatcher=dispatcher)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"Add": 2, "Mul": 1})
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.detach().numpy()})[0]
        expected = (x + x + x) * x
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
