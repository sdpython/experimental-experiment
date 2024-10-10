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


class TestOnnxExportCustomCode(ExtTestCase):

    @classmethod
    def get_custom_model_autograd(cls):
        import torch

        class Twice(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # pytorch is doing symbolic tracing. It can't know
                # that this operator is coming from an autograd.function
                # defined this way.
                return x + x

            @staticmethod
            def symbolic(g: torch.Graph, x):
                # The conversion is wrong on purpose, to check, this function was used.
                return g.op("Add", x, g.op("Add", x, x))

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = Twice.apply(x)
                return y * x

        return DummyModel, torch.rand(5, 3)

    def test_custom_code_script(self):
        import torch

        cls, x = self.get_custom_model_autograd()
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

    @classmethod
    def get_custom_model_autograd_marked(cls):
        import torch

        class mark(torch.nn.Module):
            def __init__(self, name: str, out: bool):
                super().__init__()
                self.name = name
                self.out = out

            def forward(self, x):
                return x

        class Twice(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # pytorch is doing symbolic tracing. It can't know
                # that this operator is coming from an autograd.function
                # defined this way.
                x = mark("twice", False)(x)
                y = x + x
                return mark("twice", True)(y)

            @staticmethod
            def symbolic(g: torch.Graph, x):
                # The conversion is wrong on purpose, to check, this function was used.
                return g.op("Add", x, g.op("Add", x, x))

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = Twice.apply(x)
                return y * x

        return mark, DummyModel, torch.rand(5, 3)

    @skipif_ci_windows("dynamo not working on Windows")
    @requires_torch("2.5")
    def test_custom_code_export_1(self):
        import torch

        mark, cls, x = self.get_custom_model_autograd_marked()
        model = cls()

        class MyTracer(torch.fx.Tracer):

            @staticmethod
            def same(x, **kwargs):
                return x

            def call_module(self, m, forward, args, kwargs):
                if isinstance(m, mark):
                    return self.create_proxy(
                        "call_function",
                        MyTracer.same,
                        args,
                        kwargs=dict(name=m.name, out=m.out),
                        name=None,
                        type_expr=None,
                    )
                return super().call_module(m, forward, (args), kwargs)

            def path_of_module(self, mod: torch.nn.Module) -> str:
                return ""

        tr = MyTracer()
        res = tr.trace(model)
        res_str = str(res)
        self.assertIn(".same", res_str)
        self.assertIn("{name: twice, out: True}", res_str)
        targets = [node.target for node in res.nodes]
        names = [
            (t if isinstance(t, str) else str(t)).split(".")[-1].split(" at ")[0]
            for t in targets
        ]
        self.assertEqual(
            names,
            [
                "x",
                "same",
                "<built-in function add>",
                "same",
                "<built-in function mul>",
                "output",
            ],
        )

        # This is what it should produce.
        """
        graph():
            %x : [num_users=2] = placeholder[target=x]
            %same : [num_users=1] = call_function[target=__main__.same]
                (args = (%x,), kwargs = {name: twice, out: False})
            %add : [num_users=1] = call_function[target=operator.add]
                (args = (%same, %same), kwargs = {})
            %same_1 : [num_users=1] = call_function[target=__main__.same]
                (args = (%add,), kwargs = {name: twice, out: True})
            %mul : [num_users=1] = call_function[target=operator.mul]
                (args = (%same_1, %x), kwargs = {})
            return mul
        """

        exported_program = torch.export.export(model, (x,))
        graph = exported_program.graph

        # This graph does not keep the node "same" added to the graph.
        # But it should possible to modify the graph and recompile it.
        targets = [node.target for node in graph.nodes]
        names = [(t if isinstance(t, str) else t.name()) for t in targets]
        self.assertEqual(names, ["x", "aten::add.Tensor", "aten::mul.Tensor", "output"])

        filename = "test_custom_code_export_1.onnx"
        torch.onnx.export(model, (x,), filename, input_names=["x"], opset_version=18)

    @skipif_ci_windows("dynamo not working on Windows")
    @unittest.skip("autograd.function not working with dynamo")
    def test_custom_code_dynamo(self):
        import torch

        cls, x = self.get_custom_model_autograd()
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
    @requires_torch("2.4")
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
        registry = torch.onnx.OnnxRegistry()
        registry.register_op(
            function=onnxscript_twice, namespace="testlib", op_name="twice_onnx"
        )

        self.assertNotEmpty(expected)
        filename = "test_custom_code_dynamo_2_fn_registry.onnx"
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
