import unittest
import sys
import numpy as np
from onnx import ModelProto, TensorProto
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    ignore_warnings,
    requires_torch,
)
from experimental_experiment.xbuilder.graph_builder import GraphBuilder
from experimental_experiment.torch_interpreter.oxs_opset import (
    OxsOpset,
    Var,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator


class TestFallbackOxs(ExtTestCase):
    def test_registry(self):
        from onnxscript.function_libs.torch_lib.registration import default_registry
        import onnxscript.function_libs.torch_lib.ops.core  # noqa: F401

        reg = default_registry
        self.assertIn("aten::celu", reg)

    def test_celu(self):
        from onnxscript.function_libs.torch_lib.registration import default_registry
        import onnxscript.function_libs.torch_lib.ops.core
        import onnxscript.function_libs.torch_lib.ops.nn

        mods = {
            "onnxscript.function_libs.torch_lib.ops.core": onnxscript.function_libs.torch_lib.ops.core,
            "onnxscript.function_libs.torch_lib.ops.nn": onnxscript.function_libs.torch_lib.ops.nn,
        }

        reg = default_registry
        self.assertIn("aten::celu", reg)
        f = reg["aten::celu"]
        self.assertGreater(len(f.overloads), 1)
        fct = f.overloads[0]
        mod = mods[fct.function.__module__]

        gr = GraphBuilder(18, ir_version=9)
        gr.make_tensor_input("X", TensorProto.FLOAT, ("a", "b"), is_dimension=False)

        old_value = mod.op
        mod.op = OxsOpset(gr)
        y = gr.op.Identity(fct.function("X", alpha=2.0), outputs=["Y"])
        mod.op = old_value

        gr.make_tensor_output(
            y, TensorProto.FLOAT, ("a", "b"), indexed=False, is_dimension=False
        )
        onx = gr.to_onnx()

        ext = ExtendedReferenceEvaluator(onx)
        x = np.random.rand(3, 4).astype(np.float32)
        y = ext.run(None, {"X": x})[0]
        self.assertEqual(x.shape, y.shape)

    def test_is_scalar(self):
        from onnxscript.function_libs.torch_lib.registration import default_registry
        import onnxscript.function_libs.torch_lib.ops.core
        import onnxscript.function_libs.torch_lib.ops.nn

        mods = {
            "onnxscript.function_libs.torch_lib.ops.core": onnxscript.function_libs.torch_lib.ops.core,
            "onnxscript.function_libs.torch_lib.ops.nn": onnxscript.function_libs.torch_lib.ops.nn,
        }

        reg = default_registry
        f = reg["aten::all.dim"]
        self.assertGreater(len(f.overloads), 0)
        fct = f.overloads[0]
        mod = mods[fct.function.__module__]

        gr = GraphBuilder(18, ir_version=9)
        gr.make_tensor_input("X", TensorProto.INT64, ("a", "b"), is_dimension=False)

        old_value = [mod.op, mod.IsScalar]

        mod.op = OxsOpset(gr)
        mod.IsScalar = mod.op.IsScalar
        y = gr.op.Identity(fct.function("X", dim=0, keepdim=True), outputs=["Y"])

        mod.op, mod.IsScalar = old_value

        gr.make_tensor_output(
            y, TensorProto.BOOL, (1, "b"), indexed=False, is_dimension=False
        )
        onx = gr.to_onnx()

        ext = ExtendedReferenceEvaluator(onx)
        x = np.random.rand(3, 4).astype(np.int64)
        y = ext.run(None, {"X": x})[0]
        self.assertEqual((1, 4), y.shape)

    def test_rank(self):
        from onnxscript.function_libs.torch_lib.registration import default_registry
        import onnxscript.function_libs.torch_lib.ops.core
        import onnxscript.function_libs.torch_lib.ops.nn

        mods = {
            "onnxscript.function_libs.torch_lib.ops.core": onnxscript.function_libs.torch_lib.ops.core,
            "onnxscript.function_libs.torch_lib.ops.nn": onnxscript.function_libs.torch_lib.ops.nn,
        }

        reg = default_registry
        f = reg["aten::atleast_2d"]
        self.assertGreater(len(f.overloads), 0)
        fct = f.overloads[0]
        mod = mods[fct.function.__module__]

        gr = GraphBuilder(18, ir_version=9)
        gr.make_tensor_input("X", TensorProto.INT64, ("a",), is_dimension=False)

        old_value = [mod.op, mod.Rank]

        mod.op = OxsOpset(gr)
        mod.Rank = mod.op.Rank
        y = gr.op.Identity(fct.function("X"), outputs=["Y"])

        mod.op, mod.Rank = old_value

        gr.make_tensor_output(
            y, TensorProto.INT64, (1, "a"), indexed=False, is_dimension=False
        )
        onx = gr.to_onnx()

        ext = ExtendedReferenceEvaluator(onx)
        x = np.random.rand(12).astype(np.int64)
        y = ext.run(None, {"X": x})[0]
        self.assertEqual((1, 12), y.shape)

    def test_gather(self):
        from onnxscript.function_libs.torch_lib.registration import default_registry
        import onnxscript.function_libs.torch_lib.ops.core
        import onnxscript.function_libs.torch_lib.ops.nn

        mods = {
            "onnxscript.function_libs.torch_lib.ops.core": onnxscript.function_libs.torch_lib.ops.core,
            "onnxscript.function_libs.torch_lib.ops.nn": onnxscript.function_libs.torch_lib.ops.nn,
        }

        reg = default_registry
        f = reg["aten::gather"]
        self.assertGreater(len(f.overloads), 0)
        fct = f.overloads[0]
        mod = mods[fct.function.__module__]

        # def aten_gather(self: TReal, dim: int, index: TInt, sparse_grad: bool = False) -> TReal:

        gr = GraphBuilder(18, ir_version=9)
        gr.make_tensor_input("X", TensorProto.INT64, ("a",), is_dimension=False)
        gr.make_tensor_input("I", TensorProto.INT64, ("b",), is_dimension=False)

        self.assertEqual(repr(Var("X")), "Var('X')")
        old_value = mod.op, mod.IsScalar

        mod.op = OxsOpset(gr)
        mod.IsScalar = mod.op.IsScalar
        try:
            fct.function(Var("X"), 0, Var("I"))
        except RuntimeError as e:
            self.assertIn("The function being traced", str(e))

        mod.op, mod.IsScalar = old_value

    @skipif_ci_windows("dynamo not supported on Windows")
    def test_fallback_oxs(self):
        import torch
        from experimental_experiment.torch_interpreter import to_onnx
        from experimental_experiment.torch_interpreter.oxs_dispatcher import (
            OxsDispatcher,
        )

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return torch.celu(self.linear(x))

        x = torch.rand(5, 3)
        model = Neuron(3, 1)

        onx = to_onnx(model, (x,), input_names=["x"], dispatcher=OxsDispatcher())
        ext = ExtendedReferenceEvaluator(onx)
        got = ext.run(None, {"x": x.numpy()})[0]
        self.assertEqual(got.shape, (5, 1))

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @requires_torch("2.3", "bug")
    @ignore_warnings(DeprecationWarning)
    def test_llama_model_fallback_debug(self):
        import torch
        from experimental_experiment.torch_helper.llama_helper import get_llama_model
        from experimental_experiment.xbuilder import OptimizationOptions
        from experimental_experiment.torch_interpreter import to_onnx
        from experimental_experiment.torch_interpreter.oxs_dispatcher import (
            OxsDebugDispatcher,
        )

        with torch.no_grad():
            model, input_tensors = get_llama_model()
            input_tensors = input_tensors[0]
            expected = model(*input_tensors)
            self.assertNotEmpty(expected)

            onx, out, _ = self.capture(
                lambda: to_onnx(
                    model,
                    input_tensors,
                    input_names=[f"input{i}" for i in range(len(input_tensors))],
                    options=OptimizationOptions(patterns=None),
                    verbose=0,
                    dispatcher=OxsDebugDispatcher(verbose=2, raise_exc=False),
                )
            )
            self.assertIsInstance(onx, ModelProto)
            self.assertIn("verified", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
