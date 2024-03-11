import unittest
import numpy as np
from onnx import TensorProto
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xbuilder.graph_builder import GraphBuilder
from experimental_experiment.torch_interpreter.oxs_opset import OxsOpset
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
            y, TensorProto.FLOAT, (1, "b"), indexed=False, is_dimension=False
        )
        onx = gr.to_onnx()

        ext = ExtendedReferenceEvaluator(onx)
        x = np.random.rand(3, 4).astype(np.int64)
        y = ext.run(None, {"X": x})[0]
        self.assertEqual(x.shape, y.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
