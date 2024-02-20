import os
import unittest
import onnx
from onnx import TensorProto
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_exp.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.torch_exp.graph_builder_optim import (
    GraphBuilderPatternOptimization,
)


class TestGraphPatternOptimization(ExtTestCase):
    def _get_model(self, name: str) -> onnx.ModelProto:
        p = os.path.join(os.path.dirname(__file__), "data", name)
        self.assertExists(p)
        return onnx.load(p)

    def test_type_inference0(self):
        origin = self._get_model("dort-c-custom__0.onnx")
        gr = GraphBuilder(origin)
        gro = GraphBuilderPatternOptimization(gr)
        dtype = gro.try_infer_type("_onx_tile0", exc=True)
        self.assertEqual(dtype, TensorProto.INT64)

    def test_type_inference1(self):
        origin = self._get_model("dort-c-custom__1.onnx")
        gr = GraphBuilder(origin)
        gro = GraphBuilderPatternOptimization(gr)
        dtype = gro.try_infer_type("_onx_mul028", exc=True)
        self.assertEqual(dtype, TensorProto.FLOAT)

    def test_unsqueeze_unsqueeze(self):
        origin = self._get_model("dort-c-custom__0.onnx")
        before = [node for node in origin.graph.node if node.op_type == "Unsqueeze"]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(patterns=["UnsqueezeUnsqueeze"]),
        )
        res, out, err = self.capture(lambda: gr.optimize_with_patterns(verbose=10))
        self.assertEmpty(err)
        self.assertEmpty(res)
        self.assertIn("[GraphBuilderPatternOptimization.optimize] done after", out)
        self.assertIn("UnsqueezeUnsqueezePattern", out)

        onx = gr.to_onnx(optimize=False)
        after = [node for node in onx.graph.node if node.op_type == "Unsqueeze"]
        self.assertEqual(len(after), len(before) - 2)

    def test_cast(self):
        origin = self._get_model("dort-c-custom__0.onnx")
        before = [node for node in origin.graph.node if node.op_type == "Cast"]
        gr = GraphBuilder(
            origin, optimization_options=OptimizationOptions(patterns=["Cast"])
        )
        res, out, err = self.capture(lambda: gr.optimize_with_patterns(verbose=10))
        self.assertEmpty(err)
        self.assertEmpty(res)
        self.assertIn("[GraphBuilderPatternOptimization.optimize] done after", out)
        self.assertNotIn("UnsqueezeUnsqueezePattern", out)
        self.assertIn("CastPattern", out)

        onx = gr.to_onnx(optimize=False)
        after = [node for node in onx.graph.node if node.op_type == "Cast"]
        self.assertEqual(len(before), 14)
        self.assertEqual(len(after), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
