import unittest
import numpy as np
from onnx import TensorProto
from onnx.checker import check_model
import onnx.helper as oh
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xoptim import EasyPatternOptimization
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)

T = str
TFLOAT = TensorProto.FLOAT


class AddAddPattern(EasyPatternOptimization):
    """
    Replaces ConstantOfShape + ScatterND with ScatterNDOfShape (com.domain).
    """

    def match_pattern(self, g: "GraphBuilder", x: T, y: T, z: T):  # noqa: F821
        """
        Builds the pattern to match.
        """
        tmp = g.op.Add(x, y)
        return g.op.Add(tmp, z)

    @classmethod
    def apply_pattern(cls, g: "GraphBuilder", x: T, y: T, z: T):  # noqa: F821
        """
        Builds the pattern to match.
        """
        return g.anyop.AddAdd(x, y, z, domain="ZZZ")


class TestGraphPatternBuilder(ExtTestCase):

    def _range(self, *shape, bias: float = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def test_graph_pattern_builder(self):

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["x", "y"], ["gggg"]),
                    oh.make_node("Add", ["gggg", "z"], ["final"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("x", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("y", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("z", TFLOAT, [None, None]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        check_model(model)
        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=[AddAddPattern()]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["AddAdd"],
            [n.op_type for n in opt_onx.graph.node],
        )

        feeds = {
            "shape": np.array([5, 6], dtype=np.int64),
            "indices": np.array([[0], [1], [0]], dtype=np.int64),
            "updates": np.arange(18).reshape((3, 6)).astype(np.float32),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        self.assertEqual(0, len(opt_onx.graph.initializer))
        check_model(opt_onx)
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("ZZZ", opsets)
        self.assertEqual(opsets["ZZZ"], 1)

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
