import unittest
from typing import Optional
import numpy as np
from onnx import TensorProto, helper as oh, numpy_helper as onh
from onnx.checker import check_model
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.xoptim import get_pattern_list
from experimental_experiment.reference import ExtendedReferenceEvaluator

TFLOAT = TensorProto.FLOAT
TINT64 = TensorProto.INT64


class TestGraphPatternOptimizationFix(ExtTestCase):
    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def test_get_pattern_list(self):
        res = get_pattern_list("fix")
        names = set(r.__class__.__name__ for r in res)
        self.assertIn("AddReductionScatterND", names)

    def test_add_reduction_scatter_nd(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ConstantOfShape",
                        ["shape"],
                        ["zero"],
                        value=onh.from_array(np.array([0], dtype=np.float32)),
                    ),
                    oh.make_node("ScatterND", ["zero", "indices", "updates"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("indices", TINT64, [2]),
                    oh.make_tensor_value_info("updates", TFLOAT, [2, 8]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [5, 8])],
                [onh.from_array(np.array([5, 8], dtype=np.int64), name="shape")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        check_model(model)
        feeds = dict(
            indices=np.array([[0], [2]], dtype=np.int64),
            updates=np.ones((2, 8), dtype=np.float32),
        )
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["AddReductionScatterND"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["ConstantOfShape", "ScatterND"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])
        node = opt_onx.graph.node[1]
        self.assertEqual(node.op_type, "ScatterND")
        n_checked = 0
        for att in node.attribute:
            if att.name == "reduction":
                self.assertEqual(att.s, b"add")
                n_checked += 1
        self.assertEqual(n_checked, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
