import unittest
import numpy as np
from onnx import TensorProto, helper as oh, numpy_helper as onh
from onnx.checker import check_model
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_exp.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.torch_exp.optimization_patterns import get_pattern_list
from experimental_experiment.torch_exp._onnx_helper import (
    choose_consistent_domain_opset,
    compatible_opsets,
)


class TestGraphPatternOptimizationOrt(ExtTestCase):
    def test_get_pattern_list(self):
        res = get_pattern_list("onnxruntime")
        names = set(r.__class__.__name__ for r in res)
        self.assertNotIn("ConstantScatterNDPattern", names)

    def test_choose_consistent_domain_opset(self):
        self.assertIsInstance(choose_consistent_domain_opset(""), int)
        self.assertEqual(choose_consistent_domain_opset("", {"": 10}), 10)
        self.assertEqual(choose_consistent_domain_opset("ai.onnx.ml", {"": 18}), 3)
        self.assertEqual(choose_consistent_domain_opset("com.microsoft", {"": 18}), 1)
        self.assertIsInstance(
            choose_consistent_domain_opset("", {"com.microsoft": 1}), int
        )
        self.assertRaise(
            lambda: choose_consistent_domain_opset("", {"ai.onnx.ml": 10}),
            AssertionError,
        )

    def test_compatible_opsets(self):
        self.assertTrue(compatible_opsets("", "Slice", 18, 18))
        self.assertTrue(compatible_opsets("", "Slice", 18, 17))
        self.assertFalse(compatible_opsets("", "Slice", 12, 13))
        self.assertFalse(compatible_opsets("", "Slice", 13, 12))
        self.assertFalse(compatible_opsets("", "Slice", 11, 13))
        self.assertTrue(compatible_opsets("", "Slice", 11, 12))
        self.assertFalse(compatible_opsets("", "Slice", 18, 1))

    def test_scatter_of_shape(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ConstantOfShape",
                        ["shape"],
                        ["cst"],
                        value=onh.from_array(np.array([0], dtype=np.float32)),
                    ),
                    oh.make_node("ScatterND", ["cst", "indices", "updates"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("shape", TensorProto.INT64, [None]),
                    oh.make_tensor_value_info(
                        "indices", TensorProto.INT64, [None, None]
                    ),
                    oh.make_tensor_value_info(
                        "updates", TensorProto.FLOAT, [None, None, None]
                    ),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [None, None, None])],
            )
        )
        check_model(model)
        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["ConstantOfShapeScatterND"]
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["ScatterNDOfShape"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))
        check_model(opt_onx)
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("com.microsoft", opsets)
        self.assertEqual(opsets["com.microsoft"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
