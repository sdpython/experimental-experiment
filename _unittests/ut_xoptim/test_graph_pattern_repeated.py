import os
import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from experimental_experiment.ext_test_case import ExtTestCase, hide_stdout
from experimental_experiment.xbuilder import GraphBuilder, OptimizationOptions
from experimental_experiment.xoptim.patterns_api import OnnxEasyPatternOptimization
from experimental_experiment.xoptim.repeated_optim import (
    node_type_frequency,
    find_largest_repeated_pattern,
)


class TestGraphPatternRepeated(ExtTestCase):
    def test_repeated_pattern(self):
        file = os.path.join(
            os.path.dirname(__file__),
            "data",
            "model_microsoft_Phi-3_5-mini-instruct-custom-default-d1rt1.onnx",
        )
        onx = onnx.load(file, load_external_data=False)
        h = node_type_frequency(onx)
        self.assertEqual(
            (
                {
                    ("", "Shape"): 5,
                    ("", "Squeeze"): 3,
                    ("", "Add"): 18,
                    ("", "Concat"): 22,
                    ("", "Gather"): 2,
                    ("", "Range"): 3,
                    ("", "Cast"): 3,
                    ("", "Reshape"): 14,
                    ("", "Expand"): 4,
                    ("", "Greater"): 2,
                    ("", "And"): 4,
                    ("", "Mul"): 29,
                    ("", "Slice"): 13,
                    ("", "Pow"): 6,
                    ("", "Reciprocal"): 6,
                    ("", "Unsqueeze"): 4,
                    ("", "MatMul"): 14,
                    ("", "Transpose"): 11,
                    ("", "ReduceMean"): 5,
                    ("", "Sqrt"): 5,
                    ("", "Split"): 4,
                    ("", "Neg"): 4,
                    ("", "Softmax"): 2,
                    ("", "Sigmoid"): 2,
                },
                {5: 3, 3: 13, 18: 1, 22: 1, 2: 54, 14: 2, 4: 5, 29: 1, 13: 1, 6: 5, 11: 3},
                2,
                [("", "Gather"), ("", "Greater"), ("", "Softmax"), ("", "Sigmoid")],
            ),
            h,
        )

    def test_repeated_pattern_asimple(self):
        onx = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "un"], ["a1"]),
                    oh.make_node("Neg", ["a1"], ["a2"]),
                    oh.make_node("Add", ["a2", "de"], ["a3"]),
                    oh.make_node("Add", ["a3", "un"], ["b1"]),
                    oh.make_node("Neg", ["b1"], ["b2"]),
                    oh.make_node("Add", ["b2", "de"], ["Z"]),
                ],
                "test",
                [],
                [],
            ),
        )
        h = node_type_frequency(onx)
        self.assertEqual(({("", "Add"): 4, ("", "Neg"): 2}, {4: 1, 2: 3}, 2, [("", "Neg")]), h)
        h = find_largest_repeated_pattern(onx)
        self.assertNotEmpty(h)
        indices, _nodes = h
        self.assertEqual(indices, [0, 1, 2])

    def test_repeated_pattern_asimple_match(self):
        onx = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "un"], ["a1"]),
                    oh.make_node("Neg", ["a1"], ["a2"]),
                    oh.make_node("Add", ["a2", "de"], ["a3"]),
                    oh.make_node("Add", ["a3", "un"], ["b1"]),
                    oh.make_node("Neg", ["b1"], ["b2"]),
                    oh.make_node("Add", ["b2", "de"], ["Z"]),
                ],
                "test",
                [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["a"])],
                [oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, ["a"])],
                [
                    onh.from_array(np.array([1], dtype=np.float32), name="un"),
                    onh.from_array(np.array([2], dtype=np.float32), name="de"),
                ],
            ),
        )
        for i, n in enumerate(onx.graph.node):
            n.name = f"i{i}"
        onnx.checker.check_model(onx)
        _indices, pattern = find_largest_repeated_pattern(onx)
        self.assertEqual(["Add", "Neg", "Add"], [n.op_type for n in pattern._match_model.node])
        self.assertEqual(["X", "de", "un"], pattern._match_model.input)
        self.assertEqual(["a3"], pattern._match_model.output)
        self.assertEqual(["RepeatedPattern"], [n.op_type for n in pattern._apply_model.node])
        self.assertEqual(["X", "de", "un"], pattern._apply_model.input)
        self.assertEqual(["a3"], pattern._apply_model.output)
        self.assertEqual(["X", "de", "un"], pattern._apply_model.node[0].input)
        self.assertEqual(["a3"], pattern._apply_model.node[0].output)
        builder = GraphBuilder(
            onx, optimization_options=OptimizationOptions(patterns=[pattern])
        )
        self.assertNotEmpty(builder.initializers_dict)
        builder._check([], "test")
        new_onx = builder.to_onnx()
        self.assertEqual(
            ["RepeatedPattern", "RepeatedPattern"], [n.op_type for n in new_onx.graph.node]
        )

    def test_repeated_pattern_order_equal(self):
        onx = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "un"], ["a1"]),
                    oh.make_node("Neg", ["a1"], ["a21"]),
                    oh.make_node("Abs", ["a1"], ["a22"]),
                    oh.make_node("Add", ["a21", "a22"], ["a3"]),
                    oh.make_node("Add", ["a3", "un"], ["b1"]),
                    oh.make_node("Neg", ["b1"], ["b21"]),
                    oh.make_node("Abs", ["b1"], ["b22"]),
                    oh.make_node("Add", ["b21", "b22"], ["Z"]),
                ],
                "test",
                [],
                [],
            ),
        )
        h = node_type_frequency(onx)
        self.assertEqual(
            (
                {("", "Add"): 4, ("", "Neg"): 2, ("", "Abs"): 2},
                {4: 1, 2: 4},
                2,
                [("", "Neg"), ("", "Abs")],
            ),
            h,
        )
        h = find_largest_repeated_pattern(onx)
        self.assertNotEmpty(h)
        indices, _nodes = h
        self.assertEqual(indices, [0, 1, 2, 3])

    def test_repeated_pattern_order_unequal(self):
        onx = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "un"], ["a1"]),
                    oh.make_node("Neg", ["a1"], ["a21"]),
                    oh.make_node("Abs", ["a1"], ["a22"]),
                    oh.make_node("Add", ["a21", "a22"], ["a3"]),
                    oh.make_node("Add", ["a3", "un"], ["b1"]),
                    oh.make_node("Abs", ["b1"], ["b22"]),
                    oh.make_node("Neg", ["b1"], ["b21"]),
                    oh.make_node("Add", ["b21", "b22"], ["Z"]),
                ],
                "test",
                [],
                [],
            ),
        )
        h = node_type_frequency(onx)
        self.assertEqual(
            (
                {("", "Add"): 4, ("", "Neg"): 2, ("", "Abs"): 2},
                {4: 1, 2: 4},
                2,
                [("", "Neg"), ("", "Abs")],
            ),
            h,
        )
        h = find_largest_repeated_pattern(onx)
        self.assertNotEmpty(h)
        indices, _nodes = h
        self.assertEqual(indices, [0, 1, 2, 3])

    @hide_stdout()
    def test_repeated_pattern_true_phi3_mini(self):
        file = os.path.join(
            os.path.dirname(__file__),
            "data",
            "model_microsoft_Phi-3_5-mini-instruct-custom-default-d1rt1.onnx",
        )
        onx = onnx.load(file, load_external_data=False)
        h = find_largest_repeated_pattern(onx, verbose=3, all_instances=True)
        self.assertNotEmpty(h)
        self.assertEqual(len(h), 2)
        n = 60
        self.assertEqual(len(h[0][0]), n)
        self.assertIsInstance(h[1], OnnxEasyPatternOptimization)

    @hide_stdout()
    def test_repeated_pattern_true_phi4_mini(self):
        file = os.path.join(
            os.path.dirname(__file__),
            "data",
            "microsoft_Phi-4-mini-reasoning-custom-default-f32-cpu.onnx",
        )
        onx = onnx.load(file, load_external_data=False)
        h = find_largest_repeated_pattern(onx, verbose=3, all_instances=True)
        self.assertNotEmpty(h)
        self.assertEqual(len(h), 2)
        n = 60
        self.assertEqual(len(h[0][0]), n)
        self.assertIsInstance(h[1], OnnxEasyPatternOptimization)


if __name__ == "__main__":
    unittest.main(verbosity=2)
