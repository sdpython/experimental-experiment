import itertools
import unittest
import numpy as np
from onnx import TensorProto, helper as oh, numpy_helper as onh
from onnx.checker import check_model
from experimental_experiment.ext_test_case import ExtTestCase, skipif_ci_windows
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.xoptim.patterns import get_pattern_list
from experimental_experiment.xbuilder._onnx_helper import (
    choose_consistent_domain_opset,
    compatible_opsets,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator

TFLOAT = TensorProto.FLOAT


class TestGraphPatternOptimizationOrt(ExtTestCase):
    def _range(self, *shape, bias: float = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

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

    @skipif_ci_windows("get_all_schemas_with_history returns wrong values")
    def test_compatible_opsets(self):
        self.assertTrue(compatible_opsets("", "Slice", 18, 18))
        self.assertTrue(compatible_opsets("", "Slice", 18, 17))
        self.assertFalse(compatible_opsets("", "Slice", 12, 13))
        self.assertFalse(compatible_opsets("", "Slice", 13, 12))
        self.assertFalse(compatible_opsets("", "Slice", 11, 13))
        self.assertTrue(compatible_opsets("", "Slice", 11, 12))
        self.assertFalse(compatible_opsets("", "Slice", 18, 1))

    def common_fused_matmul(self, side):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["xm1"], perm=[0, 1, 3, 2]),
                    oh.make_node(
                        "MatMul",
                        ["xm1", "Y"] if side == "left" else ["Y", "xm1"],
                        ["Z"],
                    ),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info(
                        "X",
                        TFLOAT,
                        [2, 2, 128, 32] if side == "left" else [2, 2, 32, 128],
                    ),
                    oh.make_tensor_value_info(
                        "Y",
                        TFLOAT,
                        [2, 2, 128, 64] if side == "left" else [2, 2, 64, 128],
                    ),
                ],
                [
                    oh.make_tensor_value_info(
                        "Z",
                        TFLOAT,
                        [2, 2, 32, 64] if side == "left" else [2, 2, 64, 32],
                    ),
                ],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        check_model(model)
        feeds = (
            {"X": self._range(2, 2, 128, 32), "Y": self._range(2, 2, 128, 64)}
            if side == "left"
            else {"X": self._range(2, 2, 32, 128), "Y": self._range(2, 2, 64, 128)}
        )
        if side == "left":
            assert feeds["X"][0, 0].T @ feeds["Y"][0, 0] is not None
        else:
            assert feeds["Y"][0, 0] @ feeds["X"][0, 0].T is not None
        ref = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["FusedMatMul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["FusedMatMul"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_fused_matmul_left(self):
        self.common_fused_matmul("left")

    def test_fused_matmul_right(self):
        self.common_fused_matmul("right")

    def test_fused_matmul_both(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["xm1"], perm=[0, 1, 3, 2]),
                    oh.make_node("Transpose", ["Y"], ["ym1"], perm=[0, 1, 3, 2]),
                    oh.make_node("MatMul", ["xm1", "ym1"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 2, 128, 32]),
                    oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 64, 128]),
                ],
                [
                    oh.make_tensor_value_info("Z", TFLOAT, [2, 2, 32, 64]),
                    oh.make_tensor_value_info("xm1", TFLOAT, [2, 2, 32, 128]),
                    oh.make_tensor_value_info("ym1", TFLOAT, [2, 2, 128, 64]),
                ],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 128, 32), "Y": self._range(2, 2, 64, 128)}
        assert feeds["X"][0, 0].T @ feeds["Y"][0, 0].T is not None
        ref = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["FusedMatMul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Transpose", "Transpose", "FusedMatMul"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])
        node = opt_onx.graph.node[2]
        self.assertEqual(node.op_type, "FusedMatMul")
        self.assertEqual(node.domain, "com.microsoft")
        for att in node.attribute:
            if att.name == "transA":
                self.assertEqual(att.i, 0)
            elif att.name == "transB":
                self.assertEqual(att.i, 1)

    def test_fused_matmul_both_div(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["xm1"], perm=[0, 1, 3, 2]),
                    oh.make_node("Transpose", ["Y"], ["ym1"], perm=[0, 1, 3, 2]),
                    oh.make_node("MatMul", ["xm1", "ym1"], ["zd"]),
                    oh.make_node("Div", ["zd", "deux"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 2, 128, 32]),
                    oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 64, 128]),
                ],
                [
                    oh.make_tensor_value_info("Z", TFLOAT, [2, 2, 32, 64]),
                ],
                [onh.from_array(np.array([2], dtype=np.float32), name="deux")],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 128, 32), "Y": self._range(2, 2, 64, 128)}
        assert feeds["X"][0, 0].T @ feeds["Y"][0, 0].T is not None
        ref = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["FusedMatMul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Transpose", "FusedMatMul"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])
        node = opt_onx.graph.node[1]
        self.assertEqual(node.op_type, "FusedMatMul")
        self.assertEqual(node.domain, "com.microsoft")
        for att in node.attribute:
            if att.name == "transA":
                self.assertEqual(att.i, 0)
            elif att.name == "transB":
                self.assertEqual(att.i, 1)
            elif att.name == "alpha":
                self.assertEqual(att.f, 0.5)

    def get_simplified_layer_normalization_model(self, div, dyn):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Pow", ["X", "exp"], ["x2"]),
                    oh.make_node("ReduceMean", ["x2", "axis"], ["xr"]),
                    oh.make_node("Add", ["xr", "eps"], ["xa"]),
                    oh.make_node("Sqrt", ["xa"], ["xq"]),
                    (
                        oh.make_node("Div", ["one", "xq"], ["xi"])
                        if div
                        else oh.make_node("Reciprocal", ["xq"], ["xi"])
                    ),
                    oh.make_node("Mul", ["xi", "X"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "D" if dyn else 4])],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "D" if dyn else 4])],
                [
                    onh.from_array(np.array([2], dtype=np.float32), name="exp"),
                    onh.from_array(
                        np.array([9.999999974752427e-7], dtype=np.float32), name="eps"
                    ),
                    onh.from_array(np.array([-1], dtype=np.int64), name="axis"),
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                ],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        check_model(model)
        return model

    def test_simplified_layer_normalization_model(self):
        for div, dyn in itertools.product([False, True], [False, True]):
            with self.subTest(div=div, dyn=dyn):
                model = self.get_simplified_layer_normalization_model(div=div, dyn=dyn)
                gr = GraphBuilder(
                    model,
                    infer_shapes=True,
                    optimization_options=OptimizationOptions(
                        patterns=["SimplifiedLayerNormalization"]
                    ),
                )
                opt_onx = gr.to_onnx(optimize=True)
                self.assertEqual(
                    (
                        [
                            "Shape",
                            "Gather",
                            "ConstantOfShape",
                            "SimplifiedLayerNormalization",
                        ]
                        if dyn
                        else ["SimplifiedLayerNormalization"]
                    ),
                    [n.op_type for n in opt_onx.graph.node],
                )

                feeds = {
                    "X": np.arange(20).reshape((5, 4)).astype(np.float32),
                }
                ref1 = ExtendedReferenceEvaluator(model)
                expected = ref1.run(None, feeds)

                ninits = {
                    (False, False): 1,
                    (False, True): 1,
                    (True, False): 1,
                    (True, True): 1,
                }
                self.assertEqual(ninits[div, dyn], len(opt_onx.graph.initializer))

                ref2 = ExtendedReferenceEvaluator(opt_onx)
                got = ref2.run(None, feeds)
                self.assertEqualArray(expected[0], got[0], atol=1e-5)

                if got:
                    from onnxruntime import InferenceSession

                    sess = InferenceSession(
                        opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
                    )
                    got = sess.run(None, feeds)
                    self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def get_simplified_layer_normalization_model_output(self, div, dyn):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Pow", ["X", "exp"], ["x2"]),
                    oh.make_node("ReduceMean", ["x2", "axis"], ["xr"]),
                    oh.make_node("Add", ["xr", "eps"], ["xa"]),
                    oh.make_node("Sqrt", ["xa"], ["xq"]),
                    (
                        oh.make_node("Div", ["one", "xq"], ["Z"])
                        if div
                        else oh.make_node("Reciprocal", ["xq"], ["Z"])
                    ),
                    oh.make_node("Mul", ["Z", "X"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "D" if dyn else 4])],
                [
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "D" if dyn else 4]),
                    oh.make_tensor_value_info("Z", TFLOAT, ["a", 1]),
                ],
                [
                    onh.from_array(np.array([2], dtype=np.float32), name="exp"),
                    onh.from_array(
                        np.array([9.999999974752427e-7], dtype=np.float32), name="eps"
                    ),
                    onh.from_array(np.array([-1], dtype=np.int64), name="axis"),
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                ],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        check_model(model)
        return model

    def test_simplified_layer_normalization_model_output(self):
        for div, dyn in itertools.product([False, True], [False, True]):
            with self.subTest(div=div, dyn=dyn):
                model = self.get_simplified_layer_normalization_model_output(
                    div=div, dyn=dyn
                )
                gr = GraphBuilder(
                    model,
                    infer_shapes=True,
                    optimization_options=OptimizationOptions(
                        patterns=["SimplifiedLayerNormalization"]
                    ),
                )
                opt_onx = gr.to_onnx(optimize=True)
                self.assertEqual(
                    (
                        [
                            "Shape",
                            "Gather",
                            "ConstantOfShape",
                            "SimplifiedLayerNormalization",
                        ]
                        if dyn
                        else ["SimplifiedLayerNormalization"]
                    ),
                    [n.op_type for n in opt_onx.graph.node],
                )

                feeds = {
                    "X": np.arange(20).reshape((5, 4)).astype(np.float32),
                }
                ref1 = ExtendedReferenceEvaluator(model)
                expected = ref1.run(None, feeds)

                ninits = {
                    (False, False): 1,
                    (False, True): 1,
                    (True, False): 1,
                    (True, True): 1,
                }
                self.assertEqual(ninits[div, dyn], len(opt_onx.graph.initializer))

                ref2 = ExtendedReferenceEvaluator(opt_onx)
                got = ref2.run(None, feeds)
                self.assertEqualArray(expected[0], got[0], atol=1e-5)
                self.assertEqualArray(expected[1], got[1], atol=1e-5)

                if got:
                    from onnxruntime import InferenceSession

                    sess = InferenceSession(
                        opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
                    )
                    got = sess.run(None, feeds)
                    self.assertEqualArray(expected[0], got[0], atol=1e-5)
                    self.assertEqualArray(expected[1], got[1], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
