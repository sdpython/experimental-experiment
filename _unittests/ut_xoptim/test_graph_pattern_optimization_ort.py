import itertools
import os
import unittest
from typing import Optional
import numpy as np
from onnx import (
    ModelProto,
    TensorProto,
    helper as oh,
    numpy_helper as onh,
    load as onnx_load,
    shape_inference,
)
from onnx.checker import check_model
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
    requires_cuda,
    hide_stdout,
    has_onnxruntime_training,
)
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
    InferShapesOptions,
)
from experimental_experiment.xoptim import get_pattern_list
from experimental_experiment.xoptim.patterns_ort.gather_grad import GatherGradPattern
from experimental_experiment.xoptim.patterns_ort.activation import GeluErfPattern
from experimental_experiment.xbuilder._onnx_helper import (
    choose_consistent_domain_opset,
    compatible_opsets,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx

TFLOAT = TensorProto.FLOAT
TINT64 = TensorProto.INT64


class TestGraphPatternOptimizationOrt(ExtTestCase):
    def _range(self, *shape, bias: Optional[float] = None):
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
        self.assertIsInstance(choose_consistent_domain_opset("", {"com.microsoft": 1}), int)
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

    def _get_model(self, name: str) -> ModelProto:
        p = os.path.join(os.path.dirname(__file__), "..", "ut_xbuilder", "data", name)
        if not os.path.exists(p):
            p = os.path.join(os.path.dirname(__file__), "data", name)
        self.assertExists(p)
        return onnx_load(p)

    def test_fused_matmul_pattern(self):
        origin = self._get_model("bug_fused.onnx")
        check_model(origin)
        gr = GraphBuilder(
            origin,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FusedMatMul"],
                verbose=0,  # stop_after=2
            ),
        )
        onx = gr.to_onnx(optimize=True)
        check_model(onx)
        if has_onnxruntime_training():
            self._check_with_ort(origin)
            self._check_with_ort(onx)

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
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
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
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
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
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 2, 32, 64])],
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
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
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

    def test_fused_matmul_div(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "FusedMatMul",
                        ["X", "Y"],
                        ["zd"],
                        domain="com.microsoft",
                        alpha=1.3,
                    ),
                    oh.make_node("Div", ["zd", "deux"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 2, 32, 128]),
                    oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 128, 64]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 2, 32, 64])],
                [onh.from_array(np.array([2], dtype=np.float32), name="deux")],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 32, 128), "Y": self._range(2, 2, 128, 64)}
        assert feeds["X"][0, 0] @ feeds["Y"][0, 0] is not None
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["FusedMatMulDiv"], verbose=0),
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
        node = opt_onx.graph.node[0]
        self.assertEqual(node.op_type, "FusedMatMul")
        self.assertEqual(node.domain, "com.microsoft")
        for att in node.attribute:
            if att.name == "transA":
                self.assertEqual(att.i, 0)
            elif att.name == "transB":
                self.assertEqual(att.i, 0)
            elif att.name == "alpha":
                self.assertAlmostEqual(att.f, 1.3 / 2, atol=1e-5)

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
            opset_imports=[oh.make_opsetid("", 18)],
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
                    infer_shapes_options=True,
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

                feeds = {"X": np.arange(20).reshape((5, 4)).astype(np.float32)}
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
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        check_model(model)
        return model

    def test_simplified_layer_normalization_model_output(self):
        for div, dyn in itertools.product([False, True], [False, True]):
            with self.subTest(div=div, dyn=dyn):
                model = self.get_simplified_layer_normalization_model_output(div=div, dyn=dyn)
                gr = GraphBuilder(
                    model,
                    infer_shapes_options=True,
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

                feeds = {"X": np.arange(20).reshape((5, 4)).astype(np.float32)}
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

    def test_softmax_grad(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Mul", ["dY", "Y"], ["mul"]),
                    oh.make_node("ReduceSum", ["mul", "axis"], ["red"]),
                    oh.make_node("Mul", ["red", "Y"], ["scaled"]),
                    oh.make_node("Sub", ["mul", "scaled"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("dY", TFLOAT, [2, 3]),
                    oh.make_tensor_value_info("Y", TFLOAT, [2, 3]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 3])],
                [onh.from_array(np.array([-1], dtype=np.int64), name="axis")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        check_model(model)
        feeds = {"dY": self._range(2, 3), "Y": self._range(2, 3)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["SoftmaxGrad"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["SoftmaxGrad"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        if not has_onnxruntime_training():
            raise unittest.SkipTest("no onnxruntime training")

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)
        node = opt_onx.graph.node[0]
        self.assertEqual(node.op_type, "SoftmaxGrad")
        self.assertEqual(node.domain, "com.microsoft")
        for att in node.attribute:
            if att.name == "axis":
                self.assertEqual(att.i, -1)

    def test_gather_grad(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ConstantOfShape",
                        ["shape"],
                        ["cst"],
                        value=onh.from_array(np.array([0], dtype=np.float32)),
                    ),
                    oh.make_node(
                        "ScatterND",
                        ["cst", "indices", "updates"],
                        ["Z"],
                        reduction="add",
                    ),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("shape", TensorProto.INT64, [None]),
                    oh.make_tensor_value_info("indices", TensorProto.INT64, [None, None]),
                    oh.make_tensor_value_info("updates", TFLOAT, [None, None, None]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        check_model(model)
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=[GatherGradPattern()]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["GatherGrad"],
            [n.op_type for n in opt_onx.graph.node],
        )

        feeds = {
            "shape": np.array([5, 6], dtype=np.int64),
            # np.array([[0], [1], [0]], dtype=np.int64) does not work
            "indices": np.array([[0], [1], [2]], dtype=np.int64),
            "updates": np.arange(18).reshape((3, 6)).astype(np.float32),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        self.assertEqual(0, len(opt_onx.graph.initializer))
        check_model(opt_onx)
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("com.microsoft", opsets)
        self.assertEqual(opsets["com.microsoft"], 1)

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_fused_matmul_both_div_2x(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Div", ["X", "deux"], ["half"]),
                    oh.make_node(
                        "FusedMatMul",
                        ["half", "X"],
                        ["x1"],
                        transA=1,
                        alpha=50.1,
                        domain="com.microsoft",
                    ),
                    oh.make_node(
                        "FusedMatMul",
                        ["X", "half"],
                        ["x2"],
                        transA=1,
                        alpha=0.07,
                        domain="com.microsoft",
                    ),
                    oh.make_node("Add", ["x1", "x2"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 2, 4, 4]),
                    oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 4, 4]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 2, 32, 64])],
                [onh.from_array(np.array([2], dtype=np.float32), name="deux")],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        feeds = {"X": self._range(2, 2, 4, 4), "Y": self._range(2, 2, 4, 4)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["FusedMatMulx2"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["FusedMatMul", "FusedMatMul", "Add"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_fused_matmul_transpose(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "FusedMatMul",
                        ["X", "Y"],
                        ["xy"],
                        transA=1,
                        transB=1,
                        alpha=50.1,
                        domain="com.microsoft",
                    ),
                    oh.make_node("Transpose", ["xy"], ["Z"], perm=[0, 1, 3, 2]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 2, 6, 3]),
                    oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 5, 6]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 2, None, None])],
                [onh.from_array(np.array([2], dtype=np.float32), name="deux")],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        feeds = {"X": self._range(2, 2, 6, 3), "Y": self._range(2, 2, 5, 6)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FusedMatMulTranspose"], verbose=0
            ),
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

    @ignore_warnings(UserWarning)
    def test_fast_gelu(self):
        data = os.path.join(os.path.dirname(__file__), "data", "layernorm.onnx")
        model = onnx_load(data, load_external_data=False)
        del model.opset_import[:]
        model.opset_import.append(oh.make_opsetid("", 20))
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["Cast", "Gelu", "FastGelu"], verbose=0, constant_folding=False
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertNotIn("Gelu", set(n.op_type for n in opt_onx.graph.node))
        self.assertIn("FastGelu", set(n.op_type for n in opt_onx.graph.node))
        self.assertEqual(42, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

    def test_fast_gelu18(self):
        data = os.path.join(os.path.dirname(__file__), "data", "layernorm.onnx")
        model = onnx_load(data, load_external_data=False)
        del model.opset_import[:]
        model.opset_import.append(oh.make_opsetid("", 18))
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["Cast", "GeluOrt", "FastGelu"], verbose=0, constant_folding=False
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertNotIn("Gelu", set(n.op_type for n in opt_onx.graph.node))
        self.assertIn("FastGelu", set(n.op_type for n in opt_onx.graph.node))
        self.assertEqual(42, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

    def test_bias_gelu(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "B"], ["xb"]),
                    oh.make_node("Div", ["xb", "sq2"], ["xbinv"]),
                    oh.make_node("Erf", ["xbinv"], ["xerf"]),
                    oh.make_node("Add", ["xerf", "one"], ["xerf1"]),
                    oh.make_node("Mul", ["xb", "xerf1"], ["y2"]),
                    oh.make_node("Mul", ["y2", "half"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [2, 2, 4, 8])],
                [oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 4, 8])],
                [
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                    onh.from_array(np.array([0.5], dtype=np.float32), name="half"),
                    onh.from_array(np.array([1.4140625], dtype=np.float32), name="sq2"),
                    onh.from_array(
                        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.4, -0.1], dtype=np.float32),
                        name="B",
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
        feeds = {"X": self._range(2, 2, 4, 8)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["BiasGelu"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["BiasGelu"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-4)
        node = opt_onx.graph.node[0]
        self.assertEqual(node.op_type, "BiasGelu")

    def test_bias_gelu_with_conflict(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "B"], ["xb"]),
                    oh.make_node("Div", ["xb", "sq2"], ["xbinv"]),
                    oh.make_node("Erf", ["xbinv"], ["xerf"]),
                    oh.make_node("Add", ["xerf", "one"], ["xerf1"]),
                    oh.make_node("Mul", ["xb", "xerf1"], ["y2"]),
                    oh.make_node("Mul", ["y2", "half"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [2, 2, 4, 8])],
                [oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 4, 8])],
                [
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                    onh.from_array(np.array([0.5], dtype=np.float32), name="half"),
                    onh.from_array(np.array([1.4140625], dtype=np.float32), name="sq2"),
                    onh.from_array(
                        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.4, -0.1], dtype=np.float32),
                        name="B",
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
        feeds = {"X": self._range(2, 2, 4, 8)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["AddAddMulMul", "AddAddMulMulBroadcast", "BiasGelu"],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["BiasGelu"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-4)
        node = opt_onx.graph.node[0]
        self.assertEqual(node.op_type, "BiasGelu")

    def test_gelu_erf(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Div", ["X", "sq2"], ["xd"]),
                    oh.make_node("Erf", ["xd"], ["exd"]),
                    oh.make_node("Add", ["exd", "one"], ["aexd"]),
                    oh.make_node("Mul", ["X", "aexd"], ["y2"]),
                    oh.make_node("Mul", ["half", "y2"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [2, 2, 4, 8])],
                [oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 4, 8])],
                [
                    onh.from_array(np.array([1.4140625], dtype=np.float32), name="sq2"),
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                    onh.from_array(np.array([0.5], dtype=np.float32), name="half"),
                ],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 4, 8)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=[GeluErfPattern(verbose=0)],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Gelu"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-4)
        node = opt_onx.graph.node[0]
        self.assertEqual(node.op_type, "Gelu")

    @requires_cuda()
    def test_bias_softmax(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy"]),
                    oh.make_node("Softmax", ["xy"], ["Z"], axis=-1),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [16, 8, 4, 8]),
                    oh.make_tensor_value_info("Y", TFLOAT, [16, 1, 4, 8]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [16, 8, 4, 8])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        check_model(model)
        feeds = {"X": self._range(16, 8, 4, 8), "Y": self._range(16, 1, 4, 8)}
        ref = InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["BiasSoftmax"], processor="CPU,CUDA", verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["BiasSoftmax"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CUDAExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-4)

    def test_fused_conv(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Conv",
                        ["X", "W", "B"],
                        ["c"],
                        dilations=[1, 1],
                        group=1,
                        pads=[1, 1, 1, 1],
                        strides=[1, 1],
                    ),
                    oh.make_node("Relu", ["c"], ["Y"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [1, 8, 6, 6]),
                    oh.make_tensor_value_info("W", TFLOAT, [8, 8, 3, 3]),
                    oh.make_tensor_value_info("B", TFLOAT, [8]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, [1, 8, 6, 6])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        feeds = {
            "X": self._range(1, 8, 6, 6),
            "W": self._range(8, 8, 3, 3),
            "B": self._range(8),
        }
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["FusedConv"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["FusedConv"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_quick_gelu(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sigmoid", ["X"], ["S"]),
                    oh.make_node("Mul", ["X", "S"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [1, 8, 6, 6])],
                [oh.make_tensor_value_info("Y", TFLOAT, [1, 8, 6, 6])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        feeds = {"X": self._range(1, 8, 6, 6)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["QuickGelu"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["QuickGelu"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_skip_layer_normalization_1d(self):
        from onnxruntime import InferenceSession

        for itype, dtype in [(TFLOAT, np.float32), (TensorProto.FLOAT16, np.float16)]:
            model = oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node("Add", ["X1", "X2"], ["add"]),
                        oh.make_node(
                            "LayerNormalization",
                            ["add", "scale", "bias"],
                            ["Y"],
                            axis=-1,
                        ),
                    ],
                    "dummy",
                    [
                        oh.make_tensor_value_info("X1", itype, ["a", "b", "c"]),
                        oh.make_tensor_value_info("X2", itype, ["a", "b", "c"]),
                        oh.make_tensor_value_info("scale", itype, ["c"]),
                        oh.make_tensor_value_info("bias", itype, ["c"]),
                    ],
                    [
                        oh.make_tensor_value_info("add", itype, ["a", "b", "c"]),
                        oh.make_tensor_value_info("Y", itype, ["a", "b", "c"]),
                    ],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
                ir_version=9,
            )
            feeds = {
                "X1": self._range(8, 3, 32).astype(dtype),
                "X2": self._range(8, 3, 32).astype(dtype),
                "scale": self._range(32).astype(dtype),
                "bias": self._range(32).astype(dtype),
            }
            ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            expected = ref.run(None, feeds)

            gr = GraphBuilder(
                model,
                infer_shapes_options=True,
                optimization_options=OptimizationOptions(
                    patterns=["SkipLayerNormalization"], verbose=0
                ),
            )
            opt_onx = gr.to_onnx(optimize=True)
            self.assertIn("SkipLayerNormalization", [n.op_type for n in opt_onx.graph.node])

            opt_ref = InferenceSession(
                opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            got = opt_ref.run(None, feeds)
            self.assertEqualArray(expected[0].ravel(), got[0].ravel())
            self.assertEqualArray(expected[0], got[0])

    def test_skip_layer_normalization_3d(self):
        itype, _dtype = (TFLOAT, np.float32)
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X1", "X2"], ["add"]),
                    oh.make_node(
                        "LayerNormalization",
                        ["add", "scale", "bias"],
                        ["Y"],
                        axis=-1,
                    ),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X1", itype, ["a", "b", "c"]),
                    oh.make_tensor_value_info("X2", itype, ["a", "b", "c"]),
                    oh.make_tensor_value_info("scale", itype, ["a", "b", "c"]),
                    oh.make_tensor_value_info("bias", itype, ["a", "b", "c"]),
                ],
                [
                    oh.make_tensor_value_info("add", itype, ["a", "b", "c"]),
                    oh.make_tensor_value_info("Y", itype, ["a", "b", "c"]),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["SkipLayerNormalization"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertNotIn("SkipLayerNormalization", [n.op_type for n in opt_onx.graph.node])

    def test_reshape_gemm(self):
        from onnxruntime import InferenceSession

        for transB in [0, 1]:
            with self.subTest(transB=transB):

                model = oh.make_model(
                    oh.make_graph(
                        [
                            oh.make_node("Reshape", ["A", "shape"], ["xr"]),
                            oh.make_node("Gemm", ["xr", "B"], ["Y"], transB=transB),
                        ],
                        "dummy",
                        [
                            oh.make_tensor_value_info("A", TFLOAT, ["a", "b", 8]),
                            oh.make_tensor_value_info("B", TFLOAT, [4, 8] if transB else [8, 4]),
                        ],
                        [oh.make_tensor_value_info("Y", TFLOAT, ["f", "g"])],
                        [onh.from_array(np.array([-1, 8], dtype=np.int64), name="shape")],
                    ),
                    opset_imports=[oh.make_opsetid("", 18)],
                    ir_version=9,
                )
                feeds = {
                    "A": self._range(2, 3, 8),
                    "B": self._range(*([4, 8] if transB else [8, 4])),
                }
                ref = InferenceSession(
                    model.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                expected = ref.run(None, feeds)

                gr = GraphBuilder(
                    model,
                    infer_shapes_options=True,
                    optimization_options=OptimizationOptions(patterns=["ReshapeGemm"], verbose=0),
                )
                opt_onx = gr.to_onnx(optimize=True)
                self.assertIn("FusedMatMul", [n.op_type for n in opt_onx.graph.node])

                opt_ref = InferenceSession(
                    opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                got = opt_ref.run(None, feeds)
                self.assertEqualArray(expected[0].ravel(), got[0].ravel())
                self.assertEqualArray(expected[0], got[0])

    def test_reshape_gemm_reshape(self):
        for transB in [0, 1]:
            with self.subTest(transB=transB):
                model = oh.make_model(
                    oh.make_graph(
                        [
                            oh.make_node("Shape", ["A"], ["shapeA"], start=0, end=-1),
                            oh.make_node("Concat", ["shapeA", "mone"], ["shapey"], axis=0),
                            oh.make_node("Reshape", ["A", "shape"], ["xr"]),
                            oh.make_node("Gemm", ["xr", "B"], ["y2"], transB=transB),
                            oh.make_node("Reshape", ["y2", "shapey"], ["yy"]),
                            oh.make_node("Identity", ["yy"], ["Y"]),
                        ],
                        "dummy",
                        [
                            oh.make_tensor_value_info("A", TFLOAT, ["a", "b", "c"]),
                            oh.make_tensor_value_info("B", TFLOAT, [4, 8] if transB else [8, 4]),
                        ],
                        [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"])],
                        [
                            onh.from_array(np.array([-1, 8], dtype=np.int64), name="shape"),
                            onh.from_array(np.array([-1], dtype=np.int64), name="mone"),
                        ],
                    ),
                    opset_imports=[oh.make_opsetid("", 18)],
                    ir_version=9,
                )
                feeds = {
                    "A": self._range(2, 3, 8),
                    "B": self._range(*([4, 8] if transB else [8, 4])),
                }
                ref = self._check_with_ort(model)
                expected = ref.run(None, feeds)

                gr = GraphBuilder(
                    model,
                    infer_shapes_options=True,
                    optimization_options=OptimizationOptions(
                        patterns=["ReshapeGemmReshape"], verbose=0
                    ),
                )
                opt_onx = gr.to_onnx(optimize=True)
                self.assertEqual(["FusedMatMul"], [n.op_type for n in opt_onx.graph.node])

                opt_ref = self._check_with_ort(opt_onx)
                got = opt_ref.run(None, feeds)
                self.assertEqualArray(expected[0].ravel(), got[0].ravel())
                self.assertEqualArray(expected[0], got[0])

    def test_transpose_matmul_b(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["B"], ["xr"], perm=[0, 2, 3, 1]),
                    oh.make_node("MatMul", ["A", "xr"], ["Y"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("A", TFLOAT, ["a", "b", "c", "d"]),
                    oh.make_tensor_value_info("B", TFLOAT, ["i", "j", "k", "l"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["m", "n", "o", "p"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        feeds = {
            "A": self._range(2, 3, 8, 7),
            "B": self._range(2, 8, 3, 7),
        }
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["TransposeFusedMatMulB"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertIn("FusedMatMul", [n.op_type for n in opt_onx.graph.node])

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0].ravel(), got[0].ravel())
        self.assertEqualArray(expected[0], got[0])

    def test_skip_simplified_layer_normalization(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "skip"], ["xs"]),
                    oh.make_node(
                        "SimplifiedLayerNormalization",
                        ["xs", "scale"],
                        ["Y"],
                        epsilon=1e-1,
                        axis=-1,
                    ),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "cache", 192]),
                    oh.make_tensor_value_info("skip", TFLOAT, ["batch", "cache", 192]),
                ],
                [
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "cache", 192]),
                    oh.make_tensor_value_info("xs", TFLOAT, ["batch", "cache", 192]),
                ],
                [onh.from_array(np.ones(192, dtype=np.float32), name="scale")],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        feeds = {"X": self._range(2, 128, 192), "skip": self._range(2, 128, 192)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["SkipSimplifiedLayerNormalization"]
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["SkipSimplifiedLayerNormalization"], [n.op_type for n in opt_onx.graph.node]
        )
        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualAny(expected, got)

    def test_skip_simplified_layer_normalization_mul(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "skip"], ["xs"]),
                    oh.make_node(
                        "SimplifiedLayerNormalization",
                        ["xs", "scale"],
                        ["ym"],
                        epsilon=1e-1,
                        axis=-1,
                    ),
                    oh.make_node("Mul", ["ym", "weights"], ["a"]),
                    oh.make_node("Add", ["a", "weights"], ["Y"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "cache", 192]),
                    oh.make_tensor_value_info("skip", TFLOAT, ["batch", "cache", 192]),
                ],
                [
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "cache", 192]),
                    oh.make_tensor_value_info("xs", TFLOAT, ["batch", "cache", 192]),
                ],
                [
                    onh.from_array(np.ones(192, dtype=np.float32), name="scale"),
                    onh.from_array(self._range(192, bias=1000), name="weights"),
                ],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["SkipSimplifiedLayerNormalization"]
            ),
        )
        self.dump_onnx("test_skip_simplified_layer_normalization_mul0.onnx", model)
        model2 = gr.to_onnx(optimize=True)
        feeds = {"X": self._range(2, 128, 192, bias=0.001), "skip": self._range(2, 128, 192)}
        self.dump_onnx("test_skip_simplified_layer_normalization_mul1.onnx", model2)

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)
        ref2 = InferenceSession(model2.SerializeToString(), providers=["CPUExecutionProvider"])
        expected2 = ref2.run(None, feeds)
        self.assertEqualAny(expected, expected2)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=[
                    "SkipSimplifiedLayerNormalization",
                    "SkipSimplifiedLayerNormalizationMul",
                ],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["SkipSimplifiedLayerNormalization", "Add"], [n.op_type for n in opt_onx.graph.node]
        )
        self.dump_onnx("test_skip_simplified_layer_normalization_mul.onnx", opt_onx)
        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)

        self.assertEqualArray(expected2[1], got[1])
        self.assertEqualArray(expected2[0], got[0])

    def test_simplified_layer_normalization_mul(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "skip"], ["xs"]),
                    oh.make_node(
                        "SimplifiedLayerNormalization",
                        ["xs", "scale"],
                        ["ym"],
                        epsilon=1e-1,
                        axis=-1,
                    ),
                    oh.make_node("Mul", ["ym", "weights"], ["a"]),
                    oh.make_node("Add", ["a", "weights"], ["Y"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "cache", 192]),
                    oh.make_tensor_value_info("skip", TFLOAT, ["batch", "cache", 192]),
                ],
                [
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "cache", 192]),
                    oh.make_tensor_value_info("xs", TFLOAT, ["batch", "cache", 192]),
                ],
                [
                    onh.from_array(np.ones(192, dtype=np.float32), name="scale"),
                    onh.from_array(self._range(192, bias=1000), name="weights"),
                ],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        feeds = {"X": self._range(2, 128, 192, bias=0.001), "skip": self._range(2, 128, 192)}
        sess = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = sess.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["SimplifiedLayerNormalizationMul"],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Add", "SimplifiedLayerNormalization", "Add"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.dump_onnx("test_simplified_layer_normalization_mul.onnx", opt_onx)
        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)

        self.assertEqualArray(expected[1], got[1])
        self.assertEqualArray(expected[0], got[0])

    def test_contrib_rotary_embedding_concat_after(self):
        opset = 20
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Concat", ["m1", "m1"], ["m1x2"], axis=-1),
                    oh.make_node("Concat", ["m2", "m2"], ["m2x2"], axis=-1),
                    oh.make_node("Split", ["X", "split"], ["Xh1", "Xh2"], axis=-1),
                    oh.make_node("Split", ["Xh1"], ["x1", "x2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["x2"], ["nx2"]),
                    oh.make_node("Concat", ["nx2", "x1"], ["cc"], axis=-1),
                    oh.make_node("Mul", ["cc", "m1x2"], ["cm1"]),
                    oh.make_node("Mul", ["Xh1", "m2x2"], ["cm2"]),
                    oh.make_node("Add", ["cm1", "cm2"], ["Yh"]),
                    oh.make_node("Concat", ["Yh", "Xh2"], ["Y"], axis=-1),
                ],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", 2, "c", "d"]),
                    oh.make_tensor_value_info("m1", TFLOAT, [1, 1, "c", "e"]),
                    oh.make_tensor_value_info("m2", TFLOAT, [1, 1, "c", "e"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "d"])],
                [onh.from_array(np.array([4, 6], dtype=np.int64), name="split")],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )

        shape_x = (2, 2, 3, 10)
        shape_c = (1, 1, 3, 2)
        feeds = {
            "X": ((np.arange(np.prod(shape_x)) + 1) / (np.prod(shape_x) * 10))
            .reshape(shape_x)
            .astype(np.float32),
            "m1": ((np.arange(np.prod(shape_c)) + 1) / np.prod(shape_c) * 15)
            .reshape(shape_c)
            .astype(np.float32),
            "m2": ((np.arange(np.prod(shape_c)) + 1) / np.prod(shape_c) * 5)
            .reshape(shape_c)
            .astype(np.float32),
        }
        # ExtendedReferenceEvaluator(model, verbose=10).run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=["FunctionHalfRotaryEmbedding", "ContribRotaryEmbedding"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_contrib_rotary_embedding_concat_after.onnx", opt_onx)
        self.assertIn("RotaryEmbedding", [n.op_type for n in opt_onx.graph.node])
        self.assertIn("com.microsoft", [n.domain for n in opt_onx.graph.node])

        import onnxruntime

        ref = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        z = ref.run(None, feeds)[0]
        ref = onnxruntime.InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        zz = ref.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

    def test_contrib_rotary_embedding_no_concat_after(self):
        opset = 20
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Concat", ["m1", "m1"], ["m1x2"], axis=-1),
                    oh.make_node("Concat", ["m2", "m2"], ["m2x2"], axis=-1),
                    oh.make_node("Split", ["X"], ["x1", "x2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["x2"], ["nx2"]),
                    oh.make_node("Concat", ["nx2", "x1"], ["cc"], axis=-1),
                    oh.make_node("Mul", ["cc", "m1x2"], ["cm1"]),
                    oh.make_node("Mul", ["X", "m2x2"], ["cm2"]),
                    oh.make_node("Add", ["cm1", "cm2"], ["Y"]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", 2, "c", "e*2"]),
                    oh.make_tensor_value_info("m1", TFLOAT, [1, 1, "c", "e"]),
                    oh.make_tensor_value_info("m2", TFLOAT, [1, 1, "c", "e"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "e*2"])],
                [onh.from_array(np.array([4, 6], dtype=np.int64), name="split")],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )

        shape_x = (2, 2, 3, 16)
        shape_c = (1, 1, 3, 8)
        feeds = {
            "X": ((np.arange(np.prod(shape_x)) + 1) / (np.prod(shape_x) * 10))
            .reshape(shape_x)
            .astype(np.float32),
            "m1": ((np.arange(np.prod(shape_c)) + 1) / np.prod(shape_c) * 15)
            .reshape(shape_c)
            .astype(np.float32),
            "m2": ((np.arange(np.prod(shape_c)) + 1) / np.prod(shape_c) * 5)
            .reshape(shape_c)
            .astype(np.float32),
        }
        # ExtendedReferenceEvaluator(model, verbose=10).run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=["FunctionHalfRotaryEmbedding", "ContribRotaryEmbedding"], verbose=10
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_contrib_rotary_embedding_no_concat_after.onnx", opt_onx)
        self.assertIn("RotaryEmbedding", [n.op_type for n in opt_onx.graph.node])
        self.assertIn("Concat", [n.op_type for n in opt_onx.graph.node])
        self.assertIn("com.microsoft", [n.domain for n in opt_onx.graph.node])

        import onnxruntime

        ref = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        z = ref.run(None, feeds)[0]
        ref = onnxruntime.InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        zz = ref.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

    @hide_stdout()
    def test_missing_kernels(self):
        opset = 20
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cast", ["X"], ["xc"], to=TensorProto.BFLOAT16),
                    oh.make_node("Sin", ["xc"], ["xcs"]),
                    oh.make_node("Cos", ["xc"], ["xcc"]),
                    oh.make_node("Cast", ["zero"], ["zeroc"], to=TensorProto.BFLOAT16),
                    oh.make_node("Cast", ["one"], ["onec"], to=TensorProto.BFLOAT16),
                    oh.make_node("Cast", ["five"], ["fivec"], to=TensorProto.BFLOAT16),
                    oh.make_node("Range", ["zeroc", "onec", "fivec"], ["y"]),
                    oh.make_node("Add", ["xcc", "xcs"], ["xccc"]),
                    oh.make_node("Add", ["xccc", "y"], ["yc"]),
                    oh.make_node("Cast", ["yc"], ["Y"], to=TensorProto.FLOAT),
                ],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, ["a"])],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a"])],
                [
                    onh.from_array(np.array(0, dtype=np.int64), name="zero"),
                    onh.from_array(np.array(1, dtype=np.int64), name="one"),
                    onh.from_array(np.array(5, dtype=np.int64), name="five"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )

        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=["MissingRange", "MissingCosSin"], verbose=10
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            [
                "Cast",
                "Cast",
                "Sin",
                "Cast",
                "Cast",
                "Cos",
                "Cast",
                "Range",
                "Cast",
                "Add",
                "Add",
                "Cast",
            ],
            [n.op_type for n in opt_onx.graph.node],
        )

        import onnxruntime

        # feeds = {"X": np.arange(5).astype(np.float32)}
        self.assertRaise(
            lambda: self.make_inference_session(model),
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph,
        )
        self.assertRaise(
            lambda: self.make_inference_session(opt_onx),
            onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented,  # on CPU
        )
        # sess = self.make_inference_session(opt_onx)
        # got = sess.run(None, feeds)
        # self.assertEqualArray(np.array([0, 1, 2, 3, 4], dtype=np.float32), got)

    def test_contrib_rotary_embedding_concat_after_position_ids(self):
        opset = 20
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["position_ids"], ["batch"], start=0, end=1),
                    oh.make_node("Concat", ["batch", "init11"], ["new_shape"], axis=0),
                    oh.make_node("Unsqueeze", ["weights", "init01"], ["weights_u"]),
                    oh.make_node("Expand", ["weights_u", "new_shape"], ["weights_expanded"]),
                    oh.make_node("Unsqueeze", ["position_ids", "one"], ["pids1"]),
                    oh.make_node("Cast", ["pids1"], ["cids"], to=TensorProto.FLOAT),
                    oh.make_node("Reshape", ["cids", "init0_11"], ["resh"]),
                    oh.make_node("Mul", ["weights_expanded", "resh"], ["milti"]),
                    oh.make_node("Cos", ["milti"], ["m1s"]),
                    oh.make_node("Sin", ["milti"], ["m2s"]),
                    oh.make_node("Unsqueeze", ["m1s", "one"], ["m1"]),
                    oh.make_node("Unsqueeze", ["m2s", "one"], ["m2"]),
                    oh.make_node("Concat", ["m1", "m1"], ["m1x2"], axis=-1),
                    oh.make_node("Concat", ["m2", "m2"], ["m2x2"], axis=-1),
                    oh.make_node("Split", ["X", "split"], ["Xh1", "Xh2"], axis=-1),
                    oh.make_node("Split", ["Xh1"], ["x1", "x2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["x2"], ["nx2"]),
                    oh.make_node("Concat", ["nx2", "x1"], ["cc"], axis=-1),
                    oh.make_node("Mul", ["cc", "m2x2"], ["cm1"]),
                    oh.make_node("Mul", ["Xh1", "m1x2"], ["cm2"]),
                    oh.make_node("Add", ["cm1", "cm2"], ["Yh"]),
                    oh.make_node("Concat", ["Yh", "Xh2"], ["Y"], axis=-1),
                ],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", 2, "c", "d"]),
                    oh.make_tensor_value_info("position_ids", TINT64, ["a", "e"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "d"])],
                [
                    onh.from_array(np.array([4, 6], dtype=np.int64), name="split"),
                    onh.from_array(np.array([1], dtype=np.int64), name="one"),
                    onh.from_array(np.array([0, -1, 1], dtype=np.int64), name="init0_11"),
                    onh.from_array(np.array(1, dtype=np.int64), name="one_no_dim"),
                    onh.from_array(np.array([0.1, 0.2], dtype=np.float32), name="weights"),
                    onh.from_array(np.array([0, 1], dtype=np.int64), name="init01"),
                    onh.from_array(np.array([1, 1], dtype=np.int64), name="init11"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )

        shape_x = (2, 2, 3, 10)
        feeds = {
            "X": ((np.arange(np.prod(shape_x)) + 1) / (np.prod(shape_x) * 10))
            .reshape(shape_x)
            .astype(np.float32),
            "position_ids": np.array([[1, 3, 6], [2, 4, 7]], dtype=np.int64),
        }
        ExtendedReferenceEvaluator(model, verbose=0).run(None, feeds)
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=["FunctionCosSinCache", "FunctionHalfRotaryEmbedding"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_contrib_rotary_embedding_concat_after_position_ids0.onnx", opt_onx)
        ref = self.make_inference_session(model)
        z = ref.run(None, feeds)[0]
        sess = self.make_inference_session(opt_onx)
        zz = sess.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=[
                    "FunctionCosSinCache",
                    "FunctionHalfRotaryEmbedding",
                    "ContribRotaryEmbedding",
                ],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_contrib_rotary_embedding_concat_after_position_ids.onnx", opt_onx)
        self.assertIn("RotaryEmbedding", [n.op_type for n in opt_onx.graph.node])
        self.assertIn("com.microsoft", [n.domain for n in opt_onx.graph.node])
        sess = self.make_inference_session(opt_onx)
        zz = sess.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

    def test_contrib_rotary_embedding_concat_after_position_ids_3d(self):
        opset = 20
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["Xt"], perm=[0, 2, 1, 3]),
                    oh.make_node("Shape", ["position_ids"], ["batch"], start=0, end=1),
                    oh.make_node("Concat", ["batch", "init11"], ["new_shape"], axis=0),
                    oh.make_node("Unsqueeze", ["weights", "init01"], ["weights_u"]),
                    oh.make_node("Expand", ["weights_u", "new_shape"], ["weights_expanded"]),
                    oh.make_node("Unsqueeze", ["position_ids", "one"], ["pids1"]),
                    oh.make_node("Cast", ["pids1"], ["cids"], to=TensorProto.FLOAT),
                    oh.make_node("Reshape", ["cids", "init0_11"], ["resh"]),
                    oh.make_node("Mul", ["weights_expanded", "resh"], ["milti"]),
                    oh.make_node("Cos", ["milti"], ["m1s"]),
                    oh.make_node("Sin", ["milti"], ["m2s"]),
                    oh.make_node("Unsqueeze", ["m1s", "one"], ["m1"]),
                    oh.make_node("Unsqueeze", ["m2s", "one"], ["m2"]),
                    oh.make_node("Concat", ["m1", "m1"], ["m1x2"], axis=-1),
                    oh.make_node("Concat", ["m2", "m2"], ["m2x2"], axis=-1),
                    oh.make_node("Split", ["Xt", "split"], ["Xh1", "Xh2"], axis=-1),
                    oh.make_node("Split", ["Xh1"], ["x1", "x2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["x2"], ["nx2"]),
                    oh.make_node("Concat", ["nx2", "x1"], ["cc"], axis=-1),
                    oh.make_node("Mul", ["cc", "m2x2"], ["cm1"]),
                    oh.make_node("Mul", ["Xh1", "m1x2"], ["cm2"]),
                    oh.make_node("Add", ["cm1", "cm2"], ["Yh"]),
                    oh.make_node("Concat", ["Yh", "Xh2"], ["Y"], axis=-1),
                ],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "c", 2, "d"]),
                    oh.make_tensor_value_info("position_ids", TINT64, ["a", "e"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "d"])],
                [
                    onh.from_array(np.array([4, 6], dtype=np.int64), name="split"),
                    onh.from_array(np.array([1], dtype=np.int64), name="one"),
                    onh.from_array(np.array([0, -1, 1], dtype=np.int64), name="init0_11"),
                    onh.from_array(np.array(1, dtype=np.int64), name="one_no_dim"),
                    onh.from_array(np.array([0.1, 0.2], dtype=np.float32), name="weights"),
                    onh.from_array(np.array([0, 1], dtype=np.int64), name="init01"),
                    onh.from_array(np.array([1, 1], dtype=np.int64), name="init11"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )

        shape_x = (2, 3, 2, 10)
        feeds = {
            "X": ((np.arange(np.prod(shape_x)) + 1) / (np.prod(shape_x) * 10))
            .reshape(shape_x)
            .astype(np.float32),
            "position_ids": np.array([[1, 3, 6], [2, 4, 7]], dtype=np.int64),
        }
        ExtendedReferenceEvaluator(model, verbose=0).run(None, feeds)
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=[
                    "FunctionCosSinCache",
                    "FunctionHalfRotaryEmbedding",
                    "ContribRotaryEmbedding",
                ],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx(
            "test_contrib_rotary_embedding_concat_after_position_ids_3d0.onnx", opt_onx
        )
        ref = self.make_inference_session(model)
        z = ref.run(None, feeds)[0]
        sess = self.make_inference_session(opt_onx)
        zz = sess.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=[
                    "FunctionCosSinCache",
                    "FunctionHalfRotaryEmbedding",
                    "ContribRotaryEmbedding",
                    "ContribRotaryEmbedding3D",
                ],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_contrib_rotary_embedding_concat_after_position_ids_3d.onnx", opt_onx)
        self.assertIn("RotaryEmbedding", [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(["Transpose"], [n.op_type for n in opt_onx.graph.node][-1:])
        self.assertIn("com.microsoft", [n.domain for n in opt_onx.graph.node])
        sess = self.make_inference_session(opt_onx)
        zz = sess.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

    def test_contrib_rotary_embedding_concat_after_position_ids_swap(self):
        opset = 20
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["position_ids"], ["batch"], start=0, end=1),
                    oh.make_node("Concat", ["batch", "init11"], ["new_shape"], axis=0),
                    oh.make_node("Unsqueeze", ["weights", "init02"], ["weights_u"]),
                    oh.make_node("Expand", ["weights_u", "new_shape"], ["weights_expandede"]),
                    oh.make_node(
                        "Reshape", ["weights_expandede", "init01_1"], ["weights_expanded"]
                    ),
                    oh.make_node("Unsqueeze", ["position_ids", "one"], ["pids1"]),
                    oh.make_node("Cast", ["pids1"], ["cids"], to=TensorProto.FLOAT),
                    oh.make_node("Reshape", ["cids", "init0_11"], ["resh"]),
                    oh.make_node("Mul", ["weights_expanded", "resh"], ["milti"]),
                    oh.make_node("Cos", ["milti"], ["m1s"]),
                    oh.make_node("Sin", ["milti"], ["m2s"]),
                    oh.make_node("Unsqueeze", ["m1s", "one"], ["m1"]),
                    oh.make_node("Unsqueeze", ["m2s", "one"], ["m2"]),
                    oh.make_node("Concat", ["m1", "m1"], ["m1x2"], axis=-1),
                    oh.make_node("Concat", ["m2", "m2"], ["m2x2"], axis=-1),
                    oh.make_node("Split", ["X", "split"], ["Xh1", "Xh2"], axis=-1),
                    oh.make_node("Split", ["Xh1"], ["x1", "x2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["x2"], ["nx2"]),
                    oh.make_node("Concat", ["nx2", "x1"], ["cc"], axis=-1),
                    oh.make_node("Mul", ["cc", "m2x2"], ["cm1"]),
                    oh.make_node("Mul", ["Xh1", "m1x2"], ["cm2"]),
                    oh.make_node("Add", ["cm1", "cm2"], ["Yh"]),
                    oh.make_node("Concat", ["Yh", "Xh2"], ["Y"], axis=-1),
                ],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", 2, "c", "d"]),
                    oh.make_tensor_value_info("position_ids", TINT64, ["a", "e"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "d"])],
                [
                    onh.from_array(np.array([4, 6], dtype=np.int64), name="split"),
                    onh.from_array(np.array([1], dtype=np.int64), name="one"),
                    onh.from_array(np.array([0, -1, 1], dtype=np.int64), name="init0_11"),
                    onh.from_array(np.array([0, 1, -1], dtype=np.int64), name="init01_1"),
                    onh.from_array(np.array(1, dtype=np.int64), name="one_no_dim"),
                    onh.from_array(np.array([0.1, 0.2], dtype=np.float32), name="weights"),
                    onh.from_array(np.array([0, 2], dtype=np.int64), name="init02"),
                    onh.from_array(np.array([1, 1], dtype=np.int64), name="init11"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )

        shape_x = (2, 2, 3, 10)
        feeds = {
            "X": ((np.arange(np.prod(shape_x)) + 1) / (np.prod(shape_x) * 10))
            .reshape(shape_x)
            .astype(np.float32),
            "position_ids": np.array([[1, 3, 6], [2, 4, 7]], dtype=np.int64),
        }
        ExtendedReferenceEvaluator(model, verbose=0).run(None, feeds)
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=[
                    "SwapExpandReshape",
                    "FunctionCosSinCache",
                    "FunctionHalfRotaryEmbedding",
                ],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx(
            "test_contrib_rotary_embedding_concat_after_position_ids_swap0.onnx", opt_onx
        )
        ref = self.make_inference_session(model)
        z = ref.run(None, feeds)[0]
        sess = self.make_inference_session(opt_onx)
        zz = sess.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=[
                    "SwapExpandReshape",
                    "FunctionCosSinCache",
                    "FunctionHalfRotaryEmbedding",
                    "ContribRotaryEmbedding",
                ],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx(
            "test_contrib_rotary_embedding_concat_after_position_ids_swap.onnx", opt_onx
        )
        self.assertIn("RotaryEmbedding", [n.op_type for n in opt_onx.graph.node])
        self.assertIn("com.microsoft", [n.domain for n in opt_onx.graph.node])
        sess = self.make_inference_session(opt_onx)
        zz = sess.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

    def test_contrib_rotary_embedding_concat_after_position_ids_no_match(self):
        opset = 20
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["position_ids"], ["batch"], start=0, end=1),
                    oh.make_node("Concat", ["batch", "init11"], ["new_shape"], axis=0),
                    oh.make_node("Unsqueeze", ["weights", "init01"], ["weights_u"]),
                    oh.make_node("Expand", ["weights_u", "new_shape"], ["weights_expanded"]),
                    oh.make_node("Unsqueeze", ["position_ids", "one"], ["pids1"]),
                    oh.make_node("Cast", ["pids1"], ["cids"], to=TensorProto.FLOAT),
                    oh.make_node("Reshape", ["cids", "init0_11"], ["resh"]),
                    oh.make_node("Mul", ["weights_expanded", "resh"], ["milti"]),
                    oh.make_node("Cos", ["milti"], ["m1s"]),
                    oh.make_node("Sin", ["milti"], ["m2s"]),
                    oh.make_node("Unsqueeze", ["m1s", "one"], ["m1"]),
                    oh.make_node("Unsqueeze", ["m2s", "one"], ["m2"]),
                    oh.make_node("Concat", ["m1", "m1"], ["m1x2"], axis=-1),
                    oh.make_node("Concat", ["m2", "m2"], ["m2x2"], axis=-1),
                    oh.make_node("Split", ["X", "split"], ["Xh1", "Xh2"], axis=-1),
                    oh.make_node("Split", ["Xh1"], ["x1", "x2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["x2"], ["nx2"]),
                    oh.make_node("Concat", ["nx2", "x1"], ["cc"], axis=-1),
                    oh.make_node("Mul", ["cc", "m1x2"], ["cm1"]),
                    oh.make_node("Mul", ["Xh1", "m2x2"], ["cm2"]),
                    oh.make_node("Add", ["cm1", "cm2"], ["Yh"]),
                    oh.make_node("Concat", ["Yh", "Xh2"], ["Y"], axis=-1),
                ],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", 2, "c", "d"]),
                    oh.make_tensor_value_info("position_ids", TINT64, ["a", "e"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "d"])],
                [
                    onh.from_array(np.array([4, 6], dtype=np.int64), name="split"),
                    onh.from_array(np.array([1], dtype=np.int64), name="one"),
                    onh.from_array(np.array([0, -1, 1], dtype=np.int64), name="init0_11"),
                    onh.from_array(np.array(1, dtype=np.int64), name="one_no_dim"),
                    onh.from_array(np.array([0.1, 0.2], dtype=np.float32), name="weights"),
                    onh.from_array(np.array([0, 1], dtype=np.int64), name="init01"),
                    onh.from_array(np.array([1, 1], dtype=np.int64), name="init11"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=["FunctionCosSinCache", "FunctionHalfRotaryEmbedding"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx(
            "test_contrib_rotary_embedding_concat_after_position_ids_no_match.onnx", opt_onx
        )

        shape_x = (2, 2, 3, 10)
        feeds = {
            "X": ((np.arange(np.prod(shape_x)) + 1) / (np.prod(shape_x) * 10))
            .reshape(shape_x)
            .astype(np.float32),
            "position_ids": np.array([[1, 3, 6], [2, 4, 7]], dtype=np.int64),
        }
        ExtendedReferenceEvaluator(model, verbose=0).run(None, feeds)
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=[
                    "FunctionCosSinCache",
                    "FunctionHalfRotaryEmbedding",
                    "ContribRotaryEmbedding",
                ],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertNotIn("RotaryEmbedding", [n.op_type for n in opt_onx.graph.node])

    def test_multi_head_attention(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["query"], ["t_query"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["keys"], ["t_keys"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["values"], ["t_values"], perm=[0, 2, 1, 3]),
                    oh.make_node("Concat", ["past_keys", "t_keys"], ["ct_keys"], axis=-2),
                    oh.make_node("Concat", ["past_values", "t_values"], ["ct_values"], axis=-2),
                    oh.make_node("Mul", ["t_query", "scale_sqrt"], ["query_scaled"]),
                    oh.make_node("Mul", ["ct_keys", "scale_sqrt"], ["keys_scaled"]),
                    oh.make_node(
                        "Transpose", ["keys_scaled"], ["keys_scaled_t"], perm=[0, 1, 3, 2]
                    ),
                    oh.make_node("MatMul", ["query_scaled", "keys_scaled_t"], ["qk"]),
                    oh.make_node("Where", ["mask", "zero", "minfty"], ["bias"]),
                    oh.make_node("Add", ["qk", "bias"], ["qkb"]),
                    oh.make_node("Softmax", ["qkb"], ["qkbs"], axis=-1),
                    oh.make_node("IsNaN", ["qkbs"], ["nans"]),
                    oh.make_node("Where", ["nans", "zero", "qkbs"], ["filt"]),
                    oh.make_node("MatMul", ["filt", "ct_values"], ["prob"]),
                    oh.make_node("Transpose", ["prob"], ["Y"], perm=[0, 2, 1, 3]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("query", TFLOAT, ["aq", "bq", 8, 64]),
                    oh.make_tensor_value_info("keys", TFLOAT, ["ak", "bk", 8, 64]),
                    oh.make_tensor_value_info("values", TFLOAT, ["av", "bv", 8, 64]),
                    oh.make_tensor_value_info("past_keys", TFLOAT, ["pak", 8, "pck", 64]),
                    oh.make_tensor_value_info("past_values", TFLOAT, ["pav", 8, "pcv", 64]),
                    oh.make_tensor_value_info("mask", TensorProto.BOOL, ["am", 1, "cm", "dm"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["ay", "by", "cy", "dy"])],
                [
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(np.array([-np.inf], dtype=np.float32), name="minfty"),
                    onh.from_array(np.array([0.1**0.5], dtype=np.float32), name="scale_sqrt"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", 18)],
            ir_version=10,
        )
        feeds = dict(
            query=np.random.randn(32, 128, 8, 64).astype(np.float32),
            keys=np.random.randn(32, 128, 8, 64).astype(np.float32),
            values=np.random.randn(32, 128, 8, 64).astype(np.float32),
            mask=np.random.rand(32, 1, 128, 256) >= 0.5,
            past_keys=np.random.randn(32, 8, 128, 64).astype(np.float32),
            past_values=np.random.randn(32, 8, 128, 64).astype(np.float32),
        )
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        z = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FunctionAttention", "MultiHeadAttention3D"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_multi_head_attention.onnx", opt_onx)
        self.assertEqual(
            ["Reshape", "Reshape", "Reshape", "Where", "MultiHeadAttention", "Reshape"],
            [n.op_type for n in opt_onx.graph.node],
        )
        ref = self.make_inference_session(opt_onx)
        zz = ref.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

    def test_multi_head_attention_where_add(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["query"], ["t_query"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["keys"], ["t_keys"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["values"], ["t_values"], perm=[0, 2, 1, 3]),
                    oh.make_node("Concat", ["past_keys", "t_keys"], ["ct_keys"], axis=-2),
                    oh.make_node("Concat", ["past_values", "t_values"], ["ct_values"], axis=-2),
                    oh.make_node("Mul", ["t_query", "scale_sqrt"], ["query_scaled"]),
                    oh.make_node("Mul", ["ct_keys", "scale_sqrt"], ["keys_scaled"]),
                    oh.make_node(
                        "Transpose", ["keys_scaled"], ["keys_scaled_t"], perm=[0, 1, 3, 2]
                    ),
                    oh.make_node("MatMul", ["query_scaled", "keys_scaled_t"], ["qk"]),
                    oh.make_node("Where", ["mask", "zero", "minfty"], ["bias"]),
                    oh.make_node("Add", ["qk", "bias"], ["qkb"]),
                    oh.make_node("Softmax", ["qkb"], ["qkbs"], axis=-1),
                    oh.make_node("IsNaN", ["qkbs"], ["nans"]),
                    oh.make_node("Where", ["nans", "zero", "qkbs"], ["filt"]),
                    oh.make_node("MatMul", ["filt", "ct_values"], ["prob"]),
                    oh.make_node("Transpose", ["prob"], ["Y"], perm=[0, 2, 1, 3]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("query", TFLOAT, ["aq", "bq", 8, 64]),
                    oh.make_tensor_value_info("keys", TFLOAT, ["ak", "bk", 8, 64]),
                    oh.make_tensor_value_info("values", TFLOAT, ["av", "bv", 8, 64]),
                    oh.make_tensor_value_info("past_keys", TFLOAT, ["pak", 8, "pck", 64]),
                    oh.make_tensor_value_info("past_values", TFLOAT, ["pav", 8, "pcv", 64]),
                    oh.make_tensor_value_info("mask", TensorProto.BOOL, ["am", 1, "cm", "dm"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["ay", "by", "cy", "dy"])],
                [
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(np.array([-np.inf], dtype=np.float32), name="minfty"),
                    onh.from_array(np.array([0.1**0.5], dtype=np.float32), name="scale_sqrt"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", 18)],
            ir_version=10,
        )
        feeds = dict(
            query=np.random.randn(32, 128, 8, 64).astype(np.float32),
            keys=np.random.randn(32, 128, 8, 64).astype(np.float32),
            values=np.random.randn(32, 128, 8, 64).astype(np.float32),
            mask=np.random.rand(32, 1, 128, 256) >= 0.5,
            past_keys=np.random.randn(32, 8, 128, 64).astype(np.float32),
            past_values=np.random.randn(32, 8, 128, 64).astype(np.float32),
        )
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        z = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FunctionAttention", "MultiHeadAttention3D"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_multi_head_attention.onnx", opt_onx)
        self.assertEqual(
            ["Reshape", "Reshape", "Reshape", "Where", "MultiHeadAttention", "Reshape"],
            [n.op_type for n in opt_onx.graph.node],
        )
        ref = self.make_inference_session(opt_onx)
        zz = ref.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

    def _get_model_attention(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(
                self,
                sequence_length,
                num_heads,
                kv_num_heads,
                head_size,
                softmax_scale,
                use_smooth_softmax,
            ):
                super().__init__()
                self.sequence_length = sequence_length
                self.num_heads = num_heads
                self.kv_num_heads = kv_num_heads
                self.head_size = head_size
                self.softmax_scale = softmax_scale
                self.use_smooth_softmax = use_smooth_softmax

            def concat_cache(self, past_key_cache, new_key):
                assert past_key_cache.size(0) == new_key.size(
                    0
                ), f"Batch sizes do not match, {past_key_cache.shape=}, {new_key.shape=}"
                assert past_key_cache.size(1) == new_key.size(
                    1
                ), f"Number of heads do not match, {past_key_cache.shape=}, {new_key.shape=}"
                assert past_key_cache.size(3) == new_key.size(
                    3
                ), f"Head dimensions do not match, {past_key_cache.shape=}, {new_key.shape=}"
                concatenated_keys = torch.cat((past_key_cache, new_key), dim=2)
                return concatenated_keys

            def smooth_softmax_ref(self, x, head_sink):
                assert len(x.shape) == 4
                b, n, s, _t = x.shape

                if head_sink is not None:
                    assert len(head_sink.shape) == 1
                    assert head_sink.shape[0] == x.shape[1]
                    sink = head_sink.reshape(1, n, 1, 1).expand(b, -1, s, -1)
                else:
                    sink = torch.zeros(b, n, s, 1, dtype=x.dtype)

                y = torch.cat([x, sink], dim=-1)
                y = torch.softmax(y, dim=-1)
                y = y[..., :-1]
                return y

            def group_query_attention_reference(
                self,
                query,
                key,
                value,
                scale=None,
                mask=None,
            ):
                if scale is None:
                    scale = 1.0 / (self.head_size**0.5)

                num_key_value_groups = self.num_heads // self.kv_num_heads
                value = torch.repeat_interleave(value, dim=1, repeats=num_key_value_groups)
                key = torch.repeat_interleave(key, dim=1, repeats=num_key_value_groups)
                # attn = torch.einsum("bhmd,bhnd->bhmn", query, key).float() * scale
                attn = torch.matmul(query * scale**0.5, key.transpose(2, 3) * scale**0.5)
                if mask is not None:
                    attn = attn.masked_fill(~mask, float("-inf")).to(query.dtype)
                # the exporter does not like this.
                # torch._check(attn.max().item() > -10000, lambda: "mask is only False")

                attn = (
                    self.smooth_softmax_ref(attn, None)
                    if self.use_smooth_softmax
                    else attn.softmax(-1)
                )
                attn = torch.where(
                    attn.isnan(), torch.tensor(0, dtype=query.dtype, device=query.device), attn
                )
                # attn_output = torch.einsum("bhmn,bhnd->bhmd", attn.type_as(value), value)
                attn_output = torch.matmul(attn, value)
                result = attn_output
                return result

            def forward(self, query, key, value, attention_mask, past_key, past_value):
                present_key = self.concat_cache(past_key, key)
                present_value = self.concat_cache(past_value, value)
                return self.group_query_attention_reference(
                    query,
                    present_key,
                    present_value,
                    scale=self.softmax_scale,
                    mask=attention_mask,
                )

        model = Model(
            sequence_length=23,
            num_heads=8,
            kv_num_heads=4,
            head_size=32,
            softmax_scale=None,
            use_smooth_softmax=False,
        ).eval()

        query = torch.rand((1, model.num_heads, model.sequence_length, model.head_size))
        key = torch.rand((1, model.kv_num_heads, model.sequence_length, model.head_size))
        value = torch.rand((1, model.kv_num_heads, model.sequence_length, model.head_size))
        past_key = torch.rand(
            (1, model.kv_num_heads, model.sequence_length // 2, model.head_size)
        )
        past_value = torch.rand(
            (1, model.kv_num_heads, model.sequence_length // 2, model.head_size)
        )
        attention_mask = torch.randint(
            0, 2, (model.sequence_length, model.sequence_length + model.sequence_length // 2)
        ).to(bool)

        # query[:, :, :, :] = 1
        # key[:, :, :, :] = 1
        value[:, :, :, :] = 0.7
        # past_key[:, :, :, :] = 1
        past_value[:, :, :, :] = 0.7
        # attention_mask[:, :] = False

        inputs = (query, key, value, attention_mask, past_key, past_value)
        expected = model.forward(*inputs)
        self.assertEqual((1, 8, 23, 32), expected.shape)
        ds = (
            {0: "batch", 2: "seq_length"},
            {0: "batch", 2: "seq_length"},
            {0: "batch", 2: "seq_length"},
            {0: "seq_length", 1: "total_length"},
            {0: "batch", 2: "past_length"},
            {0: "batch", 2: "past_length"},
        )
        return model, inputs, ds, expected

    @ignore_warnings((UserWarning, FutureWarning))
    @hide_stdout()
    def test_gqa_default(self):
        model, inputs, ds, expected = self._get_model_attention()
        f1 = self.get_dump_file("test_gqa.default.default.custom.onnx")
        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=ds,
            filename=f1,
            options=OptimizationOptions(patterns="default"),
        )
        # self.assertEqual(["Attention"], [f.name for f in onx.functions])
        ort = self._check_with_ort(onx, cpu=True)
        feeds = dict(zip([i.name for i in onx.graph.input], [t.detach().numpy() for t in inputs]))
        got = ort.run(None, feeds)
        self.assertEqualArray(expected, got[0], 1e-5)

    @ignore_warnings((UserWarning, FutureWarning))
    @hide_stdout()
    def test_gqa_ort(self):
        model, inputs, ds, expected = self._get_model_attention()
        f1 = self.get_dump_file("test_gqa.default.ort.custom.onnx")
        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=ds,
            filename=f1,
            options=OptimizationOptions(patterns="default+onnxruntime"),
        )
        # self.assertEqual(["Attention"], [f.name for f in onx.functions])
        ort = self._check_with_ort(onx, cpu=True)
        feeds = dict(zip([i.name for i in onx.graph.input], [t.detach().numpy() for t in inputs]))
        got = ort.run(None, feeds)
        self.assertEqualArray(expected, got[0], 1e-5)
        # if has_onnxscript("0.6.4"):
        #    from onnx_diagnostic.export.api import to_onnx as od_to_onnx
        #    f2 = self.get_dump_file("test_gqa.default.ort.onnx")
        #    od_to_onnx(model, inputs, dynamic_shapes=ds, exporter="onnx-dynamo", filename=f2)

    def test_onnx_gqa_no_rotary_4D(self):
        _mkv_ = oh.make_tensor_value_info

        num_heads = 8
        kv_num_heads = 4
        head_size = 32
        sequence_length = 23
        scale = 0.43 / head_size**0.5

        query = np.random.rand(*(1, num_heads, sequence_length, head_size))
        key = np.random.rand(*(1, kv_num_heads, sequence_length, head_size))
        value = np.random.rand(*(1, kv_num_heads, sequence_length, head_size))
        past_key = np.random.rand(*(1, kv_num_heads, sequence_length // 2, head_size))
        past_value = np.random.rand(*(1, kv_num_heads, sequence_length // 2, head_size))
        attention_mask = np.random.randint(
            0, 1, size=(sequence_length, sequence_length + sequence_length // 2)
        ).astype(bool)

        # something is wrong here
        # query[:,:,:,:] = 1
        # key[:,:,:,:] = 1
        value[:, :, :, :] = 1
        # value[-1,-1,-1,-1] = 0
        # past_key[:,:,:,:] = 1
        past_value[:, :, :, :] = 1
        # attention_mask[:,:] = False

        inputs = (
            query.astype(np.float32),
            key.astype(np.float32),
            value.astype(np.float32),
            attention_mask,
            past_key.astype(np.float32),
            past_value.astype(np.float32),
        )

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Attention",
                        ["query", "key", "value", "mask", "past_key", "past_value"],
                        ["attn", "present_key", "present_value"],
                        scale=scale,
                    ),
                    # QGA contribops
                    oh.make_node("Shape", ["query"], ["batch"], end=1),
                    oh.make_node("Where", ["mask", "zero", "infty"], ["float_mask"]),
                    oh.make_node("Unsqueeze", ["float_mask", "cst01"], ["expanded_mask"]),
                    oh.make_node("Shape", ["mask"], ["total_seqlength64"], start=-1),
                    oh.make_node(
                        "Cast", ["total_seqlength64"], ["total_seqlength"], to=TensorProto.INT32
                    ),
                    oh.make_node("Sub", ["total_seqlength", "one"], ["total_seqlength_1"]),
                    oh.make_node("Expand", ["total_seqlength_1", "batch"], ["seqlensk"]),
                    oh.make_node("Transpose", ["query"], ["query4D"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["key"], ["keys4D"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["value"], ["values4D"], perm=[0, 2, 1, 3]),
                    oh.make_node("Reshape", ["query4D", "shape00"], ["query3D"]),
                    oh.make_node("Reshape", ["keys4D", "shape00"], ["keys3D"]),
                    oh.make_node("Reshape", ["values4D", "shape00"], ["values3D"]),
                    oh.make_node(
                        "GroupQueryAttention",
                        [
                            "query3D",
                            "keys3D",
                            "values3D",
                            "past_key",
                            "past_value",
                            "seqlensk",
                            "total_seqlength",
                            "",
                            "",
                            "",
                            "expanded_mask",
                        ],
                        ["attn3D", "present_key_gqa", "present_value_gqa"],
                        do_rotary=0,
                        num_heads=num_heads,
                        kv_num_heads=kv_num_heads,
                        rotary_interleaved=0,
                        scale=scale,
                        domain="com.microsoft",
                    ),
                    oh.make_node("Reshape", ["attn3D", "shape0000"], ["attn4D"]),
                    oh.make_node("Transpose", ["attn4D"], ["attn_gqa"], perm=[0, 2, 1, 3]),
                ],
                "gqa",
                [
                    _mkv_("query", TFLOAT, ["b", "h", "l", "s"]),
                    _mkv_("key", TFLOAT, ["b", "h2", "l2", "s"]),
                    _mkv_("value", TFLOAT, ["b", "h2", "l2", "s"]),
                    _mkv_("mask", TensorProto.BOOL, ["m1", "m2"]),
                    _mkv_("past_key", TFLOAT, ["b", "h3", "lp", "s"]),
                    _mkv_("past_value", TFLOAT, ["b", "h3", "lp", "s"]),
                ],
                [
                    _mkv_("attn", TFLOAT, ["b", "h3", "l3", "s"]),
                    _mkv_("present_key", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("present_value", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("attn_gqa", TFLOAT, ["b", "h3", "l3", "s"]),
                    _mkv_("present_key_gqa", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("present_value_gqa", TFLOAT, ["b", "ho", "lo", "s"]),
                ],
                [
                    onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="shape00"),
                    onh.from_array(
                        np.array([0, 0, -1, head_size], dtype=np.int64), name="shape0000"
                    ),
                    onh.from_array(np.array([0, 1], dtype=np.int64), name="cst01"),
                    onh.from_array(np.array([1], dtype=np.int32), name="one"),
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(
                        np.array([np.finfo(np.float32).min], dtype=np.float32), name="infty"
                    ),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 24), oh.make_opsetid("com.microsoft", 1)],
            ir_version=11,
        )
        model = shape_inference.infer_shapes(model)
        check_model(model)
        self.dump_onnx("test_onnx_gqa_no_rotary_4D.onnx", model)

        sess = self._check_with_ort(model, cpu=True)
        feeds = dict(zip([i.name for i in model.graph.input], inputs))
        got = sess.run(None, feeds)
        self.assertEqualArray(got[1], got[4], atol=1e-5)
        self.assertEqualArray(got[2], got[5], atol=1e-5)
        self.assertEqualArray(got[0], got[3], atol=1e-5)

    def test_onnx_gqa_no_rotary_3D(self):
        _mkv_ = oh.make_tensor_value_info

        num_heads = 8
        kv_num_heads = 4
        head_size = 32
        sequence_length = 23
        scale = 0.43 / head_size**0.5

        query = np.random.rand(*(1, sequence_length, num_heads * head_size))
        key = np.random.rand(*(1, sequence_length, kv_num_heads * head_size))
        value = np.random.rand(*(1, sequence_length, kv_num_heads * head_size))
        past_key = np.random.rand(*(1, kv_num_heads, sequence_length // 2, head_size))
        past_value = np.random.rand(*(1, kv_num_heads, sequence_length // 2, head_size))
        attention_mask = np.random.randint(
            0, 1, size=(sequence_length, sequence_length + sequence_length // 2)
        ).astype(bool)

        # something is wrong here
        # query[:,:,:,:] = 1
        # key[:,:,:,:] = 1
        value[:, :, :] = 1
        # value[-1,-1,-1,-1] = 0
        # past_key[:,:,:,:] = 1
        past_value[:, :, :, :] = 1
        # attention_mask[:,:] = False

        inputs = (
            query.astype(np.float32),
            key.astype(np.float32),
            value.astype(np.float32),
            attention_mask,
            past_key.astype(np.float32),
            past_value.astype(np.float32),
        )

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Attention",
                        ["query", "key", "value", "mask", "past_key", "past_value"],
                        ["attn", "present_key", "present_value"],
                        scale=scale,
                        q_num_heads=num_heads,
                        kv_num_heads=kv_num_heads,
                    ),
                    # QGA contribops
                    oh.make_node("Shape", ["query"], ["batch"], end=1),
                    oh.make_node("Where", ["mask", "zero", "infty"], ["float_mask"]),
                    oh.make_node("Unsqueeze", ["float_mask", "cst01"], ["expanded_mask"]),
                    oh.make_node("Shape", ["mask"], ["total_seqlength64"], start=-1),
                    oh.make_node(
                        "Cast", ["total_seqlength64"], ["total_seqlength"], to=TensorProto.INT32
                    ),
                    oh.make_node("Sub", ["total_seqlength", "one"], ["total_seqlength_1"]),
                    oh.make_node("Expand", ["total_seqlength_1", "batch"], ["seqlensk"]),
                    oh.make_node(
                        "GroupQueryAttention",
                        [
                            "query",
                            "key",
                            "value",
                            "past_key",
                            "past_value",
                            "seqlensk",
                            "total_seqlength",
                            "",
                            "",
                            "",
                            "expanded_mask",
                        ],
                        ["attn_gqa", "present_key_gqa", "present_value_gqa"],
                        do_rotary=0,
                        num_heads=num_heads,
                        kv_num_heads=kv_num_heads,
                        rotary_interleaved=0,
                        scale=scale,
                        domain="com.microsoft",
                    ),
                ],
                "gqa",
                [
                    _mkv_("query", TFLOAT, ["b", "l", "hs"]),
                    _mkv_("key", TFLOAT, ["b", "l2", "h2s"]),
                    _mkv_("value", TFLOAT, ["b", "l2", "h2s"]),
                    _mkv_("mask", TensorProto.BOOL, ["m1", "m2"]),
                    _mkv_("past_key", TFLOAT, ["b", "h3", "lp", "s"]),
                    _mkv_("past_value", TFLOAT, ["b", "h3", "lp", "s"]),
                ],
                [
                    _mkv_("attn", TFLOAT, ["b", "l3", "h3s"]),
                    _mkv_("present_key", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("present_value", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("attn_gqa", TFLOAT, ["b", "l3", "h3s"]),
                    _mkv_("present_key_gqa", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("present_value_gqa", TFLOAT, ["b", "ho", "lo", "s"]),
                ],
                [
                    # onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="shape00"),
                    # onh.from_array(
                    #    np.array([0, 0, -1, head_size], dtype=np.int64), name="shape0000"
                    # ),
                    onh.from_array(np.array([0, 1], dtype=np.int64), name="cst01"),
                    onh.from_array(np.array([1], dtype=np.int32), name="one"),
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(
                        np.array([np.finfo(np.float32).min], dtype=np.float32), name="infty"
                    ),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 24), oh.make_opsetid("com.microsoft", 1)],
            ir_version=11,
        )
        model = shape_inference.infer_shapes(model)
        check_model(model)
        self.dump_onnx("test_onnx_gqa_no_rotary_3D.onnx", model)

        sess = self._check_with_ort(model, cpu=True)
        feeds = dict(zip([i.name for i in model.graph.input], inputs))
        got = sess.run(None, feeds)
        self.assertEqualArray(got[1], got[4], atol=1e-5)
        self.assertEqualArray(got[2], got[5], atol=1e-5)
        self.assertEqualArray(got[0], got[3], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
