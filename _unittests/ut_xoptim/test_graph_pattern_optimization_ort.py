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
)
from onnx.checker import check_model
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
    requires_onnxruntime_training,
    requires_cuda,
)
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.xoptim import get_pattern_list
from experimental_experiment.xoptim.patterns_ort.gather_grad import GatherGradPattern
from experimental_experiment.xoptim.patterns_ort.activation import GeluErfPattern
from experimental_experiment.xbuilder._onnx_helper import (
    choose_consistent_domain_opset,
    compatible_opsets,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator

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

    def _check_with_ort(self, proto: ModelProto):
        from onnxruntime import InferenceSession, get_available_providers

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        InferenceSession(proto.SerializeToString(), providers=providers)

    @requires_onnxruntime_training()
    def test_fused_matmul_pattern(self):
        origin = self._get_model("bug_fused.onnx")
        check_model(origin)
        self._check_with_ort(origin)
        gr = GraphBuilder(
            origin,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FusedMatMul"],
                verbose=0,  # stop_after=2
            ),
        )
        onx = gr.to_onnx(optimize=True)
        self._check_with_ort(onx)
        check_model(onx)

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

    @requires_onnxruntime_training()
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
                patterns=["Cast", "Gelu", "FastGelu"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertNotIn("Gelu", set(n.op_type for n in opt_onx.graph.node))
        self.assertIn("FastGelu", set(n.op_type for n in opt_onx.graph.node))
        self.assertEqual(154, len(opt_onx.graph.initializer))
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
                patterns=["Cast", "GeluOrt", "FastGelu"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertNotIn("Gelu", set(n.op_type for n in opt_onx.graph.node))
        self.assertIn("FastGelu", set(n.op_type for n in opt_onx.graph.node))
        self.assertEqual(154, len(opt_onx.graph.initializer))
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

    def test_rotary_embedding(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["xt"], perm=[0, 2, 1, 3]),
                    oh.make_node("Split", ["xt"], ["s1", "s2"], num_outputs=2, axis=-1),
                    oh.make_node("Neg", ["s2"], ["ns2"]),
                    oh.make_node("Concat", ["ns2", "s1"], ["conc"], axis=-1),
                    oh.make_node("Mul", ["xt", "cos"], ["xcos"], name="mx"),
                    oh.make_node("Mul", ["conc", "sin"], ["ysin"], name="my"),
                    oh.make_node("Add", ["xcos", "ysin"], ["Y"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 8, 3, 32]),
                    oh.make_tensor_value_info("cos", TFLOAT, [1, 3, 1, 32]),
                    oh.make_tensor_value_info("sin", TFLOAT, [1, 3, 1, 32]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, [2, 3, 8, 32])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        feeds = {
            "X": self._range(2, 8, 3, 32),
            "cos": self._range(1, 3, 1, 32),
            "sin": self._range(1, 3, 1, 32),
        }
        # cap = ExtendedReferenceEvaluator(model, verbose=10)
        # cap.run(None, feeds)
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["RotaryEmbedding"], verbose=10),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Shape", "Range", "RotaryEmbedding"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
