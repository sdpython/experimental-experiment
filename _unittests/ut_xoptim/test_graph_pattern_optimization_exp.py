import itertools
import unittest
import numpy as np
from typing import Optional
from onnx import ModelProto, TensorProto, helper as oh, numpy_helper as onh
from onnx.checker import check_model
from experimental_experiment.ext_test_case import ExtTestCase, skipif_ci_windows
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.xoptim import get_pattern_list
from experimental_experiment.xbuilder._onnx_helper import (
    choose_consistent_domain_opset,
    compatible_opsets,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator

TFLOAT = TensorProto.FLOAT
TFLOAT16 = TensorProto.FLOAT16


class TestGraphPatternOptimizationExp(ExtTestCase):
    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + np.array([bias], dtype=x.dtype)
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
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["ConstantOfShapeScatterND"], processor="CPU,CUDA"
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["ScatterNDOfShape"],
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
        self.assertIn("onnx_extended.ortops.optim.cuda", opsets)
        self.assertEqual(opsets["onnx_extended.ortops.optim.cuda"], 1)

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def _get_aamm_model(
        self,
        op_type: str,
        left: bool,
        other_type: Optional[str] = None,
        negative: bool = False,
    ) -> ModelProto:
        if other_type is None:
            other_type = op_type
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(op_type, ["Y", "X"] if negative else ["X", "Y"], ["xy"]),
                    oh.make_node(
                        other_type,
                        (
                            (["Z", "xy"] if left else ["xy", "Z"])
                            if negative
                            else (["xy", "Z"] if left else ["Z", "xy"])
                        ),
                        ["F"],
                    ),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["d"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["d"]),
                    oh.make_tensor_value_info("Z", TFLOAT, ["d"]),
                ],
                [oh.make_tensor_value_info("F", TFLOAT, ["d"])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        check_model(model)
        return model

    def test_add_add_mul_mul_pattern(self):
        for op_type, left in itertools.product(["Add", "Mul"], [True, False]):
            with self.subTest(op_type=op_type, left=left):
                model = self._get_aamm_model(op_type=op_type, left=left)
                self.assertEqual(len(model.graph.node), 2)
                gr = GraphBuilder(
                    model,
                    infer_shapes=True,
                    optimization_options=OptimizationOptions(
                        patterns=["AddAddMulMul"], processor="CPU,CUDA"
                    ),
                )
                opt_onx = gr.to_onnx(optimize=True)
                self.assertEqual([op_type * 2], [_.op_type for _ in opt_onx.graph.node])
                opsets = {v.domain: v.version for v in opt_onx.opset_import}
                self.assertIn("onnx_extended.ortops.optim.cuda", opsets)
                self.assertEqual(opsets["onnx_extended.ortops.optim.cuda"], 1)

                feeds = {
                    "X": np.array([10, 11], dtype=np.float32),
                    "Y": np.array([10, 12], dtype=np.float32),
                    "Z": np.array([10, 13], dtype=np.float32),
                }
                ref1 = ExtendedReferenceEvaluator(model)
                expected = ref1.run(None, feeds)

                ref2 = ExtendedReferenceEvaluator(opt_onx)
                got = ref2.run(None, feeds)
                self.assertEqualArray(expected[0], got[0])

    def test_mul_sigmoid(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sigmoid", ["X"], ["xs"]),
                    oh.make_node("Mul", ["X", "xs"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [None, None])],
                [oh.make_tensor_value_info("Y", TFLOAT, [None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        check_model(model)
        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["MulSigmoid"], processor="CPU,CUDA"
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["MulSigmoid"],
            [n.op_type for n in opt_onx.graph.node],
        )

        feeds = {
            "X": np.arange(18).reshape((3, 6)).astype(np.float32),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        self.assertEqual(0, len(opt_onx.graph.initializer))
        check_model(opt_onx)
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("onnx_extended.ortops.optim.cuda", opsets)
        self.assertEqual(opsets["onnx_extended.ortops.optim.cuda"], 1)

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def _simple_rotary(self, side):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Split", ["X", "splits"], ["s1", "s2"], axis=-1),
                    (
                        oh.make_node("Neg", ["s1"], ["ns1"])
                        if side == "left"
                        else oh.make_node("Neg", ["s2"], ["ns2"])
                    ),
                    (
                        oh.make_node("Concat", ["s2", "ns1"], ["Y"], axis=-1)
                        if side == "left"
                        else oh.make_node("Concat", ["ns2", "s1"], ["Y"], axis=-1)
                    ),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [None, None])],
                [oh.make_tensor_value_info("Y", TFLOAT, [None, None])],
                [onh.from_array(np.array([4, 4], dtype=np.int64), name="splits")],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        check_model(model)
        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["SimpleRotary"], processor="CPU,CUDA", verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Rotary"],
            [n.op_type for n in opt_onx.graph.node],
        )

        feeds = {
            "X": np.arange(24).reshape((3, 8)).astype(np.float32),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        self.assertEqual(1, len(opt_onx.graph.initializer))
        check_model(opt_onx)
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("onnx_extended.ortops.optim.cuda", opsets)
        self.assertEqual(opsets["onnx_extended.ortops.optim.cuda"], 1)

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_simple_rotary(self):
        self._simple_rotary("right")
        self._simple_rotary("left")

    def test_add_mul_pattern(self):
        for op_type, left in itertools.product(["Add", "Mul"], [True, False]):
            other_type = "Add" if op_type == "Mul" else "Mul"
            with self.subTest(op_type=op_type, left=left):
                model = self._get_aamm_model(
                    op_type=op_type, left=left, other_type=other_type
                )
                self.assertEqual(len(model.graph.node), 2)
                gr = GraphBuilder(
                    model,
                    infer_shapes=True,
                    optimization_options=OptimizationOptions(
                        patterns=["AddMul"], processor="CPU,CUDA"
                    ),
                )
                opt_onx = gr.to_onnx(optimize=True)
                self.assertEqual(
                    [f"{op_type}{other_type}"], [_.op_type for _ in opt_onx.graph.node]
                )
                opsets = {v.domain: v.version for v in opt_onx.opset_import}
                self.assertIn("onnx_extended.ortops.optim.cuda", opsets)
                self.assertEqual(opsets["onnx_extended.ortops.optim.cuda"], 1)

                feeds = {
                    "X": np.array([10, 11], dtype=np.float32),
                    "Y": np.array([10, 12], dtype=np.float32),
                    "Z": np.array([10, 13], dtype=np.float32),
                }
                ref1 = ExtendedReferenceEvaluator(model)
                expected = ref1.run(None, feeds)

                ref2 = ExtendedReferenceEvaluator(opt_onx)
                got = ref2.run(None, feeds)
                self.assertEqualArray(expected[0], got[0])

    def test_replace_zero(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cast", ["X"], ["xb"], to=TensorProto.BOOL),
                    oh.make_node("Where", ["xb", "cst", "X"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [None, None])],
                [oh.make_tensor_value_info("Y", TFLOAT, [None, None])],
                [onh.from_array(np.array([5.67], dtype=np.float32), name="cst")],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        check_model(model)
        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["ReplaceZero"], processor="CPU,CUDA", verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["ReplaceZero"],
            [n.op_type for n in opt_onx.graph.node],
        )

        feeds = {
            "X": (np.arange(18).reshape((3, 6)) - 3).astype(np.float32),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        self.assertEqual(0, len(opt_onx.graph.initializer))
        check_model(opt_onx)
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("onnx_extended.ortops.optim.cuda", opsets)
        self.assertEqual(opsets["onnx_extended.ortops.optim.cuda"], 1)

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_negx_plus1(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sub", ["one", "X"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [None, None])],
                [oh.make_tensor_value_info("Y", TFLOAT, [None, None])],
                [onh.from_array(np.array([1], dtype=np.float32), name="one")],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        check_model(model)
        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["NegXplus1"], processor="CPU,CUDA", verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["NegXplus1"],
            [n.op_type for n in opt_onx.graph.node],
        )

        feeds = {
            "X": (np.arange(18).reshape((3, 6)) - 3).astype(np.float32),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        self.assertEqual(0, len(opt_onx.graph.initializer))
        check_model(opt_onx)
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("onnx_extended.ortops.optim.cuda", opsets)
        self.assertEqual(opsets["onnx_extended.ortops.optim.cuda"], 1)

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_tri_matrix(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Range", ["zero", "dim", "one"], ["ar"]),
                    oh.make_node("Add", ["ar", "one"], ["ad"]),
                    oh.make_node("Reshape", ["ad", "shape1"], ["re"]),
                    oh.make_node("Less", ["ar", "re"], ["le"]),
                    oh.make_node(
                        "ConstantOfShape",
                        ["shape"],
                        ["cst"],
                        value=onh.from_array(
                            np.array([-3.4028234663852886e38], dtype=np.float32)
                        ),
                    ),
                    oh.make_node("Where", ["le", "zerof", "cst"], ["Y"]),
                ],
                "dummy",
                [],
                [oh.make_tensor_value_info("Y", TFLOAT, [None, None])],
                [
                    onh.from_array(np.array([1], dtype=np.int64), name="one"),
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1024], dtype=np.int64), name="dim"),
                    onh.from_array(np.array([0], dtype=np.float32), name="zerof"),
                    onh.from_array(np.array([1024, 1], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([1024, 1024], dtype=np.int64), name="shape"),
                ],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        check_model(model)
        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["TriMatrix"], processor="CPU,CUDA", verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["TriMatrix"],
            [n.op_type for n in opt_onx.graph.node],
        )

        feeds = {
            "X": (np.arange(18).reshape((3, 6)) - 3).astype(np.float32),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        self.assertEqual(2, len(opt_onx.graph.initializer))
        check_model(opt_onx)
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("onnx_extended.ortops.optim.cuda", opsets)
        self.assertEqual(opsets["onnx_extended.ortops.optim.cuda"], 1)

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def _transpose_cast(self, in_type, cast_before):
        out_type = TFLOAT16 if in_type == TFLOAT else TFLOAT

        if cast_before:
            nodes = [
                oh.make_node("Cast", ["X"], ["xc"], to=out_type),
                oh.make_node("Transpose", ["xc"], ["Y"], perm=[1, 0]),
            ]
        else:
            nodes = [
                oh.make_node("Transpose", ["X"], ["xt"], perm=[1, 0]),
                oh.make_node("Cast", ["xt"], ["Y"], to=out_type),
            ]

        model = oh.make_model(
            oh.make_graph(
                nodes,
                "dummy",
                [oh.make_tensor_value_info("X", in_type, ["a", "b"])],
                [oh.make_tensor_value_info("Y", out_type, ["b", "a"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["TransposeCast"], processor="CPU,CUDA", verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        suffix = "32" if out_type == TFLOAT else "16"
        self.assertEqual(
            [f"Transpose2DCastFP{suffix}"],
            [n.op_type for n in opt_onx.graph.node],
        )

        feeds = {
            "X": (np.arange(32).reshape((4, 8)) - 3).astype(np.float32),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        self.assertEqual(0, len(opt_onx.graph.initializer))
        check_model(opt_onx)
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("onnx_extended.ortops.optim.cuda", opsets)
        self.assertEqual(opsets["onnx_extended.ortops.optim.cuda"], 1)

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_transpose_cast(self):
        self._transpose_cast(TFLOAT, False)
        self._transpose_cast(TFLOAT, True)
        self._transpose_cast(TFLOAT16, False)
        self._transpose_cast(TFLOAT16, True)

    def test_sub_mul_pattern(self):
        for op_type, left, negative in itertools.product(
            ["Sub", "Mul"], [True, False], [False, True]
        ):
            other_type = "Sub" if op_type == "Mul" else "Mul"
            with self.subTest(op_type=op_type, left=left, negative=negative):
                model = self._get_aamm_model(
                    op_type=op_type,
                    left=left,
                    other_type=other_type,
                    negative=negative,
                )
                self.assertEqual(len(model.graph.node), 2)
                gr = GraphBuilder(
                    model,
                    infer_shapes=True,
                    optimization_options=OptimizationOptions(
                        patterns=["SubMul"], processor="CPU,CUDA"
                    ),
                )
                opt_onx = gr.to_onnx(optimize=True)
                self.assertEqual(
                    [f"{op_type}{other_type}"], [_.op_type for _ in opt_onx.graph.node]
                )
                opsets = {v.domain: v.version for v in opt_onx.opset_import}
                self.assertIn("onnx_extended.ortops.optim.cuda", opsets)
                self.assertEqual(opsets["onnx_extended.ortops.optim.cuda"], 1)

                feeds = {
                    "X": np.array([10, 11], dtype=np.float32),
                    "Y": np.array([10, 12], dtype=np.float32),
                    "Z": np.array([10, 13], dtype=np.float32),
                }
                ref1 = ExtendedReferenceEvaluator(model)
                expected = ref1.run(None, feeds)
                ref2 = ExtendedReferenceEvaluator(opt_onx)
                got = ref2.run(None, feeds)
                self.assertEqualArray(expected[0], got[0])

    def _get_shared_input_model(self, op_type: str, left: bool) -> ModelProto:
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(op_type, ["X", "Y"], ["F1"]),
                    oh.make_node(op_type, ["X", "Z"] if left else ["Y", "Z"], ["F2"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["d"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["d"]),
                    oh.make_tensor_value_info("Z", TFLOAT, ["d"]),
                ],
                [
                    oh.make_tensor_value_info("F1", TFLOAT, ["d"]),
                    oh.make_tensor_value_info("F2", TFLOAT, ["d"]),
                ],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        check_model(model)
        return model

    def test_add_mul_shared_input_pattern(self):
        for op_type, left in itertools.product(["Add", "Mul"], [True, False]):
            with self.subTest(op_type=op_type, left=left):
                model = self._get_shared_input_model(
                    op_type=op_type,
                    left=left,
                )
                self.assertEqual(len(model.graph.node), 2)
                gr = GraphBuilder(
                    model,
                    infer_shapes=True,
                    optimization_options=OptimizationOptions(
                        patterns=["AddMulSharedInput"], processor="CPU,CUDA"
                    ),
                )
                opt_onx = gr.to_onnx(optimize=True)
                self.assertEqual(
                    [f"{op_type}SharedInput"], [_.op_type for _ in opt_onx.graph.node]
                )
                opsets = {v.domain: v.version for v in opt_onx.opset_import}
                self.assertIn("onnx_extended.ortops.optim.cuda", opsets)
                self.assertEqual(opsets["onnx_extended.ortops.optim.cuda"], 1)

                feeds = {
                    "X": np.array([10, 11], dtype=np.float32),
                    "Y": np.array([10, 12], dtype=np.float32),
                    "Z": np.array([10, 13], dtype=np.float32),
                }
                ref1 = ExtendedReferenceEvaluator(model)
                expected = ref1.run(None, feeds)
                ref2 = ExtendedReferenceEvaluator(opt_onx)
                got = ref2.run(None, feeds)
                self.assertEqualArray(expected[0], got[0])

    def _get_add_mul_transpose_model(
        self, op_type1: str, op_type2: str, left: bool
    ) -> ModelProto:
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        f"{op_type1}{op_type2}",
                        ["X", "Y", "Z"],
                        ["F1"],
                        domain="onnx_extended.ortops.optim.cuda",
                    ),
                    oh.make_node("Transpose", ["F1"], ["final"], perm=[0, 2, 1, 3]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c", "d"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "d"]),
                    oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c", "d"]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, ["a", "b", "c", "d"])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
            ],
            ir_version=9,
        )
        check_model(model)
        return model

    def test_add_mul_transpose_pattern(self):
        for op_type, left in itertools.product(["Add", "Mul"], [True, False]):
            with self.subTest(op_type=op_type, left=left):
                op_type1 = op_type
                op_type2 = "Mul" if op_type == "Add" else "Add"
                model = self._get_add_mul_transpose_model(
                    op_type1=op_type1,
                    op_type2=op_type2,
                    left=left,
                )
                self.assertEqual(len(model.graph.node), 2)
                gr = GraphBuilder(
                    model,
                    infer_shapes=True,
                    optimization_options=OptimizationOptions(
                        patterns=["AddMulTranspose"], processor="CPU,CUDA", verbose=0
                    ),
                )
                self.capture(lambda model=model: self.print_model(model))
                opt_onx = gr.to_onnx(optimize=True)
                self.assertEqual(
                    [f"{op_type1}{op_type2}"], [_.op_type for _ in opt_onx.graph.node]
                )
                opsets = {v.domain: v.version for v in opt_onx.opset_import}
                self.assertIn("onnx_extended.ortops.optim.cuda", opsets)
                self.assertEqual(opsets["onnx_extended.ortops.optim.cuda"], 1)

                feeds = {
                    "X": self._range(2, 3, 4, 5),
                    "Y": self._range(2, 3, 4, 5),
                    "Z": self._range(2, 3, 4, 5),
                }
                ref1 = ExtendedReferenceEvaluator(model)
                expected = ref1.run(None, feeds)
                ref2 = ExtendedReferenceEvaluator(opt_onx)
                got = ref2.run(None, feeds)
                self.assertEqualArray(expected[0], got[0])

    def _masked_scatternd_of_shape_cuda(self, reduction, line, itype):
        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16

        model1 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ConstantOfShape",
                        ["shape"],
                        ["data"],
                        value=onh.from_array(np.array([0], dtype=np.float32)),
                    ),
                    oh.make_node("Equal", ["indices", "mone"], ["masked_indices"]),
                    oh.make_node(
                        "Where",
                        ["masked_indices", "zero", "updates"],
                        ["masked_updates"],
                    ),
                    oh.make_node(
                        "ScatterND",
                        inputs=["data", "indices", "masked_updates"],
                        outputs=["y"],
                        reduction=reduction,
                    ),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("shape", TensorProto.INT64, [None]),
                    oh.make_tensor_value_info("indices", TensorProto.INT64, [None, None, 1]),
                    oh.make_tensor_value_info("updates", itype, [None, None, None]),
                ],
                [oh.make_tensor_value_info("y", itype, [None, None])],
                [
                    onh.from_array(np.array([-1], dtype=np.int64), name="mone"),
                    onh.from_array(np.array([0], dtype=dtype), name="zero"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        gr = GraphBuilder(
            model1,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["ConstantOfShapeScatterND", "MaskedShapeScatterND"],
                processor="CPU,CUDA",
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        # self.print_model(opt_onx)
        self.assertEqual(["MaskedScatterNDOfShape"], [n.op_type for n in opt_onx.graph.node])

        data = np.zeros((32, 16), dtype=dtype)
        indices = np.array(
            [
                [0, 1, 2],
                [2, 3, 4],
                [-1, 30, 31],
                [-1, 7, 8],
                [10, 11, -1],
                [20, -1, 21],
            ],
            dtype=np.int64,
        )
        indices = indices[..., np.newaxis]
        shape = (6, 3, data.shape[-1])
        updates = (np.arange(np.prod(shape)).reshape(shape) + 1).astype(dtype)

        feeds2 = dict(
            shape=np.array(data.shape, dtype=np.int64), indices=indices, updates=updates
        )
        ref = ExtendedReferenceEvaluator(model1)
        expected = ref.run(None, feeds2)[0]

        sess = ExtendedReferenceEvaluator(opt_onx)
        got = sess.run(None, feeds2)[0]
        self.assertEqual(expected.tolist(), got.tolist())

    def test_masked_scatternd_of_shape_standalone_cuda(self):
        self._masked_scatternd_of_shape_cuda("add", 0, TensorProto.FLOAT)
        self._masked_scatternd_of_shape_cuda("add", 1, TensorProto.FLOAT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
