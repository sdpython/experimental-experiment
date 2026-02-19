import unittest
from typing import Optional
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import numpy as np
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    has_onnxruntime,
    hide_stdout,
    ignore_warnings,
)
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    InferShapesOptions,
    OptimizationOptions,
)
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.reference import ExtendedReferenceEvaluator

TFLOAT = onnx.TensorProto.FLOAT
TFLOAT16 = onnx.TensorProto.FLOAT16
TINT64 = onnx.TensorProto.INT64
TBOOL = onnx.TensorProto.BOOL


class TestGraphPatternOptimizationOnnxLLM(ExtTestCase):
    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def test_causal_mask(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["X"], ["d1"], start=0, end=1),
                    oh.make_node("Shape", ["X"], ["d2"], start=1, end=2),
                    oh.make_node("Squeeze", ["d1"], ["nd1"]),
                    oh.make_node("Squeeze", ["d2"], ["nd2"]),
                    oh.make_node("Range", ["zero", "nd2", "one"], ["rg1"]),
                    oh.make_node("Range", ["nd1", "nd2", "one"], ["rg2"]),
                    oh.make_node("Unsqueeze", ["rg1", "a012"], ["m1"]),
                    oh.make_node("Unsqueeze", ["rg2", "a013"], ["m2"]),
                    oh.make_node("LessOrEqual", ["m1", "m2"], ["yc"]),
                    oh.make_node("Cast", ["yc"], ["Y"], to=TINT64),
                ],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b"])],
                [oh.make_tensor_value_info("Y", TINT64, [1, 1, "c", "d"])],
                [
                    onh.from_array(np.array(0, dtype=np.int64), "zero"),
                    onh.from_array(np.array(1, dtype=np.int64), "one"),
                    onh.from_array(np.array([0, 1, 2], dtype=np.int64), "a012"),
                    onh.from_array(np.array([0, 1, 3], dtype=np.int64), "a013"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", 18)],
            ir_version=10,
        )

        feeds = {"X": (np.arange(12).reshape((3, 4))).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        z = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=False,
            optimization_options=OptimizationOptions(patterns="FunctionCausalMask", verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Shape", "Shape", "CausalMask", "Cast"],
            [n.op_type for n in opt_onx.graph.node],
        )
        ref = ExtendedReferenceEvaluator(opt_onx)
        zz = ref.run(None, feeds)[0]
        self.assertEqualArray(z, zz)

    def test_causal_mask_greater(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["X"], ["d1"], start=0, end=1),
                    oh.make_node("Shape", ["X"], ["d2"], start=1, end=2),
                    oh.make_node("Squeeze", ["d1"], ["nd1"]),
                    oh.make_node("Squeeze", ["d2"], ["nd2"]),
                    oh.make_node("Range", ["zero", "nd2", "one"], ["rg1"]),
                    oh.make_node("Range", ["nd1", "nd2", "one"], ["rg2"]),
                    oh.make_node("Unsqueeze", ["rg1", "a012"], ["m1"]),
                    oh.make_node("Unsqueeze", ["rg2", "a013"], ["m2"]),
                    oh.make_node("Sub", ["m2", "initi"], ["m2sub"]),
                    oh.make_node("Greater", ["m1", "m2sub"], ["yc"]),
                    oh.make_node("Cast", ["yc"], ["Y"], to=TINT64),
                ],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b"])],
                [oh.make_tensor_value_info("Y", TINT64, [1, 1, "c", "d"])],
                [
                    onh.from_array(np.array(0, dtype=np.int64), "zero"),
                    onh.from_array(np.array(1, dtype=np.int64), "one"),
                    onh.from_array(np.array([0, 1, 2], dtype=np.int64), "a012"),
                    onh.from_array(np.array([0, 1, 3], dtype=np.int64), "a013"),
                    onh.from_array(np.array([3], dtype=np.int64), "initi"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", 18)],
            ir_version=10,
        )

        feeds = {"X": (np.arange(12).reshape((3, 4))).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        z = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=False,
            optimization_options=OptimizationOptions(patterns="FunctionCausalMask", verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Shape", "Shape", "ShiftedCausalMask", "Cast"],
            [n.op_type for n in opt_onx.graph.node],
        )
        ref = ExtendedReferenceEvaluator(opt_onx)
        zz = ref.run(None, feeds)[0]
        self.assertEqualArray(z, zz)

    def test_causal_mask_mul_add(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["X"], ["N"], start=0, end=1),
                    oh.make_node("Shape", ["X"], ["d1"], start=1, end=2),
                    oh.make_node("Shape", ["X"], ["d2"], start=2, end=3),
                    oh.make_node("Squeeze", ["d1"], ["nd1"]),
                    oh.make_node("Squeeze", ["d2"], ["nd2"]),
                    oh.make_node("Range", ["zero", "nd1", "one"], ["rg1"]),
                    oh.make_node("Range", ["zero", "nd2", "one"], ["rg2"]),
                    oh.make_node("Unsqueeze", ["rg1", "a012"], ["m1"]),
                    oh.make_node("Unsqueeze", ["rg2", "a123"], ["m2"]),
                    oh.make_node("Mul", ["m2", "N"], ["yc"]),
                    oh.make_node("Add", ["m1", "yc"], ["yyc"]),
                    oh.make_node("Cast", ["yyc"], ["Y"], to=TINT64),
                ],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"])],
                [oh.make_tensor_value_info("Y", TINT64, ["a_", "b_", "c_", "d_"])],
                [
                    onh.from_array(np.array(0, dtype=np.int64), "zero"),
                    onh.from_array(np.array(1, dtype=np.int64), "one"),
                    onh.from_array(np.array([0, 1, 2], dtype=np.int64), "a012"),
                    onh.from_array(np.array([1, 2, 3], dtype=np.int64), "a123"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", 18)],
            ir_version=10,
        )

        feeds = {"X": (np.arange(24).reshape((2, 3, 4))).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        z = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=False,
            optimization_options=OptimizationOptions(
                patterns="FunctionCausalMaskMulAdd", verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Shape", "Shape", "Shape", "CausalMaskMulAdd", "Cast"],
            [n.op_type for n in opt_onx.graph.node],
        )
        ref = ExtendedReferenceEvaluator(opt_onx)
        zz = ref.run(None, feeds)[0]
        self.assertEqualArray(z, zz)

    def test_function_causal_mask_bug(self):
        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("local_functions", 1),
            oh.make_opsetid("intermediate", 1),
        ]
        initializers = []
        inputs = [
            oh.make_tensor_value_info(
                "past_key_values_self_attention_cache_key_3::Shape2:3",
                onnx.TensorProto.INT64,
                shape=(1,),
            ),
            oh.make_tensor_value_info(
                "SqueezeAddPattern_add_907", onnx.TensorProto.INT64, shape=(1,)
            ),
        ]
        nodes = [
            oh.make_node(
                "Constant",
                [],
                ["init7_s_0"],
                value=onh.from_array(np.array(0, dtype=np.int64), name="value"),
            ),
            oh.make_node(
                "Constant",
                [],
                ["init7_s_1"],
                value=onh.from_array(np.array(1, dtype=np.int64), name="value"),
            ),
            oh.make_node(
                "Constant",
                [],
                ["init7_s3_0_1_2"],
                value=onh.from_array(np.array([0, 1, 2], dtype=np.int64), name="value"),
            ),
            oh.make_node(
                "Constant",
                [],
                ["init7_s3_0_1_3"],
                value=onh.from_array(np.array([0, 1, 3], dtype=np.int64), name="value"),
            ),
            oh.make_node(
                "Squeeze",
                ["past_key_values_self_attention_cache_key_3::Shape2:3"],
                ["sym_size_int_196"],
            ),
            oh.make_node("Squeeze", ["SqueezeAddPattern_add_907"], ["add_907"]),
            oh.make_node("Range", ["init7_s_0", "add_907", "init7_s_1"], ["arange_8"]),
            oh.make_node("Range", ["sym_size_int_196", "add_907", "init7_s_1"], ["arange_5"]),
            oh.make_node("Unsqueeze", ["arange_8", "init7_s3_0_1_2"], ["unsqueeze_23"]),
            oh.make_node("Unsqueeze", ["arange_5", "init7_s3_0_1_3"], ["unsqueeze_20"]),
            oh.make_node("LessOrEqual", ["unsqueeze_23", "unsqueeze_20"], ["le_12"]),
        ]
        outputs = [
            oh.make_tensor_value_info(
                "le_12",
                onnx.TensorProto.BOOL,
                shape=(1, 1, "sequence_length", "s16+sequence_length"),
            ),
            oh.make_tensor_value_info(
                "arange_5", onnx.TensorProto.INT64, shape=("sequence_length",)
            ),
        ]
        graph = oh.make_graph(nodes, "pattern", inputs, outputs, initializers)
        model = oh.make_model(graph, opset_imports=opset_imports, ir_version=0)
        self.dump_onnx("test_function_causal_mask_bug.onnx", model)

        feeds = {
            "past_key_values_self_attention_cache_key_3::Shape2:3": np.array([8], dtype=np.int64),
            "SqueezeAddPattern_add_907": np.array([8], dtype=np.int64),
        }
        ref = self.check_ort(model)
        z = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=False,
            optimization_options=OptimizationOptions(patterns="FunctionCausalMask", verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_function_causal_mask_bug.opt.onnx", opt_onx)
        ref = self.check_ort(opt_onx)
        zz = ref.run(None, feeds)[0]
        self.assertEqualArray(z, zz)
        self.assertEqual(
            ["Squeeze", "Squeeze", "CausalMask", "Constant", "Range"],
            [n.op_type for n in opt_onx.graph.node],
        )

    def test_rotary_embedding_half(self):
        opset = 22
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Split", ["X"], ["x1", "x2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["x2"], ["nx2"]),
                    oh.make_node("Concat", ["nx2", "x1"], ["c"], axis=-1),
                    oh.make_node("Mul", ["c", "m1"], ["cm1"]),
                    oh.make_node("Mul", ["X", "m2"], ["cm2"]),
                    oh.make_node("Add", ["cm1", "cm2"], ["Y"]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c", "d"]),
                    oh.make_tensor_value_info("m1", TFLOAT, ["c", "d"]),
                    oh.make_tensor_value_info("m2", TFLOAT, ["c", "d"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "d"])],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )

        gr = GraphBuilder(
            model,
            infer_shapes_options=False,
            optimization_options=OptimizationOptions(
                patterns="FunctionHalfRotaryEmbedding", verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_rotary_embedding_half.onnx", opt_onx)
        self.assertIn("HalfRotaryEmbedding", [n.op_type for n in opt_onx.graph.node])

        feeds = {
            "X": (np.arange(2 * 4 * 6 * 8) / (2 * 4 * 6 * 8))
            .reshape((2, 4, 6, 8))
            .astype(np.float32),
            "m1": (np.arange(6 * 8) / 24).reshape((6, 8)).astype(np.float32),
            "m2": (np.arange(6 * 8) / 36).reshape((6, 8)).astype(np.float32),
        }

        import onnxruntime

        for cls in [
            lambda m: ExtendedReferenceEvaluator(m, verbose=0),
            lambda m: onnxruntime.InferenceSession(
                m.SerializeToString(), providers=["CPUExecutionProvider"]
            ),
        ]:
            ref = cls(model)
            z = ref.run(None, feeds)[0]
            ref = cls(opt_onx)
            zz = ref.run(None, feeds)[0]
            self.assertEqualArray(z, zz)

    def test_function_cos_sin_cache_float16(self):
        g = GraphBuilder(18, ir_version=9)
        dim1 = g.make_tensor_input("dim1", TINT64, (1,), is_dimension=False)
        dim2 = g.make_tensor_input("dim2", TINT64, (1,), is_dimension=False)
        weights = g.make_tensor_input("weights", TFLOAT, (1, 1, "a"), is_dimension=False)
        m1 = g.op.Reshape(
            g.op.Cast(
                g.op.Unsqueeze(
                    g.op.Range(g.op.Squeeze(dim1), g.op.Squeeze(dim2), g.ONE_NO_DIM),
                    np.array([0, 1], dtype=np.int64),
                ),
                to=onnx.TensorProto.FLOAT,
            ),
            np.array([0, -1, 1], dtype=np.int64),
        )
        mul = g.op.Mul(weights, m1)
        cos = g.op.Cast(g.op.Cos(mul), to=onnx.TensorProto.FLOAT16, outputs=["cos_cache"])
        sin = g.op.Cast(g.op.Sin(mul), to=onnx.TensorProto.FLOAT16, outputs=["sin_cache"])
        g.make_tensor_output(cos, TFLOAT16, (1, "seq", "a"), indexed=False, is_dimension=False)
        g.make_tensor_output(sin, TFLOAT16, (1, "seq", "a"), indexed=False, is_dimension=False)
        model = g.to_onnx(optimize=False)
        self.dump_onnx("test_function_cos_sin_cache_float16.onnx", model)

        feeds = {
            "dim1": np.array([3], dtype=np.int64),
            "dim2": np.array([6], dtype=np.int64),
            "weights": self._range(1, 1, 16),
        }
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["FunctionCosSinCache"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["CosSinCacheWithRange_to10"],
            [n.op_type for n in opt_onx.graph.node],
        )
        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_function_cos_sin_cache_float16_with_position_ids(self):
        g = GraphBuilder(18, ir_version=9)
        position_ids = g.make_tensor_input("position_ids", TINT64, ("b",), is_dimension=False)
        weights = g.make_tensor_input("weights", TFLOAT, (1, 1, "a"), is_dimension=False)
        m1 = g.op.Reshape(
            g.op.Cast(
                g.op.Unsqueeze(position_ids, np.array([1], dtype=np.int64)),
                to=onnx.TensorProto.FLOAT,
            ),
            np.array([0, -1, 1], dtype=np.int64),
        )
        mul = g.op.Mul(weights, m1)
        cos = g.op.Cast(g.op.Cos(mul), to=onnx.TensorProto.FLOAT16, outputs=["cos_cache"])
        sin = g.op.Cast(g.op.Sin(mul), to=onnx.TensorProto.FLOAT16, outputs=["sin_cache"])
        g.make_tensor_output(cos, TFLOAT16, (1, "seq", "a"), indexed=False, is_dimension=False)
        g.make_tensor_output(sin, TFLOAT16, (1, "seq", "a"), indexed=False, is_dimension=False)
        model = g.to_onnx(optimize=False)
        self.dump_onnx("test_function_cos_sin_cache_float16_position_ids.onnx", model)

        feeds = {
            "position_ids": np.array([3, 5, 6], dtype=np.int64),
            "weights": self._range(1, 1, 16),
        }
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["FunctionCosSinCache"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["CosSinCache_to10_p1"],
            [n.op_type for n in opt_onx.graph.node],
        )
        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_function_cos_sin_cache_float32(self):
        g = GraphBuilder(18, ir_version=9)
        dim1 = g.make_tensor_input("dim1", TINT64, (1,), is_dimension=False)
        dim2 = g.make_tensor_input("dim2", TINT64, (1,), is_dimension=False)
        weights = g.make_tensor_input("weights", TFLOAT, (1, 1, "a"), is_dimension=False)
        m1 = g.op.Reshape(
            g.op.Cast(
                g.op.Unsqueeze(
                    g.op.Range(g.op.Squeeze(dim1), g.op.Squeeze(dim2), g.ONE_NO_DIM),
                    np.array([0, 1], dtype=np.int64),
                ),
                to=onnx.TensorProto.FLOAT,
            ),
            np.array([0, -1, 1], dtype=np.int64),
        )
        mul = g.op.Mul(weights, m1)
        cos = g.op.Cos(mul, outputs=["cos_cache"])
        sin = g.op.Sin(mul, outputs=["sin_cache"])
        g.make_tensor_output(cos, TFLOAT, (1, "seq", "a"), indexed=False, is_dimension=False)
        g.make_tensor_output(sin, TFLOAT, (1, "seq", "a"), indexed=False, is_dimension=False)
        model = g.to_onnx(optimize=False)
        self.dump_onnx("test_function_cos_sin_cache_float32.onnx", model)

        feeds = {
            "dim1": np.array([3], dtype=np.int64),
            "dim2": np.array([6], dtype=np.int64),
            "weights": self._range(1, 1, 16),
        }
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["FunctionCosSinCache"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["CosSinCacheWithRange"],
            [n.op_type for n in opt_onx.graph.node],
        )
        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_function_cos_sin_cache_float32_with_shape_inference(self):
        g = GraphBuilder(18, ir_version=9)
        dim1 = g.make_tensor_input("dim1", TINT64, (1,), is_dimension=False)
        dim2 = g.make_tensor_input("dim2", TINT64, (1,), is_dimension=False)
        weights = g.make_tensor_input("weights", TFLOAT, (1, 1, "a"), is_dimension=False)
        m1 = g.op.Reshape(
            g.op.Cast(
                g.op.Unsqueeze(
                    g.op.Range(g.op.Squeeze(dim1), g.op.Squeeze(dim2), g.ONE_NO_DIM),
                    np.array([0, 1], dtype=np.int64),
                ),
                to=onnx.TensorProto.FLOAT,
            ),
            np.array([0, -1, 1], dtype=np.int64),
        )
        mul = g.op.Mul(weights, m1)
        cos = g.op.Exp(g.op.Cos(mul), outputs=["cos_cache"])
        sin = g.op.Exp(g.op.Sin(mul), outputs=["sin_cache"])
        g.make_tensor_output(cos, TFLOAT, (1, "seq", "a"), indexed=False, is_dimension=False)
        g.make_tensor_output(sin, TFLOAT, (1, "seq", "a"), indexed=False, is_dimension=False)
        model = g.to_onnx(optimize=False)
        self.dump_onnx("test_function_cos_sin_cache_float32.onnx", model)

        feeds = {
            "dim1": np.array([3], dtype=np.int64),
            "dim2": np.array([6], dtype=np.int64),
            "weights": self._range(1, 1, 16),
        }
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["FunctionCosSinCache"], verbose=0),
        )
        gr.optimize()
        gr.infer_shapes()
        self.assertEqual((1, "seq", "a"), gr.get_shape("cos_cache"))
        self.assertEqual((1, "seq", "a"), gr.get_shape("sin_cache"))
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["CosSinCacheWithRange", "Exp", "Exp"],
            [n.op_type for n in opt_onx.graph.node],
        )
        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_rotary_embedding_full(self):
        opset = 23
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
                patterns=["FunctionHalfRotaryEmbedding", "RotaryEmbedding"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertIn("RotaryEmbeddingPattern--Xh1::Shape", gr._known_value_shape)
        self.assertEqual(gr._known_value_shape["RotaryEmbeddingPattern--Xh1::Shape"], ("a", 1, 1))
        self.dump_onnx("test_rotary_embedding_full.onnx", opt_onx)
        self.assertIn("RotaryEmbedding", [n.op_type for n in opt_onx.graph.node])

        import onnxruntime

        for name, cls in [
            ("ref", lambda m: ExtendedReferenceEvaluator(m, verbose=0)),
            (
                "ort",
                lambda m: onnxruntime.InferenceSession(
                    m.SerializeToString(), providers=["CPUExecutionProvider"]
                ),
            ),
        ]:
            with self.subTest(name=name):
                if name == "ort" and not has_onnxruntime("1.23"):
                    raise unittest.SkipTest("onnxruntime < 1.23")
                ref = cls(model)
                z = ref.run(None, feeds)[0]
                ref = cls(opt_onx)
                zz = ref.run(None, feeds)[0]
                self.assertEqualArray(z, zz, atol=1e-4)

    def test_local_function_attention(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Mul", ["query", "scale_sqrt"], ["query_scaled"]),
                    oh.make_node("Mul", ["keys", "scale_sqrt"], ["keys_scaled"]),
                    oh.make_node(
                        "Transpose", ["keys_scaled"], ["keys_scaled_t"], perm=[0, 1, 3, 2]
                    ),
                    oh.make_node("MatMul", ["query_scaled", "keys_scaled_t"], ["qk"]),
                    oh.make_node("Where", ["mask", "zero", "minfty"], ["bias"]),
                    oh.make_node("Add", ["qk", "bias"], ["qkb"]),
                    oh.make_node("Softmax", ["qkb"], ["qkbs"], axis=-1),
                    oh.make_node("IsNaN", ["qkbs"], ["nans"]),
                    oh.make_node("Where", ["nans", "zero", "qkbs"], ["filt"]),
                    oh.make_node("MatMul", ["filt", "values"], ["Y"]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("query", TFLOAT, ["aq", "bq", "cq", "dq"]),
                    oh.make_tensor_value_info("keys", TFLOAT, ["ak", "bk", "ck", "dk"]),
                    oh.make_tensor_value_info("values", TFLOAT, ["av", "bv", "cv", "dv"]),
                    oh.make_tensor_value_info("mask", TBOOL, ["am", "bm", "cm", "dm"]),
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
        query = np.random.randn(32, 8, 128, 64).astype(np.float32)
        keys = np.random.randn(32, 8, 128, 64).astype(np.float32)
        values = np.random.randn(32, 8, 128, 64).astype(np.float32)
        mask = np.random.rand(1, 8, 128, 128) >= 0.5

        feeds = dict(query=query, keys=keys, values=values, mask=mask)
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        z = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns="FunctionAttention", verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["LocalAttention_to1"], [n.op_type for n in opt_onx.graph.node])
        ref = ExtendedReferenceEvaluator(opt_onx, verbose=0)
        zz = ref.run(None, feeds)[0]
        self.assertEqualArray(z, zz)

    def _get_gqa_model(self):
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
                    attn = attn.masked_fill((1 - mask).bool(), float("-inf")).to(query.dtype)

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
        attention_mask = torch.zeros(
            (model.sequence_length, model.sequence_length + model.sequence_length // 2)
        )
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

    @hide_stdout()
    def test_local_attention_gqa_0(self):
        model, inputs, ds, expected = self._get_gqa_model()
        f1 = self.get_dump_file("test_local_attention_gqa_0.onnx")
        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=ds,
            filename=f1,
            options=OptimizationOptions(
                patterns=[
                    "FunctionAttention",
                    "WhereAdd",
                    "SwapUnary",
                    "ShapeBasedEditDistanceReshape",
                    "Cast",
                    "SwapUnary",
                    "ShapeBasedExpandSwap",
                ],
                verbose=10,
            ),
        )
        ort = self._check_with_ort(onx)
        feeds = dict(zip([i.name for i in onx.graph.input], [t.detach().numpy() for t in inputs]))
        got = ort.run(None, feeds)
        self.assertEqualArray(expected, got[0])
        self.assertEqual(["LocalAttentionSW_to1"], [f.name for f in onx.functions])
        self.assertIn("LocalAttentionSW_to1", [n.op_type for n in onx.graph.node])

    @hide_stdout()
    def test_local_attention_gqa_1(self):
        model, inputs, ds, expected = self._get_gqa_model()
        f1 = self.get_dump_file("test_local_attention_gqa_1.onnx")
        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=ds,
            filename=f1,
            options=OptimizationOptions(
                patterns=[
                    "WhereAdd",
                    "SwapUnary",
                    "ShapeBasedEditDistanceReshape",
                    "Cast",
                    "SwapUnary",
                    "ShapeBasedExpandSwap",
                    "FunctionAttention",
                ],
                verbose=0,
            ),
        )
        ort = self._check_with_ort(onx)
        feeds = dict(zip([i.name for i in onx.graph.input], [t.detach().numpy() for t in inputs]))
        got = ort.run(None, feeds)
        self.assertEqualArray(expected, got[0])
        self.assertEqual(["LocalAttentionGQASW_to1"], [f.name for f in onx.functions])
        self.assertIn("LocalAttentionGQASW_to1", [n.op_type for n in onx.graph.node])

    @hide_stdout()
    def test_local_attention_gqa_2(self):
        model, inputs, ds, expected = self._get_gqa_model()
        f1 = self.get_dump_file("test_local_attention_gqa_2.onnx")
        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=ds,
            filename=f1,
            options=OptimizationOptions(
                patterns=[
                    "FunctionAttention",
                    "WhereAdd",
                    "SwapUnary",
                    "ShapeBasedEditDistanceReshape",
                    "Cast",
                    "SwapUnary",
                    "ShapeBasedExpandSwap",
                    "FunctionAttentionGQA",
                ],
                verbose=0,
            ),
        )
        ort = self._check_with_ort(onx)
        feeds = dict(zip([i.name for i in onx.graph.input], [t.detach().numpy() for t in inputs]))
        got = ort.run(None, feeds)
        self.assertEqualArray(expected, got[0])
        self.assertIn("LocalAttentionGQASW_to1", [f.name for f in onx.functions])
        self.assertIn("LocalAttentionGQASW_to1", [n.op_type for n in onx.graph.node])

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
                return torch.matmul(attn, value)

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
            sequence_length=1,
            num_heads=8,
            kv_num_heads=4,
            head_size=32,
            softmax_scale=None,
            use_smooth_softmax=False,
        ).eval()

        past_length = 22
        query = torch.rand((1, model.num_heads, model.sequence_length, model.head_size))
        key = torch.rand((1, model.kv_num_heads, model.sequence_length, model.head_size))
        value = torch.rand((1, model.kv_num_heads, model.sequence_length, model.head_size))
        past_key = torch.rand((1, model.kv_num_heads, past_length, model.head_size))
        past_value = torch.rand((1, model.kv_num_heads, past_length, model.head_size))
        attention_mask = torch.randint(
            0, 2, (model.sequence_length, model.sequence_length + past_length)
        ).to(bool)

        inputs = (query, key, value, attention_mask, past_key, past_value)
        expected = model.forward(*inputs)
        self.assertEqual((1, 8, model.sequence_length, 32), expected.shape)
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
    def test_attention_gqa_default_22(self):
        model, inputs, ds, expected = self._get_model_attention()
        f1 = self.get_dump_file("test_attention_gqa.22.onnx")
        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=ds,
            filename=f1,
            options=OptimizationOptions(patterns="default"),
            target_opset=22,
        )
        self.assertIn("LocalAttentionGQASW_to1", [f.op_type for f in onx.graph.node])
        self.assertNotIn("Attention", [f.op_type for f in onx.graph.node])
        ort = self._check_with_ort(onx, cpu=True)
        feeds = dict(zip([i.name for i in onx.graph.input], [t.detach().numpy() for t in inputs]))
        got = ort.run(None, feeds)
        self.assertEqualArray(expected, got[0], 1e-5)

    @ignore_warnings((UserWarning, FutureWarning))
    @hide_stdout()
    def test_attention_gqa_default_24(self):
        model, inputs, ds, expected = self._get_model_attention()
        f1 = self.get_dump_file("test_attention_gqa.24.onnx")
        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=ds,
            filename=f1,
            options=OptimizationOptions(patterns="default"),
            target_opset=24,
        )
        self.assertNotIn("LocalAttentionGQASW_to1", [f.op_type for f in onx.graph.node])
        self.assertIn("Attention", [f.op_type for f in onx.graph.node])
        ort = self._check_with_ort(onx, cpu=True)
        feeds = dict(zip([i.name for i in onx.graph.input], [t.detach().numpy() for t in inputs]))
        got = ort.run(None, feeds)
        self.assertEqualArray(expected, got[0], 1e-5)

    def test_attention_gqa_bool(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Concat", ["past_key", "key"], ["present_key"], axis=2),
                    oh.make_node("Concat", ["past_value", "value"], ["present_value"], axis=2),
                    oh.make_node("Unsqueeze", ["present_key", "two"], ["key_u"]),
                    oh.make_node("Expand", ["key_u", "t11211"], ["key_ue"]),
                    oh.make_node("Squeeze", ["key_ue", "one"], ["key_ues"]),
                    oh.make_node("Unsqueeze", ["present_value", "two"], ["value_u"]),
                    oh.make_node("Expand", ["value_u", "t11211"], ["value_ue"]),
                    oh.make_node("Squeeze", ["value_ue", "one"], ["value_ues"]),
                    oh.make_node(
                        "Attention",
                        ["query", "key_ues", "value_ues", "mask"],
                        ["Y"],
                        scale=0.11,
                    ),
                ],
                "test",
                [
                    oh.make_tensor_value_info("query", TFLOAT, ["a", 2, "c", "d"]),
                    oh.make_tensor_value_info("key", TFLOAT, ["a", 1, "c", "d"]),
                    oh.make_tensor_value_info("value", TFLOAT, ["a", 1, "c", "d"]),
                    oh.make_tensor_value_info("mask", TBOOL, ["a", 1, "c", "c+h"]),
                    oh.make_tensor_value_info("past_key", TFLOAT, ["a", 1, "h", "d"]),
                    oh.make_tensor_value_info("past_value", TFLOAT, ["a", 1, "h", "d"]),
                ],
                [
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", 2, "c_", "d"]),
                    oh.make_tensor_value_info("present_key", TFLOAT, ["a", 1, "c+h", "d"]),
                    oh.make_tensor_value_info("present_value", TFLOAT, ["a", 1, "c+h", "d"]),
                ],
                [
                    onh.from_array(np.array([1], dtype=np.int64), "one"),
                    onh.from_array(np.array([2], dtype=np.int64), "two"),
                    onh.from_array(np.array([1, 1, 2, 1, 1], dtype=np.int64), "t11211"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", 24)],
            ir_version=11,
        )
        onnx.checker.check_model(model)

        feeds = dict(
            query=self._range(1, 2, 3, 8),
            key=self._range(1, 1, 3, 8),
            value=self._range(1, 1, 3, 8),
            mask=(self._range(1, 1, 3, 5).astype(int) % 2) == 1,
            past_key=self._range(1, 1, 2, 8),
            past_value=self._range(1, 1, 2, 8),
        )
        feeds["mask"][0, 0, 0, ::2] = False
        feeds["mask"][0, 0, 0, 1::2] = True
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=False,
            optimization_options=OptimizationOptions(patterns="AttentionGQA", verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Attention"], [n.op_type for n in opt_onx.graph.node])
        ref = ExtendedReferenceEvaluator(opt_onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected[1], got[1])
        self.assertEqualArray(expected[2], got[2])
        self.assertEqualArray(expected[0], got[0])

        # onnxruntime
        sess = self._check_with_ort(model, cpu=True)
        sess_opt = self._check_with_ort(opt_onx, cpu=True)
        expected = sess.run(None, feeds)
        got = sess_opt.run(None, feeds)
        self.assertEqualArray(expected[1], got[1])
        self.assertEqualArray(expected[2], got[2])
        self.assertEqualArray(expected[0], got[0])

    def test_attention_gqa_bool_not_one(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Concat", ["past_key", "key"], ["present_key"], axis=2),
                    oh.make_node("Concat", ["past_value", "value"], ["present_value"], axis=2),
                    oh.make_node("Unsqueeze", ["present_key", "two"], ["key_u"]),
                    oh.make_node("Expand", ["key_u", "t11211"], ["key_ue"]),
                    oh.make_node("Reshape", ["key_ue", "resh"], ["key_ues"]),
                    oh.make_node("Unsqueeze", ["present_value", "two"], ["value_u"]),
                    oh.make_node("Expand", ["value_u", "t11211"], ["value_ue"]),
                    oh.make_node("Reshape", ["value_ue", "resh"], ["value_ues"]),
                    oh.make_node(
                        "Attention",
                        ["query", "key_ues", "value_ues", "mask"],
                        ["Y"],
                        scale=0.11,
                    ),
                ],
                "test",
                [
                    oh.make_tensor_value_info("query", TFLOAT, ["a", 4, "c", 8]),
                    oh.make_tensor_value_info("key", TFLOAT, ["a", 2, "c", 8]),
                    oh.make_tensor_value_info("value", TFLOAT, ["a", 2, "c", 8]),
                    oh.make_tensor_value_info("mask", TBOOL, ["a", 1, "c", "c+h"]),
                    oh.make_tensor_value_info("past_key", TFLOAT, ["a", 2, "h", 8]),
                    oh.make_tensor_value_info("past_value", TFLOAT, ["a", 2, "h", 8]),
                ],
                [
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", 4, "c_", 8]),
                    oh.make_tensor_value_info("present_key", TFLOAT, ["a", 2, "c+h", 8]),
                    oh.make_tensor_value_info("present_value", TFLOAT, ["a", 2, "c+h", 8]),
                ],
                [
                    onh.from_array(np.array([1], dtype=np.int64), "one"),
                    onh.from_array(np.array([2], dtype=np.int64), "two"),
                    onh.from_array(np.array([1, 1, 2, 1, 1], dtype=np.int64), "t11211"),
                    onh.from_array(np.array([0, 4, -1, 8], dtype=np.int64), "resh"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", 24)],
            ir_version=11,
        )
        onnx.checker.check_model(model)

        feeds = dict(
            query=self._range(1, 4, 3, 8),
            key=self._range(1, 2, 3, 8),
            value=self._range(1, 2, 3, 8),
            mask=(self._range(1, 1, 3, 5).astype(int) % 2) == 1,
            past_key=self._range(1, 2, 2, 8),
            past_value=self._range(1, 2, 2, 8),
        )
        feeds["mask"][0, 0, 0, ::2] = False
        feeds["mask"][0, 0, 0, 1::2] = True
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=False,
            optimization_options=OptimizationOptions(patterns="AttentionGQA", verbose=10),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Attention"], [n.op_type for n in opt_onx.graph.node])
        ref = ExtendedReferenceEvaluator(opt_onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected[1], got[1])
        self.assertEqualArray(expected[2], got[2])
        self.assertEqualArray(expected[0], got[0])

        # onnxruntime
        sess = self._check_with_ort(model, cpu=True)
        sess_opt = self._check_with_ort(opt_onx, cpu=True)
        expected = sess.run(None, feeds)
        got = sess_opt.run(None, feeds)
        self.assertEqualArray(expected[1], got[1])
        self.assertEqualArray(expected[2], got[2])
        self.assertEqualArray(expected[0], got[0])

    def test_attention_gqa_float(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Concat", ["past_key", "key"], ["present_key"], axis=2),
                    oh.make_node("Concat", ["past_value", "value"], ["present_value"], axis=2),
                    oh.make_node("Unsqueeze", ["present_key", "two"], ["key_u"]),
                    oh.make_node("Expand", ["key_u", "t11211"], ["key_ue"]),
                    oh.make_node("Squeeze", ["key_ue", "one"], ["key_ues"]),
                    oh.make_node("Unsqueeze", ["present_value", "two"], ["value_u"]),
                    oh.make_node("Expand", ["value_u", "t11211"], ["value_ue"]),
                    oh.make_node("Squeeze", ["value_ue", "one"], ["value_ues"]),
                    oh.make_node(
                        "Attention",
                        ["query", "key_ues", "value_ues", "mask"],
                        ["Y"],
                        scale=0.11,
                    ),
                ],
                "test",
                [
                    oh.make_tensor_value_info("query", TFLOAT, ["a", 2, "c", "d"]),
                    oh.make_tensor_value_info("key", TFLOAT, ["a", 1, "c", "d"]),
                    oh.make_tensor_value_info("value", TFLOAT, ["a", 1, "c", "d"]),
                    oh.make_tensor_value_info("mask", TFLOAT, ["a", 1, "c", "c+h"]),
                    oh.make_tensor_value_info("past_key", TFLOAT, ["a", 1, "h", "d"]),
                    oh.make_tensor_value_info("past_value", TFLOAT, ["a", 1, "h", "d"]),
                ],
                [
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", 2, "c_", "d"]),
                    oh.make_tensor_value_info("present_key", TFLOAT, ["a", 1, "c+h", "d"]),
                    oh.make_tensor_value_info("present_value", TFLOAT, ["a", 1, "c+h", "d"]),
                ],
                [
                    onh.from_array(np.array([1], dtype=np.int64), "one"),
                    onh.from_array(np.array([2], dtype=np.int64), "two"),
                    onh.from_array(np.array([1, 1, 2, 1, 1], dtype=np.int64), "t11211"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", 24)],
            ir_version=11,
        )
        onnx.checker.check_model(model)

        feeds = dict(
            query=self._range(1, 2, 3, 8),
            key=self._range(1, 1, 3, 8),
            value=self._range(1, 1, 3, 8),
            mask=np.where((self._range(1, 1, 3, 5).astype(int) % 2) == 1, 0, -np.inf).astype(
                np.float32
            ),
            past_key=self._range(1, 1, 2, 8),
            past_value=self._range(1, 1, 2, 8),
        )
        feeds["mask"][0, 0, 0, ::2] = 0
        feeds["mask"][0, 0, 0, 1::2] = -np.inf
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=False,
            optimization_options=OptimizationOptions(patterns="AttentionGQA", verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Attention"], [n.op_type for n in opt_onx.graph.node])
        ref = ExtendedReferenceEvaluator(opt_onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected[1], got[1])
        self.assertEqualArray(expected[2], got[2])
        self.assertEqualArray(expected[0], got[0])

        # onnxruntime
        sess = self._check_with_ort(model, cpu=True)
        sess_opt = self._check_with_ort(opt_onx, cpu=True)
        expected = sess.run(None, feeds)
        got = sess_opt.run(None, feeds)
        self.assertEqualArray(expected[1], got[1])
        self.assertEqualArray(expected[2], got[2])
        self.assertEqualArray(expected[0], got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
