import os
import unittest
import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper as oh, numpy_helper as onh
from onnx.checker import check_model
from onnx_array_api.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import ExtTestCase, ignore_warnings
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.xoptim.graph_builder_optim import (
    GraphBuilderPatternOptimization,
)
from experimental_experiment.xbuilder.shape_helper import (
    compatible_shapes,
    compatible_dimensions,
)
from experimental_experiment.xoptim.patterns import get_pattern_list


class TestGraphPatternOptimization(ExtTestCase):
    def test_get_pattern_list(self):
        res = get_pattern_list(negative_list=["Cast"])
        names = set(r.__class__.__name__ for r in res)
        self.assertNotIn("CastPattern", names)
        res = get_pattern_list(negative_list="default")
        self.assertEqual(res, [])

    def _check_with_ort(self, proto: ModelProto):
        from onnxruntime import InferenceSession, get_available_providers

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        InferenceSession(proto.SerializeToString(), providers=providers)

    @ignore_warnings(DeprecationWarning)
    def test_try_with_custom_model(self):
        filename = "onnx_model_1_3.onnx"
        if not os.path.exists(filename):
            raise unittest.SkipTest(f"filename={filename!r} not found")
        onx = onnx.load(filename)
        gr = GraphBuilder(
            onx,
            infer_shapes=True,
            verbose=0,
            optimization_options=OptimizationOptions(
                remove_identity=False,
                verbose=10 if __name__ == "__main__" else 0,
                patterns="default",
            ),
        )
        optimized = gr.to_onnx()
        self._check_with_ort(onx)
        if __name__ == "__main__":
            with open(f"try-{filename}-optimized.onnx", "wb") as f:
                f.write(optimized.SerializeToString())
        self._check_with_ort(optimized)

    def _get_model(self, name: str) -> onnx.ModelProto:
        p = os.path.join(os.path.dirname(__file__), "..", "ut_xbuilder", "data", name)
        self.assertExists(p)
        return onnx.load(p)

    def test_compatible_shapes(self):
        self.assertTrue(compatible_shapes((1, 2), (1, "D2")))
        self.assertFalse(compatible_shapes((1, 2), (1,)))
        self.assertTrue(compatible_shapes((1, 2), (1, 2)))
        self.assertFalse(compatible_shapes(("D2", 2), (1, "D2")))
        self.assertTrue(compatible_shapes(("D2", 2), (2, "D2")))

    def test_compatible_dimensions(self):
        self.assertTrue(compatible_dimensions(1, 1))
        self.assertTrue(compatible_dimensions(1, "D1"))
        self.assertFalse(compatible_dimensions(1, 2))
        self.assertTrue(compatible_dimensions(1, "D1", "D2"))

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

    def test_shape_inference0(self):
        origin = self._get_model("dort-c-custom__0.onnx")
        gr = GraphBuilder(origin, infer_shapes=True)
        gro = GraphBuilderPatternOptimization(gr)
        shape = gro.try_infer_shape("_onx_tile0", exc=True)
        self.assertEqual(shape, (2, 1, 1024, 1024))

    def test_unsqueeze_unsqueeze(self):
        origin = self._get_model("dort-c-custom__0.onnx")
        before = [node for node in origin.graph.node if node.op_type == "Unsqueeze"]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(
                patterns=["UnsqueezeUnsqueeze"], verbose=10
            ),
        )
        res, out, err = self.capture(lambda: gr.optimize_with_patterns())
        self.assertEmpty(err)
        self.assertNotEmpty(res)
        self.assertIn("[GraphBuilderPatternOptimization.optimize] done after", out)
        self.assertIn("UnsqueezeUnsqueezePattern", out)

        onx = gr.to_onnx(optimize=False)
        after = [node for node in onx.graph.node if node.op_type == "Unsqueeze"]
        self.assertEqual(len(after), len(before) - 2)

    def test_cast(self):
        origin = self._get_model("dort-c-custom__0.onnx")
        before = [node for node in origin.graph.node if node.op_type == "Cast"]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(patterns=["Cast"], verbose=10),
        )
        res, out, err = self.capture(lambda: gr.optimize_with_patterns())
        self.assertEmpty(err)
        self.assertNotEmpty(res)
        self.assertIn("[GraphBuilderPatternOptimization.optimize] done after", out)
        self.assertNotIn("UnsqueezeUnsqueezePattern", out)
        self.assertIn("CastPattern", out)

        onx = gr.to_onnx(optimize=False)
        after = [node for node in onx.graph.node if node.op_type == "Cast"]
        self.assertEqual(len(before), 14)
        self.assertEqual(len(after), 2)

    def test_reshape_matmul_reshape(self):
        origin = self._get_model("dort-c-custom__0.onnx")
        before = [node for node in origin.graph.node if node.op_type == "Reshape"]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(
                patterns=["ReshapeMatMulReshape"], verbose=10
            ),
            infer_shapes=True,
        )
        res, out, err = self.capture(lambda: gr.optimize_with_patterns())
        self.assertEmpty(err)
        self.assertNotEmpty(res)
        self.assertIn("[GraphBuilderPatternOptimization.optimize] done after", out)
        self.assertIn("ReshapeMatMulReshapePattern", out)

        onx = gr.to_onnx(optimize=False)
        after = [node for node in onx.graph.node if node.op_type == "Reshape"]
        self.assertEqual(len(before), 24)
        self.assertEqual(len(after), 22)

    def _range(self, *shape, bias: float = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def test_execution_static(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [32, 128]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5, 128, 64]),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [3, 5, 32, 64])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(
                        np.array([1, 32, 128], dtype=np.int64), name="shape1"
                    ),
                    onh.from_array(
                        np.array([15, 128, 64], dtype=np.int64), name="shape2"
                    ),
                    onh.from_array(
                        np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"
                    ),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr, _, __ = self.capture(
            lambda: GraphBuilder(model, infer_shapes=True, verbose=10)
        )
        s = str(gr.optimization_options)
        self.assertIn("OptimizationOptions(", s)
        self.assertIn("CastPattern", s)
        opt_onx, out, _ = self.capture(lambda: gr.to_onnx(optimize=True))
        self.assertIn("remove_initializer:shape1", out)
        self.assertIn("remove_initializer:shape2", out)
        self.assertEqual(
            ["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_execution_dynamic_1(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, ["D32", "D128"]),
                    oh.make_tensor_value_info(
                        "Y", TensorProto.FLOAT, ["batch", "channel", "D128", "D64"]
                    ),
                ],
                [
                    oh.make_tensor_value_info(
                        "Z", TensorProto.FLOAT, ["batch", "channel", "D32", "64"]
                    )
                ],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(
                        np.array([1, 32, 128], dtype=np.int64), name="shape1"
                    ),
                    onh.from_array(
                        np.array([15, 128, 64], dtype=np.int64), name="shape2"
                    ),
                    onh.from_array(
                        np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"
                    ),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(model, infer_shapes=True)
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_execution_dynamic_2(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, ["D32", "D128"]),
                    oh.make_tensor_value_info(
                        "Y", TensorProto.FLOAT, ["batch", "channel", "any", "D64"]
                    ),
                ],
                [
                    oh.make_tensor_value_info(
                        "Z", TensorProto.FLOAT, ["batch", "channel", "D32", "64"]
                    )
                ],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(
                        np.array([1, 32, 128], dtype=np.int64), name="shape1"
                    ),
                    onh.from_array(
                        np.array([15, 128, 64], dtype=np.int64), name="shape2"
                    ),
                    onh.from_array(
                        np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"
                    ),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(model, infer_shapes=True)
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_execution_keep_intermediate(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [32, 128]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5, 128, 64]),
                ],
                [
                    oh.make_tensor_value_info("Z", TensorProto.FLOAT, [3, 5, 32, 64]),
                    oh.make_tensor_value_info("xm1", TensorProto.FLOAT, [1, 32, 128]),
                ],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(
                        np.array([1, 32, 128], dtype=np.int64), name="shape1"
                    ),
                    onh.from_array(
                        np.array([15, 128, 64], dtype=np.int64), name="shape2"
                    ),
                    onh.from_array(
                        np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"
                    ),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr, _, __ = self.capture(
            lambda: GraphBuilder(model, infer_shapes=True, verbose=10)
        )
        s = str(gr.optimization_options)
        self.assertIn("OptimizationOptions(", s)
        self.assertIn("CastPattern", s)
        opt_onx, out, _ = self.capture(lambda: gr.to_onnx(optimize=True))
        self.assertIn("remove_initializer:shape2", out)
        self.assertEqual(
            ["Unsqueeze", "MatMul", "Reshape"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(2, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reshape_reshape_dort(self):
        origin = self._get_model("dort-c-custom__0.onnx")
        before = [node for node in origin.graph.node if node.op_type == "Reshape"]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(
                patterns=["ReshapeReshape"], verbose=0
            ),
            infer_shapes=True,
        )
        onx = gr.to_onnx(optimize=True)
        after = [node for node in onx.graph.node if node.op_type == "Reshape"]
        self.assertEqual(len(after), len(before) - 4)

    def test_reshape_reshape_execution(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "r1"], ["xu1"]),
                    oh.make_node("Reshape", ["xu1", "r2"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [32, 128]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5, 128, 64]),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [3, 5, 32, 64])],
                [
                    onh.from_array(np.array([-1], dtype=np.int64), name="r1"),
                    onh.from_array(np.array([-1, 128], dtype=np.int64), name="r2"),
                    onh.from_array(
                        np.array([1, 32, 128], dtype=np.int64), name="shape1"
                    ),
                    onh.from_array(
                        np.array([15, 128, 64], dtype=np.int64), name="shape2"
                    ),
                    onh.from_array(
                        np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"
                    ),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["ReshapeReshape"]),
        )
        s = str(gr.optimization_options)
        self.assertIn("OptimizationOptions(", s)
        self.assertIn("ReshapeReshapePattern", s)
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Reshape", "Reshape", "Cast", "MatMul", "Reshape"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(3, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_expand_dort(self):
        origin = self._get_model("dort-c-custom__0.onnx")
        before = [node for node in origin.graph.node if node.op_type == "Expand"]
        gr = GraphBuilder(
            origin,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["Expand"]),
        )
        onx = gr.to_onnx(optimize=True)
        after = [node for node in onx.graph.node if node.op_type == "Expand"]
        self.assertEqual(len(after), len(before) - 5)

    def test_expand_execution(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xu1"]),
                    oh.make_node("Expand", ["xu1", "expand_shape"], ["xm1"]),
                    oh.make_node("Cast", ["Y"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [32, 128]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5, 128, 64]),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [3, 5, 32, 64])],
                [
                    onh.from_array(
                        np.array([1, 32, 128], dtype=np.int64), name="shape1"
                    ),
                    onh.from_array(
                        np.array([1, 32, 128], dtype=np.int64), name="expand_shape"
                    ),
                    onh.from_array(
                        np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"
                    ),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["Expand"]),
        )
        s = str(gr.optimization_options)
        self.assertIn("OptimizationOptions(", s)
        self.assertIn("ExpandPattern", s)
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Reshape", "Cast", "MatMul", "Reshape"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(2, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_transpose_transpose_dort(self):
        origin = self._get_model("dort-c-custom__1.onnx")
        before = [node for node in origin.graph.node if node.op_type == "Transpose"]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(
                patterns=["TransposeTranspose"], verbose=0
            ),
        )
        onx = gr.to_onnx(optimize=True)
        after = [node for node in onx.graph.node if node.op_type == "Transpose"]
        self.assertEqual(len(before) - 14, len(after))

    def test_transpose_transpose_execution(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "s1"], ["xs"]),
                    oh.make_node("Transpose", ["xs"], ["r1"], perm=[1, 0, 3, 2]),
                    oh.make_node("Transpose", ["r1"], ["xm1"], perm=[1, 0, 3, 2]),
                    oh.make_node("MatMul", ["xm1", "Y"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [32, 128]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5, 128, 64]),
                ],
                [
                    oh.make_tensor_value_info("Z", TensorProto.FLOAT, [3, 5, 32, 64]),
                    oh.make_tensor_value_info("r1", TensorProto.FLOAT, [1, 1, 128, 32]),
                ],
                [onh.from_array(np.array([1, 1, 32, 128], dtype=np.int64), name="s1")],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["TransposeTranspose"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Reshape", "Transpose", "MatMul"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            self.assertEqualArray(a, b)

    def test_transpose_matmul_dort(self):
        origin = self._get_model("dort-c-custom__0.onnx")
        before = [node for node in origin.graph.node if node.op_type == "Transpose"]
        gr = GraphBuilder(
            origin,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["TransposeMatMul"]),
        )
        onx = gr.to_onnx(optimize=True)
        after = [node for node in onx.graph.node if node.op_type == "Transpose"]
        self.assertEqual(len(before), len(after))
        self.assertIn("Gemm", set(n.op_type for n in onx.graph.node))

    def test_transpose_matmul_execution_matmul(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["xm1"], perm=[1, 0]),
                    oh.make_node("MatMul", ["xm1", "Y"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [128, 32]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [128, 64]),
                ],
                [
                    oh.make_tensor_value_info("Z", TensorProto.FLOAT, [32, 64]),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(128, 32), "Y": self._range(128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["TransposeMatMul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Gemm"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_transpose_matmul_execution_gemm(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["xm1"], perm=[1, 0]),
                    oh.make_node(
                        "Gemm",
                        ["xm1", "Y"],
                        ["Z"],
                        alpha=2.0,
                        beta=0.0,
                        transA=1,
                        transB=1,
                    ),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [32, 128]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [64, 128]),
                ],
                [
                    oh.make_tensor_value_info("Z", TensorProto.FLOAT, [32, 64]),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(64, 128)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["TransposeMatMul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Gemm"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_rotary_concat_part_dort(self):
        origin = self._get_model("dort-c-custom__1.onnx")
        before = [
            node for node in origin.graph.node if node.op_type == "ConstantOfShape"
        ]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(
                patterns=["RotaryConcatPart"], verbose=0
            ),
        )
        onx = gr.to_onnx(optimize=True)
        after = [node for node in onx.graph.node if node.op_type == "ConstantOfShape"]
        self.assertEqual(len(before) - 4, len(after))

    def test_rotary_concat_part_execution_1(self):
        from onnx_array_api.light_api import start

        def mk(shape):
            return np.array(shape, dtype=np.int64)

        model = (
            start(opset=18, ir_version=9)
            .cst(mk([2, 2, 1024, 256]), "shape")
            .cst(mk([0]), "c0")
            .cst(mk([256]), "c256")
            .cst(mk([512]), "c512")
            .cst(mk([3]), "c3")
            .vin("X", TensorProto.FLOAT, ("a", "b", "c", "d"))
            .bring("shape")
            .ConstantOfShape()
            .rename("C1")
            .bring("shape")
            .ConstantOfShape()
            .rename("C2")
            .bring("X", "c256", "c512", "c3")
            .Slice()
            .rename("S1")
            .bring("C1", "S1")
            .Concat(axis=3)
            .rename("P1")
            .bring("X", "c0", "c256", "c3")
            .Slice()
            .rename("notused")
            .Neg()
            .rename("S2")
            .bring("S2", "C2")
            .Concat(axis=3)
            .rename("P2")
            .bring("P1", "P2")
            .Add()
            .rename("Y")
            .vout(TensorProto.FLOAT, ("a", "b", "c", "d"))
            .to_onnx()
        )
        check_model(model)

        feeds = {"X": self._range(2, 2, 1024, 512)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["RotaryConcatPart"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Slice", "Slice", "Neg", "Concat"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(4, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_rotary_concat_part_execution_2(self):
        from onnx_array_api.light_api import start

        def mk(shape):
            return np.array(shape, dtype=np.int64)

        model = (
            start(opset=18, ir_version=9)
            .cst(mk([2, 2, 1024, 256]), "shape")
            .cst(mk([0]), "c0")
            .cst(mk([256]), "c256")
            .cst(mk([512]), "c512")
            .cst(mk([3]), "c3")
            .vin("X", TensorProto.FLOAT, ("a", "b", "c", "d"))
            .bring("shape")
            .ConstantOfShape()
            .rename("C1")
            .bring("shape")
            .ConstantOfShape()
            .rename("C2")
            .bring("X", "c256", "c512", "c3")
            .Slice()
            .Neg()
            .rename("S1")
            .bring("C1", "S1")
            .Concat(axis=3)
            .rename("P1")
            .bring("X", "c0", "c256", "c3")
            .Slice()
            .rename("S2")
            .bring("S2", "C2")
            .Concat(axis=3)
            .rename("P2")
            .bring("P1", "P2")
            .Add()
            .rename("Y")
            .vout(TensorProto.FLOAT, ("a", "b", "c", "d"))
            .to_onnx()
        )
        check_model(model)

        feeds = {"X": self._range(2, 2, 1024, 512)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["RotaryConcatPart"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Slice", "Slice", "Neg", "Concat"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(4, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_mul_mul_mul(self):
        from onnx_array_api.light_api import start

        def mk(shape):
            return np.array(shape, dtype=np.float32)

        model = (
            start(opset=18, ir_version=9)
            .cst(mk([2]), "cst1")
            .cst(mk([3]), "cst2")
            .vin("X", TensorProto.FLOAT, ("a", "b"))
            .vin("Y", TensorProto.FLOAT, ("a", "b"))
            .bring("X", "cst1")
            .Mul()
            .rename("xc")
            .bring("Y", "cst2")
            .Mul()
            .rename("yc")
            .bring("xc", "yc")
            .Mul()
            .rename("Z")
            .vout(TensorProto.FLOAT, ("a", "b"))
            .to_onnx()
        )
        check_model(model)

        feeds = {"X": self._range(3, 3), "Y": self._range(3, 3)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["MulMulMul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Mul", "Mul"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_div_div_mul(self):
        from onnx_array_api.light_api import start

        def mk(shape):
            return np.array(shape, dtype=np.float32)

        model = (
            start(opset=18, ir_version=9)
            .cst(mk([2]), "cst1")
            .cst(mk([3]), "cst2")
            .vin("X", TensorProto.FLOAT, ("a", "b"))
            .vin("Y", TensorProto.FLOAT, ("a", "b"))
            .bring("X", "cst1")
            .Div()
            .rename("xc")
            .bring("Y", "cst2")
            .Div()
            .rename("yc")
            .bring("xc", "yc")
            .Mul()
            .rename("Z")
            .vout(TensorProto.FLOAT, ("a", "b"))
            .to_onnx()
        )
        check_model(model)

        feeds = {"X": self._range(3, 3), "Y": self._range(3, 3)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["MulMulMul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Mul", "Mul"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def common_sub1_mul(self, side):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ConstantOfShape",
                        ["shape"],
                        ["one"],
                        value=onh.from_array(np.array([1], dtype=np.float32)),
                    ),
                    oh.make_node(
                        "Sub", ["one", "X" if side == "left" else "Y"], ["i1"]
                    ),
                    oh.make_node(
                        "Mul", ["i1", "Y"] if side == "left" else ["X", "i1"], ["Z"]
                    ),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", 6]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["a", 6]),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.FLOAT, ["a", 6])],
                [
                    onh.from_array(np.array([1, 6], dtype=np.int64), name="shape"),
                ],
            )
        )
        check_model(model)

        feeds = {"X": self._range(11, 6), "Y": self._range(11, 6, bias=0.5)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["Sub1Mul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Mul", "Sub"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_sub1_mul_left(self):
        self.common_sub1_mul("left")

    def test_sub1_mul_right(self):
        self.common_sub1_mul("right")

    def test_sub1_mul_data(self):
        origin = self._get_model("basic_static_1.onnx")
        check_model(origin)
        node_list = [(n.op_type, tuple(n.output)) for n in origin.graph.node]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(patterns=["Sub1Mul"]),
        )
        onx = gr.to_onnx(optimize=True)
        check_model(onx)
        new_node_list = [(n.op_type, tuple(n.output)) for n in onx.graph.node]
        self.assertNotEqual(node_list, new_node_list)

    def test_statistics(self):
        from onnx_array_api.light_api import start

        def mk(shape):
            return np.array(shape, dtype=np.float32)

        model = (
            start(opset=18, ir_version=9)
            .cst(mk([2]), "cst1")
            .cst(mk([3]), "cst2")
            .vin("X", TensorProto.FLOAT, ("a", "b"))
            .vin("Y", TensorProto.FLOAT, ("a", "b"))
            .bring("X", "cst1")
            .Div()
            .rename("xc")
            .bring("Y", "cst2")
            .Div()
            .rename("yc")
            .bring("xc", "yc")
            .Mul()
            .rename("Z")
            .vout(TensorProto.FLOAT, ("a", "b"))
            .to_onnx()
        )
        check_model(model)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["MulMulMul"]),
        )
        stats = gr.optimize()
        self.assertEqual(
            stats,
            [
                {
                    "pattern": "MulMulMulPattern",
                    "added": 2,
                    "removed": 3,
                    "iteration": 0,
                    "match_index": 0,
                }
            ],
        )

    def test_sub2_mul_data(self):
        origin = self._get_model("dort-cus-custom__1_sub.onnx")
        node_list = [(n.op_type, tuple(n.output)) for n in origin.graph.node]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(patterns=["Sub1Mul"]),
        )
        stat = gr.optimize()
        self.assertEqual(len(stat), 3)
        onx = gr.to_onnx(optimize=False)
        csts = [i for i in onx.graph.node if "Constant" in i.op_type]
        cst_output = set(i.output[0] for i in csts)
        self.assertNotIn("fill", cst_output)
        self.assertNotIn("fill_1", cst_output)
        new_node_list = [(n.op_type, tuple(n.output)) for n in onx.graph.node]
        self.assertNotEqual(node_list, new_node_list)

    def common_expand_broadcast(self, side):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Expand", ["X", "shape"], ["i1"]),
                    oh.make_node(
                        "Mul", ["i1", "Y"] if side == "left" else ["Y", "i1"], ["Z"]
                    ),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 1]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 6]),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 4, 6])],
                [
                    onh.from_array(np.array([2, 4, 6], dtype=np.int64), name="shape"),
                ],
            )
        )
        check_model(model)

        feeds = {"X": self._range(1, 4, 1), "Y": self._range(2, 4, 6, bias=0.5)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["ExpandBroadcast"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Mul"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_expand_broadcast_left(self):
        self.common_expand_broadcast("left")

    def test_expand_broadcast_right(self):
        self.common_expand_broadcast("right")

    def test_expand_broadcast_data(self):
        origin = self._get_model("dort-cus-custom__1_sub.onnx")
        node_list = [n.op_type for n in origin.graph.node]
        gr = GraphBuilder(
            origin,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["ExpandBroadcast"]),
        )
        stat = gr.optimize()
        self.assertEqual(len(stat), 5)
        onx = gr.to_onnx(optimize=False)
        new_node_list = [n.op_type for n in onx.graph.node]
        self.assertNotEqual(node_list, new_node_list)

    def test_reshape_2of3_static_3(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["yr"]),
                    oh.make_node("Mul", ["xr", "yr"], ["xrr"]),
                    oh.make_node("Reshape", ["xrr", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4]),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3, 4])],
                [
                    onh.from_array(np.array([-1, 8], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([3, -1], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([2, 3, 4], dtype=np.int64), name="shape3"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(2, 3, 4), "Y": self._range(2, 3, 4)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["Reshape2Of3"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Mul"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reshape_2of3_static_3_keep(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["yr"]),
                    oh.make_node("Mul", ["xr", "yr"], ["xrr"]),
                    oh.make_node("Reshape", ["xrr", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4]),
                ],
                [
                    oh.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3, 4]),
                    oh.make_tensor_value_info("xrr", TensorProto.FLOAT, [3, 8]),
                ],
                [
                    onh.from_array(np.array([-1, 8], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([3, -1], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([2, 3, 4], dtype=np.int64), name="shape3"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(2, 3, 4), "Y": self._range(2, 3, 4)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["Reshape2Of3"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Mul", "Reshape"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArrays(expected, got)

    def test_reshape_2of3_static_2_left(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr1"]),
                    oh.make_node("Mul", ["xr1", "Y"], ["xr"]),
                    oh.make_node("Reshape", ["xr", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 8]),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3, 4])],
                [
                    onh.from_array(np.array([-1, 8], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([2, 3, 4], dtype=np.int64), name="shape3"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(2, 3, 4), "Y": self._range(3, 8)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["Reshape2Of3"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Reshape", "Mul"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reshape_2of3_static_2_right(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr1"]),
                    oh.make_node("Mul", ["Y", "xr1"], ["xr"]),
                    oh.make_node("Reshape", ["xr", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 8]),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3, 4])],
                [
                    onh.from_array(np.array([-1, 8], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([2, 3, 4], dtype=np.int64), name="shape3"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(2, 3, 4), "Y": self._range(3, 8)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["Reshape2Of3"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Reshape", "Mul"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reshape_2of3_static_2_left_right(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["yr1"]),
                    oh.make_node("Mul", ["xr1", "yr1"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4]),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [3, 8])],
                [
                    onh.from_array(np.array([-1, 8], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([3, -1], dtype=np.int64), name="shape2"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(2, 3, 4), "Y": self._range(2, 3, 4)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["Reshape2Of3"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Mul", "Reshape"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
