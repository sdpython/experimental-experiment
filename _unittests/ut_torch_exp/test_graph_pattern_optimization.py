import os
import unittest
import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper as oh, numpy_helper as onh
from onnx.checker import check_model
from onnx_array_api.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import ExtTestCase, ignore_warnings
from experimental_experiment.torch_exp.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.torch_exp.graph_builder_optim import (
    GraphBuilderPatternOptimization,
)
from experimental_experiment.torch_exp.annotations import (
    compatible_shapes,
    compatible_dimensions,
)


class TestGraphPatternOptimization(ExtTestCase):
    def _check_with_ort(self, proto: ModelProto):
        from onnxruntime import InferenceSession, get_available_providers

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        InferenceSession(proto.SerializeToString(), providers=providers)

    @ignore_warnings(DeprecationWarning)
    def test_try_with_custom_model(self):
        filename = "onnx_model_init_1.onnx"
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
        p = os.path.join(os.path.dirname(__file__), "data", name)
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
        self.assertEmpty(res)
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
        self.assertEmpty(res)
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
        self.assertEmpty(res)
        self.assertIn("[GraphBuilderPatternOptimization.optimize] done after", out)
        self.assertIn("ReshapeMatMulReshapePattern", out)

        onx = gr.to_onnx(optimize=False)
        after = [node for node in onx.graph.node if node.op_type == "Reshape"]
        self.assertEqual(len(before), 24)
        self.assertEqual(len(after), 22)

    def _range(self, *shape):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
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
            optimization_options=OptimizationOptions(patterns=["Expand"], verbose=0),
        )
        onx = gr.to_onnx(optimize=True)
        after = [node for node in onx.graph.node if node.op_type == "Expand"]
        self.assertEqual(len(after), len(before) - 3)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
