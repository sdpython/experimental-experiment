import os
import unittest
import numpy as np
from onnx import (
    ModelProto,
    TensorProto,
    helper as oh,
    numpy_helper as onh,
    load as load_onnx,
)
from onnx.checker import check_model
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)

TFLOAT = TensorProto.FLOAT


class TestGraphPatternCombination(ExtTestCase):

    def _check_ort_cpu(self, onx):
        import onnxruntime

        onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )

    def _get_model(self, name: str) -> ModelProto:
        p = os.path.join(os.path.dirname(__file__), "..", "ut_xbuilder", "data", name)
        if not os.path.exists(p):
            p = os.path.join(os.path.dirname(__file__), "data", name)
        self.assertExists(p)
        return load_onnx(p)

    def _range(self, *shape, bias: float = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def test_reshape_matmul_reshape_static(self):
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
                    oh.make_tensor_value_info("X", TFLOAT, [32, 128]),
                    oh.make_tensor_value_info("Y", TFLOAT, [3, 5, 128, 64]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 32, 64])],
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

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=[
                    "Cast",
                    "ReshapeMatMulReshape",
                    "UnsqueezeUnsqueeze",
                    "MatMulReshape2Of3",
                    "ReshapeReshape",
                ],
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reshape_matmul_reshape_dynamic_1(self):
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
                    oh.make_tensor_value_info("X", TFLOAT, ["D32", "D128"]),
                    oh.make_tensor_value_info(
                        "Y", TFLOAT, ["batch", "channel", "D128", "D64"]
                    ),
                ],
                [
                    oh.make_tensor_value_info(
                        "Z", TFLOAT, ["batch", "channel", "D32", "64"]
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

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=[
                    "Cast",
                    "ReshapeMatMulReshape",
                    "UnsqueezeUnsqueeze",
                    "MatMulReshape2Of3",
                    "ReshapeReshape",
                ],
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reshape_matmul_reshape_dynamic_2(self):
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
                    oh.make_tensor_value_info("X", TFLOAT, ["D32", "D128"]),
                    oh.make_tensor_value_info(
                        "Y", TFLOAT, ["batch", "channel", "any", "D64"]
                    ),
                ],
                [
                    oh.make_tensor_value_info(
                        "Z", TFLOAT, ["batch", "channel", "D32", "64"]
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

        gr = GraphBuilder(
            model,
            optimization_options=OptimizationOptions(
                patterns=[
                    "Cast",
                    "ReshapeMatMulReshape",
                    "UnsqueezeUnsqueeze",
                    "MatMulReshape2Of3",
                    "ReshapeReshape",
                ]
            ),
            infer_shapes=True,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reshape_matmul_reshape_keep_intermediate(self):
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
                    oh.make_tensor_value_info("X", TFLOAT, [32, 128]),
                    oh.make_tensor_value_info("Y", TFLOAT, [3, 5, 128, 64]),
                ],
                [
                    oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 32, 64]),
                    oh.make_tensor_value_info("xm1", TFLOAT, [1, 32, 128]),
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

        gr = GraphBuilder(
            model,
            optimization_options=OptimizationOptions(
                patterns=[
                    "Cast",
                    "ReshapeMatMulReshape",
                    "UnsqueezeUnsqueeze",
                    "MatMulReshape2Of3",
                    "ReshapeReshape",
                ],
                verbose=10,
            ),
            infer_shapes=True,
        )
        s = str(gr.optimization_options)
        self.assertIn("OptimizationOptions(", s)
        self.assertIn("CastPattern", s)
        opt_onx, out, _ = self.capture(lambda: gr.to_onnx(optimize=True))
        self.assertIn("remove_initializer:shape2", out)
        self.assertEqual(
            ["Unsqueeze", "Reshape", "MatMul"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(2, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_simplified_only(self):
        for model in ["noopt-llama-custom__0.onnx"]:
            onx = self._get_model(model)
            with self.subTest(model=model):
                gr = GraphBuilder(
                    onx,
                    optimization_options=OptimizationOptions(
                        patterns=["SimplifiedLayerNormalization"],
                        verbose=20,
                    ),
                    infer_shapes=True,
                )
                onx = gr.to_onnx(optimize=True)
                types = set([n.op_type for n in onx.graph.node])
                self.assertIn("SimplifiedLayerNormalization", types)
                self._check_ort_cpu(onx)

    def test_simplified_with_all(self):
        for model in [
            "noopt-llama-custom__0.onnx",
            "noopt-llama-custom__1.onnx",
            "noopt-phi-custom__0.onnx",
            "noopt-phi-custom__1.onnx",
        ]:
            onx = self._get_model(model)
            with self.subTest(model=model):
                gr = GraphBuilder(
                    onx,
                    optimization_options=OptimizationOptions(
                        patterns="default+onnxruntime",
                        verbose=0,
                    ),
                    infer_shapes=True,
                )
                onx = gr.to_onnx(optimize=True)
                self._check_ort_cpu(onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
