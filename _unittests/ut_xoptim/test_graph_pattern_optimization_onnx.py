"""
Use:

::

    LOG_PATTERN_OPTIMIZE=10 \\
    python _unittests/ut_xoptim/test_graph_pattern_optimization.py \\
        -k test_rotary_concat_part_plug
"""

import os
import unittest
import pprint
from typing import Optional
import numpy as np
import onnx
from onnx import (
    ModelProto,
    TensorProto,
    helper as oh,
    numpy_helper as onh,
    load as onnx_load,
)
from onnx.checker import check_model
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_onnx,
    requires_torch,
    hide_stdout,
)
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.xoptim.graph_builder_optim import (
    GraphBuilderPatternOptimization,
)
from experimental_experiment.xbuilder._shape_helper import (
    compatible_shapes,
    compatible_dimensions,
)
from experimental_experiment.xoptim import get_pattern_list
from experimental_experiment.helpers import pretty_onnx

TFLOAT = TensorProto.FLOAT
TFLOAT16 = TensorProto.FLOAT16
TINT64 = TensorProto.INT64
_mkv_ = oh.make_tensor_value_info


class TestGraphPatternOptimization(ExtTestCase):
    def test_get_pattern_list(self):
        res = get_pattern_list(negative_list=["Cast"])
        names = set(r.__class__.__name__ for r in res)
        self.assertNotIn("CastPattern", names)
        res = get_pattern_list(negative_list="default")
        self.assertEqual(res, [])

    def test_get_pattern_list_plus(self):
        res1 = get_pattern_list("default")
        self.assertGreater(len(res1), 4)
        res2 = get_pattern_list("onnxruntime")
        self.assertGreater(len(res2), 1)
        res = get_pattern_list("default+onnxruntime")
        self.assertGreater(len(res), 3)
        self.assertEqual(res1 + res2, res)

    def test_get_pattern_list_plus_list(self):
        res1 = get_pattern_list("default")
        self.assertGreater(len(res1), 4)
        res2 = get_pattern_list("onnxruntime")
        self.assertGreater(len(res2), 1)
        res = get_pattern_list(["default+onnxruntime"])
        self.assertGreater(len(res), 3)
        self.assertEqual(res1 + res2, res)

    def _check_with_ort(self, proto: ModelProto):
        from onnxruntime import InferenceSession, get_available_providers
        from onnxruntime.capi.onnxruntime_pybind11_state import Fail

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        try:
            InferenceSession(proto.SerializeToString(), providers=providers)
        except Fail as e:
            saved = self.dump_onnx("test_graph_pattern_optimization.onnx", proto)
            raise AssertionError(f"Fails due to {e}, model saved into {saved!r}")  # noqa: B904

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
                verbose=0 if __name__ == "__main__" else 0,
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
        if not os.path.exists(p):
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
        self.assertEqual(dtype, TFLOAT)

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

    def _range(self, *shape, bias: Optional[float] = None):
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
                [_mkv_("X", TFLOAT, [32, 128]), _mkv_("Y", TFLOAT, [3, 5, 128, 64])],
                [_mkv_("Z", TFLOAT, [3, 5, 32, 64])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr, _, __ = self.capture(
            lambda: GraphBuilder(
                model,
                infer_shapes=True,
                optimization_options=OptimizationOptions(
                    patterns=["Cast", "ReshapeMatMulReshape", "UnsqueezeUnsqueeze"],
                    verbose=10,
                ),
                verbose=10,
            )
        )
        s = str(gr.optimization_options)
        self.assertIn("OptimizationOptions(", s)
        self.assertIn("CastPattern", s)
        opt_onx, out, _ = self.capture(lambda: gr.to_onnx(optimize=True))
        self.assertIn("remove_initializer 3:2/6:shape1", out)
        self.assertIn("remove_initializer 4:3/6:shape2", out)
        self.assertEqual(["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node])
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
                    _mkv_("X", TFLOAT, ["D32", "D128"]),
                    _mkv_("Y", TFLOAT, ["batch", "channel", "D128", "D64"]),
                ],
                [_mkv_("Z", TFLOAT, ["batch", "channel", "D32", "64"])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
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
                patterns=["Cast", "ReshapeMatMulReshape", "UnsqueezeUnsqueeze"],
            ),
            infer_shapes=True,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node])
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
                    _mkv_("X", TFLOAT, ["D32", "D128"]),
                    _mkv_("Y", TFLOAT, ["batch", "channel", "D128", "D64"]),
                ],
                [_mkv_("Z", TFLOAT, ["batch", "channel", "D32", "64"])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
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
                patterns=["Cast", "ReshapeMatMulReshape", "UnsqueezeUnsqueeze"]
            ),
            infer_shapes=True,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node])
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
                [_mkv_("X", TFLOAT, [32, 128]), _mkv_("Y", TFLOAT, [3, 5, 128, 64])],
                [_mkv_("Z", TFLOAT, [3, 5, 32, 64]), _mkv_("xm1", TFLOAT, [1, 32, 128])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr, _, __ = self.capture(
            lambda: GraphBuilder(
                model,
                optimization_options=OptimizationOptions(
                    patterns=["Cast", "ReshapeMatMulReshape", "UnsqueezeUnsqueeze"],
                    verbose=10,
                ),
                infer_shapes=True,
                verbose=10,
            )
        )
        s = str(gr.optimization_options)
        self.assertIn("OptimizationOptions(", s)
        self.assertIn("CastPattern", s)
        opt_onx, out, _ = self.capture(lambda: gr.to_onnx(optimize=True))
        self.assertIn("remove_initializer 3:3/6:shape2", out)
        self.assertEqual(
            ["Unsqueeze", "Reshape", "MatMul"], [n.op_type for n in opt_onx.graph.node]
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
            optimization_options=OptimizationOptions(patterns=["ReshapeReshape"]),
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
                [_mkv_("X", TFLOAT, [32, 128]), _mkv_("Y", TFLOAT, [3, 5, 128, 64])],
                [_mkv_("Z", TFLOAT, [3, 5, 32, 64])],
                [
                    onh.from_array(np.array([-1], dtype=np.int64), name="r1"),
                    onh.from_array(np.array([-1, 128], dtype=np.int64), name="r2"),
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
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
                [_mkv_("X", TFLOAT, [32, 128]), _mkv_("Y", TFLOAT, [3, 5, 128, 64])],
                [_mkv_("Z", TFLOAT, [3, 5, 32, 64])],
                [
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(
                        np.array([1, 32, 128], dtype=np.int64), name="expand_shape"
                    ),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
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
            optimization_options=OptimizationOptions(patterns=["TransposeTranspose"]),
        )
        onx = gr.to_onnx(optimize=True)
        after = [node for node in onx.graph.node if node.op_type == "Transpose"]
        self.assertEqual(len(before) - 14, len(after))

    def test_transpose_transpose_execution_id(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "s1"], ["xs"]),
                    oh.make_node("Transpose", ["xs"], ["r1"], perm=[1, 0, 3, 2]),
                    oh.make_node("Transpose", ["r1"], ["xm1"], perm=[1, 0, 3, 2]),
                    oh.make_node("MatMul", ["xm1", "Y"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [32, 128]), _mkv_("Y", TFLOAT, [3, 5, 128, 64])],
                [
                    _mkv_("Z", TFLOAT, [3, 5, 32, 64]),
                    _mkv_("r1", TFLOAT, [1, 1, 128, 32]),
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

    def test_transpose_transpose_execution_one_left(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "s1"], ["xs"]),
                    oh.make_node("Transpose", ["xs"], ["r1"], perm=[1, 0, 3, 2]),
                    oh.make_node("Transpose", ["r1"], ["xm1"], perm=[0, 1, 3, 2]),
                    oh.make_node("MatMul", ["xm1", "Y"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [32, 128]), _mkv_("Y", TFLOAT, [3, 5, 128, 64])],
                [
                    _mkv_("Z", TFLOAT, [3, 5, 32, 64]),
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
                [_mkv_("X", TFLOAT, [128, 32]), _mkv_("Y", TFLOAT, [128, 64])],
                [
                    _mkv_("Z", TFLOAT, [32, 64]),
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
                [_mkv_("X", TFLOAT, [32, 128]), _mkv_("Y", TFLOAT, [64, 128])],
                [_mkv_("Z", TFLOAT, [32, 64])],
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
        before = [node for node in origin.graph.node if node.op_type == "ConstantOfShape"]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(patterns=["RotaryConcatPart"]),
            infer_shapes=True,
        )
        onx = gr.to_onnx(optimize=True)
        after = [node for node in onx.graph.node if node.op_type == "ConstantOfShape"]
        self.assertEqual(len(before) - 4, len(after))

    @unittest.skipIf(True, "not yet completed")
    def test_rotary_concat_part_plug(self):
        origin = self._get_model("dort-pres-plug_1.onnx")
        before = [node for node in origin.graph.node if node.op_type == "ConstantOfShape"]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(
                patterns=["RotaryConcatPart"], verbose=20
            ),
            infer_shapes=True,
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
            .vin("X", TFLOAT, ("a", "b", "c", "d"))
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
            .vout(TFLOAT, ("a", "b", "c", "d"))
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
            .vin("X", TFLOAT, ("a", "b", "c", "d"))
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
            .vout(TFLOAT, ("a", "b", "c", "d"))
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
            .vin("X", TFLOAT, ("a", "b"))
            .vin("Y", TFLOAT, ("a", "b"))
            .bring("X", "cst1")
            .Mul()
            .rename("xc")
            .bring("Y", "cst2")
            .Mul()
            .rename("yc")
            .bring("xc", "yc")
            .Mul()
            .rename("Z")
            .vout(TFLOAT, ("a", "b"))
            .to_onnx()
        )
        check_model(model)

        feeds = {"X": self._range(3, 3), "Y": self._range(3, 3)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["MulMulMulScalar"]),
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
            .vin("X", TFLOAT, ("a", "b"))
            .vin("Y", TFLOAT, ("a", "b"))
            .bring("X", "cst1")
            .Div()
            .rename("xc")
            .bring("Y", "cst2")
            .Div()
            .rename("yc")
            .bring("xc", "yc")
            .Mul()
            .rename("Z")
            .vout(TFLOAT, ("a", "b"))
            .to_onnx()
        )
        check_model(model)

        feeds = {"X": self._range(3, 3), "Y": self._range(3, 3)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["MulMulMulScalar"]),
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
                    oh.make_node("Sub", ["one", "X" if side == "left" else "Y"], ["i1"]),
                    oh.make_node("Mul", ["i1", "Y"] if side == "left" else ["X", "i1"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", 6]), _mkv_("Y", TFLOAT, ["a", 6])],
                [_mkv_("Z", TFLOAT, ["a", 6])],
                [onh.from_array(np.array([1, 6], dtype=np.int64), name="shape")],
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
            .vin("X", TFLOAT, ("a", "b"))
            .vin("Y", TFLOAT, ("a", "b"))
            .bring("X", "cst1")
            .Div()
            .rename("xc")
            .bring("Y", "cst2")
            .Div()
            .rename("yc")
            .bring("xc", "yc")
            .Mul()
            .rename("Z")
            .vout(TFLOAT, ("a", "b"))
            .to_onnx()
        )
        check_model(model)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["MulMulMulScalar"]),
        )
        stats = gr.optimize()
        stats = [
            {k: v for k, v in st.items() if k != "time_in"}
            for st in stats
            if "match_MulMulMulScalarPattern" in st["pattern"]
        ]
        self.assertEqual(
            stats,
            [
                {
                    "pattern": "match_MulMulMulScalarPattern",
                    "iteration": 0,
                    "instances": 1,
                    "match_index": 1,
                },
                {
                    "pattern": "match_MulMulMulScalarPattern",
                    "iteration": 1,
                    "instances": 0,
                    "match_index": 0,
                },
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
        self.assertGreater(len(stat), 20)
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
                    oh.make_node("Mul", ["i1", "Y"] if side == "left" else ["Y", "i1"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [1, 4, 1]), _mkv_("Y", TFLOAT, [2, 4, 6])],
                [_mkv_("Z", TFLOAT, [2, 4, 6])],
                [onh.from_array(np.array([2, 4, 6], dtype=np.int64), name="shape")],
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
        self.assertGreater(len(stat), 26)
        onx = gr.to_onnx(optimize=False)
        new_node_list = [n.op_type for n in onx.graph.node]
        self.assertNotEqual(node_list, new_node_list)

    def test_mul_reshape_2of3_static_3(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["yr"]),
                    oh.make_node("Mul", ["xr", "yr"], ["xrr"]),
                    oh.make_node("Reshape", ["xrr", "shape3"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [2, 3, 4]), _mkv_("Y", TFLOAT, [2, 3, 4])],
                [_mkv_("Z", TFLOAT, [2, 3, 4])],
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

    def test_mul_reshape_2of3_static_3_keep(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["yr"]),
                    oh.make_node("Mul", ["xr", "yr"], ["xrr"]),
                    oh.make_node("Reshape", ["xrr", "shape3"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [2, 3, 4]), _mkv_("Y", TFLOAT, [2, 3, 4])],
                [_mkv_("Z", TFLOAT, [2, 3, 4]), _mkv_("xrr", TFLOAT, [3, 8])],
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

    def test_mul_reshape_2of3_static_2_left(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr1"]),
                    oh.make_node("Mul", ["xr1", "Y"], ["xr"]),
                    oh.make_node("Reshape", ["xr", "shape3"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [2, 3, 4]), _mkv_("Y", TFLOAT, [3, 8])],
                [_mkv_("Z", TFLOAT, [2, 3, 4])],
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

    def test_mul_reshape_2of3_static_2_right(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr1"]),
                    oh.make_node("Mul", ["Y", "xr1"], ["xr"]),
                    oh.make_node("Reshape", ["xr", "shape3"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [2, 3, 4]), _mkv_("Y", TFLOAT, [3, 8])],
                [_mkv_("Z", TFLOAT, [2, 3, 4])],
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

    def test_mul_reshape_2of3_static_2_left_right(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["yr1"]),
                    oh.make_node("Mul", ["xr1", "yr1"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [2, 3, 4]), _mkv_("Y", TFLOAT, [2, 3, 4])],
                [_mkv_("Z", TFLOAT, [3, 8])],
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
            optimization_options=OptimizationOptions(patterns=["Reshape2Of3"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Mul", "Reshape"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_matmul_reshape_2of3_static_3(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["yr"]),
                    oh.make_node("MatMul", ["xr", "yr"], ["xrr"]),
                    oh.make_node("Reshape", ["xrr", "shape3"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [2, 2, 3, 4]), _mkv_("Y", TFLOAT, [2, 2, 4, 3])],
                [_mkv_("Z", TFLOAT, [2, 2, 3, 3])],
                [
                    onh.from_array(np.array([-1, 3, 4], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([-1, 4, 3], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([2, 2, 3, 3], dtype=np.int64), name="shape3"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 3, 4), "Y": self._range(2, 2, 4, 3)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["MatMulReshape2Of3"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["MatMul"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_matmul_reshape_2of3_static_3_keep(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["yr"]),
                    oh.make_node("MatMul", ["xr", "yr"], ["xrr"]),
                    oh.make_node("Reshape", ["xrr", "shape3"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [2, 2, 3, 4]), _mkv_("Y", TFLOAT, [2, 2, 4, 3])],
                [_mkv_("Z", TFLOAT, [2, 2, 3, 3]), _mkv_("xrr", TFLOAT, [4, 3, 3])],
                [
                    onh.from_array(np.array([-1, 3, 4], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([-1, 4, 3], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([2, 2, 3, 3], dtype=np.int64), name="shape3"),
                ],
            )
        )
        check_model(model)
        onnx.shape_inference.infer_shapes(model)
        feeds = {"X": self._range(2, 2, 3, 4), "Y": self._range(2, 2, 4, 3)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["MatMulReshape2Of3"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["MatMul", "Reshape"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArrays(expected, got)

    def test_matmul_reshape_2of3_static_2_left(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr1"]),
                    oh.make_node("MatMul", ["xr1", "Y"], ["xr"]),
                    oh.make_node("Reshape", ["xr", "shape3"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [2, 2, 3, 4]), _mkv_("Y", TFLOAT, [4, 4, 3])],
                [_mkv_("Z", TFLOAT, [2, 2, 3, 3])],
                [
                    onh.from_array(np.array([4, 3, 4], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([2, 2, 3, 3], dtype=np.int64), name="shape3"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 3, 4), "Y": self._range(4, 4, 3)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["MatMulReshape2Of3"]),
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(["Reshape", "MatMul"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_matmul_reshape_2of3_static_2_right(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr1"]),
                    oh.make_node("MatMul", ["Y", "xr1"], ["xr"]),
                    oh.make_node("Reshape", ["xr", "shape3"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [2, 2, 4, 3]), _mkv_("Y", TFLOAT, [4, 3, 4])],
                [_mkv_("Z", TFLOAT, [2, 2, 3, 3])],
                [
                    onh.from_array(np.array([-1, 4, 3], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([2, 2, 3, 3], dtype=np.int64), name="shape3"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 4, 3), "Y": self._range(4, 3, 4)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["MatMulReshape2Of3"]),
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(["Reshape", "MatMul"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_matmul_reshape_2of3_static_2_left_right(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["yr1"]),
                    oh.make_node("MatMul", ["xr1", "yr1"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [2, 2, 3, 4]), _mkv_("Y", TFLOAT, [2, 2, 4, 3])],
                [_mkv_("Z", TFLOAT, [4, 3, 3])],
                [
                    onh.from_array(np.array([-1, 3, 4], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([-1, 4, 3], dtype=np.int64), name="shape2"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 3, 4), "Y": self._range(2, 2, 4, 3)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["MatMulReshape2Of3"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["MatMul", "Reshape"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reduce_reshape(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("ReduceSum", ["X", "axes"], ["xr"], keepdims=1),
                    oh.make_node("Reshape", ["xr", "shape"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [3, 2])],
                [_mkv_("Y", TFLOAT, [3])],
                [
                    onh.from_array(np.array([1], dtype=np.int64), name="axes"),
                    onh.from_array(np.array([3], dtype=np.int64), name="shape"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(3, 2)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["ReduceReshape"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["ReduceSum"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reduce_reshape_2d(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("ReduceSum", ["X", "axes"], ["xr"], keepdims=1),
                    oh.make_node("Reshape", ["xr", "shape"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [4, 3, 2])],
                [_mkv_("Y", TFLOAT, [3])],
                [
                    onh.from_array(np.array([0, 2], dtype=np.int64), name="axes"),
                    onh.from_array(np.array([3], dtype=np.int64), name="shape"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(4, 3, 2)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["ReduceReshape"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["ReduceSum"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @requires_onnx("1.16.0", "shape inference differs")
    def test_reduce_reshape_all(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("ReduceSum", ["X"], ["xr"], keepdims=1),
                    oh.make_node("Reshape", ["xr", "shape"], ["yr"]),
                    oh.make_node("Cos", ["yr"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [3, 2])],
                [_mkv_("Y", TFLOAT, [1])],
                [
                    onh.from_array(np.array([], dtype=np.int64), name="shape"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(3, 2)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["ReduceReshape"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["ReduceSum", "Cos"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reduce_reshape_opset10(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("ReduceSum", ["X"], ["xr"], keepdims=1, axes=[1]),
                    oh.make_node("Reshape", ["xr", "shape"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [3, 2])],
                [_mkv_("Y", TFLOAT, [3])],
                [
                    onh.from_array(np.array([1], dtype=np.int64), name="axes"),
                    onh.from_array(np.array([3], dtype=np.int64), name="shape"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 10)],
            ir_version=10,
        )
        check_model(model)
        feeds = {"X": self._range(3, 2)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["ReduceReshape"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["ReduceSum"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_transpose_reshape_matmul_left(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["xt"], perm=[0, 2, 1]),
                    oh.make_node("Reshape", ["xt", "shape"], ["xts"]),
                    oh.make_node("MatMul", ["xts", "Y"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [4, 5, 7]), _mkv_("Y", TFLOAT, [2, 2, 5, 3])],
                [_mkv_("Z", TFLOAT, [2, 2, 7, 3])],
                [onh.from_array(np.array([2, 2, 7, 5], dtype=np.int64), name="shape")],
            )
        )
        check_model(model)
        feeds = {"X": self._range(4, 5, 7), "Y": self._range(2, 2, 5, 3)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["TransposeReshapeMatMul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Reshape", "Transpose", "MatMul"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_transpose_reshape_matmul_right(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["Y"], ["yt"], perm=[0, 2, 1]),
                    oh.make_node("Reshape", ["yt", "shape"], ["yts"]),
                    oh.make_node("MatMul", ["X", "yts"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [2, 2, 5, 7]), _mkv_("Y", TFLOAT, [4, 3, 7])],
                [_mkv_("Z", TFLOAT, [2, 2, 5, 3])],
                [onh.from_array(np.array([2, 2, 7, 3], dtype=np.int64), name="shape")],
            )
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 5, 7), "Y": self._range(4, 3, 7)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["TransposeReshapeMatMul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Reshape", "Transpose", "MatMul"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_expand_forward_exp(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Expand", ["X", "shape"], ["xs"]),
                    oh.make_node("Exp", ["xs"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [1, 5, 7])],
                [_mkv_("Z", TFLOAT, [3, 5, 7])],
                [onh.from_array(np.array([3, 1, 1], dtype=np.int64), name="shape")],
            )
        )
        check_model(model)
        feeds = {"X": self._range(1, 5, 7)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["ExpandSwap"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Exp", "Expand"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_expand_forward_pow(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Expand", ["X", "shape"], ["xs"]),
                    oh.make_node("Pow", ["xs", "p"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [1, 5, 7])],
                [_mkv_("Z", TFLOAT, [3, 5, 7])],
                [
                    onh.from_array(np.array([3, 1, 1], dtype=np.int64), name="shape"),
                    onh.from_array(np.array([2], dtype=np.int64), name="p"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(1, 5, 7)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["ExpandSwap"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Pow", "Expand"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(2, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_expand_forward_cast(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Expand", ["X", "shape"], ["xs"]),
                    oh.make_node("Cast", ["xs"], ["Z"], to=TFLOAT16),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [1, 5, 7])],
                [_mkv_("Z", TFLOAT16, [3, 5, 7])],
                [onh.from_array(np.array([3, 1, 1], dtype=np.int64), name="shape")],
            )
        )
        check_model(model)
        feeds = {"X": self._range(1, 5, 7)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["ExpandSwap"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Cast", "Expand"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_matmul_reshape_phi(self):
        origin = self._get_model("phi_1_good.onnx")
        check_model(origin)
        self._check_with_ort(origin)
        gr = GraphBuilder(
            origin,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["MatMulReshape2Of3"],
                verbose=0,  # stop_after=2
            ),
        )
        onx = gr.to_onnx(optimize=True)
        self._check_with_ort(onx)
        check_model(onx)

    def test_slices_split(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Slice", ["X", "zero", "sept", "un"], ["x1"]),
                    oh.make_node("Slice", ["X", "sept", "huit", "un"], ["x2"]),
                    oh.make_node("Add", ["x1", "x2"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", 8])],
                [_mkv_("Y", TFLOAT, ["a", 7])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([7], dtype=np.int64), name="sept"),
                    onh.from_array(np.array([8], dtype=np.int64), name="huit"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(11, 8)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["SlicesSplit"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Split", "Add"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_slices_split_llama(self):
        origin = self._get_model("dort-split-custom__0.onnx")
        split = [n for n in origin.graph.node if n.op_type == "Split"]
        self.assertEqual(len(split), 0)
        self._check_with_ort(origin)
        gr = GraphBuilder(
            origin,
            infer_shapes=False,
            optimization_options=OptimizationOptions(
                patterns=["SlicesSplit"],
                verbose=0,
            ),
        )
        gr.set_shape("transpose", (2, 2, 1024, 512))
        gr.set_shape("transpose_1", (2, 2, 1024, 512))
        onx = gr.to_onnx(optimize=True)
        # self.dump_onnx("test_slices_split_llama.onnx", onx)
        split = [n for n in onx.graph.node if n.op_type == "Split"]
        self.assertEqual(len(split), 2)
        self._check_with_ort(onx)

    def test_slices_split_llama_not_onnx_node_shape_inference(self):
        origin = self._get_model("dort-split-custom__0.onnx")
        split = [n for n in origin.graph.node if n.op_type == "Split"]
        self.assertEqual(len(split), 0)
        self._check_with_ort(origin)
        # ShapeInference is necessarily incomplete because the model contains
        # a couple of FusedMatMul operator (from onnxruntime).
        # The inference seems to give a wrong value in that (empty)
        # which may be considered as a empty shape.
        # Then a node after this one may be wrong in case of an empty shape.
        # The optimization may do something wrong.
        gr = GraphBuilder(
            origin,
            infer_shapes="new",
            optimization_options=OptimizationOptions(
                patterns=["SlicesSplit"],
                verbose=0,
            ),
        )
        gr.set_shape("transpose", (2, 2, 1024, 512))
        gr.set_shape("transpose_1", (2, 2, 1024, 512))
        onx = gr.to_onnx(optimize=True)
        # We delete all the shape values because some of them are wrong.
        del onx.graph.value_info[:]
        split = [n for n in onx.graph.node if n.op_type == "Split"]
        self.assertEqual(len(split), 2)
        self._check_with_ort(onx)

    def test_rotary_split_missed(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Split", ["X", "split"], ["x1", "x2"], axis=1),
                    oh.make_node("Neg", ["x1"], ["nx1"]),
                    oh.make_node("ConstantOfShape", ["shape"], ["zero"]),
                    oh.make_node("Concat", ["nx1", "zero"], ["c1"], axis=1),
                    oh.make_node("Concat", ["zero", "x2"], ["c2"], axis=1),
                    oh.make_node("Add", ["c1", "c2"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", 16])],
                [_mkv_("Y", TFLOAT, ["a", 24])],
                [
                    onh.from_array(np.array([8, 8], dtype=np.int64), name="split"),
                    onh.from_array(np.array([3, 16], dtype=np.int64), name="shape"),
                ],
            )
        )
        feeds = {"X": self._range(3, 16)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["RotaryConcatPart"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Split", "Neg", "ConstantOfShape", "Concat", "Concat", "Add"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(2, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_rotary_split(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Split", ["X", "split"], ["x1", "x2"], axis=1),
                    oh.make_node("Neg", ["x1"], ["nx1"]),
                    oh.make_node("ConstantOfShape", ["shape"], ["zero"]),
                    oh.make_node("Concat", ["nx1", "zero"], ["c1"], axis=1),
                    oh.make_node("Concat", ["zero", "x2"], ["c2"], axis=1),
                    oh.make_node("Add", ["c1", "c2"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", 16])],
                [_mkv_("Y", TFLOAT, ["a", 16])],
                [
                    onh.from_array(np.array([8, 8], dtype=np.int64), name="split"),
                    onh.from_array(np.array([3, 8], dtype=np.int64), name="shape"),
                ],
            )
        )
        feeds = {"X": self._range(3, 16)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["RotaryConcatPart"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Split", "Neg", "Concat"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_cast_cast_binary(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cast", ["X"], ["xc"], to=TensorProto.FLOAT16),
                    oh.make_node("Cast", ["Y"], ["yc"], to=TensorProto.FLOAT16),
                    oh.make_node("Add", ["xc", "yc"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", 4]), _mkv_("Y", TFLOAT, ["a", 4])],
                [_mkv_("Z", TFLOAT16, ["a", 4])],
            )
        )
        feeds = {"X": self._range(3, 4), "Y": self._range(3, 4)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["CastCastBinary"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Add", "Cast"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reshape_reshape_binary(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "sh1"], ["xc"]),
                    oh.make_node("Reshape", ["Y", "sh2"], ["yc"]),
                    oh.make_node("Add", ["xc", "yc"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", 4]), _mkv_("Y", TFLOAT, ["a", 4])],
                [_mkv_("Z", TFLOAT, ["b", 8])],
                [
                    onh.from_array(np.array([-1, 8], dtype=np.int64), name="sh1"),
                    onh.from_array(np.array([-1, 8], dtype=np.int64), name="sh2"),
                ],
            )
        )
        feeds = {"X": self._range(6, 4), "Y": self._range(6, 4)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["ReshapeReshapeBinary"]),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Add", "Reshape"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_switch_order_binary_left(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy"]),
                    oh.make_node("Add", ["xy", "Z"], ["F"]),
                ],
                "dummy",
                [
                    _mkv_("X", TFLOAT, ["a", 2, 3, 4]),
                    _mkv_("Y", TFLOAT, ["a", 1, 3, 4]),
                    _mkv_("Z", TFLOAT, ["a", 1, 3, 4]),
                ],
                [_mkv_("F", TFLOAT, ["a", 2, 3, 4])],
            )
        )
        feeds = {
            "X": self._range(2, 2, 3, 4),
            "Y": self._range(2, 2, 3, 4),
            "Z": self._range(2, 1, 3, 4),
        }
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["SwitchOrderBinary"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Add", "Add"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_switch_order_binary_right(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy"]),
                    oh.make_node("Add", ["Z", "xy"], ["F"]),
                ],
                "dummy",
                [
                    _mkv_("X", TFLOAT, ["a", 2, 3, 4]),
                    _mkv_("Y", TFLOAT, ["a", 1, 3, 4]),
                    _mkv_("Z", TFLOAT, ["a", 1, 3, 4]),
                ],
                [_mkv_("F", TFLOAT, ["a", 2, 3, 4])],
            )
        )
        feeds = {
            "X": self._range(2, 2, 3, 4),
            "Y": self._range(2, 2, 3, 4),
            "Z": self._range(2, 1, 3, 4),
        }
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["SwitchOrderBinary"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Add", "Add"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_rotary_concat_dort(self):
        origin = self._get_model("dort-llama-llama-ort_1.onnx")
        before = [node for node in origin.graph.node if node.op_type == "Transpose"]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(patterns=["RotaryConcatPart"], verbose=0),
            infer_shapes=True,
        )
        onx = gr.to_onnx(optimize=True)
        after = [node for node in onx.graph.node if node.op_type == "Transpose"]
        self.assertNotEqual(len(before), len(after))

    def test_same_children_pattern_2(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy"]),
                    oh.make_node("Cast", ["xy"], ["xy1"], to=TensorProto.FLOAT16),
                    oh.make_node("Cast", ["xy"], ["xy2"], to=TensorProto.FLOAT16),
                    oh.make_node("Add", ["xy1", "xy2"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", 2, 3, 4]), _mkv_("Y", TFLOAT, ["a", 1, 3, 4])],
                [_mkv_("Z", TFLOAT16, ["a", 2, 3, 4])],
            )
        )
        feeds = {"X": self._range(2, 3, 4), "Y": self._range(1, 3, 4)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["SameChildren"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Add", "Cast", "Add"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_same_children_pattern_3(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy"]),
                    oh.make_node("Cast", ["xy"], ["xy1"], to=TensorProto.FLOAT16),
                    oh.make_node("Cast", ["xy"], ["xy2"], to=TensorProto.FLOAT16),
                    oh.make_node("Cast", ["xy"], ["xy3"], to=TensorProto.FLOAT16),
                    oh.make_node("Add", ["xy1", "xy2"], ["xy12"]),
                    oh.make_node("Add", ["xy12", "xy3"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", 2, 3, 4]), _mkv_("Y", TFLOAT, ["a", 1, 3, 4])],
                [_mkv_("Z", TFLOAT16, ["a", 2, 3, 4])],
            )
        )
        feeds = {"X": self._range(2, 3, 4), "Y": self._range(1, 3, 4)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["SameChildren"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Add", "Cast", "Add", "Add"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_same_children_pattern_2_next(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy"]),
                    oh.make_node("Cast", ["xy"], ["xy1"], to=TensorProto.FLOAT16),
                    oh.make_node("Cast", ["xy"], ["xy2"], to=TensorProto.FLOAT16),
                    oh.make_node("Exp", ["xy1"], ["e1"]),
                    oh.make_node("Exp", ["xy2"], ["e2"]),
                    oh.make_node("Add", ["e1", "e2"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", 2, 3, 4]), _mkv_("Y", TFLOAT, ["a", 1, 3, 4])],
                [_mkv_("Z", TFLOAT16, ["a", 2, 3, 4])],
            )
        )
        feeds = {"X": self._range(2, 3, 4), "Y": self._range(1, 3, 4)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["SameChildren"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Add", "Cast", "Exp", "Add"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_same_children_pattern_4_next(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy"]),
                    oh.make_node("Cast", ["xy"], ["xy1"], to=TensorProto.FLOAT16),
                    oh.make_node("Cast", ["xy"], ["xy2"], to=TensorProto.FLOAT16),
                    oh.make_node("Cast", ["xy"], ["xy3"], to=TensorProto.FLOAT16),
                    oh.make_node("Cast", ["xy"], ["xy4"], to=TensorProto.FLOAT16),
                    oh.make_node("Exp", ["xy1"], ["e1"]),
                    oh.make_node("Exp", ["xy2"], ["e2"]),
                    oh.make_node("Exp", ["xy3"], ["e3"]),
                    oh.make_node("Exp", ["xy4"], ["e4"]),
                    oh.make_node("Add", ["e1", "e2"], ["e12"]),
                    oh.make_node("Add", ["e3", "e4"], ["e34"]),
                    oh.make_node("Add", ["e12", "e34"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", 2, 3, 4]), _mkv_("Y", TFLOAT, ["a", 1, 3, 4])],
                [_mkv_("Z", TFLOAT16, ["a", 2, 3, 4])],
            )
        )
        feeds = {"X": self._range(2, 3, 4), "Y": self._range(1, 3, 4)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["SameChildren"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Add", "Cast", "Exp", "Add", "Add"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_cast_op_cast_binary(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cast", ["Y"], ["yc"], to=TensorProto.FLOAT),
                    oh.make_node("Add", ["X", "yc"], ["zc"]),
                    oh.make_node("Cast", ["zc"], ["Z"], to=TensorProto.FLOAT16),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b"]), _mkv_("Y", TFLOAT16, ["a", "b"])],
                [_mkv_("Z", TFLOAT16, ["a", "b"])],
            )
        )
        feeds = {
            "X": self._range(2, 3).astype(np.float32),
            "Y": self._range(2, 3).astype(np.float16),
        }
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["CastOpCast"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Cast", "Add"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_computation_cast_op_cast_binary(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cast", ["Y"], ["yc"], to=TensorProto.FLOAT),
                    oh.make_node("Add", ["X", "yc"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b"]), _mkv_("Y", TFLOAT16, ["a", "b"])],
                [_mkv_("Z", TFLOAT, ["a", "b"])],
            )
        )
        feeds = {
            "X": self._range(2, 3).astype(np.float32),
            "Y": self._range(2, 3).astype(np.float16),
        }
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["ComputationCastOpCast"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Cast", "Add", "Cast"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-3)

    def test_cast_op_cast_unary(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cast", ["X"], ["xc"], to=TensorProto.FLOAT16),
                    oh.make_node("Neg", ["xc"], ["xnc"]),
                    oh.make_node("Cast", ["xnc"], ["Y"], to=TensorProto.FLOAT),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b"])],
                [_mkv_("Y", TFLOAT, ["a", "b"])],
            )
        )
        feeds = {
            "X": self._range(2, 3).astype(np.float32),
            "Y": self._range(2, 3).astype(np.float16),
        }
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["CastOpCast"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Neg"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-3)

    def test_identity_pattern_transpose(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "two"], ["x2"]),
                    oh.make_node("Mul", ["x2", "two"], ["x3"]),
                    oh.make_node("Transpose", ["x3"], ["Y"], perm=[0, 1, 2]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b", "c"])],
                [_mkv_("Y", TFLOAT, ["a", "b", "c"])],
                [onh.from_array(np.array([2], dtype=np.float32), name="two")],
            )
        )
        feeds = {"X": self._range(2, 3, 4).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["Identity"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Add", "Mul"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_identity_pattern_mul(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Constant", [], ["one"], value_float=1.0),
                    oh.make_node("Add", ["X", "two"], ["x2"]),
                    oh.make_node("Mul", ["x2", "one"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b", "c"])],
                [_mkv_("Y", TFLOAT, ["a", "b", "c"])],
                [onh.from_array(np.array([2], dtype=np.float32), name="two")],
            )
        )
        feeds = {"X": self._range(2, 3, 4).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["Identity"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Add"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_identity_pattern_add_mul_more(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "zero"], ["x2"]),
                    oh.make_node("Mul", ["x2", "one"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b", 4])],
                [_mkv_("Y", TFLOAT, ["a", "b", 4])],
                [
                    onh.from_array(np.array([0, 0, 0, 0], dtype=np.float32), name="zero"),
                    onh.from_array(np.array([1, 1, 1, 1], dtype=np.float32), name="one"),
                ],
            )
        )
        feeds = {"X": self._range(2, 3, 4).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["Identity"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Identity"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_identity_pattern_add(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Constant", [], ["zero"], value_float=0.0),
                    oh.make_node("Add", ["X", "zero"], ["x2"]),
                    oh.make_node("Mul", ["x2", "two"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b", "c"])],
                [_mkv_("Y", TFLOAT, ["a", "b", "c"])],
                [onh.from_array(np.array([2], dtype=np.float32), name="two")],
            )
        )
        feeds = {"X": self._range(2, 3, 4).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["Identity"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Mul"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reduce_sum_normalization(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cast", ["X"], ["xc"], to=TFLOAT),
                    oh.make_node("ReduceSum", ["xc", "axis"], ["red"], keepdims=1),
                    oh.make_node("Mul", ["red", "Y"], ["mul"]),
                    oh.make_node("Sub", ["xc", "mul"], ["subc"]),
                    oh.make_node("Cast", ["subc"], ["Z"], to=TFLOAT16),
                ],
                "dummy",
                [_mkv_("X", TFLOAT16, ["a", "b"]), _mkv_("Y", TFLOAT, ["a", "b"])],
                [_mkv_("Z", TFLOAT16, ["a", "b"])],
                [onh.from_array(np.array(0.0 - 1, dtype=np.int64), name="axis")],
            )
        )
        feeds = {
            "X": self._range(2, 3).astype(np.float16),
            "Y": self._range(2, 3).astype(np.float32),
        }
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["ReduceSumNormalize"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["ReduceSum", "Cast", "Mul", "Sub"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-3)

    def test_2of3_expand(self):
        origin = self._get_model("bug_2of3_s.onnx")
        split = [n for n in origin.graph.node if n.op_type == "Split"]
        self.assertEqual(len(split), 0)
        self._check_with_ort(origin)
        gr = GraphBuilder(
            origin,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["Expand", "ReshapeReshape", "MatMulReshape2Of3"],
                verbose=0,
            ),
        )
        onx = gr.to_onnx(optimize=True)
        # split = [n for n in onx.graph.node if n.op_type == "Split"]
        # self.assertEqual(len(split), 2)
        self._check_with_ort(onx)

    def test_unsqueeze_equal(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "axis"], ["Y"]),
                    oh.make_node("Equal", ["X", "mone"], ["xe"]),
                    oh.make_node("Unsqueeze", ["xe", "axis"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b"])],
                [
                    _mkv_("Y", TFLOAT, ["a", 1, "b"]),
                    _mkv_("Z", TensorProto.BOOL, ["a", 1, "b"]),
                ],
                [
                    onh.from_array(np.array([1], dtype=np.int64), name="axis"),
                    onh.from_array(np.array([-1], dtype=np.float32), name="mone"),
                ],
            )
        )
        feeds = {"X": self._range(2, 3).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        expected = ref.run(None, feeds)[0]
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["UnsqueezeEqual"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Unsqueeze", "Equal"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(2, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-3)

    def test_layer_normalization(self):
        data = os.path.join(os.path.dirname(__file__), "data", "layernorm.onnx")
        model = onnx.load(data, load_external_data=False)
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["LayerNormalization"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        # with open("gggg.onnx", "wb") as f:
        #     f.write(opt_onx.SerializeToString())
        self.assertIn("LayerNormalization", set(n.op_type for n in opt_onx.graph.node))
        self.assertEqual(217, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

    def test_gelu(self):
        data = os.path.join(os.path.dirname(__file__), "data", "layernorm.onnx")
        model = onnx_load(data, load_external_data=False)
        del model.opset_import[:]
        model.opset_import.append(oh.make_opsetid("", 20))
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["Cast", "Gelu"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertIn("Gelu", set(n.op_type for n in opt_onx.graph.node))
        self.assertEqual(154, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

    def test_dropout(self):
        data = os.path.join(os.path.dirname(__file__), "data", "layernorm.onnx")
        model = onnx_load(data, load_external_data=False)
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(patterns=["Dropout"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        with open("test_dropout.onnx", "wb") as f:
            f.write(opt_onx.SerializeToString())
        self.assertNotIn("Dropout", set(n.op_type for n in opt_onx.graph.node))
        self.assertEqual(169, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

    def _get_model_ln_scale_bias(self, **kwargs):
        return oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("LayerNormalization", ["X", "one"], ["norm"], **kwargs),
                    oh.make_node("Mul", ["norm", "scale"], ["scaled"]),
                    oh.make_node("Add", ["scaled", "bias"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b"])],
                [_mkv_("Y", TFLOAT, ["a", "b"])],
                [
                    onh.from_array(
                        np.array(
                            ([1, 1, 1, 1, 1, 1] if kwargs.get("axis", -1) == 0 else [1, 1, 1]),
                            dtype=np.float32,
                        ),
                        name="one",
                    ),
                    onh.from_array(np.array([2.5], dtype=np.float32), name="scale"),
                    onh.from_array(np.array([2], dtype=np.float32), name="bias"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 20)],
            ir_version=10,
        )

    def test_layer_normalization_scale_bias(self):
        from onnxruntime import InferenceSession

        for kwargs in [{}, dict(axis=0), dict(stash_type=1), dict(epsilon=1e-1)]:
            model = self._get_model_ln_scale_no_bias(**kwargs)
            feeds = {"X": self._range(2, 3).astype(np.float32)}

            try:
                sess = InferenceSession(
                    model.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                expected = sess.run(None, feeds)[0]
            except Exception as e:
                raise AssertionError(f"Issue with kwargs={kwargs}, model{model}") from e

            inputs = [tuple(n.input) for n in model.graph.node]

            gr = GraphBuilder(
                model,
                infer_shapes=True,
                optimization_options=OptimizationOptions(
                    patterns=["LayerNormalizationScale"], verbose=0
                ),
            )
            opt_onx = gr.to_onnx(optimize=True)
            self.assertEqual(
                ["Mul", "LayerNormalization"],
                [n.op_type for n in opt_onx.graph.node],
            )
            self.assertEqual(2, len(opt_onx.graph.initializer))
            new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
            self.assertNotEqual(inputs, new_inputs)

            try:
                opt_ref = InferenceSession(
                    opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
            except Exception as e:
                raise AssertionError(f"Issue with kwargs={kwargs}, model{opt_onx}") from e
            try:
                got = opt_ref.run(None, feeds)[0]
            except Exception as e:
                raise AssertionError(f"Issue with kwargs={kwargs}, model{opt_onx}") from e
            self.assertEqualArray(expected, got, atol=1e-3)

            import torch

            if torch.cuda.is_available():
                opt_ref = InferenceSession(
                    opt_onx.SerializeToString(),
                    providers=["CUDAExecutionProvider"],
                )
                got = opt_ref.run(None, feeds)[0]
                self.assertEqualArray(expected, got, atol=1e-3)

    def _get_model_ln_scale_no_bias(self, **kwargs):
        return oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("LayerNormalization", ["X", "s0"], ["norm"], **kwargs),
                    oh.make_node("Mul", ["norm", "scale"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b"])],
                [_mkv_("Y", TFLOAT, ["a", "b"])],
                [
                    onh.from_array(
                        np.array(
                            (
                                [-0.1, -0.01, -0.05, -0.1, -0.01, -0.08]
                                if kwargs.get("axis", -1) == 0
                                else [-0.1, -0.01, -0.05]
                            ),
                            dtype=np.float32,
                        ),
                        name="s0",
                    ),
                    onh.from_array(
                        np.array(
                            [2] if kwargs.get("axis", -1) == 0 else [2, 3, 4],
                            dtype=np.float32,
                        ),
                        name="scale",
                    ),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 20)],
            ir_version=10,
        )

    def test_layer_normalization_scale_no_bias(self):
        from onnxruntime import InferenceSession

        for kwargs in [
            dict(axis=0),
            {},
            dict(axis=1),
            dict(stash_type=1),
            dict(epsilon=1e-1),
        ]:
            model = self._get_model_ln_scale_no_bias(**kwargs)
            feeds = {"X": self._range(2, 3).astype(np.float32)}

            try:
                sess = InferenceSession(
                    model.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                expected = sess.run(None, feeds)[0]
            except Exception as e:
                raise AssertionError(f"Issue with kwargs={kwargs}, model{model}") from e

            inputs = [tuple(n.input) for n in model.graph.node]

            gr = GraphBuilder(
                model,
                infer_shapes=True,
                optimization_options=OptimizationOptions(
                    patterns=["LayerNormalizationScale"], verbose=0
                ),
            )
            opt_onx = gr.to_onnx(optimize=True)
            self.assertEqual(
                ["Mul", "LayerNormalization"],
                [n.op_type for n in opt_onx.graph.node],
            )
            self.assertEqual(2, len(opt_onx.graph.initializer))
            new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
            self.assertNotEqual(inputs, new_inputs)

            try:
                opt_ref = InferenceSession(
                    opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
            except Exception as e:
                raise AssertionError(f"Issue with kwargs={kwargs}, model{opt_onx}") from e
            try:
                got = opt_ref.run(None, feeds)[0]
            except Exception as e:
                raise AssertionError(f"Issue with kwargs={kwargs}, model{opt_onx}") from e
            self.assertEqualArray(expected, got, atol=1e-3)

            import torch

            if torch.cuda.is_available():
                opt_ref = InferenceSession(
                    opt_onx.SerializeToString(),
                    providers=["CUDAExecutionProvider"],
                )
                got = opt_ref.run(None, feeds)[0]
                self.assertEqualArray(expected, got, atol=1e-3)

    def _get_model_redln(self, dtype=None, axis=-1, fixed=True):
        itype = TFLOAT if dtype in (None, np.float32) else TensorProto.FLOAT16
        axis_ = [axis] if axis in (-1, 1) else [0, 1]
        return oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("ReduceMean", ["X", "axis"], ["mean"], keepdims=1),
                    oh.make_node("Sub", ["X", "mean"], ["xc"]),
                    oh.make_node("Pow", ["xc", "two"], ["x2"]),
                    oh.make_node("ReduceMean", ["x2", "axis"], ["mean2"], keepdims=1),
                    oh.make_node("Sqrt", ["mean2"], ["mean2s"]),
                    oh.make_node("Div", ["xc", "mean2s"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", itype, [2, 3] if fixed else ["a", "b"])],
                [_mkv_("Y", itype, [2, 3] if fixed else ["a", "b"])],
                [
                    onh.from_array(np.array(axis_, dtype=np.int64), name="axis"),
                    onh.from_array(np.array([2], dtype=np.float32).T, name="two"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 20)],
            ir_version=10,
        )

    def test_layer_normalization_simple(self):
        from onnxruntime import InferenceSession

        for dtype in [np.float32, np.float16]:
            for kwargs in [
                dict(axis=0),
                {},
                dict(axis=0, fixed=False),
                dict(fixed=False),
            ]:
                model = self._get_model_redln(dtype=dtype, **kwargs)
                feeds = {"X": (self._range(2, 3) + np.array([1, 2, 3])).astype(dtype)}

                try:
                    sess = InferenceSession(
                        model.SerializeToString(), providers=["CPUExecutionProvider"]
                    )
                    expected = sess.run(None, feeds)[0]
                except Exception as e:
                    raise AssertionError(f"Issue with kwargs={kwargs}, model{model}") from e

                inputs = [tuple(n.input) for n in model.graph.node]

                gr = GraphBuilder(
                    model,
                    infer_shapes=True,
                    optimization_options=OptimizationOptions(
                        patterns=["LayerNormalization"], verbose=0
                    ),
                )
                opt_onx = gr.to_onnx(optimize=True)
                self.assertIn(
                    [n.op_type for n in opt_onx.graph.node],
                    (
                        ["LayerNormalization"],
                        [
                            "Shape",
                            "ConstantOfShape",
                            "ConstantOfShape",
                            "LayerNormalization",
                        ],
                    ),
                )
                self.assertIn(len(opt_onx.graph.initializer), (0, 2))
                new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
                self.assertNotEqual(inputs, new_inputs)

                opt_ref = InferenceSession(
                    opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                got = opt_ref.run(None, feeds)[0]
                self.assertEqualArray(expected, got, atol=1e-3)

                import torch

                if torch.cuda.is_available():
                    opt_ref = InferenceSession(
                        opt_onx.SerializeToString(),
                        providers=["CUDAExecutionProvider"],
                    )
                    got = opt_ref.run(None, feeds)[0]
                    self.assertEqualArray(expected, got, atol=1e-3)

    def test_mul_mul_matmul(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Mul", ["X", "c"], ["a"]),
                    oh.make_node("Mul", ["d", "Y"], ["b"]),
                    oh.make_node("MatMul", ["a", "b"], ["z"]),
                    oh.make_node("Add", ["z", "z"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [32, 16]), _mkv_("Y", TFLOAT, [16, 64])],
                [_mkv_("Z", TFLOAT, [32, 64])],
                [
                    onh.from_array(np.array([0.4], dtype=np.float32), name="c"),
                    onh.from_array(np.array([0.6], dtype=np.float32), name="d"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 16), "Y": self._range(16, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["MulMulMatMul"],
                verbose=0,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["MatMul", "Mul", "Add"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    def test_transpose_reshape_transpose_after(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["xt"], perm=[0, 1, 3, 2, 4, 5]),
                    oh.make_node("Reshape", ["xt", "shape"], ["xts"]),
                    oh.make_node("Transpose", ["xts"], ["Y"], perm=[0, 3, 1, 2]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [32, 4, 4, 14, 14, 128])],
                [_mkv_("Y", TFLOAT, [32, 128, 56, 56])],
                [onh.from_array(np.array([32, 56, 56, 128], dtype=np.int64), name="shape")],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 4, 4, 14, 14, 128)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["TransposeReshapeTranspose"],
                verbose=0,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Transpose", "Transpose", "Reshape"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    def test_transpose_reshape_transpose_before(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["xt"], perm=[0, 2, 3, 1]),
                    oh.make_node("Reshape", ["xt", "shape"], ["xts"]),
                    oh.make_node("Transpose", ["xts"], ["Y"], perm=[0, 1, 3, 2, 4, 5]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [32, 256, 28, 26])],
                [_mkv_("Y", TFLOAT, [32, 2, 2, 14, 13, 256])],
                [
                    onh.from_array(
                        np.array([32, 2, 14, 2, 13, 256], dtype=np.int64), name="shape"
                    )
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 256, 28, 26)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["TransposeReshapeTranspose"],
                verbose=0,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Reshape", "Transpose", "Transpose"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    def test_cast_layer_normalization_cast(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cast", ["X"], ["xc"], to=TensorProto.FLOAT),
                    oh.make_node(
                        "LayerNormalization",
                        ["xc", "scale", "bias"],
                        ["norm"],
                        stash_type=1,
                    ),
                    oh.make_node("Cast", ["norm"], ["Y"], to=TensorProto.FLOAT16),
                ],
                "dummy",
                [_mkv_("X", TFLOAT16, [3, 3])],
                [_mkv_("Y", TFLOAT16, [3, 3])],
                [
                    onh.from_array(np.array([0.5, 0.6, 0.7], dtype=np.float32), name="scale"),
                    onh.from_array(
                        np.array([-0.5, -0.6, -0.7], dtype=np.float32), name="bias"
                    ),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {"X": self._range(3, 3).astype(np.float16)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["CastLayerNormalizationCast"],
                verbose=0,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Cast", "Cast", "LayerNormalization"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(2, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)
        self._check_with_ort(opt_onx)

    def test_leaky_relu(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Greater", ["X", "zero"], ["xpos"]),
                    oh.make_node("Mul", ["X", "slope"], ["xmul"]),
                    oh.make_node("Where", ["xpos", "X", "xmul"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [3, 3])],
                [_mkv_("Y", TFLOAT, [3, 3])],
                [
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(np.array([0.76], dtype=np.float32), name="slope"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {"X": self._range(3, 3).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["LeakyRelu"],
                verbose=0,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["LeakyRelu"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)
        # self._check_with_ort(opt_onx)

    def test_leaky_relu_2(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Greater", ["X", "zero"], ["xpos"]),
                    oh.make_node("Mul", ["X", "slope"], ["xmul"]),
                    oh.make_node("Where", ["xpos", "X", "xmul"], ["X1"]),
                    oh.make_node("Greater", ["X1", "zero"], ["xpos2"]),
                    oh.make_node("Mul", ["X1", "slope2"], ["xmul2"]),
                    oh.make_node("Where", ["xpos2", "X1", "xmul2"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [3, 3])],
                [_mkv_("Y", TFLOAT, [3, 3])],
                [
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(np.array([0.76], dtype=np.float32), name="slope"),
                    onh.from_array(np.array([-0.33], dtype=np.float32), name="slope2"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {"X": self._range(3, 3).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["LeakyRelu"],
                verbose=0,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["LeakyRelu", "LeakyRelu"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)
        # self._check_with_ort(opt_onx)

    @requires_torch("2.6")
    def test_conv_null_bias(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Conv",
                        ["X", "W", "B"],
                        ["Y"],
                        dilations=[1, 1],
                        group=1,
                        kernel_shape=[4, 4],
                        pads=[1, 1, 1, 1],
                        strides=[2, 2],
                    )
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [512, 3, 64, 64]), _mkv_("W", TFLOAT, [64, 3, 4, 4])],
                [_mkv_("Y", TFLOAT, [512, 64, 32, 32])],
                [
                    onh.from_array(np.zeros((64,), dtype=np.float32), name="B"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {
            "X": self._range(512, 3, 64, 64).astype(np.float32),
            "W": self._range(64, 3, 4, 4).astype(np.float32),
        }
        from onnxruntime import InferenceSession

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["ConvBiasNull"],
                verbose=0,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Conv"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        excs = []
        for _ in range(2):
            opt_ref = InferenceSession(
                opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            for __ in range(2):
                got = opt_ref.run(None, feeds)[0]
                try:
                    self.assertEqualArray(expected, got, atol=1e-2)
                except Exception as e:
                    excs.append((_, __, str(e)))
        if excs:
            raise AssertionError(f"{len(excs)} validations failed\n{pprint.pformat(excs)}")

    def test_conv_null_bias_shape_expand(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["B"], ["Bshape"], start=0, end=1),
                    oh.make_node("Expand", ["zero", "Bshape"], ["B2"]),
                    oh.make_node(
                        "Conv",
                        ["X", "W", "B2"],
                        ["Y"],
                        dilations=[1, 1],
                        group=1,
                        kernel_shape=[4, 4],
                        pads=[1, 1, 1, 1],
                        strides=[2, 2],
                    ),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [512, 3, 64, 64]), _mkv_("W", TFLOAT, [64, 3, 4, 4])],
                [_mkv_("Y", TFLOAT, [512, 64, 32, 32])],
                [
                    onh.from_array(np.zeros((64,), dtype=np.float32), name="B"),
                    onh.from_array(np.zeros((1,), dtype=np.float32), name="zero"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {
            "X": np.abs(self._range(512, 3, 64, 64).astype(np.float32)),
            "W": np.abs(self._range(64, 3, 4, 4).astype(np.float32)),
        }
        from onnxruntime import InferenceSession

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["ConvBiasNull"],
                verbose=0,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Conv"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    @requires_torch("2.6")
    def test_folding_with_conv_no_bias(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["B"], ["Bshape"], start=0, end=1),
                    oh.make_node("Expand", ["zero", "Bshape"], ["B2"]),
                    oh.make_node(
                        "Conv",
                        ["X", "W", "B2"],
                        ["Y"],
                        dilations=[1, 1],
                        group=1,
                        kernel_shape=[4, 4],
                        pads=[1, 1, 1, 1],
                        strides=[2, 2],
                    ),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [512, 3, 64, 64]), _mkv_("W", TFLOAT, [64, 3, 4, 4])],
                [_mkv_("Y", TFLOAT, [512, 64, 32, 32])],
                [
                    onh.from_array(np.zeros((64,), dtype=np.float32), name="B"),
                    onh.from_array(np.zeros((1,), dtype=np.float32), name="zero"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {
            "X": np.abs(self._range(512, 3, 64, 64).astype(np.float32)),
            "W": np.abs(self._range(64, 3, 4, 4).astype(np.float32)),
        }
        from onnxruntime import InferenceSession

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["Cast"], verbose=0, constant_folding=True
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Conv"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_batch_normalization_identity(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "BatchNormalization",
                        ["X", "scale", "B", "input_mean", "input_var"],
                        ["Y"],
                        epsilon=0.0,
                    ),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [1024, 16])],
                [_mkv_("Y", TFLOAT, [1024, 16])],
                [
                    onh.from_array(np.zeros((16,), dtype=np.float32), name="B"),
                    onh.from_array(np.zeros((16,), dtype=np.float32), name="input_mean"),
                    onh.from_array(np.ones((16,), dtype=np.float32), name="input_var"),
                    onh.from_array(np.ones((16,), dtype=np.float32), name="scale"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {"X": self._range(1024, 16).astype(np.float32)}
        from onnxruntime import InferenceSession

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["BatchNormalization"], verbose=0, constant_folding=True
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Identity"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_batch_normalization_no_training(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "BatchNormalization",
                        ["X", "scale", "B", "input_mean", "input_var"],
                        ["Y", "unused1", "unused2"],
                        epsilon=0.5,
                        momentum=1.0,
                        training_mode=1,
                    ),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [2, 3, 4, 5])],
                [_mkv_("Y", TFLOAT, [2, 3, 4, 5])],
                [
                    onh.from_array((np.arange(3) + 1).astype(np.float32), name="scale"),
                    onh.from_array((np.arange(3) + 100).astype(np.float32), name="B"),
                    onh.from_array((np.arange(3) + 20).astype(np.float32), name="input_mean"),
                    onh.from_array((np.arange(3) + 2).astype(np.float32), name="input_var"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {"X": self._range(2, 3, 4, 5).astype(np.float32)}
        from onnxruntime import InferenceSession

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["BatchNormalizationTraining"], verbose=0, constant_folding=True
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertIn("ReduceMean", pretty_onnx(opt_onx))
        self.assertNotEqual(
            ["BatchNormalization"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(4, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_batch_normalization_no_training_no_reshape(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "BatchNormalization",
                        ["X", "scale", "B", "input_mean", "input_var"],
                        ["Y", "unused1", "unused2"],
                        epsilon=0.5,
                        momentum=1.0,
                        training_mode=1,
                    ),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [2, 3, 4, 5])],
                [_mkv_("Y", TFLOAT, [2, 3, 4, 5])],
                [
                    onh.from_array(
                        (self._range(1, 3, 4, 5) + 1).astype(np.float32), name="scale"
                    ),
                    onh.from_array(
                        (self._range(1, 3, 4, 1) + 100).astype(np.float32), name="B"
                    ),
                    onh.from_array((np.arange(3) + 20).astype(np.float32), name="input_mean"),
                    onh.from_array((np.arange(3) + 2).astype(np.float32), name="input_var"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {"X": self._range(2, 3, 4, 5).astype(np.float32)}
        from onnxruntime import InferenceSession

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["BatchNormalizationTraining"], verbose=0, constant_folding=True
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertIn("ReduceMean", pretty_onnx(opt_onx))
        self.assertNotEqual(
            ["BatchNormalization"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(4, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)[0]
        self.assertEqual(got.shape, feeds["X"].shape)

    def test_matmul_add_1(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MatMul", ["X1", "X2"], ["Y"]),
                    oh.make_node("Add", ["Y", "B"], ["Z"]),
                ],
                "dummy",
                [
                    _mkv_("X1", TFLOAT, [2, 3]),
                    _mkv_("X2", TFLOAT, [3, 2]),
                    _mkv_("B", TFLOAT, [2]),
                ],
                [_mkv_("Z", TFLOAT, [2, 2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {
            "X1": self._range(2, 3).astype(np.float32),
            "X2": self._range(3, 2).astype(np.float32),
            "B": self._range(2).astype(np.float32),
        }
        from onnxruntime import InferenceSession

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["MatMulAdd"], verbose=0, constant_folding=True
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(["Gemm"], [n.op_type for n in opt_onx.graph.node])

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_matmul_add_2(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Gemm", ["X1", "X2"], ["Y"], transB=1),
                    oh.make_node("Add", ["Y", "B"], ["Z"]),
                ],
                "dummy",
                [
                    _mkv_("X1", TFLOAT, [2, 3]),
                    _mkv_("X2", TFLOAT, [2, 3]),
                    _mkv_("B", TFLOAT, [2]),
                ],
                [_mkv_("Z", TFLOAT, [2, 2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {
            "X1": self._range(2, 3).astype(np.float32),
            "X2": self._range(2, 3).astype(np.float32),
            "B": self._range(2).astype(np.float32),
        }
        from onnxruntime import InferenceSession

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["MatMulAdd"], verbose=0, constant_folding=True
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(["Gemm"], [n.op_type for n in opt_onx.graph.node])

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_matmul_add_3(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Gemm", ["X1", "X2", "B1"], ["Y"], transB=1),
                    oh.make_node("Add", ["Y", "B2"], ["Z"]),
                ],
                "dummy",
                [
                    _mkv_("X1", TFLOAT, [2, 3]),
                    _mkv_("X2", TFLOAT, [2, 3]),
                    _mkv_("B1", TFLOAT, [2]),
                    _mkv_("B2", TFLOAT, [2]),
                ],
                [_mkv_("Z", TFLOAT, [2, 2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {
            "X1": self._range(2, 3).astype(np.float32),
            "X2": self._range(2, 3).astype(np.float32),
            "B1": self._range(2).astype(np.float32),
            "B2": (self._range(2) + 10).astype(np.float32),
        }
        from onnxruntime import InferenceSession

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["MatMulAdd"], verbose=0, constant_folding=True
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(["Add", "Gemm"], [n.op_type for n in opt_onx.graph.node])

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_matmul_add_reshape_1(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MatMul", ["X1", "X2"], ["Y"]),
                    oh.make_node("Add", ["Y", "B"], ["Z"]),
                ],
                "dummy",
                [
                    _mkv_("X1", TFLOAT, [4, 2, 3]),
                    _mkv_("X2", TFLOAT, [3, 2]),
                    _mkv_("B", TFLOAT, [2]),
                ],
                [_mkv_("Z", TFLOAT, [4, 2, 2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {
            "X1": self._range(4, 2, 3).astype(np.float32),
            "X2": self._range(3, 2).astype(np.float32),
            "B": self._range(2).astype(np.float32),
        }
        from onnxruntime import InferenceSession

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["MatMulAdd"], verbose=0, constant_folding=True
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(
            ["Reshape", "Gemm", "Reshape"], [n.op_type for n in opt_onx.graph.node]
        )

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_matmul_add_reshape_2(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MatMul", ["X1", "X2"], ["Y"]),
                    oh.make_node("Add", ["Y", "B"], ["Z"]),
                ],
                "dummy",
                [
                    _mkv_("X1", TFLOAT, [4, 2, 3]),
                    _mkv_("X2", TFLOAT, [3, 2]),
                    _mkv_("B", TFLOAT, [4, 2, 2]),
                ],
                [_mkv_("Z", TFLOAT, [4, 2, 2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {
            "X1": self._range(4, 2, 3).astype(np.float32),
            "X2": self._range(3, 2).astype(np.float32),
            "B": self._range(4, 2, 2).astype(np.float32),
        }
        from onnxruntime import InferenceSession

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["MatMulAdd"], verbose=0, constant_folding=True
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(
            ["Reshape", "Reshape", "Gemm", "Reshape"], [n.op_type for n in opt_onx.graph.node]
        )

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    @hide_stdout()
    def test_matmul_add_reshape_2_dyn(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MatMul", ["X1", "X2"], ["Y"]),
                    oh.make_node("Add", ["Y", "B"], ["Z"]),
                ],
                "dummy",
                [
                    _mkv_("X1", TFLOAT, ["a", "b", 3]),
                    _mkv_("X2", TFLOAT, [3, "d"]),
                    _mkv_("B", TFLOAT, ["a", "b", "d"]),
                ],
                [_mkv_("Z", TFLOAT, ["a", "b", "d"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {
            "X1": self._range(4, 2, 3).astype(np.float32),
            "X2": self._range(3, 2).astype(np.float32),
            "B": self._range(4, 2, 2).astype(np.float32),
        }
        from onnxruntime import InferenceSession

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["MatMulAdd"],
                verbose=10,
                constant_folding=True,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(
            ["Reshape", "Shape", "Concat", "Reshape", "Shape", "Concat", "Gemm", "Reshape"],
            [n.op_type for n in opt_onx.graph.node],
        )

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_matmul_transpose_cst(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Gemm", ["X", "B"], ["Z"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, [2, 3])],
                [_mkv_("Z", TFLOAT, [2, 2])],
                [onh.from_array(self._range(3, 2).astype(np.float32), name="B")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {"X": self._range(2, 3).astype(np.float32)}
        from onnxruntime import InferenceSession

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["GemmTranspose"],
                verbose=0,
                constant_folding=False,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(["Transpose", "Gemm"], [n.op_type for n in opt_onx.graph.node])

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_slice_slice_nostep(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Slice", ["X", "zero", "one", "zero"], ["x1"]),
                    oh.make_node("Slice", ["x1", "zero", "one", "one"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b"])],
                [_mkv_("Y", TFLOAT, ["c", "d"])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="one"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {"X": self._range(2, 3).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["SliceSlice"],
                verbose=0,
                constant_folding=True,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(["Slice"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(len(opt_onx.graph.initializer), 3)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_slice_slice_steps1(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Slice", ["X", "zero", "one", "zero", "one"], ["x1"]),
                    oh.make_node("Slice", ["x1", "zero", "one", "one"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b"])],
                [_mkv_("Y", TFLOAT, ["c", "d"])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="one"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {"X": self._range(2, 3).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["SliceSlice"],
                verbose=0,
                constant_folding=True,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(["Slice"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(len(opt_onx.graph.initializer), 4)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_slice_slice_steps2(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Slice", ["X", "zero", "one", "zero"], ["x1"]),
                    oh.make_node("Slice", ["x1", "zero", "one", "one", "one"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b"])],
                [_mkv_("Y", TFLOAT, ["c", "d"])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="one"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {"X": self._range(2, 3).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["SliceSlice"],
                verbose=0,
                constant_folding=True,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(["Slice"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(len(opt_onx.graph.initializer), 4)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_slice_slice_steps3(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Slice", ["X", "zero", "one", "zero", "one"], ["x1"]),
                    oh.make_node("Slice", ["x1", "zero", "one", "one", "one"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b"])],
                [_mkv_("Y", TFLOAT, ["c", "d"])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="one"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {"X": self._range(2, 3).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["SliceSlice"],
                verbose=0,
                constant_folding=True,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(["Slice"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(len(opt_onx.graph.initializer), 4)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_clip_clip(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Clip", ["X", "zero"], ["x1"]),
                    oh.make_node("Clip", ["x1", "", "one"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b"])],
                [_mkv_("Y", TFLOAT, ["c", "d"])],
                [
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        feeds = {"X": self._range(2, 3).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["ClipClip"],
                verbose=0,
                constant_folding=True,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(["Clip"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(len(opt_onx.graph.initializer), 2)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_sequence_construct(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("SequenceConstruct", ["X1", "X2"], ["seq"]),
                    oh.make_node("SequenceAt", ["seq", "i0"], ["Y1"]),
                    oh.make_node("SequenceAt", ["seq", "i1"], ["Y2"]),
                ],
                "dummy",
                [_mkv_("X1", TFLOAT, ["a", "b"]), _mkv_("X2", TFLOAT, ["c", "d"])],
                [_mkv_("Y1", TFLOAT, ["a", "b"]), _mkv_("Y2", TFLOAT, ["c", "d"])],
                [
                    onh.from_array(np.array(0, dtype=np.int64), name="i0"),
                    onh.from_array(np.array(1, dtype=np.int64), name="i1"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        from onnxruntime import InferenceSession

        feeds = {
            "X1": self._range(2, 3).astype(np.float32),
            "X2": self._range(3, 4).astype(np.float32),
        }
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["SequenceConstructAt"],
                verbose=0,
                constant_folding=True,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)

        self.assertEqual(["Identity", "Identity"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(len(opt_onx.graph.initializer), 0)

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)
        self.assertEqualArray(expected[1], got[1], atol=1e-5)

    @hide_stdout()
    def test_shape_eval(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["ids_weight"], ["shape"], start=0, end=2),
                    oh.make_node("Concat", ["shape", "init328"], ["new_shape"], axis=0),
                    oh.make_node("MatMul", ["ids_weight", "A"], ["A1"]),
                    oh.make_node("MatMul", ["ids_weight", "B"], ["B1"]),
                    oh.make_node("MatMul", ["ids_weight", "C"], ["C1"]),
                    oh.make_node("Reshape", ["A1", "new_shape"], ["Areshaped"]),
                    oh.make_node("Reshape", ["B1", "new_shape"], ["Breshaped"]),
                    oh.make_node("Reshape", ["C1", "new_shape"], ["Creshaped"]),
                    oh.make_node("Transpose", ["Areshaped"], ["At"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["Breshaped"], ["Bt"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["Creshaped"], ["Ct"], perm=[0, 2, 1, 3]),
                ],
                "dummy",
                [_mkv_("ids_weight", TFLOAT, ["batch", "seq", 256])],
                [
                    _mkv_("At", TFLOAT, ["batch", 32, "seq", 8]),
                    _mkv_("Bt", TFLOAT, ["batch", 32, "seq", 8]),
                    _mkv_("Ct", TFLOAT, ["batch", 32, "seq", 8]),
                ],
                [
                    onh.from_array(np.array([32, 8], dtype=np.int64), name="init328"),
                    onh.from_array(np.random.randn(256, 256).astype(np.float32), name="A"),
                    onh.from_array(np.random.randn(256, 256).astype(np.float32), name="B"),
                    onh.from_array(np.random.randn(256, 256).astype(np.float32), name="C"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)
        gr = GraphBuilder(
            model,
            infer_shapes=True,
            verbose=5,
        )
        shapes = gr._known_shapes
        self.assertEqual(shapes["Areshaped"], ("batch", "seq", 32, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
