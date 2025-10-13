import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.xshape.shape_builder import ShapeBuilder
from experimental_experiment.xshape.shape_builder_impl import BasicShapeBuilder

TFLOAT = onnx.TensorProto.FLOAT
TFLOAT16 = onnx.TensorProto.FLOAT16
TINT64 = onnx.TensorProto.INT64
_mkv_ = oh.make_tensor_value_info


class TestShapeBuilder(ExtTestCase):
    def test_shape_builder(self):
        builder = ShapeBuilder()
        for me in dir(builder):
            if me.startswith("get_") and not me.startswith("get_att"):
                self.assertRaise(lambda me=me: getattr(builder, me)(""), NotImplementedError)
            if me.startswith("set_"):
                self.assertRaise(
                    lambda me=me: getattr(builder, me)("", None), NotImplementedError
                )

    def test_basic_shape_builder(self):
        b = BasicShapeBuilder()
        msg = b.get_debug_msg()
        self.assertIn("--SHAPE--", msg)

    def test_check_shape(self):
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
        onnx.shape_inference.infer_shapes(model)
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertEqual(builder._input_names, ["X", "Y"])
        self.assertEqual(
            builder._known_ranks,
            {
                "zero": 1,
                "un": 1,
                "shape1": 1,
                "shape2": 1,
                "shape3": 1,
                "X": 2,
                "Y": 4,
                "xu1": 3,
                "xu2": 4,
                "xm1": 3,
                "xm2c": 3,
                "xm2": 3,
                "xm": 3,
                "Z": 4,
            },
        )
        self.assertEqual(
            builder._known_shapes,
            {
                "zero": (1,),
                "un": (1,),
                "shape1": (3,),
                "shape2": (3,),
                "shape3": (4,),
                "X": ("D32", "D128"),
                "Y": ("batch", "channel", "D128", "D64"),
                "xu1": (1, "D32", "D128"),
                "xu2": (1, 1, "D32", "D128"),
                "xm1": (1, 32, 128),
                "xm2c": (15, 128, 64),
                "xm2": (15, 128, 64),
                "xm": (15, 32, 64),
                "Z": (3, 5, 32, 64),
            },
        )
        self.assertEqual(
            builder._known_types,
            {
                "zero": 7,
                "un": 7,
                "shape1": 7,
                "shape2": 7,
                "shape3": 7,
                "X": 1,
                "Y": 1,
                "xu1": 1,
                "xu2": 1,
                "xm1": 1,
                "xm2c": 1,
                "xm2": 1,
                "xm": 1,
                "Z": 1,
            },
        )
        self.assertEqualAny(
            builder.constants_computed_,
            {
                "shape1": np.array([1, 32, 128], dtype=np.int64),
                "shape2": np.array([15, 128, 64], dtype=np.int64),
                "shape3": np.array([3, 5, 32, 64], dtype=np.int64),
                "un": np.array([1], dtype=np.int64),
                "zero": np.array([0], dtype=np.int64),
            },
        )
        self.assertEqual(builder.constraints_, {})
        self.assertEqual(
            builder.dynamic_dimensions_,
            {
                "D128": {"D128"},
                "D32": {"D32"},
                "D64": {"D64"},
                "batch": {"batch"},
                "channel": {"channel"},
            },
        )
        self.assertEqual(builder._known_value_shape, {"zero": (0,), "un": (1,)})
        self.assertEqual(builder._output_names, ["Z"])

    def test_reshape_reshape(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr"]),
                    oh.make_node("Reshape", ["xr", "shape2"], ["xrr"]),
                    oh.make_node("Add", ["xrr", "one"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b", "c"])],
                [_mkv_("Y", TFLOAT, ["a", "b", "c"])],
                [
                    onh.from_array(np.array([0, 0, 2, -1], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                ],
            )
        )
        onnx.shape_inference.infer_shapes(model)
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertEqual(builder._input_names, ["X"])
        self.assertEqual(
            builder._known_ranks,
            {"X": 3, "Y": 3, "one": 1, "shape1": 1, "shape2": 1, "xr": 4, "xrr": 3},
        )
        self.assertEqual(
            builder._known_shapes,
            {
                "shape1": (4,),
                "shape2": (3,),
                "one": (1,),
                "X": ("a", "b", "c"),
                "xr": ("a", "b", 2, "c//2"),
                "xrr": ("a", "b", "c"),
                "Y": ("a", "b", "c"),
            },
        )
        self.assertEqual(
            builder._known_types,
            {"shape1": 7, "shape2": 7, "one": 1, "X": 1, "xr": 1, "xrr": 1, "Y": 1},
        )
        self.assertEqualAny(
            builder.constants_computed_,
            {"shape1": np.array([0, 0, 2, -1]), "shape2": np.array([0, 0, -1])},
        )
        self.assertEqual(builder.constraints_, {})
        self.assertEqual(builder.dynamic_dimensions_, {"a": {"a"}, "b": {"b"}, "c": {"c"}})
        self.assertEqual(builder._known_value_shape, {})
        self.assertEqual(builder._output_names, ["Y"])

    def test_value_as_shape(self):
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
        onnx.shape_inference.infer_shapes(model)
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertEqual(
            builder._known_value_shape,
            {"init328": (32, 8), "new_shape": ("batch", "seq", 32, 8), "shape": ("batch", "seq")},
        )
        self.assertEqual(builder._input_names, ["ids_weight"])
        self.assertEqual(
            builder._known_ranks,
            {
                "A": 2,
                "A1": 3,
                "Areshaped": 4,
                "At": 4,
                "B": 2,
                "B1": 3,
                "Breshaped": 4,
                "Bt": 4,
                "C": 2,
                "C1": 3,
                "Creshaped": 4,
                "Ct": 4,
                "ids_weight": 3,
                "init328": 1,
                "new_shape": 1,
                "shape": 1,
            },
        )
        self.assertEqual(
            builder._known_types,
            {
                "init328": 7,
                "A": 1,
                "B": 1,
                "C": 1,
                "ids_weight": 1,
                "A1": 1,
                "B1": 1,
                "C1": 1,
                "Areshaped": 1,
                "Breshaped": 1,
                "Creshaped": 1,
                "At": 1,
                "Bt": 1,
                "Ct": 1,
            },
        )
        self.assertEqual(
            builder._known_shapes,
            {
                "init328": (2,),
                "A": (256, 256),
                "B": (256, 256),
                "C": (256, 256),
                "ids_weight": ("batch", "seq", 256),
                "shape": (2,),
                "new_shape": (4,),
                "A1": ("batch", "seq", 256),
                "B1": ("batch", "seq", 256),
                "C1": ("batch", "seq", 256),
                "Areshaped": ("batch", "seq", 32, 8),
                "Breshaped": ("batch", "seq", 32, 8),
                "Creshaped": ("batch", "seq", 32, 8),
                "At": ("batch", 32, "seq", 8),
                "Bt": ("batch", 32, "seq", 8),
                "Ct": ("batch", 32, "seq", 8),
            },
        )
        self.assertEqualAny(
            builder.constants_computed_, {"init328": np.array([32, 8], dtype=np.int64)}
        )
        self.assertEqual(builder.constraints_, {})
        self.assertEqual(builder.dynamic_dimensions_, {"batch": {"batch"}, "seq": {"seq"}})
        self.assertEqual(builder._output_names, ["At", "Bt", "Ct"])

    def test_evaluate_shape(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Concat", ["X", "Y"], ["Z"], axis=1),
                ],
                "dummy",
                [_mkv_("Y", TFLOAT, ["batch", "seq1"]), _mkv_("X", TFLOAT, ["batch", "seq2"])],
                [_mkv_("Z", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertEqual(
            builder._known_shapes,
            {"Y": ("batch", "seq1"), "X": ("batch", "seq2"), "Z": ("batch", "seq2+seq1")},
        )
        feeds = dict(
            X=np.random.rand(3, 5).astype(np.float32), Y=np.random.rand(3, 6).astype(np.float32)
        )
        got = ExtendedReferenceEvaluator(model).run(None, feeds)
        res = builder.compare_with_true_inputs(feeds, got)
        self.assertEqual(res, {"Z": (("batch", 3, 3), ("seq2+seq1", 11, 11))})


if __name__ == "__main__":
    unittest.main(verbosity=2)
