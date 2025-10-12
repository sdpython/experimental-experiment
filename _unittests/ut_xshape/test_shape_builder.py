import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xshape.shape_builder import ShapeBuilder, BasicShapeBuilder

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
            {"X": 2, "Y": 4, "Z": 4, "xm": 3, "xm1": 3, "xm2": 3, "xm2c": 3, "xu1": 3, "xu2": 4},
        )
        self.assertEqual(
            builder._known_shapes,
            {
                "X": ("D32", "D128"),
                "Y": ("batch", "channel", "D128", "D64"),
                "Z": (3, 5, 32, 64),
                "xm": (15, 32, 64),
                "xm1": (1, 32, 128),
                "xm2": (15, 128, 64),
                "xm2c": (15, 128, 64),
                "xu1": (1, "D32", "D128"),
                "xu2": (1, 1, "D32", "D128"),
            },
        )
        self.assertEqual(
            builder._known_types,
            {"X": 1, "Y": 1, "Z": 1, "xm": 1, "xm1": 1, "xm2": 1, "xm2c": 1, "xu1": 1, "xu2": 1},
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
        self.assertEqual(builder._known_value_shape, {})
        self.assertEqual(builder._output_names, ["Z"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
