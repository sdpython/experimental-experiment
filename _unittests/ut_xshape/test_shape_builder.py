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
            if me.startswith("get_"):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
