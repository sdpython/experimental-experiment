import unittest
import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import TensorProto
from onnx.checker import check_model
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.onnx_tools import onnx_lighten, onnx_unlighten
from experimental_experiment._command_lines_parser import _cmd_lighten, _cmd_unlighten
from experimental_experiment.torch_test_helper import check_model_ort

TFLOAT = TensorProto.FLOAT


class TestOnnxTools(ExtTestCase):
    def _get_model(self):
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
                [oh.make_tensor_value_info("X", TFLOAT, [320, 1280])],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 320, 640])],
                [
                    onh.from_array(
                        np.random.rand(3, 5, 1280, 640).astype(np.float32), name="Y"
                    ),
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 320, 1280], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 1280, 640], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 320, 640], dtype=np.int64), name="shape3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        return model

    def test_un_lighten_model(self):
        model = self._get_model()
        check_model(model)
        size1 = len(model.SerializeToString())
        (onx, stats), out, _ = self.capture(lambda: onnx_lighten(model, verbose=1))
        self.assertIsInstance(stats, dict)
        self.assertEqual(len(stats), 1)
        self.assertIsInstance(stats["Y"], dict)
        self.assertIn("remove initializer", out)
        # check_model(onx)
        new_model = onnx_unlighten(onx, stats)
        check_model(new_model)
        size2 = len(new_model.SerializeToString())
        self.assertEqual(size1, size2)
        check_model_ort(model)

    def test_cmd_un_lighten_model(self):
        model = self._get_model()
        check_model(model)
        name = "test_cmd_un_lighten_model.onnx"
        with open(name, "wb") as f:
            f.write(model.SerializeToString())
        name2 = "test_cmd_un_lighten_model.opt.onnx"

        cmd = ["", "-i", name, "-o", name2, "-v", "1"]
        _, out, _ = self.capture(lambda: _cmd_lighten(cmd))
        self.assertIn("done", out)
        self.assertExists(name2)
        self.assertExists(name2 + ".stats")

        name3 = f"{name}.2.onnx"
        cmd = ["", "-i", name2, "-o", name3, "-v", "1"]
        _, out, _ = self.capture(lambda: _cmd_unlighten(cmd))
        self.assertIn("done", out)
        self.assertExists(name3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
