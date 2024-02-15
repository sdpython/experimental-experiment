import itertools
import os
import unittest
import numpy as np
import onnx.helper as oh
from onnx import TensorProto, load
from onnx.numpy_helper import from_array
from experimental_experiment.ext_test_case import ExtTestCase, ignore_warnings


def has_cuda():
    import torch

    return torch.cuda.is_available()


class TestCheckOrtFloat16(ExtTestCase):
    def common_scatter(self, providers, dtype, reduction, expected_names):
        from onnxruntime import InferenceSession, SessionOptions

        assert dtype in (np.float16, np.float32)
        itype = TensorProto.FLOAT if dtype == np.float32 else TensorProto.FLOAT16
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "I"], ["data"]),
                    oh.make_node(
                        "ScatterElements",
                        inputs=["data", "indices", "updates"],
                        outputs=["sy"],
                        axis=0,
                        reduction=reduction,
                    ),
                    oh.make_node("Sub", ["sy", "I"], ["Y"]),
                ],
                "name",
                [
                    oh.make_tensor_value_info("X", itype, [None, None]),
                    oh.make_tensor_value_info(
                        "indices", TensorProto.INT64, [None, None]
                    ),
                    oh.make_tensor_value_info("updates", itype, [None, None]),
                ],
                [oh.make_tensor_value_info("Y", itype, [None, None])],
                [from_array(np.array([1], dtype=dtype), name="I")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        if not os.path.exists("temp_dump"):
            os.mkdir("temp_dump")
        for name in os.listdir("temp_dump"):
            os.remove(os.path.join("temp_dump", name))

        filename = f"temp_dump/scatter_elements_{providers[0]}_{itype}.onnx"
        opts = SessionOptions()
        opts.optimized_model_filepath = filename
        sess = InferenceSession(model.SerializeToString(), opts, providers=providers)
        self.assertTrue(sess is not None)
        self.assertExists(filename)
        onx = load(filename)
        names = [n.op_type for n in onx.graph.node]
        self.assertEqual(names, expected_names)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    @ignore_warnings(DeprecationWarning)
    def test_scatter_cuda(self):
        default_value = [
            "Add",
            "MemcpyToHost",
            "ScatterElements",
            "MemcpyFromHost",
            "Sub",
        ]
        expected = {
            (np.float32, "none"): default_value,
            (np.float16, "none"): default_value,
            (np.float32, "add"): default_value,
            (np.float16, "add"): default_value,
        }
        for dtype, reduction in itertools.product(
            [np.float32, np.float16], ["none", "add"]
        ):
            with self.subTest(dtype=dtype, reduction=reduction):
                self.common_scatter(
                    ["CUDAExecutionProvider"],
                    np.float32,
                    reduction,
                    expected[dtype, reduction],
                )

    @ignore_warnings(DeprecationWarning)
    def test_scatter_cpu(self):
        default_value = [
            "Add",
            "ScatterElements",
            "Sub",
        ]
        expected = {
            (np.float32, "none"): default_value,
            (np.float16, "none"): default_value,
            (np.float32, "add"): default_value,
            (np.float16, "add"): default_value,
        }
        for dtype, reduction in itertools.product(
            [np.float32, np.float16], ["none", "add"]
        ):
            with self.subTest(dtype=dtype, reduction=reduction):
                self.common_scatter(
                    ["CPUExecutionProvider"],
                    np.float32,
                    reduction,
                    expected[dtype, reduction],
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
