import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.xoptim.patterns.onnx_reshape import (
    EditDistanceReshapePattern as EDRP,
)

align = EDRP._align_shapes


class TestGraphPatternOptimizationUtils(ExtTestCase):
    def test_edit_distance_reshape(self):
        self.assertEqual((6, -1), align((2, 3, "d"), ("a", "d")))
        self.assertEqual((6, -1), align((2, 3, "d"), (6, "d")))
        self.assertEqual((-1, 12, 196, 64), align(("A", 196, 64), ("B", 12, 196, 64)))
        self.assertEqual((-1, 196, 64), align(("A", 196, 64), ("B", 196, 64)))
        self.assertEqual((32, 196, 64), align((32, 196, 64), (32, 196, 64)))
        self.assertEqual((4, 8, 196, 64), align((32, 196, 64), (4, 8, 196, 64)))
        self.assertEqual((32, 196, 64), align((4, 8, 196, 64), (32, 196, 64)))
        self.assertEqual((0, 196, 64), align(("A", 196, 64), ("A", 196, 64)))
        self.assertEqual((0, 196, 2, 32), align(("A", 196, 64), ("A", 196, 2, 32)))

    def test_reshape_zero(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Reshape", ["X", "shape"], ["Y"])],
                "test",
                [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["A", 4, 5])],
                [oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["A", 2, 2, 5])],
                [onh.from_array(np.array([0, 2, 2, 5], dtype=np.int64), name="shape")],
            ),
            opset_imports=[oh.make_opsetid("", 20)],
            ir_version=10,
        )
        inputs = {"X": np.random.rand(7, 4, 5).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, inputs)
        self.assertEqual((7, 2, 2, 5), got[0].shape)

        import onnxruntime

        ref = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = ref.run(None, inputs)
        self.assertEqual((7, 2, 2, 5), got[0].shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
