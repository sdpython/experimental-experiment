import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.xoptim.patterns.onnx_expand import (
    ShapeBasedExpandBroadcastPattern as EBP,
    ShapeBasedExpandSwapPattern as SBES,
)
from experimental_experiment.xoptim.patterns.onnx_reshape import (
    ShapeBasedEditDistanceReshapePattern as EDRP,
    ShapeBasedReshapeIsSqueezePattern as RISP,
)


class TestGraphPatternOptimizationUtils(ExtTestCase):
    def test_edit_distance_reshape(self):
        align = EDRP._align_shapes
        self.assertEqual((0, 1024, -1), align(("d1", 4, 256, "d2"), ("d1", 1024, "d2")))
        self.assertEqual((0, 0, 1024), align(("d1", "d2", 4, 256), ("d1", "d2", 1024)))
        self.assertEqual((6, -1), align((2, 3, "d1"), ("a", "d1")))
        self.assertEqual((6, -1), align((2, 3, "d1"), (6, "d1")))
        self.assertEqual((-1, 12, 196, 64), align(("d1", 196, 64), ("d2", 12, 196, 64)))
        self.assertEqual((-1, 196, 64), align(("d1", 196, 64), ("d2", 196, 64)))
        self.assertEqual((32, 196, 64), align((32, 196, 64), (32, 196, 64)))
        self.assertEqual((4, 8, 196, 64), align((32, 196, 64), (4, 8, 196, 64)))
        self.assertEqual((32, 196, 64), align((4, 8, 196, 64), (32, 196, 64)))
        self.assertEqual((0, 196, 64), align(("d1", 196, 64), ("d1", 196, 64)))
        self.assertEqual((0, 196, 2, 32), align(("d1", 196, 64), ("d1", 196, 2, 32)))

    def test_reshape_is_squeeze(self):
        sqsx = RISP._squeeze_axes
        self.assertEqual(("Squeeze", (1,)), sqsx(("d1", 1, 256, "d2"), ("d1", 256, "d2")))
        self.assertEqual(("Squeeze", (1, 2)), sqsx(("d1", 1, 1, 256, "d2"), ("d1", 256, "d2")))
        self.assertEqual(("Unsqueeze", (1,)), sqsx(("d1", 256, "d2"), ("d1", 1, 256, "d2")))
        self.assertEqual(
            ("Unsqueeze", (1, 2)), sqsx(("d1", 256, "d2"), ("d1", 1, 1, 256, "d2"))
        )

    def test_reshape_zero(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Reshape", ["X", "shape"], ["Y"])],
                "test",
                [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["d1", 4, 5])],
                [oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["d1", 2, 2, 5])],
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

    def test_is_compatible_shapes_for_expand(self):
        comp = EBP._is_compatible_shapes_for_expand
        self.assertFalse(comp((), (1,), (1, 2, 3, 4)))
        self.assertTrue(comp((1, 4, "d1"), (4, 1, "d1"), (4, 4, "d1")))
        self.assertFalse(comp((1, 4, "d1"), (4, 1, "d2"), (4, 4, "d3")))
        self.assertFalse(comp((1, 4, "d1"), (4, 1, "d2"), (4, 4, "d2")))
        self.assertTrue(
            comp(
                ("batch", 1, "seq_length", "cache_length+seq_length"),
                (1, 1, 1, "cache_length+seq_length"),
                ("batch", 1, "seq_length", "cache_length+seq_length"),
            )
        )

    def test_get_compatible_expand_shape_for_expand_swap(self):
        mk = SBES._get_compatible_expand_shape_for_expand_swap
        self.assertEqual(
            (1, 1, 0, 1),
            mk(
                ("batch", 1, 1, 1),
                ("batch", 1, "seq_length", "cache_length+seq_length"),
                (1, 1, 1, "cache_length+seq_length"),
                ("batch", 1, "seq_length", "cache_length+seq_length"),
                ("batch", 1, "seq_length", "cache_length+seq_length"),
            ),
        )
        self.assertEqual(
            "expand_arg",
            mk(
                ("batch", 1, 1, 1),
                ("batch", 1, "seq_length", "cache_length+seq_length"),
                (1,),
                None,
                ("batch", 1, "seq_length", "cache_length+seq_length"),
            ),
        )
        self.assertEqual(
            None,
            mk(
                (1, 1, "seq_length", 1),
                ("batch", 1, "seq_length", "cache_length+seq_length"),
                (1, 1, 1, "cache_length+seq_length"),
                None,
                ("batch", 1, "seq_length", "cache_length+seq_length"),
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
