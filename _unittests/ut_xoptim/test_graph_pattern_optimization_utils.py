import unittest
from experimental_experiment.ext_test_case import ExtTestCase
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
