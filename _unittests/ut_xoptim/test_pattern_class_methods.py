import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xoptim.patterns_exp.binary_operators import (
    AddAddMulMulPattern,
)


class TestPatternClassMethod(ExtTestCase):

    def _check_same_shape(self, sh1, sh2, broadcast=False):
        class MockGB:
            def has_shape(self, _):
                return True

            def get_shape(self_, name):
                if name == "a":
                    return sh1
                if name == "b":
                    return sh2

        return AddAddMulMulPattern._same_shape(MockGB(), "a", "b", broadcast=broadcast)

    def test_same_shape(self):
        self.assertTrue(self._check_same_shape((1, 3), (1, 3)))
        self.assertFalse(self._check_same_shape((1, 3), (3, 3)))

    def test_same_shape_broadcast(self):
        self.assertTrue(self._check_same_shape((1, 1, 3), (3, 1, 3), broadcast=True))
        self.assertFalse(self._check_same_shape((1, 2, 3), (3, 1, 3), broadcast=True))
        self.assertTrue(self._check_same_shape((1, 3), (1, 3), broadcast=True))
        self.assertTrue(self._check_same_shape((1, 3), (3, 3), broadcast=True))
        self.assertTrue(self._check_same_shape((1, 1, 3), (1, 3, 3), broadcast=True))
        self.assertTrue(self._check_same_shape((1, 3, 3), (1, 1, 3), broadcast=True))
        self.assertFalse(self._check_same_shape((3, 2, 3), (3, 1, 3), broadcast=True))
        self.assertFalse(self._check_same_shape((3, 1, 3), (3, 2, 3), broadcast=True))
        self.assertTrue(
            self._check_same_shape(
                (2, 32, 1024, 128), (1, 1, 1024, 128), broadcast=True
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
