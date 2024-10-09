import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xbuilder._shape_helper import (
    is_static_shape,
    all_float,
)


class TestTools(ExtTestCase):
    def test_all_float(self):
        self.assertTrue(all_float([6.7, 7.8]))

    def test_is_static_shape(self):
        self.assertFalse(is_static_shape(None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
