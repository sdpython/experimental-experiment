import math
import unittest
from experimental_experiment.ext_test_case import ExtTestCase, measure_time
from experimental_experiment.checks import print_import_time


class TestUnitTest(ExtTestCase):
    def test_print_import_time(self):
        _, out, err = self.capture(lambda: print_import_time())
        self.assertIn("time to", out)

    def test_measure_time(self):
        res = measure_time(lambda: math.cos(5) + 1, repeat=10, number=10, div_by_number=True)
        self.assertIsInstance(res, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
