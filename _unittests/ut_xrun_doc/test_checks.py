import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.checks import print_import_time


class TestUnitTest(ExtTestCase):
    def test_print_import_time(self):
        _, out, err = self.capture(lambda: print_import_time())
        self.assertIn("time to", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
