import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xoptim.patterns import SwitchOrderBinaryPattern


class TestGraphPatternFunctions(ExtTestCase):

    def test_switch_order(self):

        pattern = SwitchOrderBinaryPattern()
        case = pattern.switch_order(
            (2, 1024, 1024), (2, 1, 1024), (2, 1024, 1024), (1024,), side=0
        )
        self.assertEqual(case, 2)
        case = pattern.switch_order(
            (2, 1, 1024), (2, 1024, 1024), (2, 1024, 1024), (1024,), side=1
        )
        self.assertEqual(case, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
