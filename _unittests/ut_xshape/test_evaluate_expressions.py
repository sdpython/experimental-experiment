import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xshape.evaluate_expressions import evaluate_expression


class TestEvaluateExpressions(ExtTestCase):
    def test_evaluate_expression(self):
        self.assertEqual(-1, evaluate_expression("x - y", dict(x=5, y=6)))
        self.assertEqual(-5, evaluate_expression("- x", dict(x=5)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
