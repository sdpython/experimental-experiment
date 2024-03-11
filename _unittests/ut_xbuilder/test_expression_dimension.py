import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xbuilder.expression_dimension import (
    Expression,
    parse_expression,
)


class TestDimension(ExtTestCase):
    def test_parse_expression(self):
        expr = "a*b*c"
        self.assertRaise(lambda: parse_expression(expr), AssertionError)
        e = parse_expression(expr, dict(a=4, b=5, c=6))
        self.assertIsInstance(e, Expression)
        self.assertEqual(repr(e), "Expression('a*b*c')")


if __name__ == "__main__":
    unittest.main(verbosity=2)
