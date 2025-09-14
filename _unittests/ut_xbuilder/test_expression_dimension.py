import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xbuilder.expression_dimension import (
    Expression,
    parse_expression,
    simplify_expression,
    simplify_two_expressions,
    rename_expression,
)


class TestDimension(ExtTestCase):
    def test_parse_expression(self):
        expr = "a*b*c"
        self.assertRaise(lambda: parse_expression(expr), AssertionError)
        e = parse_expression(expr, dict(a=4, b=5, c=6))
        self.assertIsInstance(e, Expression)
        self.assertEqual(repr(e), "Expression('a*b*c')")

    def test_parse_expression_div(self):
        import torch

        expr = torch.SymInt("32//s3")
        e = parse_expression(expr, dict(s3=8))
        self.assertIsInstance(e, Expression)
        self.assertEqual(repr(e), "Expression('32//s3')")

    def test_parse_expression_node(self):
        import torch
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        expr = torch.fx.experimental.sym_node.SymNode("32//s3", ShapeEnv(), int, 3)
        e = parse_expression(expr, dict(s3=8))
        self.assertIsInstance(e, Expression)
        self.assertEqual(repr(e), "Expression('32//s3')")

    def test_simplify_expression(self):
        self.assertEqual(simplify_expression("x - y + y"), "x")
        self.assertEqual(simplify_expression("2*x + 3*x - x"), "4*x")
        self.assertEqual(simplify_expression("a + b - a"), "b")
        self.assertEqual(simplify_expression("5 + x - 2 + 3"), "x+6")
        self.assertEqual(simplify_expression("x - x"), "0")

    def test_simplify_expression2(self):
        self.assertEqual(simplify_expression("5 + x - (2 + 3)"), "x")

    def test_simplify_two_expressions(self):
        self.assertEqual(
            simplify_two_expressions("s52+seq_length", "s52+s70"), {"s70": -1, "seq_length": 1}
        )

    def test_rename_expression(self):
        self.assertEqual("B+seq_length", rename_expression("s52+seq_length", {"s52": "B"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
