import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xshape.expressions import Expression
from experimental_experiment.xshape.expressions_torch import parse_expression


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
