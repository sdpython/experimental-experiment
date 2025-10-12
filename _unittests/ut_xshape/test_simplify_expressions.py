import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xshape.simplify_expressions import (
    simplify_expression,
    simplify_two_expressions,
)
from experimental_experiment.xshape._shape_helper import is_static_shape, all_float


class TestDimension(ExtTestCase):
    def test_simplify_expression(self):
        self.assertEqual(simplify_expression("x - y + y"), "x")
        self.assertEqual(simplify_expression("2*x + 3*x - x"), "4*x")
        self.assertEqual(simplify_expression("a + b - a"), "b")
        self.assertEqual(simplify_expression("5 + x - 2 + 3"), "x+6")
        self.assertEqual(simplify_expression("x - x"), "0")

    def test_simplify_expression2(self):
        self.assertEqual(simplify_expression("5 + x - (2 + 3)"), "x")

    def test_simplify_expression3(self):
        self.assertEqual(simplify_expression("x - 1"), "x-1")
        self.assertEqual(simplify_expression("1 - x"), "-x+1")

    def test_simplify_two_expressions(self):
        self.assertEqual(
            simplify_two_expressions("s52+seq_length", "s52+s70"), {"s70": -1, "seq_length": 1}
        )

    def test_all_float(self):
        self.assertTrue(all_float([6.7, 7.8]))

    def test_is_static_shape(self):
        self.assertFalse(is_static_shape(None))

    def test_simplify_expression_bracket(self):
        self.assertEqual("x", simplify_expression("2*x//2"))
        self.assertEqual("x", simplify_expression("(2*x)//2"))
        self.assertEqual("x", simplify_expression("(x*y)//y"))
        self.assertEqual("x", simplify_expression("(x*(y+1))//(y+1)"))
        self.assertEqual("c//2", simplify_expression("((c)//(2))"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
