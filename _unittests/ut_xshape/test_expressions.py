import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xshape.expressions import (
    simplify_expression,
    simplify_two_expressions,
    rename_expression,
    rename_dynamic_dimensions,
    rename_dynamic_expression,
)
from experimental_experiment.xshape._shape_helper import (
    is_static_shape,
    all_float,
)


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

    def test_rename_expression(self):
        self.assertEqual("B+seq_length", rename_expression("s52+seq_length", {"s52": "B"}))

    def test_all_float(self):
        self.assertTrue(all_float([6.7, 7.8]))

    def test_is_static_shape(self):
        self.assertFalse(is_static_shape(None))

    def test_rename_dynamic_dimension(self):
        constraints = {
            "batch": {"s0", "s12", "s14", "s10", "s8", "s2"},
            "s0": {"s12", "s14", "s10", "s8", "batch"},
            "seq_length": {"s1"},
            "s1": {"seq_length"},
            "s2": {"batch"},
            "DYN0": {"s3"},
            "s3": {"DYN0"},
            "s9+seq_length": {"s1 + s9", "DYN0", "s1 + s13", "s1 + s15", "s1 + s11"},
            "s1 + s9": {"s9+seq_length"},
            "s13+seq_length": {"s1 + s13"},
            "s1 + s13": {"s13+seq_length", "s9+seq_length"},
            "s8": {"s0", "batch"},
            "s12": {"s0", "batch"},
            "s11+seq_length": {"s1 + s11"},
            "s1 + s11": {"s11+seq_length", "s9+seq_length"},
            "s15+seq_length": {"s1 + s15"},
            "s1 + s15": {"s9+seq_length", "s15+seq_length"},
            "s10": {"s0", "batch"},
            "s14": {"s0", "batch"},
            "cache_length": {"s11", "s15", "s9", "s13"},
            "s11": {"cache_length"},
            "s15": {"cache_length"},
            "s9": {"cache_length"},
            "s13": {"cache_length"},
        }

        dynamic_dimensions_source = {
            "DYN0": [{"axis": 1, "input_name": "attention_mask"}],
            "batch": [
                {"axis": 0, "input_name": "input_ids"},
                {"axis": 0, "input_name": "attention_mask"},
                {"axis": 0, "input_name": (3, 0, 0)},
                {"axis": 0, "input_name": (3, 0, 1)},
                {"axis": 0, "input_name": (3, 1, 0)},
                {"axis": 0, "input_name": (3, 1, 1)},
            ],
            "cache_length": [
                {"axis": 2, "input_name": (3, 0, 0)},
                {"axis": 2, "input_name": (3, 0, 1)},
                {"axis": 2, "input_name": (3, 1, 0)},
                {"axis": 2, "input_name": (3, 1, 1)},
            ],
            "seq_length": [{"axis": 1, "input_name": "input_ids"}],
        }

        renamed = rename_dynamic_dimensions(constraints, original=set(dynamic_dimensions_source))
        self.assertEqual(
            renamed,
            {
                "seq_length": "seq_length",
                "batch": "batch",
                "DYN0": "DYN0",
                "cache_length": "cache_length",
                "s0": "batch",
                "s12": "batch",
                "s8": "batch",
                "s14": "batch",
                "s10": "batch",
                "s1": "seq_length",
                "s2": "batch",
                "s11": "cache_length",
                "s15": "cache_length",
                "s9": "cache_length",
                "s13": "cache_length",
            },
        )

    def test_rename_dynamic_expression(self):
        replacements = {
            "DYN0": "DYN0",
            "batch": "batch",
            "cache_length": "cache_length",
            "s0": "batch",
            "s1": "seq_length",
            "s1 + s11": "DYN0",
            "s1 + s13": "DYN0",
            "s1 + s15": "DYN0",
            "s1 + s9": "DYN0",
            "s10": "batch",
            "s12": "batch",
            "s14": "batch",
            "s2": "batch",
            "s3": "DYN0",
            "s8": "batch",
            "s9+seq_length": "DYN0",
            "seq_length": "seq_length",
            "s11": "cache_length",
            "s15": "cache_length",
            "s9": "cache_length",
            "s13": "cache_length",
        }
        expression = "s9+seq_length"
        renamed = rename_dynamic_expression(expression, replacements)
        self.assertEqual(renamed, "cache_length+seq_length")


if __name__ == "__main__":
    unittest.main(verbosity=2)
