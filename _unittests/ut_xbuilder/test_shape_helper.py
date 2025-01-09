import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.helpers import rename_dynamic_dimensions
from experimental_experiment.xbuilder._shape_helper import (
    is_static_shape,
    all_float,
)


class TestShapeHelper(ExtTestCase):
    def test_all_float(self):
        self.assertTrue(all_float([6.7, 7.8]))

    def test_is_static_shape(self):
        self.assertFalse(is_static_shape(None))

    def test_rename_dynamic_dimension(self):
        constraints = {
            "DYN0": {"s3"},
            "batch": {"s2", "s10", "s0", "s8", "s14", "s12"},
            "s0": {"batch", "s10", "s8", "s14", "s12"},
            "s1": {"seq_length"},
            "s1 + s11": {"s11+seq_length", "s9+seq_length"},
            "s1 + s13": {"s9+seq_length", "s13+seq_length"},
            "s1 + s15": {"s15+seq_length", "s9+seq_length"},
            "s1 + s9": {"s9+seq_length"},
            "s10": {"batch", "s0"},
            "s11+seq_length": {"s1 + s11"},
            "s12": {"batch", "s0"},
            "s13+seq_length": {"s1 + s13"},
            "s14": {"batch", "s0"},
            "s15+seq_length": {"s1 + s15"},
            "s2": {"batch"},
            "s3": {"DYN0"},
            "s8": {"batch", "s0"},
            "s9+seq_length": {"s1 + s9", "DYN0", "s1 + s13", "s1 + s11", "s1 + s15"},
            "seq_length": {"s1"},
        }

        dynamic_dimensions_source = {
            "DYN0": [{"axis": 1, "input_name": "attention_mask"}],
            "batch": [
                {"axis": 0, "input_name": "input_ids"},
                {"axis": 0, "input_name": "attention_mask"},
                {"axis": 0, "input_name": (3, 0)},
                {"axis": 0, "input_name": (3, 0)},
                {"axis": 0, "input_name": (3, 1)},
                {"axis": 0, "input_name": (3, 1)},
            ],
            "cache_length": [
                {"axis": 2, "input_name": (3, 0)},
                {"axis": 2, "input_name": (3, 0)},
                {"axis": 2, "input_name": (3, 1)},
                {"axis": 2, "input_name": (3, 1)},
            ],
            "seq_length": [{"axis": 1, "input_name": "input_ids"}],
        }

        renamed = rename_dynamic_dimensions(
            constraints, original=set(dynamic_dimensions_source)
        )
        self.assertEqual(
            renamed,
            {
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
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
