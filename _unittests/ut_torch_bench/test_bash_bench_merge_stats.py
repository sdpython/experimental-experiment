import os
import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_bench._bash_bench_benchmark_runner import (
    merge_benchmark_reports,
)


class TestBashBenchMergeStats(ExtTestCase):

    def test_merge_stats0(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "keep.csv"), os.path.join(ddata, "keep2.csv")]
        df = merge_benchmark_reports(data, excel_output="test_merge_stats0.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("pass", set(df))

    def test_merge_stats1(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "output_data_bash_bench_huggingface.py.temp.csv")]
        df = merge_benchmark_reports(data, excel_output="test_merge_stats1.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("pass", set(df))

    def test_merge_stats3(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "dummy_benchmark.csv")]
        df = merge_benchmark_reports(data, excel_output="test_merge_stats3.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("pass", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("op_torch", set(df))


if __name__ == "__main__":
    unittest.main(verbosity=2)
