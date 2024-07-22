import os
import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_bench._bash_bench_benchmark_runner_agg import (
    merge_benchmark_reports,
)


class TestBashBenchMergeStats(ExtTestCase):

    def test_merge_stats0(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "keep.csv"), os.path.join(ddata, "keep2.csv")]
        df = merge_benchmark_reports(data, excel_output="test_merge_stats0.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))

    def test_merge_stats1(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "output_data_bash_bench_huggingface.py.temp.csv")]
        df = merge_benchmark_reports(data, excel_output="test_merge_stats1.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))

    def test_merge_stats3(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "dummy_benchmark.csv")]
        df = merge_benchmark_reports(data, excel_output="test_merge_stats3.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("op_torch", set(df))

    def test_merge_stats_gpu_mem(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "gpu_mem.csv")]
        df = merge_benchmark_reports(data, excel_output="test_merge_stats_gpu_mem.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("op_torch", set(df))

    def test_merge_stats_gpu_big(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [
            os.path.join(ddata, "huggingface_benchmark_v100_main.csv"),
            os.path.join(ddata, "huggingface_benchmark_v100_default_opt.csv"),
            os.path.join(ddata, "huggingface_benchmark_v100_custom.csv"),
        ]
        df = merge_benchmark_reports(data, excel_output="test_merge_stats_gpu_big.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("op_torch", set(df))

    def test_merge_stats_duplicate(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [
            os.path.join(ddata, "duplicate0.csv"),
            os.path.join(ddata, "duplicate2.csv"),
        ]
        df = merge_benchmark_reports(
            data, excel_output="test_merge_stats_duplicate.xlsx"
        )
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        # self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("op_torch", set(df))

    def test_merge_stats_bug_timm(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "bug_timm.csv")]
        df = merge_benchmark_reports(
            data, excel_output="test_merge_stats_bug_timm.xlsx"
        )
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("op_torch", set(df))

    def test_merge_stats_bug_one_export(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "bug_one_export.csv")]
        df = merge_benchmark_reports(
            data, excel_output="test_merge_stats_bug_one_export.xlsx"
        )
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("op_torch", set(df))


if __name__ == "__main__":
    unittest.main(verbosity=2)
