import os
import unittest
import numpy as np
from pandas.errors import PerformanceWarning
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
)
from experimental_experiment.torch_bench._bash_bench_benchmark_runner_agg import (
    merge_benchmark_reports,
)


class TestBashBenchMergeStats(ExtTestCase):

    @ignore_warnings((PerformanceWarning, FutureWarning))
    def test_merge_stats0(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "keep.csv"), os.path.join(ddata, "keep2.csv")]
        df = merge_benchmark_reports(data, excel_output="test_merge_stats0.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))

    @ignore_warnings((PerformanceWarning, FutureWarning))
    def test_merge_stats1(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "output_data_bash_bench_huggingface.py.temp.csv")]
        df = merge_benchmark_reports(data, excel_output="test_merge_stats1.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))

    @ignore_warnings((PerformanceWarning, FutureWarning))
    def test_merge_stats3(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "dummy_benchmark.csv")]
        df = merge_benchmark_reports(data, excel_output="test_merge_stats3.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("op_torch", set(df))

    @ignore_warnings((PerformanceWarning, FutureWarning))
    def test_merge_stats_gpu_mem(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "gpu_mem.csv")]
        df = merge_benchmark_reports(data, excel_output="test_merge_stats_gpu_mem.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("op_torch", set(df))

    @ignore_warnings((PerformanceWarning, FutureWarning))
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

    @ignore_warnings((PerformanceWarning, FutureWarning))
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

    @ignore_warnings((PerformanceWarning, FutureWarning))
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

    @ignore_warnings((PerformanceWarning, FutureWarning))
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

    @ignore_warnings((PerformanceWarning, FutureWarning))
    def test_merge_stats_bug_op_onnx(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "bug_op_onnx.csv")]
        df = merge_benchmark_reports(
            data, excel_output="test_merge_stats_bug_op_onnx.xlsx"
        )
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("op_torch", set(df))
        dfo = df["op_onnx"]
        self.assertNotIn("op_onnx_MatMul", str(dfo.columns))

    @ignore_warnings((PerformanceWarning, FutureWarning))
    def test_merge_stats_empty(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "bug_timm_empty.csv")]
        df = merge_benchmark_reports(data, excel_output="test_merge_stats_empty.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("op_torch", set(df))
        self.assertIn("ERR", set(df))

    @skipif_ci_windows("pandas and *.csv do not work well on Windows")
    @ignore_warnings((FutureWarning,))
    def test_merge_stats_many_days(self):
        ddata = os.path.join(os.path.dirname(__file__), "data", "rawdata")
        data = [os.path.join(ddata, "*.csv")]
        df = merge_benchmark_reports(
            data, excel_output="test_merge_stats_many_days.xlsx"
        )
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("ERR", set(df))

    @ignore_warnings((FutureWarning,))
    def test_merge_stats_bug_speedup(self):
        data = os.path.join(os.path.dirname(__file__), "data", "bug_speed_up.csv")
        df = merge_benchmark_reports(
            data, excel_output="test_merge_stats_bug_speedup.xlsx"
        )
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("ERR", set(df))
        self.assertIn("AGG", set(df))
        self.assertIn("SUMMARY", set(df))
        agg = df["AGG"].reset_index(drop=False)
        sp = agg[(agg["cat"] == "speedup") & (agg["agg"] == "GEO-MEAN")]
        values = sp["HuggingFace"].values
        self.assertEqualArray(
            np.array([0.952044, 0.000001, 1.020653]), values, atol=1e-5
        )
        summary = df["SUMMARY"]
        self.assertNotIn("_dummy_", summary.columns)
        values = summary.values
        self.assertEqual(0.9520435772282563, values[4, 3])
        self.assertEqual("x", values[4, 4])
        metrics = set(summary["METRIC"])
        self.assertIn("number of running models", metrics)
        self.assertIn("pass rate", metrics)
        self.assertIn("average export time", metrics)

    @ignore_warnings((FutureWarning,))
    def test_merge_stats_bug_cpu_cuda(self):
        data = os.path.join(os.path.dirname(__file__), "data", "bug_cpu_cuda.csv")
        df = merge_benchmark_reports(
            data, excel_output="test_merge_stats_bug_cpu_cuda.xlsx"
        )
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("ERR", set(df))
        self.assertIn("AGG", set(df))
        self.assertIn("SUMMARY", set(df))


if __name__ == "__main__":
    unittest.main(verbosity=2)
