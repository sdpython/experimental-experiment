import os
import unittest
import numpy as np
from pandas.errors import PerformanceWarning
from experimental_experiment.torch_bench._bash_bench_benchmark_runner_agg import (
    merge_benchmark_reports,
)
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
)


class TestBashBenchMergeStats(ExtTestCase):
    @ignore_warnings((PerformanceWarning, FutureWarning))
    def test_merge_stats0(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "keep.csv"), os.path.join(ddata, "keep2.csv")]
        df = merge_benchmark_reports(
            data,
            excel_output="test_merge_stats0.xlsx",
            export_simple="test_merge_stats0_simple.csv",
            export_correlations="test_merge_stats0_corrs.xlsx",
        )
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
        df = merge_benchmark_reports(data, excel_output="test_merge_stats_duplicate.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        # self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("op_torch", set(df))

    @ignore_warnings((PerformanceWarning, FutureWarning))
    def test_merge_stats_bug_timm(self):
        ddata = os.path.join(os.path.dirname(__file__), "data")
        data = [os.path.join(ddata, "bug_timm.csv")]
        df = merge_benchmark_reports(data, excel_output="test_merge_stats_bug_timm.xlsx")
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
        df = merge_benchmark_reports(data, excel_output="test_merge_stats_bug_op_onnx.xlsx")
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
        df = merge_benchmark_reports(data, excel_output="test_merge_stats_many_days.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("ERR", set(df))

    @ignore_warnings((FutureWarning,))
    def test_merge_stats_bug_speedup_summary(self):
        data = os.path.join(os.path.dirname(__file__), "data", "bug_speed_up.csv")
        df = merge_benchmark_reports(
            data, excel_output="test_merge_stats_bug_speedup_summary.xlsx"
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
            np.array([0.952044, 0.000001, 1.020653]), values.astype(float), atol=1e-5
        )
        summary = df["SUMMARY"]
        self.assertNotIn("_dummy_", summary.columns)
        values = summary.values
        self.assertEqual(0.9520435772282563, values[13, 1])
        self.assertEqual("x", values[13, 4])
        metrics = set(summary["METRIC"])
        self.assertIn("number of running models", metrics)
        self.assertIn("export number", metrics)
        self.assertIn("average export time", metrics)

    @ignore_warnings((FutureWarning,))
    def test_merge_stats_bug_cpu_cuda(self):
        data = os.path.join(os.path.dirname(__file__), "data", "bug_cpu_cuda.csv")
        df = merge_benchmark_reports(data, excel_output="test_merge_stats_bug_cpu_cuda.xlsx")
        self.assertIsInstance(df, dict)
        self.assertIn("status", set(df))
        self.assertIn("memory", set(df))
        self.assertIn("op_onnx", set(df))
        self.assertIn("ERR", set(df))
        self.assertIn("AGG", set(df))
        self.assertIn("SUMMARY", set(df))

    @ignore_warnings((FutureWarning,))
    def test_merge_stats_filter(self):
        data = os.path.join(os.path.dirname(__file__), "data", "bug_cpu_cuda.csv")
        df1 = merge_benchmark_reports(
            data,
            excel_output="test_merge_stats_filter1.xlsx",
            filter_in="device:cpu;cuda",
        )
        df2 = merge_benchmark_reports(
            data, excel_output="test_merge_stats_filter2.xlsx", filter_in="device:cuda"
        )
        df3 = merge_benchmark_reports(
            data, excel_output="test_merge_stats_filter3.xlsx", filter_out="device:cuda"
        )
        self.assertIsInstance(df1, dict)
        self.assertIsInstance(df2, dict)
        self.assertIsInstance(df3, dict)
        for k in df1:
            if k not in df2:
                continue
            sh1 = df1[k].shape
            sh2 = df2[k].shape
            sh3 = df3[k].shape
            if k == "speedup":
                self.assertEqual(sh1, (1, 6))
                self.assertEqual(sh2, (1, 3))
                self.assertEqual(sh3, (1, 3))
            if k == "0main":
                self.assertNotIn(["device", "cpu"], df2[k].values.tolist())
                self.assertIn(["device", "cpu"], df3[k].values.tolist())
                self.assertNotIn(["device", "cuda"], df3[k].values.tolist())
                self.assertIn(["device", "cuda"], df2[k].values.tolist())

    @ignore_warnings((FutureWarning,))
    def test_merge_stats_filter_hg(self):
        data = os.path.join(os.path.dirname(__file__), "data", "bug_speed_up.csv")
        df = merge_benchmark_reports(
            data,
            excel_output="test_merge_stats_filter_hg.xlsx",
            filter_in="model_name:HG",
        )
        self.assertIsInstance(df, dict)
        for k, v in df.items():
            sh = v.shape
            if k == "speedup":
                self.assertEqual(sh, (12, 3))
        df = merge_benchmark_reports(
            data,
            excel_output="test_merge_stats_filter_hg_none.xlsx",
            filter_in="model_name:NONE",
        )
        self.assertEqual(df, {})

    @ignore_warnings((FutureWarning,))
    def test_merge_stats_diff(self):
        base = os.path.join(os.path.dirname(__file__), "data", "baseline.csv")
        data = os.path.join(os.path.dirname(__file__), "data", "baseline2.csv")
        dfs = merge_benchmark_reports(
            data,
            excel_output="test_merge_stats_diff.xlsx",
            baseline=base,
        )
        self.assertIn("SUMMARY2_diff", dfs)
        self.assertIn("MODELS_diff", dfs)
        self.assertIn("SUMMARY_diff", dfs)

    @ignore_warnings((FutureWarning,))
    def test_merge_stats_broken(self):
        data = os.path.join(
            os.path.dirname(__file__), "data", "huggingface_benchmark_v100_custom.csv"
        )
        dfs = merge_benchmark_reports(
            data,
            excel_output="test_merge_stats_broken.xlsx",
            broken=True,
        )
        self.assertNotEmpty(dfs)

    @ignore_warnings((FutureWarning,))
    def test_merge_stats_slow(self):
        data = os.path.join(
            os.path.dirname(__file__), "data", "huggingface_benchmark_v100_custom.csv"
        )
        dfs = merge_benchmark_reports(
            data,
            excel_output="test_merge_stats_low.xlsx",
            slow=0.9,
        )
        self.assertNotEmpty(dfs)

    @ignore_warnings((FutureWarning,))
    def test_merge_stats_subset(self):
        data = os.path.join(
            os.path.dirname(__file__), "data", "huggingface_benchmark_v100_custom.csv"
        )
        dfs = merge_benchmark_reports(
            data,
            excel_output="test_merge_stats_subset.xlsx",
            slow=0.9,
            fast=1.1,
            slow_script=0.9,
            fast_script=1.1,
            disc=0.1,
        )
        self.assertNotEmpty(dfs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
