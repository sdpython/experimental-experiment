import contextlib
import io
import logging
import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_torch,
    requires_onnxruntime_training,
    skipif_ci_windows,
)


class TestBashBenchRunnerCmd(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        logger = logging.getLogger("onnxscript.optimizer.constant_folding")
        logger.setLevel(logging.ERROR)
        ExtTestCase.setUpClass()
        cls.is_grad_enabled = torch.is_grad_enabled()

    @classmethod
    def tearDownClass(cls):
        import torch

        torch.set_grad_enabled(cls.is_grad_enabled)

    def _huggingface_export_bench_cpu(
        self,
        exporter,
        models,
        verbose=0,
        debug=False,
        optimization=None,
        dump_ort=False,
        process=False,
        tag=None,
        timeout=600,
    ):
        from experimental_experiment.torch_bench.bash_bench_huggingface import main

        args = [
            "--model",
            models,
            "--device",
            "cpu",
            "--exporter",
            exporter,
            "--verbose",
            str(verbose),
            "--quiet",
            "0",
            "-w",
            "1",
            "-r",
            "1",
            "--dump_ort",
            "1" if dump_ort else "0",
            "--dump_folder",
            "dump_test_bash_bench",
            "--timeout",
            str(timeout),
        ]
        if process:
            args.extend(["--process", "1"])
        if optimization:
            args.extend(["--opt_patterns", optimization])
        if tag:
            args.extend(["--tag", tag])
        if debug:
            print("CMD")
            print(" ".join(args))
        st = io.StringIO()
        with contextlib.redirect_stderr(st), contextlib.redirect_stdout(st):
            main(args=args)
        out = st.getvalue()
        if debug:
            print(out)
        if "," in models:
            self.assertIn("Prints", out)
        else:
            self.assertIn(":model_name,", out)
        self.assertNotIn(":discrepancies_abs,inf;", out)
        if tag:
            self.assertIn(f":version_tag,{tag};", out)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu(self):
        self._huggingface_export_bench_cpu("custom", "101Dummy")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu_2_outputs(self):
        self._huggingface_export_bench_cpu("custom", "101Dummy2Outputs")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    @requires_onnxruntime_training()
    def test_huggingface_export_bench_cort_cpu(self):
        self._huggingface_export_bench_cpu("cort", "101Dummy", process=True, verbose=20)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    @requires_onnxruntime_training()
    def test_huggingface_export_bench_cortgrad_cpu(self):
        self._huggingface_export_bench_cpu("cortgrad", "101Dummy", process=True, verbose=20)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu_dump_ort(self):
        self._huggingface_export_bench_cpu("custom", "101Dummy", dump_ort=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_huggingface_export_bench_eager_cpu(self):
        self._huggingface_export_bench_cpu("eager", "101Dummy")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu2(self):
        self._huggingface_export_bench_cpu("custom", "101Dummy,101Dummy16")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu2_timeout(self):
        self._huggingface_export_bench_cpu(
            "custom", "101Dummy,101Dummy16", timeout=1, verbose=0, debug=True
        )

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu_first(self):
        self._huggingface_export_bench_cpu("custom", "0")

    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_huggingface_export_bench_script_cpu(self):
        self._huggingface_export_bench_cpu("torch_script", "101Dummy")

    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_huggingface_export_bench_script_cpu_tag(self):
        self._huggingface_export_bench_cpu("torch_script", "101Dummy", tag="taggy")

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.7")
    def test_huggingface_export_bench_onnx_dynamo_cpu(self):
        self._huggingface_export_bench_cpu("onnx_dynamo", "101Dummy")

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_dynamo_cpu(self):
        self._huggingface_export_bench_cpu("dynamo_export", "101Dummy")

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_dynamo_cpu_optimize(self):
        self._huggingface_export_bench_cpu(
            "dynamo_export", "101Dummy", optimization="default"
        )

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu_tuple(self):
        self._huggingface_export_bench_cpu("custom", "101DummyTuple")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_custom_cpu_electra(self):
        self._huggingface_export_bench_cpu(
            "custom", "ElectraForQuestionAnswering", verbose=3
        )

    def _timm_export_bench_cpu(
        self,
        exporter,
        models,
        verbose=0,
        debug=False,
        optimization=None,
        dump_ort=False,
    ):
        from experimental_experiment.torch_bench.bash_bench_timm import main

        args = [
            "--model",
            models,
            "--device",
            "cpu",
            "--exporter",
            exporter,
            "--verbose",
            str(verbose),
            "--quiet",
            "0",
            "-w",
            "1",
            "-r",
            "1",
            "--dump_ort",
            "1" if dump_ort else "0",
            "--dump_folder",
            "dump_test_bash_bench",
        ]
        if optimization:
            args.extend(["--opt_patterns", optimization])
        if debug:
            print("CMD")
            print(" ".join(args))
        st = io.StringIO()
        with contextlib.redirect_stdout(st):
            main(args=args)
        out = st.getvalue()
        if debug:
            print(out)
        if "," in models:
            self.assertIn("Prints", out)
        else:
            self.assertIn(":model_name,", out)
        self.assertNotIn(":discrepancies_abs,inf;", out)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_timm_export_bench_script_cpu(self):
        self._timm_export_bench_cpu("torch_script", "mobilenetv2_100")


if __name__ == "__main__":
    unittest.main(verbosity=2)
