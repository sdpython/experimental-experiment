import contextlib
import io
import logging
import os
import unittest
import onnx
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    is_windows,
    requires_torch,
    requires_transformers,
    requires_onnx_diagnostic,
)
from experimental_experiment.torch_models.phi3_helper import has_phi3


class TestBashBenchRunnerCmdUntrained(ExtTestCase):
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

    def _untrained_export(
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
        dynamic=False,
        check_file=True,
        unique_first_dim=1,
        attn_impl=None,
        dtype="",
        rtopt=1,
        opt_patterns="",
    ):
        assert attn_impl in (
            None,
            "eager",
            "sdpa",
        ), f"unexpected value {attn_impl!r} for attn_impl"
        if is_windows():
            raise unittest.SkipTest("export does not work on Windows")
        from experimental_experiment.torch_bench.bash_bench_untrained import main

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
            "--output_data",
            "",
            "--rtopt",
            str(rtopt),
        ]
        if attn_impl:
            args.extend(["--attn_impl", attn_impl])
        if opt_patterns:
            args.extend(["--opt_patterns", opt_patterns])
        if dtype:
            args.extend(["--dtype", dtype])
        if dynamic:
            args.extend(["--dynamic", "1"])
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
        if debug or int(os.environ.get("ONNXVERBOSE", "0")) > 0:
            main(args=args)
        else:
            with contextlib.redirect_stderr(st), contextlib.redirect_stdout(st):
                main(args=args)
        out = st.getvalue()
        if debug:
            print(out)
        self.assertIn(":model_name,", out)
        self.assertNotIn(":discrepancies_abs,inf;", out)
        if tag:
            self.assertIn(f":version_tag,{tag};", out)
        filename = None
        for line in out.split("\n"):
            if line.startswith(":filename,"):
                filename = line.replace(":filename,", "").strip(";")
        if check_file:
            self.assertExists(filename)
        if (
            dynamic
            and exporter.startswith(("custom", "onnx", "torch"))
            and models != "arnir0/Tiny-LLM"
        ):
            onx = onnx.load(filename)
            input_values = []
            for i in onx.graph.input:
                shape = i.type.tensor_type.shape
                value = tuple(d.dim_param or d.dim_value for d in shape.dim)
                self.assertIn(
                    value[0],
                    (
                        ("batch", "s0", "seq_length")
                        if unique_first_dim == 2
                        else ("batch", "s0")
                    ),
                )
                input_values.append(value[0])
            assert (
                len(set(input_values)) == unique_first_dim
            ), f"no unique value: input_values={input_values}"
            for i in onx.graph.output:
                shape = i.type.tensor_type.shape
                value = tuple(d.dim_param or d.dim_value for d in shape.dim)
                self.assertIn(value[0], ("batch", "s0"))
                self.assertEqual(input_values[0], value[0])

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.6")
    @requires_onnx_diagnostic("0.7.5")
    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    def test_untrained_export_bench_custom_cpu(self):
        self._untrained_export("custom-dec", "Phi2LM_1Layer", verbose=1, debug=False)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.6")
    @requires_onnx_diagnostic("0.7.5")
    def test_untrained_export_bench_custom_tiny_llm_cpu(self):
        self._untrained_export("custom", "TinyLLM", verbose=1, debug=False)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.9")
    @requires_transformers("4.50.9999")
    def test_untrained_export_bench_export_cpu(self):
        self._untrained_export(
            "export-nostrict", "Phi2LM_1Layer", verbose=1, debug=False, check_file=False
        )

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.9")
    @requires_transformers("4.49.9999")
    def test_untrained_export_bench_export_cpu_dynamic(self):
        self._untrained_export(
            "export-nostrict",
            "Phi2LM_1Layer",
            verbose=1,
            debug=False,
            check_file=False,
            dynamic=True,
        )

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.7.9999")
    @requires_transformers("4.49.9999")
    @requires_onnx_diagnostic("0.7.5")
    def test_untrained_export_bench_export_cpu_diag(self):
        self._untrained_export(
            "export-nostrict",
            "arnir0/Tiny-LLM",
            verbose=1,
            debug=False,
            check_file=False,
            dynamic=True,
        )

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.7.9999")
    @requires_transformers("4.49.9999")
    @requires_onnx_diagnostic("0.7.5")
    def test_untrained_export_bench_export_cpu_custom(self):
        self._untrained_export(
            "custom",
            "arnir0/Tiny-LLM",
            verbose=1,
            debug=False,
            check_file=False,
            dynamic=True,
        )

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.7.9999")
    @requires_transformers("4.49.9999")
    @requires_onnx_diagnostic("0.7.5")
    def test_untrained_export_bench_export_cpu_custom_sdpa(self):
        self._untrained_export(
            "custom",
            "arnir0/Tiny-LLM",
            verbose=1,
            debug=False,
            check_file=False,
            dynamic=True,
            attn_impl="sdpa",
        )

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.7.9999")
    @requires_transformers("4.49.9999")
    @requires_onnx_diagnostic("0.7.5")
    def test_untrained_export_bench_export_cpu_custom_torchrt(self):
        self._untrained_export(
            "custom",
            "arnir0/Tiny-LLM",
            verbose=1,
            debug=False,
            check_file=False,
            dynamic=True,
            rtopt=9,
            opt_patterns="default",
        )

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.7.9999")
    @requires_transformers("4.55.9999")
    def test_untrained_export_bench_export_cpu_whisper(self):
        self._untrained_export(
            "export-nostrict",
            "openai/whisper-tiny",
            verbose=1,
            debug=False,
            check_file=False,
            dynamic=True,
        )

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.7.9999")
    @requires_transformers("4.55.9999")
    def test_untrained_export_bench_custom_cpu_whisper(self):
        self._untrained_export(
            "custom",
            "openai/whisper-tiny",
            verbose=1,
            debug=False,
            check_file=False,
            dynamic=True,
            unique_first_dim=2,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
