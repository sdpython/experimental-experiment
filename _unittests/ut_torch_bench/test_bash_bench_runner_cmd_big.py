import contextlib
import io
import logging
import unittest
import onnx
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_torch,
    skipif_ci_windows,
)


class TestBashBenchRunnerCmdBig(ExtTestCase):
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

    def _huggingface_export_bench_big_cpu(
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
    ):
        from experimental_experiment.torch_bench.bash_bench_huggingface_big import main

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
        if debug:
            main(args=args)
        else:
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
        filename = None
        for line in out.split("\n"):
            if line.startswith(":filename,"):
                filename = line.replace(":filename,", "").strip(";")
        if check_file:
            self.assertExists(filename)
        if dynamic:
            onx = onnx.load(filename)
            input_values = []
            for i in onx.graph.input:
                shape = i.type.tensor_type.shape
                value = tuple(d.dim_param or d.dim_value for d in shape.dim)
                self.assertIn(value[0], ("batch", "s0"))
                input_values.append(value[0])
            assert (
                len(set(input_values)) == 1
            ), f"no unique value: input_values={input_values}"
            for i in onx.graph.output:
                shape = i.type.tensor_type.shape
                value = tuple(d.dim_param or d.dim_value for d in shape.dim)
                self.assertIn(value[0], ("batch", "s0"))
                self.assertEqual(input_values[0], value[0])

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_custom_cpu(self):
        self._huggingface_export_bench_big_cpu(
            "custom", "all_MiniLM_L6_v1", verbose=0, debug=False
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)