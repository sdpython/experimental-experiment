import contextlib
import io
import logging
import os
import unittest
import onnx
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_torch,
    requires_transformers,
    requires_pyinstrument,
    is_windows,
)


class TestBashBenchRunnerCmdOptions(ExtTestCase):
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

    def _export_cmd(
        self,
        exporter,
        models,
        verbose=1,
        debug=False,
        optimization=None,
        dump_ort=False,
        process=False,
        tag=None,
        timeout=600,
        dynamic=False,
        check_file=True,
    ):
        if is_windows():
            raise unittest.SkipTest("export does not work on Windows")
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
            "--output_data",
            "",
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
        if dynamic:
            onx = onnx.load(filename)
            input_values = []
            for posi, i in enumerate(onx.graph.input):
                if posi > 0 and "DynamicCache" in models:
                    # Only the first input is static
                    break
                shape = i.type.tensor_type.shape
                value = tuple(d.dim_param or d.dim_value for d in shape.dim)
                self.assertIn(value[0], ("batch", "s0", "s1"))
                input_values.append(value[0])
            assert len(set(input_values)) <= 2, f"no unique value: input_values={input_values}"
            for i in onx.graph.output:
                shape = i.type.tensor_type.shape
                value = tuple(d.dim_param or d.dim_value for d in shape.dim)
                self.assertIn(value[0], ("batch", "s0", "s1"))
                self.assertEqual(input_values[0], value[0])

    # export

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_export(self):
        self._export_cmd("export", "101Dummy", check_file=False)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_export_jit(self):
        self._export_cmd("export-jit", "101Dummy", check_file=False)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_export_dec(self):
        self._export_cmd("export-dec", "101Dummy", check_file=False)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_export_nostrict(self):
        self._export_cmd("export-nostrict", "101Dummy", check_file=False)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_export_fallback(self):
        self._export_cmd("export-fallback", "101Dummy", check_file=False)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_export_fallback_dec(self):
        self._export_cmd("export-fallback-dec", "101Dummy", check_file=False)

    # custom

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_custom(self):
        self._export_cmd("custom", "101Dummy", check_file=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_custom_jit(self):
        self._export_cmd("custom-jit", "101Dummy", check_file=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_custom_dec(self):
        self._export_cmd("custom-dec", "101Dummy", check_file=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_custom_nostrict(self):
        self._export_cmd("custom-nostrict", "101Dummy", check_file=True)

    # @ignore_warnings((DeprecationWarning, UserWarning))
    # @requires_torch("2.4")
    # def test_custom_tracing(self):
    #    self._export_cmd("custom-tracing", "101Dummy", check_file=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_custom_fallback(self):
        self._export_cmd("custom-fallback", "101Dummy", check_file=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_custom_fallback_dec(self):
        self._export_cmd("export-fallback-dec", "101Dummy", check_file=False)

    # onnx_dynamo

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_onnx_dynamo(self):
        self._export_cmd("onnx_dynamo", "101Dummy", check_file=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_onnx_dynamo_fallback(self):
        self._export_cmd("onnx_dynamo-fallback", "101Dummy", check_file=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    @requires_pyinstrument()
    def test_onnx_dynamo_detailed(self):
        self._export_cmd("onnx_dynamo-detailed", "101Dummy", check_file=True)

    # kind of inputs

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_eager_list(self):
        self._export_cmd("eager", "101DummyIList", check_file=False)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_eager_int(self):
        self._export_cmd("eager", "101DummyIInt", check_file=False)

    # int, none

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.7")
    @requires_transformers("4.49.9999")
    def test_eager_none_int(self):
        for exporter in ["eager", "export"]:
            with self.subTest(exporter=exporter):
                self._export_cmd(exporter, "101DummyNoneInt", dynamic=False, check_file=False)

    # int, none, default

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_eager_none_int_default(self):
        for exporter in ["eager", "export"]:
            with self.subTest(exporter=exporter):
                self._export_cmd(
                    exporter, "101DummyNoneIntDefault", dynamic=False, check_file=False
                )

    # int, list, none

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_eager_none_list_int(self):
        for exporter in ["eager", "export"]:
            with self.subTest(exporter=exporter):
                if exporter == "export":
                    raise unittest.SkipTest("this one does not work")
                self._export_cmd(
                    exporter, "101DummyNoneListInt", dynamic=False, check_file=False
                )

    # dict, none, int

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_eager_none_int_dict(self):
        for exporter in ["eager", "export"]:
            with self.subTest(exporter=exporter):
                self._export_cmd(
                    exporter, "101DummyNoneIntDict", dynamic=False, check_file=False
                )

    # DynamicCache

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.7")
    @requires_transformers("4.51.9999")
    def test_dynamic_cache_eager(self):
        for exporter in ["export", "eager"]:
            with self.subTest(exporter=exporter):
                self._export_cmd(
                    exporter,
                    "101DummyDynamicCache",
                    dynamic=False,
                    check_file=False,
                    debug=False,
                )

    @unittest.skip("issue https://github.com/pytorch/pytorch/issues/142161")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_dynamic_cache_custom_dynamic(self):
        for exporter in ["custom"]:
            for dyn in [True]:
                with self.subTest(exporter=exporter, dynamic=dyn):
                    self._export_cmd(
                        exporter,
                        "101DummyDynamicCache",
                        dynamic=dyn,
                        check_file=False,
                        debug=False,
                        verbose=30,
                    )

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_dynamic_cache_custom_static(self):
        for exporter in ["custom"]:
            for dyn in [False]:
                with self.subTest(exporter=exporter, dynamic=dyn):
                    self._export_cmd(
                        exporter,
                        "101DummyDynamicCache",
                        dynamic=dyn,
                        check_file=False,
                        debug=False,
                        verbose=30,
                    )

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_mamba_cache_custom_static(self):
        for exporter in ["custom"]:
            for dyn in [False]:
                with self.subTest(exporter=exporter, dynamic=dyn):
                    self._export_cmd(
                        exporter,
                        "101DummyMambaCache",
                        dynamic=dyn,
                        check_file=False,
                        debug=False,
                        verbose=30,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
