import contextlib
import io
import itertools
import logging
import os
import unittest
import onnx
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_onnxruntime_training,
    requires_torch,
    requires_onnx_diagnostic,
    skipif_ci_linux,
    is_windows,
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

    def _hg_export_bench_cpu(
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
        output_data=False,
        check_slice_input=False,
        use_bfloat16=False,
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
        ]
        if use_bfloat16:
            args.extend(["--dtype", "bfloat16"])
        if not output_data:
            args.extend(["--output_data", ""])
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
        if "," in models and output_data:
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
        onx = None
        if dynamic:
            onx = onnx.load(filename)
            input_values = []
            for i in onx.graph.input:
                shape = i.type.tensor_type.shape
                value = tuple(d.dim_param or d.dim_value for d in shape.dim)
                if value not in ((1,), tuple()):
                    assert value[0] and (
                        value[0] == "batch" or value[0][0] == "s"
                    ), f"Unexpected value={value[0]!r}"
                    input_values.append(value[0])
            assert len(set(input_values)) <= 3, f"no unique value: input_values={input_values}"
            for i in onx.graph.output:
                shape = i.type.tensor_type.shape
                value = tuple(d.dim_param or d.dim_value for d in shape.dim)
                if value != (1,):
                    assert value[0] and (
                        value[0] == "batch" or value[0][0] == "s"
                    ), f"Unexpected value={value[0]!r}"
                self.assertEqual(input_values[0], value[0])
        if check_slice_input:
            if onx is None:
                onx = onnx.load(filename)
            slices = [n for n in onx.graph.node if n.op_type in ("Slice", "Gather")]
            self.assertEqual(len(slices), 1)
            ends = slices[0].input[1]
            input_names = [i.name for i in onx.graph.input]
            for n in onx.graph.node:
                if n.op_type == "Identity" and n.input[0] in input_names:
                    input_names.append(n.output[0])
            self.assertIn(ends, input_names)

    def _explicit_export_bench_cpu(
        self,
        exporter,
        models,
        verbose=0,
        debug=False,
        optimization=None,
        tag=None,
        timeout=600,
        output_data=False,
    ):
        if is_windows():
            raise unittest.SkipTest("export does not work on Windows")
        from experimental_experiment.torch_bench.bash_bench_explicit import main

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
            "1",
            "-w",
            "1",
            "-r",
            "1",
            "--dump_folder",
            "dump_test_bash_bench",
            "--timeout",
            str(timeout),
        ]
        if not output_data:
            args.extend(["--output_data", ""])
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
        if "," in models and output_data:
            self.assertIn("Prints", out)
        else:
            self.assertIn(":model_name,", out)
        self.assertNotIn(":discrepancies_abs,inf;", out)
        if tag:
            self.assertIn(f":version_tag,{tag};", out)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu(self):
        self._hg_export_bench_cpu("custom", "101Dummy")

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_export_bench_onnx_dynamo_cpu_dynamic_1_input(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101Dummy", dynamic=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    @requires_onnx_diagnostic("0.7.13")
    def test_export_bench_custom_cpu_dynamic_1_input_dummy16(self):
        self._hg_export_bench_cpu("custom", "101Dummy16", dynamic=True, debug=False)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    @requires_onnx_diagnostic("0.7.13")
    def test_export_bench_onnx_dynamo_cpu_dynamic_1_input_dummy16(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101Dummy16", dynamic=True, debug=False)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.11")
    def test_export_bench_onnx_dynamo_cpu_dynamic_2_inputs(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101Dummy2Inputs", dynamic=True, debug=False)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    @requires_onnx_diagnostic("0.7.13")
    def test_export_bench_custom_cpu_dynamic_1_input(self):
        self._hg_export_bench_cpu("custom", "101Dummy", dynamic=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.6")
    @requires_onnx_diagnostic("0.7.13")
    def test_export_bench_custom_cpu_dynamic_2_inputs(self):
        self._hg_export_bench_cpu("custom", "101Dummy2Inputs", dynamic=True, debug=False)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu_fail(self):
        self._explicit_export_bench_cpu("custom", "1001Fail,1001Fail2", output_data=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_onnx_dynamo_cpu_fail(self):
        self._explicit_export_bench_cpu("onnx_dynamo", "1001Fail,1001Fail2", output_data=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_onnx_dynamo_cpu_dynamic(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101Dummy", dynamic=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu_2_outputs(self):
        self._hg_export_bench_cpu("custom", "101Dummy2Outputs")

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    @requires_onnxruntime_training()
    def test_huggingface_export_bench_cort_cpu(self):
        self._hg_export_bench_cpu("cort", "101Dummy", process=True, verbose=20, check_file=False)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    @requires_onnxruntime_training()
    def test_huggingface_export_bench_cortgrad_cpu(self):
        self._hg_export_bench_cpu(
            "cortgrad", "101Dummy", process=True, verbose=20, check_file=False
        )

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu_dump_ort(self):
        self._hg_export_bench_cpu("custom", "101Dummy", dump_ort=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_eager_cpu(self):
        self._hg_export_bench_cpu("eager", "101Dummy", check_file=False)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu2(self):
        self._hg_export_bench_cpu(
            "custom", "101Dummy,101Dummy16", check_file=False, output_data=True
        )

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu2_timeout(self):
        self._hg_export_bench_cpu(
            "custom",
            "101Dummy,101Dummy16",
            timeout=1,
            verbose=0,
            check_file=False,
            output_data=True,
        )

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_dynamo_cpu2_timeout(self):
        self._hg_export_bench_cpu(
            "onnx_dynamo",
            "101Dummy,101Dummy16",
            timeout=1,
            verbose=0,
            check_file=False,
            output_data=True,
        )

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu_first(self):
        self._hg_export_bench_cpu("custom", "0")

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_script_cpu(self):
        self._hg_export_bench_cpu("torch_script", "101Dummy")

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_script_cpu_tag(self):
        self._hg_export_bench_cpu("torch_script", "101Dummy", tag="taggy")

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.7")
    def test_huggingface_export_bench_onnx_dynamo_cpu(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101Dummy")

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu_tuple(self):
        self._hg_export_bench_cpu("custom", "101DummyTuple")

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_dynamo_cpu_optimize_ort(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101Dummy", optimization="ort")

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_dynamo_cpu_optimize_ir(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101Dummy", optimization="ir")

    @skipif_ci_linux("too long")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_custom_cpu_electra(self):
        self._hg_export_bench_cpu("custom", "ElectraForQuestionAnswering", verbose=3)

    def _timm_export_bench_cpu(
        self,
        exporter,
        models,
        verbose=0,
        debug=False,
        optimization=None,
        dump_ort=False,
    ):
        if is_windows():
            raise unittest.SkipTest("export does not work on Windows")
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
        if debug or int(os.environ.get("ONNXVERBOSE", "0")) > 0:
            main(args=args)
        else:
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

    @ignore_warnings((DeprecationWarning, UserWarning, RuntimeWarning))
    @requires_torch("2.7")
    def test_timm_export_bench_script_cpu(self):
        self._timm_export_bench_cpu("torch_script", "mobilenetv2_100")

    def test_model_list_explicit(self):
        from experimental_experiment.torch_bench.bash_bench_explicit import main

        args = ["--model", ""]
        st = io.StringIO()
        with contextlib.redirect_stderr(st), contextlib.redirect_stdout(st):
            main(args=args)
        out = st.getvalue()
        self.assertIn("Llama2Layer", out)

    def test_model_helper(self):
        from experimental_experiment.torch_bench._bash_bench_models_helper import (
            get_dummy_model,
            get_llama_model_layer,
            get_speech2text2_causal_ml_not_trained_model,
        )

        get_dummy_model()
        get_llama_model_layer()
        get_speech2text2_causal_ml_not_trained_model()

    # name1

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.6")
    @requires_onnx_diagnostic("0.7.13")
    def test_huggingface_export_bench_cpu_dummy_name1(self):
        for exporter, dynamic in itertools.product(["custom"], [True, False]):
            with self.subTest(exporter=exporter, dynamic=dynamic):
                if dynamic and exporter == "torch_script":
                    raise unittest.SkipTest(f"this input fails with {exporter!r}")
                self._hg_export_bench_cpu(exporter, "101DummyNamed1", dynamic=dynamic)

    # name2

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.6")
    @requires_onnx_diagnostic("0.7.13")
    def test_huggingface_export_bench_cpu_dummy_name2(self):
        for exporter, dynamic in itertools.product(["custom"], [True, False]):
            with self.subTest(exporter=exporter, dynamic=dynamic):
                if dynamic and exporter == "torch_script":
                    raise unittest.SkipTest(f"this input fails with {exporter!r}")
                self._hg_export_bench_cpu(exporter, "101DummyNamed2", dynamic=dynamic)

    # name dict

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.6")
    @requires_onnx_diagnostic("0.7.13")
    def test_huggingface_export_bench_cpu_dummy_name_dict(self):
        for exporter, dynamic in itertools.product(["custom"], [True, False]):
            with self.subTest(exporter=exporter, dynamic=dynamic):
                if dynamic and exporter == "torch_script":
                    raise unittest.SkipTest(f"this input fails with {exporter!r}")
                self._hg_export_bench_cpu(exporter, "101DummyNamedDict", dynamic=dynamic)

    # list

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.10.99")
    @requires_onnx_diagnostic("0.7.13")
    def test_huggingface_export_bench_cpu_dummy_list(self):
        for exporter, dynamic in itertools.product(["custom"], [True, False]):
            with self.subTest(exporter=exporter, dynamic=dynamic):
                if dynamic and exporter == "torch_script":
                    raise unittest.SkipTest("integer input fails with list")
                self._hg_export_bench_cpu(exporter, "101DummyIList", dynamic=dynamic)

    # int, list, none

    @unittest.skip("torch.expot.export does not work")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_cpu_dummy_none_list_int(self):
        for exporter, dynamic in itertools.product(
            ["custom", "onnx_dynamo", "torch_script"], [True, False]
        ):
            with self.subTest(exporter=exporter, dynamic=dynamic):
                self._hg_export_bench_cpu(
                    exporter,
                    "101DummyNoneListInt",
                    dynamic=dynamic,
                    check_slice_input=True,
                    debug=True,
                    verbose=4,
                )

    # dict, none, int

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_cpu_dummy_none_int_dict(self):
        for exporter, dynamic in itertools.product(["custom", "onnx_dynamo"], [True, False]):
            with self.subTest(exporter=exporter, dynamic=dynamic):
                if exporter == "onnx_dynamo" or (dynamic and exporter == "torch_script"):
                    raise unittest.SkipTest(f"this input fails with {exporter!r}")
                self._hg_export_bench_cpu(exporter, "101DummyNoneIntDict", dynamic=dynamic)


if __name__ == "__main__":
    unittest.main(verbosity=2)
