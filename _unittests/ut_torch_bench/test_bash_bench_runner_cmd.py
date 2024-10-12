import contextlib
import io
import logging
import unittest
import onnx
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_onnxruntime_training,
    requires_torch,
    skipif_ci_windows,
    skipif_ci_linux,
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
        if debug:
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
        if dynamic:
            onx = onnx.load(filename)
            input_values = []
            for i in onx.graph.input:
                shape = i.type.tensor_type.shape
                value = tuple(d.dim_param or d.dim_value for d in shape.dim)
                if value != (1,):
                    self.assertIn(value[0], ("batch", "s0", "s1", "s2"))
                    input_values.append(value[0])
            assert (
                len(set(input_values)) <= 3
            ), f"no unique value: input_values={input_values}"
            for i in onx.graph.output:
                shape = i.type.tensor_type.shape
                value = tuple(d.dim_param or d.dim_value for d in shape.dim)
                if value != (1,):
                    self.assertIn(value[0], ("batch", "s0", "s1", "s2"))
                self.assertEqual(input_values[0], value[0])

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

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu(self):
        self._hg_export_bench_cpu("custom", "101Dummy")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_export_bench_onnx_dynamo_cpu_dynamic_1_input(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101Dummy", dynamic=True, debug=False)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_export_bench_custom_cpu_dynamic_1_input_dummy16(self):
        self._hg_export_bench_cpu("custom", "101Dummy16", dynamic=True, debug=False)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_export_bench_onnx_dynamo_cpu_dynamic_1_input_dummy16(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101Dummy16", dynamic=True, debug=False)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_export_bench_onnx_dynamo_cpu_dynamic_2_inputs(self):
        self._hg_export_bench_cpu(
            "onnx_dynamo", "101Dummy2Inputs", dynamic=True, debug=False
        )

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_export_bench_custom_cpu_dynamic_1_input(self):
        self._hg_export_bench_cpu("custom", "101Dummy", dynamic=True, debug=False)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_export_bench_custom_cpu_dynamic_2_inputs(self):
        self._hg_export_bench_cpu("custom", "101Dummy2Inputs", dynamic=True, debug=False)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu_fail(self):
        self._explicit_export_bench_cpu("custom", "1001Fail,1001Fail2", output_data=True)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_onnx_dynamo_cpu_fail(self):
        self._explicit_export_bench_cpu(
            "onnx_dynamo", "1001Fail,1001Fail2", output_data=True
        )

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_onnx_dynamo_cpu_dynamic(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101Dummy", dynamic=True)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu_2_outputs(self):
        self._hg_export_bench_cpu("custom", "101Dummy2Outputs")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_dynamo_cpu_2_outputs(self):
        self._hg_export_bench_cpu("dynamo_export", "101Dummy2Outputs")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    @requires_onnxruntime_training()
    def test_huggingface_export_bench_cort_cpu(self):
        self._hg_export_bench_cpu(
            "cort", "101Dummy", process=True, verbose=20, check_file=False
        )

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    @requires_onnxruntime_training()
    def test_huggingface_export_bench_cortgrad_cpu(self):
        self._hg_export_bench_cpu(
            "cortgrad", "101Dummy", process=True, verbose=20, check_file=False
        )

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu_dump_ort(self):
        self._hg_export_bench_cpu("custom", "101Dummy", dump_ort=True)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_dynamo_cpu_dump_ort(self):
        self._hg_export_bench_cpu("dynamo_export", "101Dummy", dump_ort=True)

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_eager_cpu(self):
        self._hg_export_bench_cpu("eager", "101Dummy", check_file=False)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu2(self):
        self._hg_export_bench_cpu(
            "custom", "101Dummy,101Dummy16", check_file=False, output_data=True
        )

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_dynamo_cpu2(self):
        self._hg_export_bench_cpu(
            "dynamo_export", "101Dummy,101Dummy16", check_file=False, output_data=True
        )

    @skipif_ci_windows("exporter does not work on Windows")
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

    @skipif_ci_windows("exporter does not work on Windows")
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

    @skipif_ci_windows("exporter does not work on Windows")
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
    @requires_torch("2.5")
    def test_huggingface_export_bench_dynamo_cpu(self):
        self._hg_export_bench_cpu("dynamo_export", "101Dummy")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu_tuple(self):
        self._hg_export_bench_cpu("custom", "101DummyTuple")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_dynamo_cpu_optimize(self):
        self._hg_export_bench_cpu("dynamo_export", "101Dummy", optimization="default")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_dynamo_cpu_tuple(self):
        self._hg_export_bench_cpu("dynamo_export", "101DummyTuple")

    @skipif_ci_windows("exporter does not work on Windows")
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
    @ignore_warnings((DeprecationWarning, UserWarning, RuntimeWarning))
    @requires_torch("2.4")
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

    # static

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_custom_cpu_dummy_name1(self):
        self._hg_export_bench_cpu("custom", "101DummyNamed1")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_custom_cpu_dummy_name2(self):
        self._hg_export_bench_cpu("custom", "101DummyNamed2")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_custom_cpu_dummy_name_dict(self):
        self._hg_export_bench_cpu("custom", "101DummyNamedDict")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_onnx_dynamo_cpu_dummy_name1(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101DummyNamed1")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_onnx_dynamo_cpu_dummy_name2(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101DummyNamed2")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_onnx_dynamo_cpu_dummy_name_dict(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101DummyNamedDict")

    # dynamic

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_custom_cpu_dummy_name1_dynamic(self):
        self._hg_export_bench_cpu("custom", "101DummyNamed1", dynamic=True)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_custom_cpu_dummy_name2_dynamic(self):
        self._hg_export_bench_cpu("custom", "101DummyNamed2", dynamic=True)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_custom_cpu_dummy_name_dict_dynamic(self):
        self._hg_export_bench_cpu("custom", "101DummyNamedDict", dynamic=True)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_onnx_dynamo_cpu_dummy_name1_dynamic(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101DummyNamed1", dynamic=True)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_torch_script_cpu_dummy_name1_dynamic(self):
        self._hg_export_bench_cpu("torch_script", "101DummyNamed1", dynamic=True)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_onnx_dynamo_cpu_dummy_name2_dynamic(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101DummyNamed2", dynamic=True)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_onnx_dynamo_cpu_dummy_name_dict_dynamic(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101DummyNamedDict", dynamic=True)

    # list

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_custom_cpu_dummy_list(self):
        self._hg_export_bench_cpu("custom", "101DummyIList", dynamic=False)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.6")
    def test_huggingface_export_bench_custom_cpu_dummy_list_dynamic(self):
        self._hg_export_bench_cpu("custom", "101DummyIList", dynamic=True)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.7")
    def test_huggingface_export_bench_script_cpu_dummy_list(self):
        self._hg_export_bench_cpu("torch_script", "101DummyIList", dynamic=False)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_script_cpu_dummy_list_dynamic(self):
        self._hg_export_bench_cpu("torch_script", "101DummyIList", dynamic=True)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_dynamo_cpu_dummy_list(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101DummyIList", dynamic=False)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_dynamo_cpu_dummy_list_dynamic(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101DummyIList", dynamic=True)

    # int

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_custom_cpu_dummy_int(self):
        self._hg_export_bench_cpu("custom", "101DummyIInt", dynamic=False)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.6")
    def test_huggingface_export_bench_custom_cpu_dummy_int_dynamic(self):
        self._hg_export_bench_cpu("custom", "101DummyIInt", dynamic=True)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.6")
    def test_huggingface_export_bench_script_cpu_dummy_int(self):
        self._hg_export_bench_cpu("torch_script", "101DummyIInt", dynamic=False)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    @unittest.skip("should that work?")
    def test_huggingface_export_bench_script_cpu_dummy_int_dynamic(self):
        self._hg_export_bench_cpu("torch_script", "101DummyIInt", dynamic=True)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    @unittest.skip("investigate")
    def test_huggingface_export_bench_dynamo_cpu_dummy_int(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101DummyIInt", dynamic=False)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    @unittest.skip("investigate")
    def test_huggingface_export_bench_dynamo_cpu_dummy_int_dynamic(self):
        self._hg_export_bench_cpu("onnx_dynamo", "101DummyIInt", dynamic=True)

    # int, none

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_custom_cpu_dummy_none_int(self):
        self._hg_export_bench_cpu("custom", "101DummyNoneInt", dynamic=False)

    # int, list, none

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_custom_cpu_dummy_none_list_int(self):
        self._hg_export_bench_cpu("custom", "101DummyNoneListInt", dynamic=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
