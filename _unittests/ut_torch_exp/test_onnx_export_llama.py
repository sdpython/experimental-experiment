import contextlib
import io
import sys
import unittest
import warnings
import packaging.version as pv
import onnxruntime  # noqa: F401
from onnx.reference import ReferenceEvaluator
from experimental_experiment.ext_test_case import ExtTestCase, ignore_warnings
from experimental_experiment.torch_exp.onnx_export import to_onnx
from experimental_experiment.torch_helper.llama_helper import (
    get_llama_attention,
    get_llama_decoder,
    get_llama_model,
)


def torch_version():
    import torch

    return ".".join(torch.__version__.split(".")[:2])


def export_script(filename, model, *args):
    import torch

    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, args, filename, input_names=["input"])


def export_dynamo(filename, model, *args):
    import torch

    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            export_output = torch.onnx.dynamo_export(model, *args)
            model = export_output.model_proto
            with open(filename, "wb") as f:
                f.write(model.SerializeToString())


def export_utils(
    prefix,
    model,
    *args,
    remove_unused=False,
    constant_folding=True,
    verbose=0,
    return_builder=False,
    dynamo=True,
):
    export_script(f"{prefix}.onnx", model, *args)
    if dynamo:
        export_dynamo(f"{prefix}.dynamo.onnx", model, *args)
    onx = to_onnx(
        model,
        tuple(args),
        input_names=[f"input{i}" for i in range(len(args))],
        remove_unused=remove_unused,
        constant_folding=constant_folding,
        verbose=verbose,
        return_builder=return_builder,
    )
    with open(f"{prefix}.custom.onnx", "wb") as f:
        f.write((onx[0] if return_builder else onx).SerializeToString())
    return onx


class TestOnnxExportLlama(ExtTestCase):
    def check_model_ort(self, onx):
        from onnxruntime import InferenceSession

        if isinstance(onx, str):
            try:
                InferenceSession(onx, providers=["CPUExecutionProvider"])
            except Exception as e:
                import onnx
                from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

                raise AssertionError(
                    f"onnxruntime cannot load the model "
                    f"due to {e}\n{onnx_simple_text_plot(onnx.load(onx))}"
                )
            return
        try:
            InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

            raise AssertionError(
                f"onnxruntime cannot load the model"
                f"due to {e}\n{onnx_simple_text_plot(onx)}"
            )

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings(DeprecationWarning)
    def test_llama_attention(self):
        model, input_tensors = get_llama_attention(input_dims=[(2, 1024)])
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)
        onx, builder = export_utils(
            "test_llama_attention", model, *input_tensors, return_builder=True
        )
        with open("test_llama_attention.custom.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        xp = [x.numpy() for x in input_tensors]
        feeds = {f"input{i}": x for i, x in enumerate(xp)}
        ref = ReferenceEvaluator(onx)
        try:
            results = ref.run(None, feeds)
        except Exception as e:
            print("--------------------")
            try:
                ReferenceEvaluator(onx, verbose=10).run(None, feeds)
            except Exception:
                pass
            print("--------------------")
            msg = "\n".join(
                [
                    f"input shapes: {[i.shape for i in input_tensors]}",
                    builder.get_debug_msg(),
                ]
            )
            raise AssertionError(msg) from e
        self.assertEqualArray(expected.detach().numpy(), results[0], atol=1e-5)
        self.check_model_ort(onx)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings(DeprecationWarning)
    def test_llama_decoder(self):
        model, input_tensors = get_llama_decoder()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)
        onx = export_utils("test_llama_decoder", model, *input_tensors)
        xp = [x.numpy() for x in input_tensors]
        feeds = {f"input{i}": x for i, x in enumerate(xp)}
        ref = ReferenceEvaluator(onx)
        results = ref.run(None, feeds)
        self.assertEqualArray(expected.detach().numpy(), results[0], atol=1e-5)
        self.check_model_ort(onx)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(pv.Version(torch_version()) < pv.Version("2.3"), reason="bug")
    @ignore_warnings(DeprecationWarning)
    def test_llama_model(self):
        model, input_tensors = get_llama_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)
        onx = export_utils("test_llama_model", model, *input_tensors, dynamo=False)
        xp = [x.numpy() for x in input_tensors]
        feeds = {f"input{i}": x for i, x in enumerate(xp)}
        ref = ReferenceEvaluator(onx)
        results = ref.run(None, feeds)
        self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)
        with open("test_llama_model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        self.check_model_ort(onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
