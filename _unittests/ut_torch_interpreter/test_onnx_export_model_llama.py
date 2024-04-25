import contextlib
import io
import sys
import unittest
import warnings
import onnxruntime  # noqa: F401
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_torch,
)
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.torch_models.llama_helper import (
    get_llama_attention,
    get_llama_decoder,
    get_llama_model,
)


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
    use_dynamo=True,
):
    export_script(f"{prefix}.onnx", model, *args)
    if dynamo:
        export_dynamo(f"{prefix}.dynamo.onnx", model, *args)
    onx = to_onnx(
        model,
        tuple(args),
        input_names=[f"input{i}" for i in range(len(args))],
        options=OptimizationOptions(
            remove_unused=remove_unused,
            constant_folding=constant_folding,
            verbose=verbose,
            patterns=None,
        ),
        verbose=verbose,
        return_builder=return_builder,
        api_two=use_dynamo,
    )
    with open(f"{prefix}.custom.onnx", "wb") as f:
        f.write((onx[0] if return_builder else onx).SerializeToString())
    return onx


class TestOnnxExportLlama(ExtTestCase):
    def setUp(self):
        import torch

        torch._dynamo.reset()

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
    @requires_torch("2.3", "dynamic_shapes are not well carries")
    def test_llama_attention(self):
        model, input_tensors = get_llama_attention(input_dims=[(2, 1024)])
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)
        # torch._dynamo.reset()
        # torch._dynamo.export(model, tracing_mode="symbolic")(*input_tensors)
        # torch.export.export(model, input_tensors)
        onx, builder = export_utils(
            "test_llama_attention", model, *input_tensors, return_builder=True
        )
        # with open("test_llama_attention.custom.onnx", "wb") as f:
        #     f.write(onx.SerializeToString())
        xp = [x.numpy() for x in input_tensors]
        feeds = {f"input{i}": x for i, x in enumerate(xp)}
        ref = ExtendedReferenceEvaluator(onx)
        try:
            results = ref.run(None, feeds)
        except Exception as e:
            print("--------------------")
            try:
                ExtendedReferenceEvaluator(onx, verbose=10).run(None, feeds)
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
    @requires_torch("2.3", "dynamic_shapes are not well carries")
    def test_llama_decoder(self):
        import torch

        with torch.no_grad():
            model, input_tensors = get_llama_decoder()
            input_tensors = input_tensors[0]
            expected = model(*input_tensors)
            onx = export_utils("test_llama_decoder", model, *input_tensors)
            xp = [x.numpy() for x in input_tensors]
            feeds = {f"input{i}": x for i, x in enumerate(xp)}
            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, feeds)
            self.assertEqualArray(expected.detach().numpy(), results[0], atol=1e-5)
            self.check_model_ort(onx)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @requires_torch("2.3", "bug")
    @ignore_warnings(DeprecationWarning)
    @unittest.skipIf(True, reason="torch._dynamo.export does not work")
    def test_llama_model_dynamo_true(self):
        import torch

        with torch.no_grad():
            model, input_tensors = get_llama_model()
            input_tensors = input_tensors[0]
            expected = model(*input_tensors)
            onx = export_utils(
                "test_llama_model_dynamo_true",
                model,
                *input_tensors,
                dynamo=False,
                use_dynamo=True,
                remove_unused=True,
                verbose=0,
            )
            xp = [x.numpy() for x in input_tensors]
            feeds = {f"input{i}": x for i, x in enumerate(xp)}
            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, feeds)
            self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)
            # with open("test_llama_model.onnx", "wb") as f:
            # s    f.write(onx.SerializeToString())
            self.check_model_ort(onx)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @requires_torch("2.4", "Unable to find input 'x' in known results")
    @ignore_warnings(DeprecationWarning)
    def test_llama_model_dynamo_false(self):
        import torch

        with torch.no_grad():
            model, input_tensors = get_llama_model()
            input_tensors = input_tensors[0]
            expected = model(*input_tensors)
            onx = export_utils(
                "test_llama_model_dynamo_false",
                model,
                *input_tensors,
                dynamo=False,
                use_dynamo=False,
                remove_unused=True,
                verbose=0,
            )
            xp = [x.numpy() for x in input_tensors]
            feeds = {f"input{i}": x for i, x in enumerate(xp)}
            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, feeds)
            self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)
            # with open("test_llama_model.onnx", "wb") as f:
            #     f.write(onx.SerializeToString())
            self.check_model_ort(onx)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @requires_torch("2.3", "bug")
    @ignore_warnings(DeprecationWarning)
    def test_nn_dynamo_true(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x, y):
                return torch.sigmoid(self.linear(x + y))

        with torch.no_grad():
            model, input_tensors = Neuron(3, 1), [(torch.rand(2, 3), torch.rand(2, 3))]
            input_tensors = input_tensors[0]
            expected = model(*input_tensors)
            onx = export_utils(
                "test_nn_dynamo_true",
                model,
                *input_tensors,
                dynamo=False,
                use_dynamo=True,
                remove_unused=True,
                verbose=0,
            )
            xp = [x.numpy() for x in input_tensors]
            feeds = {f"input{i}": x for i, x in enumerate(xp)}
            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, feeds)
            self.assertEqualArray(expected.detach().numpy(), results[0], atol=1e-5)
            # with open("test_llama_model.onnx", "wb") as f:
            # s    f.write(onx.SerializeToString())
            self.check_model_ort(onx)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @requires_torch("2.4", "Unable to find input 'x' in known results")
    @ignore_warnings(DeprecationWarning)
    def test_nn_dynamo_false(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x, y):
                return torch.sigmoid(self.linear(x + y))

        with torch.no_grad():
            model, input_tensors = Neuron(3, 1), [(torch.rand(2, 3), torch.rand(2, 3))]
            input_tensors = input_tensors[0]
            expected = model(*input_tensors)
            onx = export_utils(
                "test_nn_dynamo_false",
                model,
                *input_tensors,
                dynamo=False,
                use_dynamo=False,
                remove_unused=True,
                verbose=0,
            )
            xp = [x.numpy() for x in input_tensors]
            feeds = {f"input{i}": x for i, x in enumerate(xp)}
            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, feeds)
            self.assertEqualArray(expected.detach().numpy(), results[0], atol=1e-5)
            # with open("test_llama_model.onnx", "wb") as f:
            # s    f.write(onx.SerializeToString())
            self.check_model_ort(onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
