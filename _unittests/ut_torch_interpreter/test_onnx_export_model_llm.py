import contextlib
import io
import unittest
import warnings
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_torch,
    skipif_ci_windows,
)
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.helpers import pretty_onnx


def export_script(filename, model, *args):
    import torch

    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, args, filename, input_names=["input"], dynamo=False)


def export_utils(
    prefix,
    model,
    *args,
    remove_unused=False,
    constant_folding=True,
    verbose=0,
    return_builder=False,
    export_options=None,
    dtype: str = "",
):
    if dtype:
        import torch

        dt = getattr(torch, dtype)
        model = model.to(dt)
        args = tuple((i.to(dt) if i.dtype in {torch.float32, torch.float16} else i) for i in args)

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
        export_options=export_options,
    )
    with open(f"{prefix}.custom.onnx", "wb") as f:
        f.write((onx[0] if return_builder else onx).SerializeToString())
    if return_builder:
        with open(f"{prefix}.custom.onnx.txt", "w") as f:
            f.write(onx[1].get_debug_msg())

    return onx


class TestOnnxExportLlama(ExtTestCase):
    def setUp(self):
        import torch

        torch._dynamo.reset()

    def check_model_ort(self, onx, providers=None):
        from onnxruntime import InferenceSession

        if providers is None:
            providers = ["CPUExecutionProvider"]

        if isinstance(onx, str):
            try:
                InferenceSession(onx, providers=providers)
            except Exception as e:
                import onnx

                raise AssertionError(  # noqa: B904
                    f"onnxruntime cannot load the model "
                    f"due to {e}\n{pretty_onnx(onnx.load(onx))}"
                )
            return
        try:
            InferenceSession(onx.SerializeToString(), providers=providers)
        except Exception as e:
            raise AssertionError(  # noqa: B904
                f"onnxruntime cannot load the model due to {e}\n{pretty_onnx(onx)}"
            )

    @skipif_ci_windows("not supported yet on Windows")
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
