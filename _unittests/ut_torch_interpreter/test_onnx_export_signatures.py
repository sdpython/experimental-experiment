import unittest
from typing import Any, List
import onnx
from experimental_experiment.ext_test_case import ExtTestCase, skipif_ci_windows
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


class TestOnnxExportSignatures(ExtTestCase):

    def _check_exporter(
        self,
        test_name: str,
        exporter: str,
        model: "torch.nn.Module",  # noqa: F821
        inputs: List[Any],
        decomposition: bool = False,
        verbose: int = 0,
        optimize: bool = False,
        atol=1e-5,
    ) -> str:
        if isinstance(exporter, tuple):
            for export in exporter:
                with self.subTest(exporter=exporter):
                    self._check_exporter(
                        test_name,
                        export,
                        model,
                        inputs,
                        decomposition=decomposition,
                        verbose=verbose,
                        optimize=optimize,
                    )
        import torch

        expected = model(*inputs)

        filename = f"{test_name}_{exporter}_{'dec' if decomposition else ''}.onnx"
        if exporter == "script":
            torch.onnx.export(model, inputs, filename, opset_version=18)
        elif exporter == "dynamo":
            torch.onnx.export(model, inputs, filename, dynamo=True)
        else:
            export_options = ExportOptions(
                decomposition_table="all" if decomposition else None,
                strict="-nostrict" not in exporter,
                tracing="-tracing" in exporter,
            )
            to_onnx(
                model,
                inputs,
                filename=filename,
                export_options=export_options,
                verbose=verbose,
                optimize=optimize,
            )

        # feeds
        onx = onnx.load(filename)
        onnx.checker.check_model(onx)
        names = [i.name for i in onx.graph.input]
        if len(names) == len(inputs):
            feeds = {}
            for name, xi in zip(names, inputs):
                if isinstance(xi, torch.Tensor):
                    feeds[name] = xi.detach().numpy()
                else:
                    raise AssertionError(f"not implemented names={name}, type={type(xi)}")
        else:
            raise AssertionError(f"not implemented names={names}, n_inputs={len(inputs)}")

        from onnxruntime import InferenceSession

        sess = InferenceSession(filename, providers=["CPUExecutionProvider"])
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=atol)

    @skipif_ci_windows("not working on windows")
    def test_signature_s1(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x):
                return torch.sigmoid(self.linear(x)) - self.buff

        x = (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32)
        self._check_exporter("test_signature_s1", ("custom", "custom-tracing"), Neuron(), (x,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
