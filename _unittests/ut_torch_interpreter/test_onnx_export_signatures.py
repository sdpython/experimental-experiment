import inspect
import unittest
from typing import Any, Optional, Tuple
import onnx
import numpy as np
from experimental_experiment.ext_test_case import ExtTestCase, skipif_ci_windows
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
from experimental_experiment.helpers import get_onnx_signature


class TestOnnxExportSignatures(ExtTestCase):

    def _check_exporter(
        self,
        test_name: str,
        model: "torch.nn.Module",  # noqa: F821
        inputs: Tuple[Any, ...],
        expected_signature: Tuple[Tuple[str, Any], ...],
        exporter: str = ("custom", "custom-tracing"),
        decomposition: bool = False,
        verbose: int = 0,
        optimize: bool = False,
        dynamic_shapes: Optional[Any] = None,
        atol: float = 1e-5,
        target_opset: int = 18,
    ) -> str:
        if isinstance(exporter, tuple):
            for export in exporter:
                if verbose:
                    print(f"test_name={test_name!r}, exporter={export!r}")
                with self.subTest(exporter=export):
                    self._check_exporter(
                        test_name=test_name,
                        model=model,
                        inputs=inputs,
                        expected_signature=expected_signature,
                        exporter=export,
                        decomposition=decomposition,
                        verbose=verbose,
                        optimize=optimize,
                        dynamic_shapes=dynamic_shapes,
                    )
            return
        import torch

        expected = model(*inputs)

        filename = f"{test_name}_{exporter}_{'dec' if decomposition else ''}.onnx"
        if exporter == "script":
            torch.onnx.export(model, inputs, filename, opset_version=18)
        elif exporter == "dynamo":
            torch.onnx.export(
                model,
                inputs,
                filename,
                dynamo=True,
                dynamic_shapes=dynamic_shapes,
                target_opset=target_opset,
            )
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
                dynamic_shapes=dynamic_shapes,
                target_opset=target_opset,
            )

        # feeds
        onx = onnx.load(filename)
        onnx.checker.check_model(onx)
        names = [i.name for i in onx.graph.input]
        sig = get_onnx_signature(onx)
        self.assertEqual(expected_signature, sig)

        if len(names) == len(inputs):
            feeds = {}
            for name, xi in zip(names, inputs):
                if isinstance(xi, torch.Tensor):
                    feeds[name] = xi.detach().numpy()
                elif isinstance(xi, int):
                    feeds[name] = np.array([xi], dtype=np.int64)
                else:
                    raise AssertionError(f"not implemented names={name}, type={type(xi)}")
        else:
            raise AssertionError(f"not implemented names={names}, n_inputs={len(inputs)}")

        from onnxruntime import InferenceSession

        sess = InferenceSession(filename, providers=["CPUExecutionProvider"])
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=atol)

    @skipif_ci_windows("not working on windows")
    def test_signature_s1s_r(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x):
                return torch.sigmoid(self.linear(x)) - self.buff

        x = (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32)
        sig = (("x", onnx.TensorProto.FLOAT, (4, 3)),)
        sname = inspect.currentframe().f_code.co_name
        self._check_exporter(sname, Neuron(), (x,), sig)

    @skipif_ci_windows("not working on windows")
    def test_signature_s1d_r(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x):
                return torch.sigmoid(self.linear(x)) - self.buff

        x = (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32)
        sig = (("x", onnx.TensorProto.FLOAT, ("batch", 3)),)
        dyn = ({0: torch.export.Dim("batch")},)
        sname = inspect.currentframe().f_code.co_name
        self._check_exporter(sname, Neuron(), (x,), sig, dynamic_shapes=dyn)

    @skipif_ci_windows("not working on windows")
    def test_signature_s2d_r(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x, y):
                return torch.sigmoid(self.linear(x)) - self.buff + y

        inputs = (
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            (torch.arange(4) + 10).reshape((-1, 1)).to(torch.float32),
        )
        sig = (
            ("x", onnx.TensorProto.FLOAT, ("batch", 3)),
            ("y", onnx.TensorProto.FLOAT, ("batch", 1)),
        )
        dyn = ({0: torch.export.Dim("batch")}, {0: torch.export.Dim("batch")})
        sname = inspect.currentframe().f_code.co_name
        self._check_exporter(sname, Neuron(), inputs, sig, dynamic_shapes=dyn)

    @skipif_ci_windows("not working on windows")
    def test_signature_s1d_i_r_v1(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x, i: int = 2):
                return torch.sigmoid(self.linear(x)) - self.buff + x[:, i : i + 1]

        inputs = ((torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32), 1)
        sig = (
            ("x", onnx.TensorProto.FLOAT, ("batch", 3)),
            ("i", onnx.TensorProto.INT64, (1,)),
        )
        dyn = {
            "x": {0: torch.export.Dim("batch")},
            "i": None,
        }  # torch.export.Dim("ii", min=0, max=3)}
        sname = inspect.currentframe().f_code.co_name
        self._check_exporter(
            sname, Neuron(), inputs, sig, dynamic_shapes=dyn, exporter="custom-tracing"
        )

    @skipif_ci_windows("not working on windows")
    @unittest.skip("Something like [a:b, i] is not implemented yet.")
    def test_signature_s1d_i_r_v2(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x, i: int = 2):
                return torch.sigmoid(self.linear(x)) - self.buff + x[:, i]

        inputs = ((torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32), 1)
        sig = (
            ("x", onnx.TensorProto.FLOAT, ("batch", 3)),
            ("i", onnx.TensorProto.INT64, (1,)),
        )
        dyn = {
            "x": {0: torch.export.Dim("batch")},
            "i": None,
        }  # torch.export.Dim("ii", min=0, max=3)}
        sname = inspect.currentframe().f_code.co_name
        self._check_exporter(
            sname, Neuron(), inputs, sig, dynamic_shapes=dyn, exporter="custom-tracing"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
