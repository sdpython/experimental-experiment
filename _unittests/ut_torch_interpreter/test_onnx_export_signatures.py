import inspect
import unittest
from typing import Any, Dict, List, Optional, Tuple
import onnx
import numpy as np
from experimental_experiment.ext_test_case import ExtTestCase, skipif_ci_windows
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
from experimental_experiment.helpers import get_onnx_signature, string_type


class TestOnnxExportSignatures(ExtTestCase):

    def _flatten_inputs(self, inputs):
        import torch

        flattened_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                flattened_inputs.append(inp)
            elif isinstance(inp, list):
                assert all(
                    isinstance(t, torch.Tensor) for t in inp
                ), f"flatten not implemented for nested lists {string_type(inputs)}"
                flattened_inputs.extend(inp)
            else:
                raise AssertionError(f"Not implemented for type {type(inp)}")
        return flattened_inputs

    def _make_feeds(
        self,
        names: List[str],
        inputs: Tuple[Any, ...],
        tracing: bool,
        exporter: str = "",
        flatten_inputs: bool = False,
    ):
        import torch

        new_inputs = self._flatten_inputs(inputs) if flatten_inputs else inputs

        if len(names) == len(new_inputs):
            feeds = {}
            for name, xi in zip(names, new_inputs):
                if isinstance(xi, torch.Tensor):
                    feeds[name] = xi.detach().numpy()
                elif tracing:
                    if isinstance(xi, int):
                        feeds[name] = np.array([xi], dtype=np.int64)
                    elif isinstance(xi, list):
                        feeds[name] = [xii.detach().numpy() for xii in xi]
                    else:
                        raise AssertionError(f"not implemented names={name}, type={type(xi)}")
                else:
                    raise AssertionError(
                        f"not implemented for exporter={exporter!r}, "
                        f"names={name}, type={type(xi)}"
                    )
        else:
            raise AssertionError(f"not implemented names={names}, n_inputs={len(inputs)}")
        return feeds

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
        others: Optional[Tuple[Any, ...]] = None,
        flatten_inputs: bool = False,
        feeds: Optional[Dict[str, Any]] = None,
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
                        flatten_inputs=flatten_inputs,
                        others=others,
                        feeds=feeds,
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

        # model
        onx = onnx.load(filename)
        onnx.checker.check_model(onx)
        names = [i.name for i in onx.graph.input]
        sig = get_onnx_signature(onx)
        if expected_signature != "NOCHECK":
            self.assertEqual(expected_signature, sig)

        # feeds
        tracing = "-tracing" in exporter
        if feeds is None:
            feeds = self._make_feeds(
                names, inputs, tracing, exporter=exporter, flatten_inputs=flatten_inputs
            )

        from onnxruntime import InferenceSession

        sess = InferenceSession(filename, providers=["CPUExecutionProvider"])
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=atol)

        if others:
            expected = model(*others)
            feeds = self._make_feeds(
                names, others, tracing, exporter=exporter, flatten_inputs=flatten_inputs
            )
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
        x2 = (torch.arange(8 * 3) + 10).reshape((-1, 3)).to(torch.float32)
        sig = (("x", onnx.TensorProto.FLOAT, ("batch", 3)),)
        dyn = ({0: torch.export.Dim("batch")},)
        sname = inspect.currentframe().f_code.co_name
        self._check_exporter(sname, Neuron(), (x,), sig, dynamic_shapes=dyn, others=(x2,))

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
        inputs2 = (
            (torch.arange(8 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            (torch.arange(8) + 10).reshape((-1, 1)).to(torch.float32),
        )
        sig = (
            ("x", onnx.TensorProto.FLOAT, ("batch", 3)),
            ("y", onnx.TensorProto.FLOAT, ("batch", 1)),
        )
        dyn = ({0: torch.export.Dim("batch")}, {0: torch.export.Dim("batch")})
        sname = inspect.currentframe().f_code.co_name
        self._check_exporter(sname, Neuron(), inputs, sig, dynamic_shapes=dyn, others=inputs2)

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
        inputs2 = ((torch.arange(8 * 3) + 10).reshape((-1, 3)).to(torch.float32), 2)
        sig = (
            ("x", onnx.TensorProto.FLOAT, ("batch", 3)),
            ("i", onnx.TensorProto.INT64, (1,)),
        )
        dyn = {
            "x": {0: torch.export.Dim("batch")},
            "i": None,  # torch.export.Dim("ii", min=0, max=3)}
        }
        sname = inspect.currentframe().f_code.co_name
        self._check_exporter(
            sname,
            Neuron(),
            inputs,
            sig,
            dynamic_shapes=dyn,
            exporter="custom-tracing",
            others=inputs2,
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
            "i": None,  # torch.export.Dim("ii", min=0, max=3)}
        }
        sname = inspect.currentframe().f_code.co_name
        self._check_exporter(
            sname, Neuron(), inputs, sig, dynamic_shapes=dyn, exporter="custom-tracing"
        )

    @skipif_ci_windows("not working on windows")
    def test_signature_s1d_ls_r_custom(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x, lx):
                return (
                    torch.sigmoid(self.linear(x))
                    - self.buff
                    + lx[0] * lx[1].sum(axis=1, keepdim=True)
                )

        inputs = (
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(4) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(4 * 2) + 10).reshape((-1, 2)).to(torch.float32),
            ],
        )
        inputs2 = (
            (torch.arange(8 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(8) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(8 * 2) + 10).reshape((-1, 2)).to(torch.float32),
            ],
        )
        dyn = {
            "x": {0: torch.export.Dim("batch")},
            "lx": [{0: torch.export.Dim("batch")}, {0: torch.export.Dim("batch")}],
        }
        sname = inspect.currentframe().f_code.co_name
        sig_custom = (("x", 1, ("batch", 3)), ("lx_0", 1, ("s1", 1)), ("lx_1", 1, ("s2", 2)))
        self._check_exporter(
            sname,
            Neuron(),
            inputs,
            sig_custom,
            dynamic_shapes=dyn,
            exporter="custom",
            flatten_inputs=True,
            others=inputs2,
        )

    @skipif_ci_windows("not working on windows")
    def test_signature_s1d_ls_r_tracing(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x, lx):
                return (
                    torch.sigmoid(self.linear(x))
                    - self.buff
                    + lx[0] * lx[1].sum(axis=1, keepdim=True)
                )

        inputs = (
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(4) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(4 * 2) + 10).reshape((-1, 2)).to(torch.float32),
            ],
        )
        inputs2 = (
            (torch.arange(8 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(8) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(8 * 2) + 10).reshape((-1, 2)).to(torch.float32),
            ],
        )
        dyn = {
            "x": {0: torch.export.Dim("batch")},
            "lx": [{0: torch.export.Dim("batch")}, {0: torch.export.Dim("batch")}],
        }
        sname = inspect.currentframe().f_code.co_name
        sig_tracing = (("x", 1, ("batch", 3)), ("lx", [("lx", 1, ("batch", 1))]))
        self._check_exporter(
            sname,
            Neuron(),
            inputs,
            sig_tracing,
            dynamic_shapes=dyn,
            exporter="custom-tracing",
            others=inputs2,
        )

    @skipif_ci_windows("not working on windows")
    def test_signature_index_s_r(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x, y):
                t = torch.sigmoid(self.linear(x)) + x
                return t[:, : y.shape[1]]

        inputs = (
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            (torch.arange(4 * 2) + 10).reshape((-1, 2)).to(torch.float32),
        )
        sname = inspect.currentframe().f_code.co_name
        sig_tracing = (("x", 1, (4, 3)), ("y", 1, (4, 2)))
        self._check_exporter(sname, Neuron(), inputs, sig_tracing)

    @skipif_ci_windows("not working on windows")
    def test_signature_index_d_r(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x, y):
                t = torch.sigmoid(self.linear(x)) + x
                return t[:, : y.shape[1]]

        inputs = (
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            (torch.arange(4 * 2) + 10).reshape((-1, 2)).to(torch.float32),
        )
        inputs2 = (
            (torch.arange(8 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            (torch.arange(8 * 1) + 10).reshape((-1, 1)).to(torch.float32),
        )
        dim = torch.export.Dim("batch", min=0, max=1024)
        dyn = {
            "x": {0: dim},
            "y": {0: dim, 1: torch.export.Dim("length", min=0, max=2)},
        }
        sname = inspect.currentframe().f_code.co_name
        sig_tracing = (("x", 1, ("batch", 3)), ("y", 1, ("batch", "length")))
        self._check_exporter(
            sname, Neuron(), inputs, sig_tracing, dynamic_shapes=dyn, others=inputs2
        )

    @skipif_ci_windows("not working on windows")
    def test_signature_llm_s_tracing(self):
        from experimental_experiment.torch_test_helper import dummy_llm

        if False:
            for cls_name in ["AttentionBlock", "MultiAttentionBlock", "DecoderLayer"]:
                model, inputs = dummy_llm(cls_name)
                sname = inspect.currentframe().f_code.co_name
                self._check_exporter(
                    f"{sname}_{cls_name}",
                    model,
                    inputs,
                    expected_signature="NOCHECK",
                    optimize=True,
                    exporter="custom",
                )

        for cls_name in ["AttentionBlock", "MultiAttentionBlock", "DecoderLayer"]:
            model, inputs = dummy_llm(cls_name)
            sname = inspect.currentframe().f_code.co_name
            self._check_exporter(
                f"{sname}_{cls_name}",
                model,
                inputs,
                expected_signature="NOCHECK",
                optimize=False,
                exporter="custom-tracing",
            )

    @skipif_ci_windows("not working on windows")
    def test_signature_llm_s_r(self):
        from experimental_experiment.torch_test_helper import dummy_llm

        model, inputs = dummy_llm()
        sname = inspect.currentframe().f_code.co_name
        self._check_exporter(
            sname,
            model,
            inputs,
            expected_signature=(("input_ids", 7, (1, 30)),),
        )

    @skipif_ci_windows("not working on windows")
    def test_signature_llm_d_r(self):
        import torch
        from experimental_experiment.torch_test_helper import dummy_llm

        model, inputs, dyn = dummy_llm(dynamic_shapes=True)
        others = (torch.randint(0, 1024, (4, 50)).to(torch.int64),)
        sname = inspect.currentframe().f_code.co_name
        self._check_exporter(
            sname,
            model,
            inputs,
            dynamic_shapes=dyn,
            expected_signature=(("input_ids", 7, ("batch", "length")),),
            others=others,
        )

    @skipif_ci_windows("not working on windows")
    def test_signature_simple_none(self):
        import torch

        class Neuron(torch.nn.Module):
            def forward(self, x=None, y=None, z=None):
                if y is None:
                    return x * z
                return x + y - z

        inputs = (
            (torch.arange(4 * 3) + 1).reshape((-1, 3)).to(torch.float32),
            None,
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
        )
        dyn = {}
        sname = inspect.currentframe().f_code.co_name
        sig_tracing = "NOCHECK"
        self._check_exporter(
            sname,
            Neuron(),
            inputs,
            sig_tracing,
            dynamic_shapes=dyn,
            others=None,
            feeds=dict(x=inputs[0].numpy(), z=inputs[2].numpy()),
            exporter="custom-nostrict",
        )

    @skipif_ci_windows("not working on windows")
    def test_signature_list_none(self):
        import torch

        class Neuron(torch.nn.Module):
            def forward(self, x=None, y=None, z=None, w=None):
                return x * (z[0] + z[1]) + w

        inputs = (
            (torch.arange(4 * 3) + 1).reshape((-1, 3)).to(torch.float32),
            None,
            [
                (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
                (torch.arange(4 * 3) + 5).reshape((-1, 3)).to(torch.float32),
            ],
            (torch.arange(4 * 3) + 0.5).reshape((-1, 3)).to(torch.float32),
        )
        dyn = {}
        sname = inspect.currentframe().f_code.co_name
        sig_tracing = "NOCHECK"
        self._check_exporter(
            sname,
            Neuron(),
            inputs,
            sig_tracing,
            dynamic_shapes=dyn,
            others=None,
            feeds=dict(
                x=inputs[0].numpy(),
                z_0=inputs[2][0].numpy(),
                z_1=inputs[2][1].numpy(),
                w=inputs[3].numpy(),
            ),
            exporter="custom-nostrict",
        )

    @skipif_ci_windows("not working on windows")
    def test_signature_dc_none(self):
        import torch
        import transformers
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )

        class Neuron(torch.nn.Module):
            def forward(self, x=None, y=None, z=None, w=None, ww=None):
                return x * (z.key_cache[0] + z.value_cache[0]) + ww

        cache = transformers.cache_utils.DynamicCache(1)
        cache.update(
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            (torch.arange(4 * 3) + 5).reshape((-1, 3)).to(torch.float32),
            0,
        )

        inputs = (
            (torch.arange(4 * 3) + 1).reshape((-1, 3)).to(torch.float32),
            None,
            cache,
            None,
            (torch.arange(4 * 3) + 0.5).reshape((-1, 3)).to(torch.float32),
        )
        dyn = {}
        sname = inspect.currentframe().f_code.co_name
        sig_tracing = "NOCHECK"
        with bypass_export_some_errors():
            self._check_exporter(
                sname,
                Neuron(),
                inputs,
                sig_tracing,
                dynamic_shapes=dyn,
                others=None,
                feeds=dict(
                    x=inputs[0].numpy(),
                    z_key_cache_0=inputs[2].key_cache[0].numpy(),
                    z_value_cache_0=inputs[2].value_cache[0].numpy(),
                    ww=inputs[4].numpy(),
                ),
                exporter="custom-nostrict",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
