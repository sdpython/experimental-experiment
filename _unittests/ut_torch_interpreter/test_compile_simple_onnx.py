import sys
import unittest
from typing import List
from onnx.reference import ReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_torch,
)
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.xbuilder import OptimizationOptions


def return_module_cls_pool():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, 5)
            # self.conv2 = nn.Conv2d(16, 16, 5)
            self.fc1 = nn.Linear(61504, 8)
            self.fc3 = nn.Linear(8, 10)

        def forward(self, x):
            c1 = self.conv1(x)
            f1 = F.relu(c1)
            t2 = F.max_pool2d(f1, (2, 2))
            # t3 = F.max_pool2d(F.relu(self.conv2(t2)), 2)
            xf = torch.flatten(t2, 1)
            xfr = F.relu(self.fc1(xf))
            y = self.fc3(xfr)
            return y

    input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)
    return MyModel(), input_tensor


class TestDynamoCompileOnnx(ExtTestCase):
    def setUp(self):
        import torch
        from torch.onnx import _OrtBackend as OrtBackend

        super().setUp()
        torch._dynamo.reset()
        OrtBackend.clear_cached_instances()

    def tearDown(self):
        import torch
        from torch.onnx import _OrtBackend as OrtBackend

        super().tearDown()
        torch._dynamo.reset()
        OrtBackend.clear_cached_instances()

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @requires_torch("2.2", "export fails")
    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_simple_dort_0(self):
        import torch

        model, input_tensor = return_module_cls_pool()
        expected = model(input_tensor)
        optimized_mod = torch.compile(model, backend="onnxrt")
        got = optimized_mod(input_tensor)
        self.assertEqual(expected.shape, got.shape)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqualArray(
            expected.detach().numpy(), got.detach().numpy(), atol=1e-5
        )

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(True, reason="export fails")
    def test_simple_dort_1(self):
        import torch
        from onnxruntime import InferenceSession

        def onnxscript_compiler(model: torch.fx.GraphModule, args: List[torch.Tensor]):
            export_output = torch.onnx.dynamo_export(model, *args)
            onx = export_output.model_proto
            sess = InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            names = [i.name for i in onx.graph.input]

            def run(*inputs, sess=sess, names=names):
                # not efficient
                xnp = [x.detach().numpy() for x in inputs]
                feeds = dict(zip(names, xnp))
                res = tuple(torch.Tensor(y) for y in sess.run(None, feeds))
                return res

            return run

        model, input_tensor = return_module_cls_pool()
        expected = model(input_tensor)
        optimized_mod = torch.compile(model, backend=onnxscript_compiler)
        got = optimized_mod(input_tensor)
        self.assertEqual(expected.shape, got.shape)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqualArray(expected.detach().numpy(), got.detach().numpy())

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @requires_torch("2.2", "export fails")
    def test_simple_dort_2_onnx(self):
        import torch

        def onnx_compiler(graph_module: torch.fx.GraphModule, args: List[torch.Tensor]):
            input_names = (
                ["input"] if len(args) == 1 else [f"input{i}" for i in range(len(args))]
            )

            onx = to_onnx(
                graph_module,
                tuple(args),
                input_names=input_names,
                options=OptimizationOptions(
                    remove_unused=True, constant_folding=True, verbose=4
                ),
                verbose=4,
            )
            try:
                sess = ReferenceEvaluator(onx, verbose=10)
            except Exception as e:
                from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

                raise AssertionError(
                    f"Unable to run onnx graph ({str(e)})\n{onnx_simple_text_plot(onx)}"
                ) from e
            names = [i.name for i in onx.graph.input]

            def run(*inputs, sess=sess, names=names):
                # not efficient
                xnp = [x.detach().numpy() for x in inputs]
                feeds = dict(zip(names, xnp))
                res = tuple(torch.Tensor(y) for y in sess.run(None, feeds))
                return res

            return run

        def lf():
            model, input_tensor = return_module_cls_pool()
            torch.export.export(model, (input_tensor,))
            expected = model(input_tensor)
            optimized_mod = torch.compile(model, backend=onnx_compiler)
            got = optimized_mod(input_tensor)
            return expected, got

        (expected, got), out, _ = self.capture(lf)
        self.assertIn("[GraphBuilder", out)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqualArray(
            expected.detach().numpy().ravel(), got.detach().numpy().ravel(), atol=1e-5
        )
        self.assertEqual(expected.shape, got.shape)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @requires_torch("2.2", "export fails")
    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_simple_dort_2_ort(self):
        import torch

        def onnx_compiler(graph_module: torch.fx.GraphModule, args: List[torch.Tensor]):
            from onnxruntime import InferenceSession

            input_names = (
                ["input"] if len(args) == 1 else [f"input{i}" for i in range(len(args))]
            )

            onx = to_onnx(
                graph_module,
                tuple(args),
                input_names=input_names,
                options=OptimizationOptions(
                    remove_unused=True,
                    constant_folding=True,
                ),
            )
            try:
                sess = InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
            except Exception as e:
                from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

                raise AssertionError(
                    f"Unable to run onnx graph ({str(e)})\n{onnx_simple_text_plot(onx)}"
                ) from e
            names = [i.name for i in onx.graph.input]

            def run(*inputs, sess=sess, names=names):
                # not efficient
                xnp = [x.detach().numpy() for x in inputs]
                feeds = dict(zip(names, xnp))
                res = tuple(torch.Tensor(y) for y in sess.run(None, feeds))
                return res

            return run

        model, input_tensor = return_module_cls_pool()
        torch.export.export(model, (input_tensor,))
        expected = model(input_tensor)
        optimized_mod = torch.compile(model, backend=onnx_compiler)
        got = optimized_mod(input_tensor)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqualArray(
            expected.detach().numpy().ravel(), got.detach().numpy().ravel(), atol=1e-5
        )
        self.assertEqual(expected.shape, got.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
