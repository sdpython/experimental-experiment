import sys
import unittest
from typing import List
import packaging.version as pv
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_exp.onnx_export import to_onnx


def torch_recent_enough():
    import torch

    return pv.Version(".".join(torch.__version__.split(".")[:2])) >= pv.Version("2.2")


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
    @unittest.skipIf(not torch_recent_enough(), reason="export fails")
    def test_simple_dort_0(self):
        import torch

        model, input_tensor = return_module_cls_pool()
        expected = model(input_tensor)
        optimized_mod = torch.compile(model, backend="onnxrt")
        got = optimized_mod(input_tensor)
        print(expected)
        print(got)
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
                if len(res) == 1:
                    return res[0]
                return res

            return run

        model, input_tensor = return_module_cls_pool()
        expected = model(input_tensor)
        optimized_mod = torch.compile(model, backend=onnxscript_compiler)
        got = optimized_mod(input_tensor)
        print(expected)
        print(got)
        self.assertEqual(expected.shape, got.shape)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqualArray(expected.detach().numpy(), got.detach().numpy())

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not torch_recent_enough(), reason="export fails")
    def test_simple_dort_2(self):
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
                remove_unused=True,
                constant_folding=True,
            )
            sess = InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            names = [i.name for i in onx.graph.input]

            def run(*inputs, sess=sess, names=names):
                # not efficient
                xnp = [x.detach().numpy() for x in inputs]
                feeds = dict(zip(names, xnp))
                res = tuple(torch.Tensor(y) for y in sess.run(None, feeds))
                if len(res) == 1:
                    return res[0]
                return res

            return run

        model, input_tensor = return_module_cls_pool()
        torch.export.export(model, (input_tensor,))
        expected = model(input_tensor)
        optimized_mod = torch.compile(model, backend=onnx_compiler)
        got = optimized_mod(input_tensor)
        self.assertEqual(expected.shape, got.shape)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqualArray(expected.detach().numpy(), got.detach().numpy())


if __name__ == "__main__":
    unittest.main(verbosity=2)
