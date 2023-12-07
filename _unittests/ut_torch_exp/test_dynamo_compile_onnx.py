import sys
import unittest
from typing import List
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_exp.onnx_export import to_onnx


def return_module_cls_pool():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class MyModel(nn.Module):
        def __init__(self, n_lin_layers=2):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, 5)
            self.conv2 = nn.Conv2d(16, 16, 5)
            self.fc1 = nn.Linear(13456, 8)
            self.fc3 = nn.Linear(8, 10)

        def forward(self, x):
            c1 = self.conv1(x)
            f1 = F.relu(c1)
            x = F.max_pool2d(f1, (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc3(x)
            return x

    input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)
    return MyModel(), input_tensor


class TestDynamoCompileOnnx(ExtTestCase):
    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skip("fails")
    def test_simple_dort_1(self):
        import torch
        from onnxruntime import InferenceSession

        def onnxscript_compiler(model: torch.fx.GraphModule, args: List[torch.Tensor]):
            export_output = torch.onnx.dynamo_export(model, *args)
            onx = export_output.to_model_proto()
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
    def test_simple_dort_2(self):
        import torch

        def onnx_compiler(model: torch.fx.GraphModule, args: List[torch.Tensor]):
            from onnxruntime import InferenceSession

            input_names = (
                ["input"] if len(args) == 1 else [f"input{i}" for i in range(len(args))]
            )

            onx = to_onnx(
                model,
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
        expected = model(input_tensor)
        optimized_mod = torch.compile(model, backend=onnx_compiler)
        got = optimized_mod(input_tensor)
        print(expected)
        print(got)
        self.assertEqual(expected.shape, got.shape)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqualArray(expected.detach().numpy(), got.detach().numpy())


if __name__ == "__main__":
    unittest.main(verbosity=2)
