import sys
import unittest
from experimental_experiment.ext_test_case import ExtTestCase, ignore_warnings


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
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc3(x)
            return x

    input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)
    return MyModel(), input_tensor


class TestDynamoOnnxRtBackend(ExtTestCase):
    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_onnxrt_tutorial_0a(self):
        from onnxruntime import InferenceSession
        import torch
        import torch.onnx
        import torch._dynamo

        torch._dynamo.reset()

        if not torch.onnx.is_onnxrt_backend_supported():
            return

        model, input_tensor = return_module_cls_pool()
        expected = model(input_tensor)
        optimized_mod = torch.compile(model)
        got = optimized_mod(input_tensor)
        self.assertEqual(expected.shape, got.shape)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqualArray(
            expected.detach().numpy(), got.detach().numpy(), atol=1e-5
        )

        export = torch.onnx.dynamo_export(model, input_tensor)
        onx = export.model_proto
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        with open("dummy_baseline_a.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        name = onx.graph.input[0].name
        got = sess.run(None, {name: input_tensor.detach().numpy()})[0]
        self.assertEqualArray(expected.detach().numpy(), got, atol=1e-5)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_onnxrt_tutorial_0b(self):
        from onnxruntime import InferenceSession
        import torch
        import torch.onnx
        import torch._dynamo

        torch._dynamo.reset()

        if not torch.onnx.is_onnxrt_backend_supported():
            return

        model, input_tensor = return_module_cls_pool()

        def f(x):
            return model(x)

        expected = f(input_tensor)
        optimized_mod = torch.compile(f)
        got = optimized_mod(input_tensor)
        self.assertEqual(expected.shape, got.shape)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqualArray(
            expected.detach().numpy(), got.detach().numpy(), atol=1e-5
        )

        export = torch.onnx.dynamo_export(f, input_tensor)
        onx = export.model_proto
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        with open("dummy_baseline_b.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        name = onx.graph.input[0].name
        got = sess.run(None, {name: input_tensor.detach().numpy()})[0]
        self.assertEqualArray(expected.detach().numpy(), got, atol=1e-5)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_onnxrt_tutorial_1(self):
        import torch
        import torch.onnx
        import torch._dynamo

        torch._dynamo.reset()

        if not torch.onnx.is_onnxrt_backend_supported():
            return

        model, input_tensor = return_module_cls_pool()
        expected = model(input_tensor)

        @torch.compile(backend="onnxrt")
        def f(x):
            return model(x)

        got = f(input_tensor)
        self.assertEqual(expected.shape, got.shape)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqualArray(expected.detach().numpy(), got.detach().numpy())

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_onnxrt_tutorial_2(self):
        import torch
        import torch.onnx
        import torch._dynamo

        torch._dynamo.reset()

        if not torch.onnx.is_onnxrt_backend_supported():
            return

        model, input_tensor = return_module_cls_pool()
        expected = model(input_tensor)
        self.assertNotEmpty(torch.compile(model))
        optimized_mod = torch.compile(model, backend="onnxrt")
        got = optimized_mod(input_tensor)
        self.assertEqual(expected.shape, got.shape)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqualArray(expected.detach().numpy(), got.detach().numpy())

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_simple_dort(self):
        import torch
        import torch.onnx
        import torch._dynamo

        torch._dynamo.reset()
        print(torch.compiler.list_backends())

        if not torch.onnx.is_onnxrt_backend_supported():
            return

        from torch.onnx._internal.onnxruntime import torch_compile_backend

        model, input_tensor = return_module_cls_pool()
        expected = model(input_tensor)
        optimized_mod = torch.compile(model, backend=torch_compile_backend)
        got = optimized_mod(input_tensor)
        print(expected)
        print(got)
        self.assertEqual(expected.shape, got.shape)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqualArray(expected.detach().numpy(), got.detach().numpy())


if __name__ == "__main__":
    unittest.main(verbosity=2)
