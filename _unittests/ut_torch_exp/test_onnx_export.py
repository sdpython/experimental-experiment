import os
import sys
import unittest
import warnings
from onnx.reference import ReferenceEvaluator
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_exp.onnx_export import to_onnx


def return_module_cls_conv():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
        from torch import nn

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 128, 5)

        def forward(self, x):
            return self.conv1(x).abs()

    input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)
    return MyModel(), input_tensor


def return_module_cls_relu():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
        from torch import nn

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 128, 5)

        def forward(self, x):
            return torch.relu(self.conv1(x))

    input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)
    return MyModel(), input_tensor


def return_module_cls_pool():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 128, 5)
            self.conv2 = nn.Conv2d(128, 16, 5)
            self.fc1 = nn.Linear(13456, 1024)
            self.fc2 = nn.Linear(1024, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)
    return MyModel(), input_tensor


def export_utils(prefix, model, *args, remove_unused=False):
    import torch

    names = []
    name = f"{prefix}_script.onnx"
    if os.path.exists(name):
        os.remove(name)
    torch.onnx.export(model, *args, name, input_names=["input"])
    names.append(name)

    name = f"{prefix}_simple.onnx"
    if os.path.exists(name):
        os.remove(name)
    onx = to_onnx(
        model, tuple(args), input_names=["input"], remove_unused=remove_unused
    )
    with open(name, "wb") as f:
        f.write(onx.SerializeToString())
    names.append(name)
    return names


class TestMockExperimental(ExtTestCase):
    def check_model_ort(self, name):
        from onnxruntime import InferenceSession

        if isinstance(name, str):
            InferenceSession(name, providers=["CPUExecutionProvider"])
            return
        InferenceSession(name.SerializeToString(), providers=["CPUExecutionProvider"])

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    def test_simple_export_conv(self):
        model, input_tensor = return_module_cls_conv()
        names = export_utils("test_simple_export_conv", model, input_tensor)
        x = input_tensor.numpy()
        results = []
        for name in names:
            ref = ReferenceEvaluator(name)
            results.append(ref.run(None, {"input": x})[0])
            self.check_model_ort(name)
        self.assertEqualArray(results[0], results[1])

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    def test_simple_export_relu(self):
        model, input_tensor = return_module_cls_relu()
        names = export_utils("test_simple_export_relu", model, input_tensor)
        x = input_tensor.numpy()
        results = []
        for name in names:
            ref = ReferenceEvaluator(name)
            results.append(ref.run(None, {"input": x})[0])
            self.check_model_ort(name)
        self.assertEqualArray(results[0], results[1])

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    def test_simple_export_pool(self):
        from onnxruntime import InferenceSession

        model, input_tensor = return_module_cls_pool()
        names = export_utils("test_simple_export_pool", model, input_tensor)
        x = input_tensor.numpy()
        results = []
        for name in names:
            ref = InferenceSession(name, providers=["CPUExecutionProvider"])
            results.append(ref.run(None, {"input": x})[0])
        self.assertEqualArray(results[0], results[1])

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    def test_remove_unused_nodes(self):
        model, input_tensor = return_module_cls_pool()
        onx1 = to_onnx(model, (input_tensor,), input_names=["input"])
        onx2 = to_onnx(
            model, (input_tensor,), input_names=["input"], remove_unused=True
        )
        self.assertGreater(len(onx1.graph.node), len(onx2.graph.node))
        sub1 = [n for n in onx1.graph.node if n.op_type == "Sub"]
        sub2 = [n for n in onx2.graph.node if n.op_type == "Sub"]
        self.assertEqual(len(sub1), 2)
        self.assertEqual(len(sub2), 0)
        p1 = [n for n in onx1.graph.node if n.op_type == "MaxPool"]
        p2 = [n for n in onx2.graph.node if n.op_type == "MaxPool"]
        self.assertEqual(len(p1), 4)
        self.assertEqual(len(p2), 2)
        self.check_model_ort(onx2)
        from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

        print(onnx_simple_text_plot(onx1))
        print(onnx_simple_text_plot(onx2))

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    def test_simple_export_pool_unused(self):
        from onnxruntime import InferenceSession

        model, input_tensor = return_module_cls_pool()
        names = export_utils(
            "test_simple_export_pool_unused", model, input_tensor, remove_unused=True
        )
        x = input_tensor.numpy()
        results = []
        for name in names:
            ref = InferenceSession(name, providers=["CPUExecutionProvider"])
            results.append(ref.run(None, {"input": x})[0])
        self.assertEqualArray(results[0], results[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
