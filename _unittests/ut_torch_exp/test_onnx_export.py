import unittest
import warnings
from onnx.reference import ReferenceEvaluator
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_exp.onnx_export import to_onnx


def return_module_cls():
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


def return_module_cls_full():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


def export_utils(prefix, model, *args):
    import torch

    names = []
    name = f"{prefix}_script.onnx"
    torch.onnx.export(model, *args, name, input_names=["input"])
    names.append(name)

    name = f"{prefix}_simple.onnx"
    onx = to_onnx(model, tuple(args), input_names=["input"])
    with open(name, "wb") as f:
        f.write(onx.SerializeToString())
    names.append(name)
    return names


class TestMockExperimental(ExtTestCase):
    def test_simple_export(self):
        model, input_tensor = return_module_cls()
        names = export_utils("test_simple_export", model, input_tensor)
        x = input_tensor.numpy()
        results = []
        for name in names:
            ref = ReferenceEvaluator(name)
            results.append(ref.run(None, {"input": x})[0])
        self.assertEqualArray(results[0], results[1])

    def test_simple_export_full(self):
        model, input_tensor = return_module_cls_full()
        names = export_utils("test_simple_export_full", model, input_tensor)
        x = input_tensor.numpy()
        results = []
        for name in names:
            ref = ReferenceEvaluator(name)
            results.append(ref.run(None, {"input": x})[0])
        self.assertEqualArray(results[0], results[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
