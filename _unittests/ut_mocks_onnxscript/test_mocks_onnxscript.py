import unittest
from onnx.reference import ReferenceEvaluator

try:
    from experimental_common.ext_test_case import ExtTestCase
except ImportError:
    from unittest import TestCase as ExtTestCase
try:
    from experimental_mocks.onnxscript.values import Opset
    from experimental_mocks.onnxscript.onnx_opset import all_opsets
except ImportError:
    from onnxscript.values import Opset
    from onnxscript.onnx_opset import all_opsets


def return_module_cls():
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

    name = f"{prefix}_dynamo.onnx"
    export_output = torch.onnx.dynamo_export(model, *args)
    export_output.save(name)
    names.append(name)
    return names


class TestMockExperimental(ExtTestCase):
    def test_install(self):
        self.assertIn(("", 18), all_opsets)
        self.assertIsInstance(all_opsets[("", 18)], Opset)

    def test_mock_dynamo_export(self):
        model, input_tensor = return_module_cls()
        names = export_utils("test_dynamo_export", model, input_tensor)
        x = input_tensor.numpy()
        results = []
        for name in names:
            ref = ReferenceEvaluator(name)
            results.append(ref.run(None, {"input": x})[0])
        self.assertEqualArray(results[0], results[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
