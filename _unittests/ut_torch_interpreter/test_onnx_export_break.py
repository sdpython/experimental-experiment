import os
import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    requires_torch,
)
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_interpreter import to_onnx


def return_module_cls_pool():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class MyModel(nn.Module):
        def __init__(self, n_lin_layers=2):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, 5)
            self.conv2 = nn.Conv2d(16, 16, 5)
            if n_lin_layers == 2:
                self.fc1 = nn.Linear(13456, 8)
                self.fc3 = nn.Linear(8, 10)
            else:
                self.fc1 = nn.Linear(13456, 8)
                self.fc2 = nn.Linear(8, 8)
                self.fc3 = nn.Linear(8, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            if not hasattr(self, "fc2"):
                x = self.fc3(x)
            else:
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
            return x

    input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)
    return MyModel(), input_tensor


def return_module_cls_explicit_break():
    import torch
    from torch import nn

    class ModelWithBreaks(nn.Module):
        def __init__(self):
            super().__init__()

            def create_sequential():
                return nn.Sequential(
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                )

            self.mod1 = create_sequential()
            self.mod2 = create_sequential()
            self.mod3 = create_sequential()
            self.mod4 = create_sequential()

        def forward(self, inp):
            mod1 = self.mod1(inp)
            torch._dynamo.graph_break()
            mod2 = self.mod2(mod1)
            torch._dynamo.graph_break()
            mod3 = self.mod3(mod2)
            torch._dynamo.graph_break()
            mod4 = self.mod4(mod3)
            return mod4

    input_tensor = torch.randn((128, 128), dtype=torch.float32)
    return ModelWithBreaks(), input_tensor


def export_utils(prefix, model, *args, remove_unused=False, constant_folding=True):
    names = []
    name = f"{prefix}_simple.onnx"
    if os.path.exists(name):
        os.remove(name)
    onx = to_onnx(
        model,
        tuple(args),
        input_names=["input"],
        options=OptimizationOptions(
            remove_unused=remove_unused,
            constant_folding=constant_folding,
            patterns=None,
        ),
    )
    with open(name, "wb") as f:
        f.write(onx.SerializeToString())
    names.append(name)
    return names


class TestOnnxExportBreak(ExtTestCase):
    def check_model_ort(self, name):
        from onnxruntime import InferenceSession

        if isinstance(name, str):
            InferenceSession(name, providers=["CPUExecutionProvider"])
            return
        InferenceSession(name.SerializeToString(), providers=["CPUExecutionProvider"])

    @skipif_ci_windows("not supported yet on Windows")
    @requires_torch("2.12")
    def test_simple_export_pool(self):
        from onnxruntime import InferenceSession

        model, input_tensor = return_module_cls_pool()
        names = export_utils("test_simple_export_break", model, input_tensor)
        x = input_tensor.numpy()
        results = []
        for name in names:
            ref = InferenceSession(name, providers=["CPUExecutionProvider"])
            results.append(ref.run(None, {"input": x})[0])
        self.assertEqualArray(results[0], results[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
