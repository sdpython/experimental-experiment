import os
import sys
import unittest
import warnings
import onnx
from onnx.reference import ReferenceEvaluator
from experimental_experiment.ext_test_case import ExtTestCase, ignore_warnings
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
            self.conv1 = nn.Conv2d(1, 16, 5)
            self.conv2 = nn.Conv2d(16, 16, 5)
            self.fc1 = nn.Linear(13456, 8)
            self.fc2 = nn.Linear(8, 8)
            self.fc3 = nn.Linear(8, 10)

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


def export_utils(
    prefix, model, *args, remove_unused=False, constant_folding=True, verbose=0
):
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
        model,
        tuple(args),
        input_names=["input"],
        remove_unused=remove_unused,
        constant_folding=constant_folding,
        verbose=verbose,
    )
    with open(name, "wb") as f:
        f.write(onx.SerializeToString())
    names.append(name)
    return names


class TestOnnxExport(ExtTestCase):
    def check_model_ort(self, name):
        from onnxruntime import InferenceSession

        if isinstance(name, str):
            try:
                InferenceSession(name, providers=["CPUExecutionProvider"])
            except Exception as e:
                import onnx
                from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

                raise AssertionError(
                    f"onnxruntime cannot load the model "
                    f"due to {e}\n{onnx_simple_text_plot(onnx.load(name))}"
                )
            return
        try:
            InferenceSession(
                name.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

            raise AssertionError(
                f"onnxruntime cannot load the model"
                f"due to {e}\n{onnx_simple_text_plot(name)}"
            )

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
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
    @ignore_warnings((UserWarning, DeprecationWarning))
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
    @ignore_warnings((UserWarning, DeprecationWarning))
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
    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_remove_unused_nodes(self):
        from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

        model, input_tensor = return_module_cls_pool()
        onx1 = to_onnx(model, (input_tensor,), input_names=["input"])
        onx2 = to_onnx(
            model, (input_tensor,), input_names=["input"], remove_unused=True
        )
        self.assertGreater(len(onx1.graph.node), len(onx2.graph.node))
        p1 = [n for n in onx1.graph.node if n.op_type == "Identity"]
        p2 = [n for n in onx2.graph.node if n.op_type == "Identity"]
        self.assertEqual(len(p1), 0)
        self.assertEqual(len(p2), 0)

        sub1 = [n for n in onx1.graph.node if n.op_type == "Sub"]
        sub2 = [n for n in onx2.graph.node if n.op_type == "Sub"]
        self.assertEqual(len(sub1), 2)
        self.assertEqual(len(sub2), 0)

        p1 = [n for n in onx1.graph.node if n.op_type == "MaxPool"]
        p2 = [n for n in onx2.graph.node if n.op_type == "MaxPool"]
        self.assertEqual(len(p1), 4)
        self.assertEqual(
            len(p2), 2, f"Mismatch number of MaxPool, {onnx_simple_text_plot(onx2)}"
        )
        self.check_model_ort(onx2)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
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

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_constant_folding(self):
        try:
            from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
        except ImportError:
            onnx_simple_text_plot = str
        model, input_tensor = return_module_cls_pool()
        onx1 = to_onnx(
            model,
            (input_tensor,),
            input_names=["input"],
            remove_unused=True,
            constant_folding=False,
        )
        self.assertGreater(
            len(onx1.graph.node),
            5,
            msg=f"Mismath number of node {len(onx1.graph.node)}, "
            f"{onnx_simple_text_plot(onx1)}",
        )
        self.assertEqual(len(onx1.graph.input), 1)
        self.assertEqual(len(onx1.graph.output), 1)
        with open("dummy.onnx", "wb") as f:
            f.write(onx1.SerializeToString())
        onx2 = to_onnx(
            model,
            (input_tensor,),
            input_names=["input"],
            remove_unused=True,
            constant_folding=True,
        )
        self.assertGreater(len(onx2.graph.node), 5)
        self.assertGreaterOrEqual(len(onx1.graph.node), len(onx2.graph.node))
        self.assertEqual(len(onx2.graph.input), 1)
        self.assertEqual(len(onx2.graph.output), 1)

        p1 = [n for n in onx1.graph.node if n.op_type == "Identity"]
        p2 = [n for n in onx2.graph.node if n.op_type == "Identity"]
        self.assertEqual(len(p1), 0)
        self.assertEqual(len(p2), 0)

        p1 = [n for n in onx1.graph.node if n.op_type == "MaxPool"]
        p2 = [n for n in onx2.graph.node if n.op_type == "MaxPool"]
        self.assertEqual(len(p1), 2)
        self.assertEqual(
            len(p2), 2, f"Mismatch number of MaxPool, {onnx_simple_text_plot(onx2)}"
        )
        self.check_model_ort(onx2)

        p1 = [n for n in onx1.graph.node if n.op_type == "Transpose"]
        p2 = [n for n in onx2.graph.node if n.op_type == "Transpose"]
        self.assertEqual(
            len(p1), 3, f"Mismatch Transpose\n{onnx_simple_text_plot(onx1)}"
        )
        self.assertEqual(
            len(p2), 0, f"Mismatch Transpose\n{onnx_simple_text_plot(onx2)}"
        )
        self.check_model_ort(onx2)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_simple_export_pool_constant_folding(self):
        from onnxruntime import InferenceSession

        model, input_tensor = return_module_cls_pool()
        names = export_utils(
            "test_simple_export_pool_unused_noopt",
            model,
            input_tensor,
            remove_unused=True,
            constant_folding=True,
        )
        onx1 = onnx.load(names[-1])
        names, out, _ = self.capture(
            lambda: export_utils(
                "test_simple_export_pool_unused_opt",
                model,
                input_tensor,
                remove_unused=True,
                constant_folding=True,
                verbose=1,
            )
        )
        self.assertIn("[GraphBuilder", out)
        onx2 = onnx.load(names[-1])
        for att in ["node", "initializer"]:
            v1 = getattr(onx1.graph, att)
            v2 = getattr(onx2.graph, att)
            self.assertEqual(len(v1), len(v2))
        x = input_tensor.numpy()
        results = []
        for name in names:
            ref = InferenceSession(name, providers=["CPUExecutionProvider"])
            results.append(ref.run(None, {"input": x})[0])
        self.assertEqualArray(results[0], results[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
