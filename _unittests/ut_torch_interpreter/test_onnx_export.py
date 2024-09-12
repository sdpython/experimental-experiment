import contextlib
import os
import unittest
import warnings
from io import StringIO
import onnx
from onnx.reference import ReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
)
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_interpreter import to_onnx, Dispatcher


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
    prefix,
    model,
    *args,
    remove_unused=False,
    constant_folding=True,
    verbose=0,
    rename_input=True,
    expected_weights=None,
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
        input_names=["input"] if rename_input else None,
        options=OptimizationOptions(
            remove_unused=remove_unused,
            constant_folding=constant_folding,
            verbose=verbose,
            patterns=None,
        ),
        verbose=verbose,
    )
    if expected_weights is not None:
        assert len(onx.graph.initializer) == expected_weights, (
            f"The model has {len(onx.graph.initializer)} initiliazers, "
            f"expecting {expected_weights}, inputs are "
            f"{[_.name for _ in onx.graph.input]}."
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
            InferenceSession(name.SerializeToString(), providers=["CPUExecutionProvider"])
        except Exception as e:
            from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

            raise AssertionError(
                f"onnxruntime cannot load the model"
                f"due to {e}\n{onnx_simple_text_plot(name)}"
            )

    @skipif_ci_windows("torch dynamo not supported on windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_simple_export_conv_rename(self):
        model, input_tensor = return_module_cls_conv()
        names = export_utils(
            "test_simple_export_conv_rename", model, input_tensor, expected_weights=2
        )
        x = input_tensor.numpy()
        results = []
        for name in names:
            with self.subTest(name=name):
                ref = ReferenceEvaluator(name)
                results.append(ref.run(None, {"input": x})[0])
                self.check_model_ort(name)
        if len(names) == len(results):
            self.assertEqualArray(results[0], results[1])

    @skipif_ci_windows("torch dynamo not supported on windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_simple_export_conv_norename(self):
        model, input_tensor = return_module_cls_conv()
        names = export_utils(
            "test_simple_export_conv_norename",
            model,
            input_tensor,
            rename_input=False,
            expected_weights=2,
            verbose=0,
        )
        x = input_tensor.numpy()
        results = []
        for name in names:
            with self.subTest(name=name):
                ref = ReferenceEvaluator(name)
                input_name = ref.input_names[0]
                results.append(ref.run(None, {input_name: x})[0])
                self.check_model_ort(name)
        if len(names) == len(results):
            self.assertEqualArray(results[0], results[1])

    @skipif_ci_windows("torch dynamo not supported on windows")
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

    @skipif_ci_windows("torch dynamo not supported on windows")
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

    @skipif_ci_windows("torch dynamo not supported on windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_remove_unused_nodes(self):
        from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

        model, input_tensor = return_module_cls_pool()
        onx1 = to_onnx(
            model,
            (input_tensor,),
            input_names=["input"],
            options=OptimizationOptions(remove_unused=False, patterns=None),
        )
        onx2 = to_onnx(
            model,
            (input_tensor,),
            input_names=["input"],
            options=OptimizationOptions(remove_unused=True),
        )
        self.assertGreater(len(onx1.graph.node), len(onx2.graph.node))
        p1 = [n for n in onx1.graph.node if n.op_type == "Identity"]
        p2 = [n for n in onx2.graph.node if n.op_type == "Identity"]
        self.assertEqual(len(p1), 0)
        self.assertEqual(len(p2), 0)

        sub1 = [n for n in onx1.graph.node if n.op_type == "Sub"]
        sub2 = [n for n in onx2.graph.node if n.op_type == "Sub"]
        self.assertIn(len(sub1), {0, 2})
        self.assertEqual(len(sub2), 0)

        p1 = [n for n in onx1.graph.node if n.op_type == "MaxPool"]
        p2 = [n for n in onx2.graph.node if n.op_type == "MaxPool"]
        self.assertIn(len(p1), {2, 4})
        self.assertEqual(
            len(p2), 2, f"Mismatch number of MaxPool, {onnx_simple_text_plot(onx2)}"
        )
        self.check_model_ort(onx2)

    @skipif_ci_windows("torch dynamo not supported on windows")
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

    @skipif_ci_windows("torch dynamo not supported on windows")
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
            options=OptimizationOptions(constant_folding=False, patterns=None),
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
            options=OptimizationOptions(constant_folding=True, patterns=None),
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
        self.assertEqual(len(p1), 3, f"Mismatch Transpose\n{onnx_simple_text_plot(onx1)}")
        self.assertEqual(len(p2), 0, f"Mismatch Transpose\n{onnx_simple_text_plot(onx2)}")
        self.check_model_ort(onx2)

    @skipif_ci_windows("torch dynamo not supported on windows")
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

    @skipif_ci_windows("torch dynamo not supported on windows")
    def test_dispatcher_function(self):
        import torch

        T = str

        class NotFoundUTError(Exception):
            pass

        def aten_celu(g, sts, outputs, x: T, alpha=1.0, inplace=False) -> T:
            assert not inplace, f"not implemented if inplace=True{g.get_debug_msg()}"
            raise NotFoundUTError("not implemented")

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return torch.celu(self.linear(x))

        x = torch.rand(5, 3)
        model = Neuron(3, 1)

        dispatcher = Dispatcher({"aten::celu": aten_celu}, verbose=4)

        s = StringIO()
        with contextlib.redirect_stdout(s):
            self.assertRaise(
                lambda: to_onnx(model, (x,), input_names=["x"], dispatcher=dispatcher),
                NotFoundUTError,
            )
        self.assertIn("[Dispatcher.find_function]", s.getvalue())

    @skipif_ci_windows("torch dynamo not supported on windows")
    def test_dispatcher_method(self):
        import torch

        T = str

        class NotFoundUTError(Exception):
            pass

        def aten_celu(g, sts, outputs, x: T, alpha=1.0, inplace=False) -> T:
            assert not inplace, f"not implemented if inplace=True{g.get_debug_msg()}"
            raise NotFoundUTError("not implemented")

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return torch.celu(self.linear(x))

        x = torch.rand(5, 3)
        model = Neuron(3, 1)

        dispatcher = Dispatcher({"aten::celu": aten_celu}, verbose=4)

        s = StringIO()
        with contextlib.redirect_stdout(s):
            self.assertRaise(
                lambda: to_onnx(model, (x,), input_names=["x"], dispatcher=dispatcher),
                NotFoundUTError,
            )
        self.assertIn("[Dispatcher.find_function]", s.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
