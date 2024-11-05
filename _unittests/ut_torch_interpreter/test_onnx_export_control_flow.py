import unittest
import warnings
from collections import Counter
import onnx
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    skipif_ci_windows,
    hide_stdout,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


class TestOnnxExportControlFlow(ExtTestCase):

    @classmethod
    def get_custom_model(cls):
        import torch

        class Bad1Fixed(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x)

                def false_fn(x):
                    return torch.cos(x)

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        return Bad1Fixed, torch.rand(5, 3)

    @requires_torch("2.7")
    def test_controlflow_script(self):
        import torch

        cls, x = self.get_custom_model()
        model = cls()
        filename = "test_controlflow_script.onnx"
        torch.onnx.export(model, (x,), filename, input_names=["x"], opset_version=18)
        with open(filename, "rb") as f:
            onx = onnx.load(f)
        # co = Counter([n.op_type for n in onx.graph.node])
        # self.assertEqual(co, {"Add": 2, "Mul": 1})

        for _x in (x, -x):
            expected = model(_x)
            ref = ExtendedReferenceEvaluator(onx)
            got = ref.run(None, {"x": _x.detach().numpy()})[0]
            self.assertEqualArray(expected, got)

    @requires_torch("2.7")
    @skipif_ci_windows("not yet supported on Windows")
    def test_controlflow_dynamo(self):
        import torch

        cls, x = self.get_custom_model()
        model = cls()
        filename = "test_controlflow_dynamo.onnx"
        torch.onnx.export(
            model,
            (x,),
            filename,
            input_names=["x"],
            opset_version=18,
            dynamo=True,
            fallback=False,
        )
        with open(filename, "rb") as f:
            onx = onnx.load(f)
        # co = Counter([n.op_type for n in onx.graph.node])
        # self.assertEqual(co, {"Add": 2, "Mul": 1})

        for _x in (x, -x):
            expected = model(_x)
            ref = ExtendedReferenceEvaluator(onx)
            got = ref.run(None, {"x": _x.detach().numpy()})[0]
            self.assertEqualArray(expected, got)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_controlflow_custom_if_1(self):
        import onnxruntime

        cls, x = self.get_custom_model()
        model = cls()
        onx = to_onnx(model, (x,))
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 2)

        for _x in (x, -x):
            expected = model(_x)
            ref = ExtendedReferenceEvaluator(onx)
            got = ref.run(None, {"x": _x.detach().numpy()})[0]
            self.assertEqualArray(expected, got, atol=1e-5)

            sess = onnxruntime.InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            got = sess.run(None, {"x": _x.detach().numpy()})[0]
            self.assertEqualArray(expected, got, atol=1e-5)

    @classmethod
    def get_custom_model_2(cls):
        import torch

        class Bad2Fixed(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x), torch.cos(x)

                def false_fn(x):
                    return torch.cos(x), torch.sin(x)

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        return Bad2Fixed, torch.rand(5, 3)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_controlflow_custom_if_2(self):
        cls, x = self.get_custom_model_2()
        model = cls()
        onx = to_onnx(model, (x,))
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 2)

        for _x in (x, -x):
            expected = model(_x)
            ref = ExtendedReferenceEvaluator(onx)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqual(len(expected), len(got))
            for e, g in zip(expected, got):
                self.assertEqualArray(e, g, atol=1e-5)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    @hide_stdout()
    def test_controlflow_custom_if_inline(self):
        cls, x = self.get_custom_model()
        model = cls()
        onx = to_onnx(model, (x,), inline=True, verbose=2)
        self.assertEqual(len(onx.functions), 0)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 0)

        for _x in (x, -x):
            expected = model(_x)
            ref = ExtendedReferenceEvaluator(onx)
            got = ref.run(None, {"x": _x.detach().numpy()})[0]
            self.assertEqualArray(expected, got, atol=1e-5)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_controlflow_custom_if_two_inputs(self):
        import torch

        class TwoInputs(torch.nn.Module):
            def forward(self, x, y):
                def true_fn(x, y):
                    return torch.sin(x), torch.cos(x) + y

                def false_fn(x, y):
                    return torch.cos(x), torch.sin(x) + y

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x, y])

        x, y = torch.rand(5, 3), torch.rand(5, 3)
        model = TwoInputs()
        onx = to_onnx(model, (x, y))
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 2)

        for _x in (x, -x):
            expected = model(_x, y)
            ref = ExtendedReferenceEvaluator(onx)
            got = ref.run(None, {"x": _x.detach().numpy(), "y": y.detach().numpy()})
            self.assertEqual(len(expected), len(got))
            for e, g in zip(expected, got):
                self.assertEqualArray(e, g, atol=1e-5)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_controlflow_custom_if_two_cond(self):
        import torch

        class TwoConds(torch.nn.Module):
            def forward(self, x):
                def true_fn2(x):
                    def true_fn1(x):
                        return torch.sin(x)

                    def false_fn1(x):
                        return torch.cos(x)

                    return torch.cond(x.sum() < 0, true_fn1, false_fn1, [x])

                def false_fn2(x):
                    return -x

                return torch.cond(x.sum() > 0, true_fn2, false_fn2, [x])

        x = torch.rand(5, 3)
        model = TwoConds()
        model(x)
        model(-x)
        torch.export.export(model, (x,))
        onx = to_onnx(model, (x,))
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 4)

        for _x in (x, -x):
            expected = model(_x)
            ref = ExtendedReferenceEvaluator(onx)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_controlflow_custom_raw_test(self):
        import torch

        class RawTest(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x)

                def false_fn(x):
                    return torch.cos(x)

                if x.sum() > 0:
                    return true_fn(x)
                return false_fn(x)

        x = torch.rand(5, 3)
        model = RawTest()
        filename = "test_controlflow_custom_raw_test.onnx"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, (x,), filename, input_names=["x"], opset_version=18)
            onx = to_onnx(model, (x,), export_options=ExportOptions(jit=True))
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"Sin": 1})
        self.assertEqual(len(onx.functions), 0)

        for _x in (x,):
            expected = model(_x)
            ref = ExtendedReferenceEvaluator(onx)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_controlflow_custom_fallback(self):
        import torch

        class RawTest(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x)

                def false_fn(x):
                    return torch.cos(x)

                if x.sum() > 0:
                    return true_fn(x)
                return false_fn(x)

        x = torch.rand(5, 3)
        model = RawTest()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onx = to_onnx(model, (x,), export_options=ExportOptions(strategy="fallback"))
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"Sin": 1})
        self.assertEqual(len(onx.functions), 0)

        for _x in (x,):
            expected = model(_x)
            ref = ExtendedReferenceEvaluator(onx)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_controlflow_custom_initializer(self):
        import torch

        class RawTest(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x) - torch.ones(x.shape, dtype=x.dtype)

                def false_fn(x):
                    return torch.cos(x) + torch.ones((1, 1024), dtype=x.dtype)

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        x = torch.rand(1024, 1024)
        model = RawTest()

        # not inlined
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onx = to_onnx(model, (x,))
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 2)

        for _x in (x, -x):
            expected = model(_x)
            ref = ExtendedReferenceEvaluator(onx)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)

        # inlined
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onx = to_onnx(model, (x,), inline=True)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 0)

        for _x in (x, -x):
            expected = model(_x)
            ref = ExtendedReferenceEvaluator(onx)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
