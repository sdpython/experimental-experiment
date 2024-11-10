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
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x)
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
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x)
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
        ref = ExtendedReferenceEvaluator(onx)
        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        for _x in (x, -x):
            expected = model(_x)
            got = ref.run(None, {"x": _x.detach().numpy()})[0]
            self.assertEqualArray(expected, got, atol=1e-5)

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
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x)
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
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x)
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
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x, y)
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
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x)
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
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x,):
            expected = model(_x)
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
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x,):
            expected = model(_x)
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
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)

        # inlined
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onx = to_onnx(model, (x,), inline=True)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 0)
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)

    def test_nested_cond(self):
        import onnxruntime
        import torch

        class Submodule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Nested weight
                self.weight = torch.nn.Parameter(torch.tensor([100.0]))

            def forward(self, x):
                def true_fn(x):
                    return x * self.weight

                def false_fn(x):
                    return x / self.weight

                y = torch.cond(torch.abs(x).sum() > 100, true_fn, false_fn, [x])
                return y

        class CondModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submodule = Submodule()
                self.weight = torch.nn.Parameter(torch.tensor([42.0]))

            def forward(self, x):
                def true_fn(x):
                    return self.submodule(x)

                def false_fn(x):
                    return x - self.weight

                y = torch.cond(x.sum() > 0, true_fn, false_fn, [x])
                return y

        x = torch.tensor([-1, 2])
        model = CondModel()
        onx = to_onnx(model, (x,))
        names = [(f.domain, f.name) for f in onx.functions]
        self.assertEqual(len(names), len(set(names)))
        ref = ExtendedReferenceEvaluator(onx)
        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        for _x in (x, -x, -x * 1000, x * 1000):
            expected = model(_x)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)
            got = sess.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)

    @requires_torch("2.6")
    def test_scan_1(self):
        import torch

        def add(carry: torch.Tensor, y: torch.Tensor):
            next_carry = carry + y
            return [next_carry, next_carry]

        class ScanModel(torch.nn.Module):
            def forward(self, x):
                init = torch.zeros_like(x[0])
                carry, out = torch.ops.higher_order.scan(
                    add, [init], [x], dim=0, reverse=False, additional_inputs=[]
                )
                return carry

        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        model = ScanModel()
        expected = model(x)
        self.assertEqualArray(expected, x.sum(axis=0))
        self.assertNotEmpty(torch.export.export(model, (x,), strict=True).graph)

        onx = to_onnx(model, (x,))
        names = [(f.domain, f.name) for f in onx.functions]
        self.assertEqual(len(names), len(set(names)))

        ref = ExtendedReferenceEvaluator(onx)

        for _x in (-x, x):
            expected = model(_x)
            feeds = {"x": _x.detach().numpy()}
            got = ref.run(None, feeds)
            self.assertEqualArray(expected, got[0], atol=1e-5)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        for _x in (-x, x):
            expected = model(_x)
            feeds = {"x": _x.detach().numpy()}
            got = sess.run(None, feeds)
            self.assertEqualArray(expected, got[0], atol=1e-5)

    @requires_torch("2.6")
    def test_scan_2(self):
        import torch

        def add(
            carry1: torch.Tensor, carry2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor
        ):
            next_carry1 = carry1 + y1
            next_carry2 = carry2 * y2
            return [next_carry1, next_carry2, next_carry1, next_carry2]

        class ScanModel(torch.nn.Module):
            def forward(self, x):
                init1 = torch.zeros_like(x[0])
                init2 = torch.ones_like(x[0])
                carry1, carry2, out1, out2 = torch.ops.higher_order.scan(
                    add,
                    [init1, init2],
                    [x, x * 2],
                    dim=0,
                    reverse=False,
                    additional_inputs=[],
                )
                return carry1, carry2, out1, out2

        x = torch.tensor([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32)
        model = ScanModel()
        expected = model(x)
        self.assertEqualArray(expected[0], x.sum(axis=0))
        self.assertNotEmpty(torch.export.export(model, (x,), strict=True).graph)

        onx = to_onnx(model, (x,))
        names = [(f.domain, f.name) for f in onx.functions]
        self.assertEqual(len(names), len(set(names)))

        ref = ExtendedReferenceEvaluator(onx)

        for _x in (-x, x):
            expected = model(_x)
            feeds = {"x": _x.detach().numpy()}
            got = ref.run(None, feeds)
            for e, g in zip(expected, got):
                self.assertEqualArray(e, g, atol=1e-5)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        for _x in (-x, x):
            expected = model(_x)
            feeds = {"x": _x.detach().numpy()}
            got = sess.run(None, feeds)
            for e, g in zip(expected, got):
                self.assertEqualArray(e, g, atol=1e-5)

    @requires_torch("2.6")
    def test_scan_cdist_carry(self):
        import torch

        def dist(carry: torch.Tensor, x: torch.Tensor):
            sub = carry - x.reshape((1, -1))
            sq = sub * sub
            rd = sq.sum(axis=1) ** 0.5
            # clone --> UnsupportedAliasMutationException:
            # Combine_fn might be aliasing the input!
            return [carry.clone(), rd]

        class ScanModel(torch.nn.Module):
            def forward(self, x):
                carry, out = torch.ops.higher_order.scan(
                    dist,
                    [x],
                    [x],
                    dim=0,
                    reverse=False,
                    additional_inputs=[],
                )
                return out

        x = torch.tensor([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32)
        model = ScanModel()
        expected = model(x)
        self.assertEqual(expected.shape, (3, 3))
        self.assertEqualArray(expected, torch.cdist(x, x))
        self.assertNotEmpty(torch.export.export(model, (x,), strict=True).graph)

        onx = to_onnx(model, (x,))
        names = [(f.domain, f.name) for f in onx.functions]
        self.assertEqual(len(names), len(set(names)))

        ref = ExtendedReferenceEvaluator(onx)

        for _x in (-x, x):
            expected = model(_x)
            feeds = {"x": _x.detach().numpy()}
            got = ref.run(None, feeds)
            self.assertEqualArray(expected, got[0], atol=1e-5)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        for _x in (-x, x):
            expected = model(_x)
            feeds = {"x": _x.detach().numpy()}
            got = sess.run(None, feeds)
            self.assertEqualArray(expected, got[0], atol=1e-5)

    @requires_torch("2.6")
    def test_scan_cdist_add(self):
        import torch

        def dist(unused: torch.Tensor, x: torch.Tensor, samex: torch.Tensor):
            sub = samex - x.reshape((1, -1))
            sq = sub * sub
            rd = torch.sqrt(sq.sum(axis=1))
            # clone --> UnsupportedAliasMutationException:
            # Combine_fn might be aliasing the input!
            return [unused.clone(), rd]

        class ScanModel(torch.nn.Module):
            def forward(self, x):
                z = torch.tensor([0], dtype=torch.float32)
                y = x.clone()
                out = torch.ops.higher_order.scan(
                    dist,
                    [z],
                    [x],
                    dim=0,
                    reverse=False,
                    additional_inputs=[y],
                )
                return out[1]

        x = torch.tensor([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32)
        model = ScanModel()
        expected = model(x)
        self.assertEqual(expected.shape, (3, 3))
        self.assertEqualArray(expected, torch.cdist(x, x))
        self.assertNotEmpty(torch.export.export(model, (x,), strict=True).graph)

        for optimize in [False, True]:
            onx = to_onnx(model, (x,), optimize=optimize)
            # with open(f"test_scan_cdist_add_{int(optimize)}.onnx", "wb") as f:
            #     f.write(onx.SerializeToString())

            names = [(f.domain, f.name) for f in onx.functions]
            self.assertEqual(len(names), len(set(names)))

            # ReferenceEvaluator does not work in this graph
            import onnxruntime

            sess = onnxruntime.InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            for _x in (-x, x):
                expected = model(_x)
                feeds = {"x": _x.detach().numpy()}
                got = sess.run(None, feeds)
                self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
