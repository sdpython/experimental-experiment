import unittest
from collections import Counter
import onnx
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    skipif_ci_windows,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx


class TestControlFlow(ExtTestCase):

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

    @unittest.skip("cond not supported by torch_script")
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

    @unittest.skip("torch.ops.higher_order.cond not supported yet")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
