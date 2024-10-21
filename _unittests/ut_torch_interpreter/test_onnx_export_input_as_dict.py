import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    skipif_ci_windows,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx


class TestOnnxExportInputDict(ExtTestCase):

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_input_dict(self):
        import torch

        class RawTest(torch.nn.Module):
            def forward(self, input_x=None, input_y=None):
                if input_y is None:
                    return input_x
                if input_x is None:
                    return input_y
                return input_x - input_y

        x = torch.rand(1024, 1024)
        y = torch.rand(1024, 1024)
        model = RawTest()

        gr1 = torch.export.export(model, (x,))
        self.assertNotIn("input_y", str(gr1.graph))
        gr2 = torch.export.export(model, tuple(), dict(input_x=x))
        self.assertNotIn("input_y", str(gr2.graph))

        # case 1: input_x=x

        onx = to_onnx(model, kwargs=dict(input_x=x), verbose=0)
        names = [i.name for i in onx.graph.input]
        self.assertEqual(names, ["input_x"])
        feeds = dict(input_x=x.detach().numpy())
        shapes = [
            tuple(d.dim_value for d in i.type.tensor_type.shape.dim) for i in onx.graph.input
        ]
        self.assertEqual(set(shapes), {(1024, 1024)})

        expected = model(input_x=x)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

        # case 2: input_x=y

        onx = to_onnx(model, kwargs=dict(input_y=x), verbose=0)
        names = [i.name for i in onx.graph.input]
        self.assertEqual(names, ["input_y"])
        feeds = dict(input_y=x.detach().numpy())
        shapes = [
            tuple(d.dim_value for d in i.type.tensor_type.shape.dim) for i in onx.graph.input
        ]
        self.assertEqual(set(shapes), {(1024, 1024)})

        expected = model(input_x=x)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

        # case 3: input_x=x, input_y=y

        onx = to_onnx(model, kwargs=dict(input_x=x, input_y=y), verbose=0)
        names = [i.name for i in onx.graph.input]
        self.assertEqual(names, ["input_x", "input_y"])
        feeds = dict(input_x=x.detach().numpy(), input_y=y.detach().numpy())
        shapes = [
            tuple(d.dim_value for d in i.type.tensor_type.shape.dim) for i in onx.graph.input
        ]
        self.assertEqual(set(shapes), {(1024, 1024)})

        expected = model(input_x=x, input_y=y)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
