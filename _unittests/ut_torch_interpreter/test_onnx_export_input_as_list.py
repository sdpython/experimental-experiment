import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    skipif_ci_windows,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx


class TestOnnxExportInputList(ExtTestCase):

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_input_list(self):
        import torch

        class RawTest(torch.nn.Module):
            def forward(self, x, list_yz):
                return x + list_yz[0] + list_yz[1]

        x = torch.rand(1024, 1024)
        list_yz = [torch.rand(1024, 1024), torch.rand(1024, 1024)]
        model = RawTest()

        onx = to_onnx(model, (x, list_yz))
        names = [i.name for i in onx.graph.input]
        feeds = dict(zip(names, [_.detach().numpy() for _ in [x, *list_yz]]))
        shapes = [
            tuple(d.dim_value for d in i.type.tensor_type.shape.dim) for i in onx.graph.input
        ]
        self.assertEqual(set(shapes), {(1024, 1024)})

        expected = model(x, list_yz)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_input_list_dynamic(self):
        import torch

        class RawTest(torch.nn.Module):
            def forward(self, x, list_yz):
                return x + list_yz[0] + list_yz[1]

        x = torch.rand(1024, 1024)
        list_yz = [torch.rand(1024, 1024), torch.rand(1024, 1024)]
        model = RawTest()

        batch = torch.export.Dim("batch", min=1, max=2048)
        onx = to_onnx(
            model,
            (x, list_yz),
            dynamic_shapes={"x": {0: batch}, "list_yz": [{0: batch}, {0: batch}]},
        )
        names = [i.name for i in onx.graph.input]
        shapes = [
            tuple(
                (d.dim_value if d.dim_value else d.dim_param)
                for d in i.type.tensor_type.shape.dim
            )
            for i in onx.graph.input
        ]
        self.assertEqual(shapes, [("batch", 1024), ("s1", 1024), ("s2", 1024)])
        feeds = dict(zip(names, [_.detach().numpy() for _ in [x, *list_yz]]))

        expected = model(x, list_yz)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
