import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    requires_onnx_diagnostic,
    skipif_ci_windows,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


class TestOnnxExportInputDictList(ExtTestCase):
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
    @requires_onnx_diagnostic("0.7.13")
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
        self.assertEqual(shapes, [("batch", 1024), ("batch", 1024), ("batch", 1024)])
        feeds = dict(zip(names, [_.detach().numpy() for _ in [x, *list_yz]]))

        expected = model(x, list_yz)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @requires_onnx_diagnostic("0.8.8")
    def test_tensor_input_tracer(self):
        import torch

        class RawTest(torch.nn.Module):
            def forward(self, x, t1, t2):
                return x + t1 + t2

        x = torch.rand(1024, 1024)
        list_yz = [torch.rand(1024, 1024), torch.rand(1024, 1024)]
        model = RawTest()

        batch = torch.export.Dim("batch", min=1, max=2048)
        onx = to_onnx(
            model,
            (x, *list_yz),
            dynamic_shapes=({0: batch}, {0: batch}, {0: batch}),
            export_options=ExportOptions(tracing=True),
        )
        names = [i.name for i in onx.graph.input]
        shapes = [
            tuple(
                (d.dim_value if d.dim_value else d.dim_param)
                for d in i.type.tensor_type.shape.dim
            )
            for i in onx.graph.input
        ]
        self.assertEqual(shapes, [("batch", 1024), ("batch", 1024), ("batch", 1024)])
        feeds = dict(zip(names, [_.detach().numpy() for _ in [x, *list_yz]]))

        expected = model(x, *list_yz)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @requires_onnx_diagnostic("0.8.8")
    @requires_torch("2.9.99")
    def test_list_input_tracer(self):
        import torch

        class RawTest(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ppp = torch.nn.Parameter(torch.tensor([2], dtype=torch.float32))

            def forward(selff, x, list_yz):
                return x + list_yz[0] + list_yz[1] + selff.ppp

        x = torch.rand(4, 4)
        list_yz = [torch.rand(4, 1), torch.rand(4, 1)]
        model = RawTest()

        batch = torch.export.Dim("batch", min=1, max=2048)
        onx = to_onnx(
            model,
            (x, list_yz),
            dynamic_shapes=({0: batch}, [{0: batch}, {0: batch}]),
            export_options=ExportOptions(tracing=True),
            verbose=10,
        )
        import onnx

        onnx.save(onx, self.get_dump_file("test_list_input_tracer.onnx"))
        shapes = [
            tuple(
                (d.dim_value if d.dim_value else d.dim_param)
                for d in i.type.tensor_type.shape.dim
            )
            for i in onx.graph.input
        ]
        self.assertEqual(shapes[0], ("batch", 4))
        expected = model(x, list_yz)
        ref = self.check_ort(onx)
        feeds = dict(zip([i.name for i in ref.get_inputs()], [_.numpy() for _ in [x, *list_yz]]))
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @requires_onnx_diagnostic("0.8.8")
    def test_neuron_tracer(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

            def _get_random_inputs(self, device: str):
                return (torch.randn(2, 5).to(device),)

        x = torch.rand(2, 5)
        model = Neuron()

        batch = torch.export.Dim("batch", min=1, max=2048)
        onx = to_onnx(
            model,
            (x,),
            dynamic_shapes=({0: batch},),
            export_options=ExportOptions(tracing=True),
        )
        import onnx

        onnx.save(onx, self.get_dump_file("test_neuron_tracer.onnx"))
        feeds = {"x": x.numpy()}

        expected = model(x)

        from onnxruntime import InferenceSession

        ref = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
