import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    skipif_ci_windows,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import (
    to_onnx,
    match_input_parameters,
    ExportOptions,
)


class TestOnnxExportInputList(ExtTestCase):

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_export_with_fake_tensor(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return torch.relu(self.linear(x))

        fake_mode = FakeTensorMode()
        converter = fake_mode.fake_tensor_converter

        fake_x = converter.from_real_tensor(fake_mode, torch.rand(2, 5))
        with fake_mode:
            model = Neuron(5, 3)
            onx = to_onnx(model, (fake_x,))

        names = [i.name for i in onx.graph.input]
        self.assertEqual(len(names), 3)
        not_fake_model = Neuron(5, 3)
        x = torch.rand(2, 5)
        expected = not_fake_model(x)
        pfeeds = match_input_parameters(not_fake_model, names, (x,))
        nfeeds = {k: v.detach().numpy() for k, v in pfeeds.items()}
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, nfeeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    def test_export_with_option_fake(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return torch.relu(self.linear(x))

        model = Neuron(5, 3)
        x = torch.rand(2, 5)
        expected = model(x)
        onx, _out, _err = self.capture(
            lambda: to_onnx(
                model,
                (x,),
                dynamic_shapes=({0: "batch"},),
                export_options=ExportOptions(fake=True),
                verbose=10,
            )
        )
        self.assertIn("[ExportOptions.export] fake args=(F1s", _out)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
