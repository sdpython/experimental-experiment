import unittest
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
    requires_cuda,
)


class TestOnnxExportComplex(ExtTestCase):
    @skipif_ci_windows("torch dynamo not supported on windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    @requires_cuda()
    def test_export_polar(self):
        import torch

        class Neuron(torch.nn.Module):
            def forward(self, x, angle):
                return torch.polar(x, angle)

        model, x, angle = Neuron(), torch.rand(4, 4), torch.rand(4, 4)
        expected = model(x, angle)
        onx = to_onnx(model, (x, angle))
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.numpy(), "angle": angle.numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
