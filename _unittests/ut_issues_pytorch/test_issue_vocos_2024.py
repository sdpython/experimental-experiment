import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    requires_vocos,
    ignore_warnings,
)
from experimental_experiment.torch_interpreter import to_onnx


class TestIssuesPytorch2024(ExtTestCase):

    @requires_torch("2.7")
    @requires_vocos()
    def test_vocos_1_onnxscript(self):
        import torch
        from vocos import Vocos

        # https://github.com/pytorch/pytorch/issues/137864
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

        def forward(x):
            return vocos.decode(x)

        vocos.forward = forward
        dummy_input = torch.randn(512, 100, 7)
        program = torch.onnx.export(vocos, dummy_input, dynamo=True)
        self.assertNotEmpty(program)

    @requires_torch("2.5")
    @requires_vocos()
    @ignore_warnings(FutureWarning)
    def test_vocos_1_custom(self):
        import torch
        from vocos import Vocos

        # https://github.com/pytorch/pytorch/issues/137864
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

        def forward(x):
            return vocos.decode(x)

        vocos.forward = forward
        dummy_input = torch.randn(512, 100, 7)
        onx = to_onnx(vocos, (dummy_input,))
        self.assertNotEmpty(onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
