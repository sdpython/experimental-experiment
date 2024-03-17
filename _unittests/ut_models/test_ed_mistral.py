import sys
import unittest
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_torch,
)
from experimental_experiment.torch_helper.mistral_helper import get_mistral_model
from experimental_experiment.torch_test_helper import export_to_onnx, check_model_ort


def has_cuda():
    import onnxruntime

    available_providers = [
        provider for provider in onnxruntime.get_available_providers()
    ]
    return "CUDAExecutionProvider" in available_providers


class TestEdMistral(ExtTestCase):

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @requires_torch("2.3", "bug")
    @ignore_warnings(DeprecationWarning)
    def test_mistral_model(self):
        model, input_tensors = get_mistral_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)
        ret = export_to_onnx(model, *input_tensors)
        onx = ret["proto"]
        xp = [x.numpy() for x in input_tensors]
        feeds = {f"input{i}": x for i, x in enumerate(xp)}
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, feeds)
        self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)
        if has_cuda():
            check_model_ort(onx, providers="cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
