import sys
import unittest
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    # requires_torch,
)
from experimental_experiment.torch_helper.mistral_helper import get_mistral_model
from experimental_experiment.torch_test_helper import export_to_onnx, check_model_ort
from experimental_experiment.torch_bench._dort_cmd_common import create_compiled_model
from experimental_experiment.torch_helper.training_helper import train_loop


def has_cuda():
    import onnxruntime

    available_providers = [
        provider for provider in onnxruntime.get_available_providers()
    ]
    return "CUDAExecutionProvider" in available_providers


class TestEdMistral(ExtTestCase):

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings(DeprecationWarning)
    def test_mistral_export(self):
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
            sess = check_model_ort(onx, providers="cuda")
            results = sess.run(None, feeds)
            self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_mistral_cort_static(self):
        model, input_tensors = get_mistral_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)

        compiled_model, storage = create_compiled_model(
            model,
            backend="debug",
            use_dynamic=False,
            target_opset=18,
            verbose=0,
            return_storage=True,
        )
        results = compiled_model(*input_tensors)
        self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)
        instances = storage["instance"]
        self.assertEqual(len(instances), 1)  # forward

        train_loop(model, *input_tensors)
        train_loop(compiled_model, *input_tensors)
        instances = storage["instance"]
        self.assertEqual(len(instances), 2)  # forward + backward

        if __name__ == "__main__":
            for i, inst in enumerate(instances):
                self.dump_onnx(f"test_mistral_cort_static_{i}.onnx", inst["onnx"])

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_mistral_cort_dynamic(self):
        model, input_tensors = get_mistral_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)

        compiled_model, storage = create_compiled_model(
            model,
            backend="debug",
            use_dynamic=True,
            target_opset=18,
            verbose=0,
            return_storage=True,
        )
        results = compiled_model(*input_tensors)
        self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)
        instances = storage["instance"]
        # self.assertEqual(len(instances), 1)  # forward

        train_loop(model, *input_tensors)
        train_loop(compiled_model, *input_tensors)
        instances = storage["instance"]
        self.assertEqual(len(instances), 2)  # forward + backward

        if __name__ == "__main__":
            for i, inst in enumerate(instances):
                self.dump_onnx(f"test_mistral_cort_dynamic_{i}.onnx", inst["onnx"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
