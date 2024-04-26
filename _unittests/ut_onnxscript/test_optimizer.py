import os
import unittest
from typing import Union
import onnx
from onnx.inliner import inline_local_functions
from experimental_experiment.ext_test_case import ExtTestCase, hide_stdout


class TestOnnxScriptOptimizer(ExtTestCase):

    def _check_ort(self, name: Union[str, onnx.ModelProto]):
        from onnxruntime import InferenceSession

        if isinstance(name, str):
            InferenceSession(name, providers=["CPUExecutionProvider"])
        else:
            InferenceSession(
                name.SerializeToString(), providers=["CPUExecutionProvider"]
            )

    @hide_stdout()
    def test_optimizer(self):
        filename = os.path.join(os.path.dirname(__file__), "data", "llama_forward.onnx")
        self.assertExists(filename)

        from onnxscript import optimizer
        from onnxscript.rewriter import onnxruntime as ort_rewriter

        onnx_model = onnx.load(filename)
        self._check_ort(onnx_model)
        onnx_model = optimizer.optimize(onnx_model)
        self._check_ort(onnx_model)
        onnx_model = ort_rewriter.rewrite(onnx_model)
        self._check_ort(onnx_model)

        # inline
        onnx_model = inline_local_functions(onnx_model)
        self._check_ort(onnx_model)

        # third time
        onnx_model = optimizer.optimize(onnx_model)
        self._check_ort(onnx_model)
        onnx_model = ort_rewriter.rewrite(onnx_model)
        self._check_ort(onnx_model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
