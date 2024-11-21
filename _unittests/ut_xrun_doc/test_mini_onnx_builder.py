import unittest
import numpy as np
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_test_helper import create_onnx_model_from_input_tensors
from experimental_experiment.reference import ExtendedReferenceEvaluator


class TestMiniOnnxBuilder(ExtTestCase):
    def test_mini_onnx_builder_tuple(self):
        import torch

        tensors = (np.array([1, 2], dtype=np.int64), torch.tensor([4, 5], dtype=torch.float32))
        model = create_onnx_model_from_input_tensors(tensors)
        self.assertEqual(["arg_0", "arg_1"], [o.name for o in model.graph.output])
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {})
        for i in range(len(tensors)):
            self.assertEqualArray(tensors[i], got[i])

    def test_mini_onnx_builder_dict(self):
        import torch

        tensors = {
            "t1": np.array([1, 2], dtype=np.int64),
            "t2": torch.tensor([4, 5], dtype=torch.float32),
        }
        model = create_onnx_model_from_input_tensors(tensors)
        self.assertEqual(["t1", "t2"], [o.name for o in model.graph.output])
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {})
        self.assertEqualArray(tensors["t1"], got[0])
        self.assertEqualArray(tensors["t2"], got[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
