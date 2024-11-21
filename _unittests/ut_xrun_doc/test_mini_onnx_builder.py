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

    def test_mini_onnx_builder_tuple_list(self):
        import torch

        tensors = (
            np.array([1, 2], dtype=np.int64),
            torch.tensor([4, 5], dtype=torch.float32),
            [np.array([-1, -2], dtype=np.int64), torch.tensor([-4, -5], dtype=torch.float32)],
            [],
        )
        model = create_onnx_model_from_input_tensors(tensors)
        self.assertEqual(
            ["arg_0", "arg_1", "arg_2", "arg_3"], [o.name for o in model.graph.output]
        )
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {})
        for i in range(len(tensors)):
            if isinstance(tensors[i], list):
                self.assertIsInstance(got[i], list)
                self.assertEqual(len(tensors[i]), len(got[i]))
                for k in range(len(tensors[i])):
                    self.assertEqualArray(tensors[i][k], got[i][k])

            else:
                self.assertEqualArray(tensors[i], got[i])

    def test_mini_onnx_builder_dict_list(self):
        import torch

        tensors = {
            "t1": np.array([1, 2], dtype=np.int64),
            "t2": torch.tensor([4, 5], dtype=torch.float32),
            "l1": [
                np.array([-1, -2], dtype=np.int64),
                torch.tensor([-4, -5], dtype=torch.float32),
            ],
            "l2": [],
        }
        model = create_onnx_model_from_input_tensors(tensors)
        self.assertEqual(["t1", "t2", "l1", "l2"], [o.name for o in model.graph.output])
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {})
        for i, t in enumerate(tensors.values()):
            if isinstance(t, list):
                self.assertIsInstance(got[i], list)
                self.assertEqual(len(t), len(got[i]))
                for k in range(len(t)):
                    self.assertEqualArray(t[k], got[i][k])

            else:
                self.assertEqualArray(t, got[i])

    def test_mini_onnx_builder_tuple_dict(self):
        import torch

        tensors = (
            np.array([1, 2], dtype=np.int64),
            torch.tensor([4, 5], dtype=torch.float32),
            {
                "tt1": np.array([-1, -2], dtype=np.int64),
                "tt2": torch.tensor([-4, -5], dtype=torch.float32),
            },
            {},
        )
        model = create_onnx_model_from_input_tensors(tensors)
        names = ["arg_0", "arg_1", "arg_2_keys", "arg_2_values", "arg_3_keys", "arg_3_values"]
        self.assertEqual(names, [o.name for o in model.graph.output])
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {})
        for i, g in enumerate(got):
            if i < 2:
                self.assertEqualArray(tensors[i], g)
            else:
                name = names[i]

                if name == "arg_2_keys":
                    self.assertEqual(["tt1", "tt2"], g.tolist())
                    continue
                if name == "arg_3_keys":
                    self.assertEqual([], g.tolist())
                    continue
                if name == "arg_2_values":
                    expected = tensors[2]
                else:
                    expected = tensors[3]
                for k in range(len(expected)):
                    self.assertEqualArray(list(expected.values())[k], g[k])

    def test_mini_onnx_builder_dict_dict(self):
        import torch

        tensors = {
            "t1": np.array([1, 2], dtype=np.int64),
            "t2": torch.tensor([4, 5], dtype=torch.float32),
            "d1": {
                "tt1": np.array([-1, -2], dtype=np.int64),
                "tt2": torch.tensor([-4, -5], dtype=torch.float32),
            },
            "d2": {},
        }
        model = create_onnx_model_from_input_tensors(tensors)
        names = ["t1", "t2", "d1_keys", "d1_values", "d2_keys", "d2_values"]
        self.assertEqual(names, [o.name for o in model.graph.output])
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {})
        for i, g in enumerate(got):
            if i < 2:
                self.assertEqualArray(tensors[names[i]], g)
            else:
                name = names[i]

                if name == "d1_keys":
                    self.assertEqual(["tt1", "tt2"], g.tolist())
                    continue
                if name == "d2_keys":
                    self.assertEqual([], g.tolist())
                    continue
                if name == "d1_values":
                    expected = tensors["d1"]
                else:
                    expected = tensors["d2"]
                for k in range(len(expected)):
                    self.assertEqualArray(list(expected.values())[k], g[k])


if __name__ == "__main__":
    unittest.main(verbosity=2)
