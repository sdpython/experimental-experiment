import unittest
import numpy as np
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.mini_onnx_builder import (
    create_onnx_model_from_input_tensors,
    create_input_tensors_from_onnx_model,
    MiniOnnxBuilder,
)
from experimental_experiment.helpers import string_type


class TestMiniOnnxBuilder(ExtTestCase):
    def test_mini_onnx_builder_sequence(self):
        builder = MiniOnnxBuilder()
        builder.append_output_sequence("name", [np.array([6, 7])])
        onx = builder.to_onnx()
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {})
        self.assertEqualAny([np.array([6, 7])], got)

    def test_mini_onnx_builder(self):
        import torch

        data = [
            (
                np.array([1, 2], dtype=np.int64),
                torch.tensor([4, 5], dtype=torch.float32),
                {
                    "tt1": np.array([-1, -2], dtype=np.int64),
                    "tt2": torch.tensor([-4, -5], dtype=torch.float32),
                },
                {},
            ),
            {
                "t1": np.array([1, 2], dtype=np.int64),
                "t2": torch.tensor([4, 5], dtype=torch.float32),
                "d1": {
                    "tt1": np.array([-1, -2], dtype=np.int64),
                    "tt2": torch.tensor([-4, -5], dtype=torch.float32),
                },
                "d2": {},
            },
            (
                np.array([1, 2], dtype=np.int64),
                torch.tensor([4, 5], dtype=torch.float32),
                (
                    np.array([-1, -2], dtype=np.int64),
                    torch.tensor([-4, -5], dtype=torch.float32),
                ),
                tuple(),
            ),
            {
                "t1": np.array([1, 2], dtype=np.int64),
                "t2": torch.tensor([4, 5], dtype=torch.float32),
                "l1": (
                    np.array([-1, -2], dtype=np.int64),
                    torch.tensor([-4, -5], dtype=torch.float32),
                ),
                "l2": tuple(),
            },
            # nested
            (
                {
                    "t1": np.array([1, 2], dtype=np.int64),
                    "t2": torch.tensor([4, 5], dtype=torch.float32),
                    "l1": (
                        np.array([-1, -2], dtype=np.int64),
                        torch.tensor([-4, -5], dtype=torch.float32),
                    ),
                    "l2": tuple(),
                },
                (
                    np.array([1, 2], dtype=np.int64),
                    torch.tensor([4, 5], dtype=torch.float32),
                    (
                        np.array([-1, -2], dtype=np.int64),
                        torch.tensor([-4, -5], dtype=torch.float32),
                    ),
                    tuple(),
                ),
            ),
            # simple
            np.array([1, 2], dtype=np.int64),
            torch.tensor([4, 5], dtype=torch.float32),
            (np.array([1, 2], dtype=np.int64), torch.tensor([4, 5], dtype=torch.float32)),
            [np.array([1, 2], dtype=np.int64), torch.tensor([4, 5], dtype=torch.float32)],
            {
                "t1": np.array([1, 2], dtype=np.int64),
                "t2": torch.tensor([4, 5], dtype=torch.float32),
            },
            (
                np.array([1, 2], dtype=np.int64),
                torch.tensor([4, 5], dtype=torch.float32),
                [
                    np.array([-1, -2], dtype=np.int64),
                    torch.tensor([-4, -5], dtype=torch.float32),
                ],
                [],
            ),
            {
                "t1": np.array([1, 2], dtype=np.int64),
                "t2": torch.tensor([4, 5], dtype=torch.float32),
                "l1": [
                    np.array([-1, -2], dtype=np.int64),
                    torch.tensor([-4, -5], dtype=torch.float32),
                ],
                "l2": [],
            },
        ]

        for inputs in data:
            with self.subTest(types=string_type(inputs)):
                model = create_onnx_model_from_input_tensors(inputs)
                restored = create_input_tensors_from_onnx_model(model)
                self.assertEqualAny(inputs, restored)


if __name__ == "__main__":
    unittest.main(verbosity=2)