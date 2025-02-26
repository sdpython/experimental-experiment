import unittest
import numpy as np
import sklearn
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    ignore_warnings,
    skipif_ci_windows,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.skl import to_onnx
from experimental_experiment.torch_interpreter import to_onnx as tto_onnx


class TestSklConvert(ExtTestCase):

    @ignore_warnings(DeprecationWarning)
    @hide_stdout()
    def test_linear_regression(self):
        X = np.random.randn(10, 3)
        y = np.random.randn(10, 1)
        lr = sklearn.linear_model.LinearRegression()
        lr.fit(X, y)
        onx = to_onnx(lr, verbose=10)
        self.print_model(onx)
        ref = ExtendedReferenceEvaluator(onx)
        self.assertEqualArray(
            lr.predict(X), ref.run(None, {ref.input_names[0]: X})[0], atol=1e-5
        )

    @ignore_warnings(DeprecationWarning)
    @hide_stdout()
    @skipif_ci_windows("not working")
    def test_logistic_regression(self):
        X = np.random.randn(20, 3)
        y = (np.random.randn(20, 1) >= 0.5).astype(int)
        lr = sklearn.linear_model.LogisticRegression()
        lr.fit(X, y)
        onx = to_onnx(lr, verbose=10)
        self.print_model(onx)
        ref = ExtendedReferenceEvaluator(onx)
        self.assertEqualAny(
            [lr.predict(X), lr.predict_proba(X)],
            ref.run(None, {ref.input_names[0]: X}),
            atol=1e-5,
        )

    @hide_stdout()
    def test_input_names(self):
        X = np.random.randn(10, 3)
        y = np.random.randn(10, 1)
        lr = sklearn.linear_model.LinearRegression()
        lr.fit(X, y)
        onx = to_onnx(lr, verbose=10, input_names=["S"])
        self.assertEqual(onx.graph.input[0].name, "S")
        ref = ExtendedReferenceEvaluator(onx)
        self.assertEqualArray(
            lr.predict(X), ref.run(None, {ref.input_names[0]: X})[0], atol=1e-5
        )

    def test_nan_to_euclidean(self):
        import torch

        class NanEuclidean(torch.nn.Module):

            def forward(self, X, Y, squared=False):
                X = X.clone()
                Y = Y.clone()
                missing_X = torch.isnan(X)
                missing_Y = torch.isnan(Y)

                # set missing values to zero
                X[missing_X] = 0
                Y[missing_Y] = 0

                # Adjust distances for missing values
                XX = X * X
                YY = Y * Y

                distances = -2 * X @ Y.T + XX.sum(1, keepdim=True) + YY.sum(1, keepdim=True).T

                distances -= XX @ missing_Y.to(X.dtype).T
                distances -= missing_X.to(X.dtype) @ YY.T

                distances = torch.clip(distances, 0, None)

                present_X = 1 - missing_X.to(X.dtype)
                present_Y = ~missing_Y
                present_count = present_X @ present_Y.to(X.dtype).T
                distances[present_count == 0] = torch.nan
                # avoid divide by zero
                present_count = torch.maximum(
                    torch.tensor([1], dtype=present_count.dtype), present_count
                )
                distances /= present_count
                distances *= X.shape[1]

                if not squared:
                    distances = distances.sqrt()

                return distances

        model = NanEuclidean()
        X = torch.randn((5, 2))
        Y = torch.randn((5, 2))
        for i in range(5):
            X[i, i % 2] = torch.nan
        for i in range(4):
            Y[i + 1, i % 2] = torch.nan
        d1 = sklearn.metrics.nan_euclidean_distances(X.numpy(), Y.numpy())
        d2 = model(X, Y)
        self.assertEqualArray(d1, d2, atol=1e-4)

        onx = tto_onnx(
            model, (X, Y), dynamic_shapes=({0: "batch", 1: "dx"}, {0: "batch", 1: "dy"})
        )
        # self.print_model(onx)
        self.dump_onnx("test_nan_to_euclidean.onnx", onx)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, dict(zip(ref.input_names, [X.numpy(), Y.numpy()])))
        self.assertEqualArray(d1, got[0], atol=2e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
