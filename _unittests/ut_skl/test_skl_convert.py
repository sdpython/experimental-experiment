import unittest
import numpy as np
import sklearn
from experimental_experiment.ext_test_case import ExtTestCase, hide_stdout, ignore_warnings
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.skl import to_onnx


class TestSklConvert(ExtTestCase):

    @ignore_warnings(DeprecationWarning)
    @hide_stdout
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
    @hide_stdout
    def test_logistic_regression(self):
        X = np.random.randn(10, 3)
        y = (np.random.randn(10, 1) >= 0.5).astype(int)
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

    @hide_stdout
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
