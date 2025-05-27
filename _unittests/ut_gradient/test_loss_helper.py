import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_onnxruntime_training,
    requires_sklearn,
)
import numpy
import onnx.defs
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss

try:
    from skl2onnx.algebra.onnx_ops import OnnxIdentity
    from skl2onnx.common.data_types import DoubleTensorType
    from skl2onnx import to_onnx
except ImportError:
    OnnxIdentity = None
from onnx.reference import ReferenceEvaluator
from experimental_experiment.helpers import pretty_onnx
from experimental_experiment.gradient.loss_helper import (
    add_loss_output,
    get_train_initializer,
)

opset = min(18, onnx.defs.onnx_opset_version() - 2)


class TestOrtTraining(ExtTestCase):
    @unittest.skipIf(OnnxIdentity is None, "sklearn-onnx not recent enough")
    @requires_onnxruntime_training()
    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_add_loss_output_reg(self):
        X, y = make_regression(100, n_features=10, bias=2)  # pylint: disable=W0632
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset, black_op={"LinearRegressor"})
        onx_loss = add_loss_output(onx)
        oinf = ReferenceEvaluator(onx_loss)
        output = oinf.run(None, {"X": X_test, "label": y_test.reshape((-1, 1))})
        loss = output[0]
        skl_loss = mean_squared_error(reg.predict(X_test), y_test)
        self.assertLess(numpy.abs(skl_loss - loss[0, 0]), 1e-5)

    @unittest.skipIf(OnnxIdentity is None, "sklearn-onnx not recent enough")
    @requires_onnxruntime_training()
    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_add_loss_output_reg_l1(self):
        X, y = make_regression(100, n_features=10, bias=2)  # pylint: disable=W0632
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset, black_op={"LinearRegressor"})
        onx_loss = add_loss_output(onx, "l1")
        oinf = ReferenceEvaluator(onx_loss)
        output = oinf.run(None, {"X": X_test, "label": y_test.reshape((-1, 1))})
        loss = output[0]
        skl_loss = mean_squared_error(reg.predict(X_test), y_test)
        self.assertLess(numpy.abs(skl_loss - loss[0, 0]), 1e-2)

    @unittest.skipIf(OnnxIdentity is None, "sklearn-onnx not recent enough")
    @requires_onnxruntime_training()
    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_get_train_initializer(self):
        X, y = make_regression(100, n_features=10, bias=2)  # pylint: disable=W0632
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset, black_op={"LinearRegressor"})
        inits = get_train_initializer(onx)
        self.assertEqual({"intercept", "coef"}, set(inits))

    @unittest.skipIf(OnnxIdentity is None, "sklearn-onnx not recent enough")
    @requires_onnxruntime_training()
    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_add_log_loss(self):
        ide = OnnxIdentity("X", op_version=opset, output_names=["Y"])
        onx = ide.to_onnx(
            inputs={"X": DoubleTensorType()},
            outputs={"Y": DoubleTensorType()},
            target_opset=opset,
        )
        onx_loss = add_loss_output(onx, "log", eps=1e-6)
        x1 = numpy.array([0, 0, 0.2, 0.5, 0.8, 1, 1])
        X = numpy.vstack([1 - x1, x1]).T.astype(numpy.float64)
        y = numpy.array([0, 1, 0, 1, 1, 1, 0], dtype=numpy.int64)
        oinf = ReferenceEvaluator(onx_loss)
        output = oinf.run(None, {"X": X, "label": y.reshape((-1, 1))})
        loss = output[0]
        eps = 1e-6
        xp = numpy.maximum(eps, numpy.minimum(1 - eps, X[:, 1]))
        skl_loss = log_loss(y, xp)
        self.assertLess(numpy.abs(skl_loss - loss[0, 0]), 1e-5)

    @unittest.skipIf(OnnxIdentity is None, "sklearn-onnx not recent enough")
    @requires_onnxruntime_training()
    @ignore_warnings((DeprecationWarning, FutureWarning))
    @requires_sklearn("1.6.0")
    def test_add_loss_output_cls(self):
        X, y = make_classification(100, n_features=10)  # pylint: disable=W0632
        X = X.astype(numpy.float32)
        y = y.astype(numpy.int64)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        reg = LogisticRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(
            reg,
            X_train,
            target_opset=opset,
            black_op={"LinearClassifier"},
            options={"zipmap": False},
        )
        eps = 1e-6
        onx_loss = add_loss_output(onx, "log", output_index="probabilities", eps=eps)
        try:
            text = pretty_onnx(onx_loss)
        except RuntimeError:
            text = ""
        if text:
            self.assertIn("Clip(probabilities", text)

        oinf = ReferenceEvaluator(onx_loss)
        output = oinf.run(None, {"X": X_test, "label": y_test.reshape((-1, 1))})
        loss = output[0]
        xp = numpy.maximum(eps, numpy.minimum(1 - eps, reg.predict_proba(X_test)))
        skl_loss = log_loss(y_test, xp)
        self.assertLess(numpy.abs(skl_loss - loss[0, 0]), 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
