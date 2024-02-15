import onnxruntime  # noqa: F401
import unittest
import logging
import numpy
import onnx.defs
from onnx.reference import ReferenceEvaluator

from onnxruntime import InferenceSession

try:
    from onnxruntime import training
except ImportError:
    # onnxruntime not training
    training = None

try:
    from onnxruntime.capi._pybind_state import GradientGraphBuilder
except ImportError:
    GradientGraphBuilder = None
from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxMul, OnnxIdentity
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
from experimental_experiment.gradient.grad_helper import (
    random_feed,
    onnx_derivative,
    DerivativeOptions,
)
from experimental_experiment.gradient.loss_helper import add_loss_output
from experimental_experiment.gradient.ops import new_ops
from experimental_experiment.ext_test_case import ExtTestCase, ignore_warnings

opset = min(onnx.defs.onnx_opset_version() - 2, 18)


class TestGradHelper(ExtTestCase):
    def check_runtime(self, onx, name, atol=1e-5, verbose=False):
        feeds = random_feed(onx.graph.input)
        n = 0
        for _, v in feeds.items():
            if v.shape[0] > 5:
                n += 1
        if n == 0:
            raise AssertionError(f"No input with more than 5 rows: {feeds!r}.")
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        try:
            got = sess.run(None, feeds)
        except OrtFail as e:
            with open(f"fail_{name}.onnx", "wb") as f:
                f.write(onx.SerializeToString())
            raise AssertionError(
                "Unable to run onnx graph %r." % ("fail_%s.onnx" % name)
            ) from e
        oinf = ReferenceEvaluator(onx, new_ops=new_ops)
        pygot = oinf.run(None, feeds)
        output_names = [o.name for o in onx.graph.output]
        self.assertGreater(len(output_names), 0)
        for i, o in enumerate(output_names):
            self.assertEqualArray(got[i], pygot[i], atol=atol)
        if verbose:
            print(
                "%s - input=%r output=%r"
                % (
                    name,
                    [o.name for o in onx.graph.input],
                    [o.name for o in onx.graph.output],
                )
            )
            with open(f"verbose_{name}.onnx", "wb") as f:
                f.write(onx.SerializeToString())

    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger("skl2onnx")
        logger.setLevel(logging.WARNING)
        logging.basicConfig(level=logging.WARNING)
        ExtTestCase.setUpClass()

    @unittest.skipIf(training is None, reason="not training")
    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_grad_helper_keep_yield(self):
        opv = opset
        node = OnnxAdd(
            "X",
            numpy.array([1], dtype=numpy.float32),
            op_version=opv,
            output_names=["Y"],
        )
        onx = node.to_onnx(
            {"X": FloatTensorType([None, 10])},
            {"Y": FloatTensorType([None, 10])},
            target_opset=opv,
        )
        onx.ir_version = 9
        new_onx = onnx_derivative(onx, options=DerivativeOptions.KeepYieldOp)
        types = set(n.op_type for n in new_onx.graph.node)
        self.assertIn("YieldOp", types)
        with open(f"verbose_{'yield'}.onnx", "wb") as f:
            f.write(new_onx.SerializeToString())

    @unittest.skipIf(training is None, reason="not training")
    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_grad_helper(self):
        opv = opset
        node = OnnxAdd(
            "X",
            numpy.array([1], dtype=numpy.float32),
            op_version=opv,
            output_names=["Y"],
        )
        onx = node.to_onnx(
            {"X": FloatTensorType([None, 10])},
            {"Y": FloatTensorType([None, 10])},
            target_opset=opv,
        )
        onx.ir_version = 9
        new_onx = onnx_derivative(onx)
        self.assertEqual(new_onx.ir_version, onx.ir_version)
        out_names = [o.name for o in new_onx.graph.output]
        self.assertNotIn("Y", out_names)
        self.check_runtime(new_onx, "test_grad_helper")

    @unittest.skipIf(training is None, reason="not training")
    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_grad_helper_nooutput(self):
        opv = opset
        node = OnnxAdd(
            "X",
            numpy.array([1], dtype=numpy.float32),
            op_version=opv,
            output_names=["Y"],
        )
        onx = node.to_onnx(
            {"X": FloatTensorType([None, 10])},
            {"Y": FloatTensorType([None, 10])},
            target_opset=opv,
        )
        onx.ir_version = 9
        new_onx = onnx_derivative(onx, options=DerivativeOptions.KeepOutputs)
        self.check_runtime(new_onx, "test_grad_helper_nooutput")

    @unittest.skipIf(training is None, reason="not training")
    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_grad_helper_mul(self):
        opv = opset
        xi = OnnxIdentity("X", op_version=opv)
        node = OnnxMul(xi, xi, op_version=opv, output_names=["Y"])
        onx = node.to_onnx(
            {"X": FloatTensorType([None, 10])},
            {"Y": FloatTensorType([None, 10])},
            target_opset=opv,
        )
        onx.ir_version = 9
        new_onx = onnx_derivative(onx)
        self.check_runtime(new_onx, "test_grad_helper_mul")

    @unittest.skipIf(training is None, reason="not training")
    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_grad_helper_noweight(self):
        opv = opset
        node = OnnxAdd(
            "X",
            numpy.array([1], dtype=numpy.float32),
            op_version=opv,
            output_names=["Y"],
        )
        onx = node.to_onnx(
            {"X": FloatTensorType([None, 10])},
            {"Y": FloatTensorType([None, 10])},
            target_opset=opv,
        )
        onx.ir_version = 9
        new_onx = onnx_derivative(onx, weights=[])
        self.check_runtime(new_onx, "test_grad_helper_noweight")

    @unittest.skipIf(training is None, reason="not training")
    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_grad_helper_fillgrad(self):
        opv = opset
        node = OnnxAdd(
            "X",
            numpy.array([1], dtype=numpy.float32),
            op_version=opv,
            output_names=["Y"],
        )
        onx = node.to_onnx(
            {"X": FloatTensorType([None, 10])},
            {"Y": FloatTensorType([None, 10])},
            target_opset=opv,
        )
        self.assertRaise(
            lambda: onnx_derivative(
                onx, weights=[], options=DerivativeOptions.FillGrad
            ),
            AssertionError,
        )
        onx.ir_version = 9
        new_onx = onnx_derivative(
            onx,
            weights=[],
            options=(DerivativeOptions.FillGrad | DerivativeOptions.KeepOutputs),
        )
        input_names = set(i.name for i in new_onx.graph.input)
        self.assertNotIn("Y_grad", input_names)
        self.check_runtime(new_onx, "test_grad_helper_fillgrad", verbose=False)

    @ignore_warnings((DeprecationWarning, FutureWarning))
    def test_grad_helper_exc(self):
        opv = opset
        node = OnnxAdd(
            "X",
            numpy.array([1], dtype=numpy.float32),
            op_version=opv,
            output_names=["Y"],
        )
        onx = node.to_onnx(
            {"X": FloatTensorType([None, 10])},
            {"Y": FloatTensorType([None, 10])},
            target_opset=opv,
        )
        self.assertRaise(
            lambda: onnx_derivative(onx, weights=[], options=1), AssertionError
        )

    @unittest.skipIf(training is None, reason="not training")
    @unittest.skipIf(GradientGraphBuilder is None, reason="not recent")
    def test_grad_helper_loss(self):
        grad_file = "test_grad_helper_loss.onnx"
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0
        )
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        reg = LinearRegression()
        reg.fit(X, y)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X, target_opset=opset, black_op={"LinearRegressor"})
        onx_loss = add_loss_output(onx)
        text1 = onnx_simple_text_plot(onx_loss)
        new_onx = onnx_derivative(
            onx,
            options=DerivativeOptions.Loss,
            label="variable",
            loss="loss",
            path_name=grad_file,
        )
        text2 = onnx_simple_text_plot(new_onx)
        self.assertNotEqual(text1, text2)


if __name__ == "__main__":
    unittest.main()
