import os
import pickle
import unittest
from typing import Optional
import numpy as np
from onnx import ModelProto, TensorProto
from onnx.checker import check_model
import onnx.helper as oh
import onnx.numpy_helper as onh
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_onnxruntime_training,
    hide_stdout,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator, OrtEval

TFLOAT = TensorProto.FLOAT


class TestOrtEval(ExtTestCase):
    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def _get_model(self) -> ModelProto:
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [32, 128]),
                    oh.make_tensor_value_info("Y", TFLOAT, [3, 5, 128, 64]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 32, 64])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
                ],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        check_model(model)
        return model

    @ignore_warnings(DeprecationWarning)
    @hide_stdout()
    def test_ort_eval(self):
        model = self._get_model()

        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model, verbose=10)
        expected, out, _ = self.capture(lambda: ref.run(None, feeds)[0])
        self.assertIn("Reshape(xm, shape3) -> Z", out)

        ort_eval = OrtEval(model, verbose=10)
        got, out, _ = self.capture(lambda: ort_eval.run(None, feeds)[0])
        self.assertEqualArray(expected, got, atol=1e-4)
        self.assertIn("Reshape(xm, shape3) -> Z", out)

    @ignore_warnings(DeprecationWarning)
    @requires_onnxruntime_training()
    @hide_stdout()
    def test_ort_eval_dlpack(self):
        import torch

        model = self._get_model()

        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model, verbose=10)
        expected, out, _ = self.capture(lambda: ref.run(None, feeds)[0])
        self.assertIn("Reshape(xm, shape3) -> Z", out)

        feeds = {k: torch.Tensor(v) for k, v in feeds.items()}
        ort_eval = OrtEval(model, verbose=10)
        got, out, _ = self.capture(lambda: ort_eval.run_dlpack(None, feeds)[0])
        self.assertEqualArray(expected, got, atol=1e-4)
        self.assertIn("Reshape(xm, shape3) -> Z", out)

    @ignore_warnings(DeprecationWarning)
    @hide_stdout()
    def test_ort_eval_whole(self):
        model = self._get_model()

        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model, verbose=10)
        expected, out, _ = self.capture(lambda: ref.run(None, feeds)[0])
        self.assertIn("Reshape(xm, shape3) -> Z", out)

        ort_eval = OrtEval(model, verbose=10, whole=True)
        got, out, _ = self.capture(lambda: ort_eval.run(None, feeds)[0])
        self.assertEqualArray(expected, got, atol=1e-4)
        self.assertNotIn("Reshape(xm, shape3) -> Z", out)

    @ignore_warnings(DeprecationWarning)
    @requires_onnxruntime_training()
    @hide_stdout()
    def test_ort_eval_dlpack_whole(self):
        import torch

        model = self._get_model()

        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model, verbose=10)
        expected, out, _ = self.capture(lambda: ref.run(None, feeds)[0])
        self.assertIn("Reshape(xm, shape3) -> Z", out)

        feeds = {k: torch.Tensor(v) for k, v in feeds.items()}
        ort_eval = OrtEval(model, verbose=10, whole=True)
        got, out, _ = self.capture(lambda: ort_eval.run_dlpack(None, feeds)[0])
        self.assertEqualArray(expected, got, atol=1e-4)
        self.assertNotIn("Reshape(xm, shape3) -> Z", out)

    @ignore_warnings(DeprecationWarning)
    @hide_stdout()
    def test_ort_eval_incremental(self):
        model = self._get_model()

        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model, verbose=10)
        expected, out, _ = self.capture(lambda: ref.run(None, feeds)[0])
        self.assertIn("Reshape(xm, shape3) -> Z", out)

        ort_eval = OrtEval(model, verbose=10, incremental=True)
        got, out, _ = self.capture(lambda: ort_eval.run(None, feeds)[0])
        self.assertEqualArray(expected, got, atol=1e-4)
        self.assertIn("Reshape(xm, shape3) -> Z", out)

    @ignore_warnings(DeprecationWarning)
    @requires_onnxruntime_training()
    @hide_stdout()
    def test_ort_eval_dlpack_incremental(self):
        import torch

        model = self._get_model()

        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model, verbose=10)
        expected, out, _ = self.capture(lambda: ref.run(None, feeds)[0])
        self.assertIn("Reshape(xm, shape3) -> Z", out)

        feeds = {k: torch.Tensor(v) for k, v in feeds.items()}
        ort_eval = OrtEval(model, verbose=10, incremental=True)
        got, out, _ = self.capture(lambda: ort_eval.run_dlpack(None, feeds)[0])
        self.assertEqualArray(expected, got, atol=1e-4)
        self.assertIn("Reshape(xm, shape3) -> Z", out)

    @ignore_warnings(DeprecationWarning)
    @requires_onnxruntime_training()
    def test_debug_pkl(self):
        model = "dump_sdpa_dis_llama/dort-llama-sdpa-custom-no__1.onnx"
        inputs = "dump_sdpa_dis_llama/dort-llama-sdpa-custom-no__1.txt.pkl"
        if not os.path.exists(model) or not os.path.exists(inputs):
            raise unittest.SkipTest(f"Unable to find {model!r} and {inputs!r}.")
        with open(inputs, "rb") as f:
            feeds = pickle.load(f)
        input_names, values, output_names = feeds

        from onnxruntime.training.ortmodule.torch_cpp_extensions import (
            aten_op_executor,
        )
        from onnxruntime.capi import _pybind_state as _C

        _C.register_aten_op_executor(
            str(aten_op_executor.is_tensor_argument_address()),
            str(aten_op_executor.execute_aten_operator_address()),
        )

        ort_eval = OrtEval(
            model,
            providers="CUDA",
            verbose=11,
            # whole=True,
            # optimized_model_filepath=model + ".optimized.onnx",
            incremental=True,
        )
        # ort_eval.run_dlpack(
        #    None, dict(zip(input_names, [t.detach().cpu().numpy() for t in values]))
        # )
        ort_eval.run_dlpack(None, dict(zip(input_names, values)))

    def test_local_function(self):
        new_domain = "custom"

        linear_regression = oh.make_function(
            new_domain,
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [
                oh.make_node("MatMul", ["x", "a"], ["xa"]),
                oh.make_node("Add", ["xa", "b"], ["y"]),
            ],
            [oh.make_opsetid("", 14)],
            [],
        )

        graph = oh.make_graph(
            [
                oh.make_node("LinearRegression", ["X", "A", "B"], ["Y1"], domain=new_domain),
                oh.make_node("Abs", ["Y1"], ["Y"]),
            ],
            "example",
            [
                oh.make_tensor_value_info("X", TFLOAT, [None, None]),
                oh.make_tensor_value_info("A", TFLOAT, [None, None]),
                oh.make_tensor_value_info("B", TFLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Y", TFLOAT, None)],
        )

        onnx_model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            functions=[linear_regression],
            ir_version=10,
        )
        feeds = {
            "X": np.random.randn(3, 3).astype(np.float32),
            "A": np.random.randn(3, 3).astype(np.float32),
            "B": np.random.randn(3, 3).astype(np.float32),
        }
        ref = ExtendedReferenceEvaluator(onnx_model)
        ort_eval = OrtEval(onnx_model)
        expected = ref.run(None, feeds)
        got = ort_eval.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
