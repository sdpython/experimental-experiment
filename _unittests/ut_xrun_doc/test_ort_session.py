import unittest
from typing import Dict, Optional, Tuple
import numpy as np
import onnx
import onnx.helper as oh
import torch
from torch._C import _from_dlpack
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_onnxruntime_training,
    requires_cuda,
)
from experimental_experiment.ort_session import (
    InferenceSessionForNumpy,
    InferenceSessionForTorch,
    investigate_onnxruntime_issue,
)

TFLOAT = onnx.TensorProto.FLOAT


class TestOrtSession(ExtTestCase):

    @classmethod
    def _range(cls, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return torch.from_numpy(x.reshape(tuple(shape)).astype(np.float32))

    @classmethod
    def _get_model(
        cls,
    ) -> Tuple[onnx.ModelProto, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["x", "y"], ["gggg"]),
                    oh.make_node("Add", ["gggg", "z"], ["final"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("x", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("y", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("z", TFLOAT, [None, None]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        onnx.checker.check_model(model)
        feeds = {"x": cls._range(5, 6), "y": cls._range(5, 6), "z": cls._range(5, 6)}
        return model, feeds, (feeds["x"] + feeds["y"] + feeds["z"],)

    def test_ort_value_dlpack_numpy(self):
        import onnxruntime as onnxrt
        from onnxruntime.capi import _pybind_state as C
        from onnxruntime.capi.onnxruntime_pybind11_state import OrtValue as C_OrtValue

        numpy_arr_input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input)
        self.assertEqual(numpy_arr_input.shape, tuple(ortvalue.shape()))
        ptr = ortvalue._ortvalue.data_ptr()

        dlp = ortvalue._ortvalue.to_dlpack()
        self.assertFalse(C.is_dlpack_uint8_tensor(dlp))
        ortvalue2 = C_OrtValue.from_dlpack(dlp, False)
        self.assertEqual(ptr, ortvalue2.data_ptr())
        new_array = ortvalue2.numpy()
        self.assertEqualArray(numpy_arr_input, new_array)

        dlp = ortvalue._ortvalue.__dlpack__()
        self.assertFalse(C.is_dlpack_uint8_tensor(dlp))
        ortvalue2 = C_OrtValue.from_dlpack(dlp, False)
        self.assertEqual(ptr, ortvalue2.data_ptr())
        new_array = ortvalue2.numpy()
        self.assertEqualArray(numpy_arr_input, new_array)

        device = ortvalue._ortvalue.__dlpack_device__()
        self.assertEqual((1, 0), device)

    def test_ort_value_dlpack_torch(self):
        from onnxruntime.capi.onnxruntime_pybind11_state import OrtValue as C_OrtValue

        torch_arr_input = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32
        )
        shape = torch_arr_input.shape
        ptr = torch_arr_input.data_ptr()
        dlp = torch_arr_input.__dlpack__()
        ortvalue = C_OrtValue.from_dlpack(dlp, False)
        self.assertEqual(shape, tuple(ortvalue.shape()))
        ptr2 = ortvalue.data_ptr()
        self.assertEqual(ptr, ptr2)

        tv = _from_dlpack(ortvalue.to_dlpack())
        self.assertEqual(tv.shape, shape)
        ptr3 = tv.data_ptr()
        self.assertEqual(ptr, ptr3)

    def test_numpy(self):
        model, feeds, expected = self._get_model()
        wrap = InferenceSessionForNumpy(model, providers="cpu")
        got = wrap.run(None, {k: v.numpy() for k, v in feeds.items()})
        self.assertIsInstance(got[0], np.ndarray)
        self.assertEqualArray(expected[0], got[0])

    def test_numpy_no_optimization(self):
        model, feeds, expected = self._get_model()
        wrap = InferenceSessionForNumpy(model, providers="cpu", graph_optimization_level=False)
        got = wrap.run(None, {k: v.numpy() for k, v in feeds.items()})
        self.assertIsInstance(got[0], np.ndarray)
        self.assertEqualArray(expected[0], got[0])

    def test_torch_guess_cpu(self):
        model, feeds, expected = self._get_model()
        wrap = InferenceSessionForTorch(model, providers="cpu", use_training_api=True)
        got = wrap.run(None, feeds)
        self.assertIsInstance(got[0], torch.Tensor)
        self.assertEqualArray(expected[0], got[0])

    @requires_onnxruntime_training(True)
    def test_torch_training_cpu(self):
        model, feeds, expected = self._get_model()
        wrap = InferenceSessionForTorch(model, providers="cpu", use_training_api=True)
        got = wrap.run(None, feeds)
        self.assertIsInstance(got[0], torch.Tensor)
        self.assertEqualArray(expected[0], got[0])

    def test_torch_notraining_cpu(self):
        model, feeds, expected = self._get_model()
        wrap = InferenceSessionForTorch(model, providers="cpu", use_training_api=False)
        got = wrap.run(None, feeds)
        self.assertIsInstance(got[0], torch.Tensor)
        self.assertEqualArray(expected[0], got[0])

    @requires_cuda()
    def test_torch_guess_cuda(self):
        model, feeds, expected = self._get_model()
        feeds = {k: v.to("cuda") for k, v in feeds.items()}
        expected = tuple(t.to("cuda") for t in expected)
        wrap = InferenceSessionForTorch(model, providers="cuda", use_training_api=True)
        got = wrap.run(None, feeds)
        self.assertIsInstance(got[0], torch.Tensor)
        self.assertEqualArray(expected[0], got[0])
        self.assertEqual(got[0].get_device(), 0)

    @requires_cuda()
    @requires_onnxruntime_training(True)
    def test_torch_training_cuda(self):
        model, feeds, expected = self._get_model()
        feeds = {k: v.to("cuda") for k, v in feeds.items()}
        expected = tuple(t.to("cuda") for t in expected)
        wrap = InferenceSessionForTorch(model, providers="cuda", use_training_api=True)
        got = wrap.run(None, feeds)
        self.assertIsInstance(got[0], torch.Tensor)
        self.assertEqualArray(expected[0], got[0])
        self.assertEqual(got[0].get_device(), 0)

    @requires_cuda()
    def test_torch_notraining_cuda(self):
        model, feeds, expected = self._get_model()
        feeds = {k: v.to("cuda") for k, v in feeds.items()}
        expected = tuple(t.to("cuda") for t in expected)
        wrap = InferenceSessionForTorch(model, providers="cuda", use_training_api=False)
        got = wrap.run(None, feeds)
        self.assertIsInstance(got[0], torch.Tensor)
        # The output is not necessarily on CUDA.
        self.assertEqualArray(expected[0].cpu(), got[0].cpu())
        # self.assertEqual(got[0].get_device(), 0)

    @hide_stdout()
    def test_investigate_onnxruntime_issue_torch(self):
        model, feeds, _expected = self._get_model()
        investigate_onnxruntime_issue(
            model,
            feeds=feeds,
            verbose=10,
            dump_filename="test_investigate_onnxruntime_issue_torch.onnx",
        )

    @hide_stdout()
    def test_investigate_onnxruntime_issue_numpy(self):
        model, feeds, _expected = self._get_model()
        feeds = {k: v.numpy() for k, v in feeds.items()}
        investigate_onnxruntime_issue(
            model,
            feeds=feeds,
            verbose=10,
            dump_filename="test_investigate_onnxruntime_issue_numpy.onnx",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
