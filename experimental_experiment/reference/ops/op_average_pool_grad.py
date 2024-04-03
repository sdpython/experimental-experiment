import numpy as np
from onnx import TensorProto
import onnx.helper as oh
from onnx.reference.op_run import OpRun


class AveragePoolGrad(OpRun):

    def _run(
        self,
        X,
        auto_pad=None,
        ceil_mode=None,
        count_include_pad=None,
        kernel_shape=None,
        pads=None,
        strides=None,
    ):
        assert auto_pad is not None, "auto_pad is None"
        assert ceil_mode is not None, "ceil_mode is None"
        assert count_include_pad is not None, "count_include_pad is None"
        assert kernel_shape is not None, "kernel_shape is None"
        assert pads is not None, "pads is None"
        assert strides is not None, "strides is None"

        if not hasattr(self, "_cache"):
            self._cache = {}

        key = (
            auto_pad,
            ceil_mode,
            count_include_pad,
            tuple(kernel_shape),
            tuple(pads),
            tuple(strides),
            len(X.shape),
        )
        if key not in self._cache:
            from onnxruntime import InferenceSession

            TFLOAT = TensorProto.FLOAT

            model = oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node(
                            "AveragePoolGrad",
                            ["X"],
                            ["Y"],
                            auto_pad=auto_pad,
                            ceil_mode=ceil_mode,
                            count_include_pad=count_include_pad,
                            kernel_shape=kernel_shape,
                            pads=pads,
                            strides=strides,
                        )
                    ],
                    "single",
                    [oh.make_tensor_value_info("X", TFLOAT, [None] * len(X.shape))],
                    [oh.make_tensor_value_info("Y", TFLOAT, [None] * len(X.shape))],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
                ir_version=9,
            )

            sess = InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            self._cache[key] = lambda x, _sess=sess: _sess.run(
                None, {"X": x.astype(np.float32)}
            )[0]

        fct = self._cache[key]
        y = fct(X)
        assert set(y.shape) != {0}, (
            f"Issue with X={X.dtype}|{X.shape}, and key={key}",
            f"output is y={y.dtype}|{y.shape}",
        )
        return (y.astype(X.dtype),)
