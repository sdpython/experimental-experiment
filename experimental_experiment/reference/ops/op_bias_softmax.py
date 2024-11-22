import numpy as np
from onnx.reference.op_run import OpRun


class BiasSoftmax(OpRun):
    op_domain = "com.microsoft"

    def _run(self, x, y, axis=None, is_inner_broadcast=None):  # type: ignore
        assert (
            is_inner_broadcast == 0
        ), f"Not implemented for is_inner_broadcast={is_inner_broadcast}"
        z = x + y
        tmp = z - z.max(axis=axis, keepdims=1)  # type: ignore
        w = np.exp(tmp)
        w /= w.sum(axis=axis, keepdims=1)  # type: ignore
        return (w.astype(x.dtype),)
