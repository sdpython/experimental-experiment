import numpy as np
from onnx.reference.op_run import OpRun


class TriMatrix(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, shape, csts):
        lower, diag, upper = list(csts)
        dtype = csts.dtype
        mat = np.empty(tuple(shape), dtype=dtype)
        i = np.arange(shape[0], dtype=np.int32).reshape((-1, 1))
        j = np.arange(shape[1], dtype=np.int32).reshape((1, -1))
        mat[i > j] = lower
        mat[i < j] = upper
        mat[i == j] = diag
        return (mat,)
