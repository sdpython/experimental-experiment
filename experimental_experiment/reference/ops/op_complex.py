import numpy as np
from onnx.reference.op_run import OpRun


class ToComplex(OpRun):
    op_domain = "ai.onnx.complex"

    def _run(self, x):
        assert x.shape[-1] in (
            1,
            2,
        ), f"Unexpected shape {x.shape}, it should a tensor (..., 2)"
        if x.shape[-1] == 1:
            return (x[..., 0] + 0j,)
        return (x[..., 0] + 1j * x[..., 1],)


class ComplexModule(OpRun):
    op_domain = "ai.onnx.complex"

    def _run(self, x):
        assert x.dtype in (
            np.complex64,
            np.complex128,
        ), f"Unexpected type {x.dtype}, it should a complex tensor"
        return (np.abs(x),)
