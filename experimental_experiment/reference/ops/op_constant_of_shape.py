import numpy as np
from onnx.reference.op_run import OpRun

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


class ConstantOfShape(OpRun):
    @staticmethod
    def _process(value):
        if (
            value is not None
            and ml_dtypes is not None
            and value.dtype == (np.uint16, [("bfloat16", "<u2")])
        ):
            value = value.view(ml_dtypes.bfloat16)
        cst = value[0] if isinstance(value, np.ndarray) and value.size > 0 else value
        if isinstance(value, np.ndarray):
            if not value.shape:
                cst = value
            elif value.size > 0:
                cst = value.ravel()[0]
            else:
                raise ValueError(f"Unexpected fill_value={value!r}")
        if isinstance(cst, bool):
            cst = np.bool_(cst)
        elif isinstance(cst, int):
            cst = np.int64(cst)
        elif isinstance(cst, float):
            cst = np.float64(cst)
        elif cst is None:
            cst = np.float32(0)
        if ml_dtypes is not None and isinstance(cst, ml_dtypes.bfloat16):
            return cst
        if not isinstance(
            cst,
            (
                np.float16,
                np.float32,
                np.float64,
                np.int64,
                np.int32,
                np.int16,
                np.int8,
                np.uint64,
                np.uint32,
                np.uint16,
                np.uint8,
                np.bool_,
            ),
        ):
            raise TypeError(f"value must be a real not {type(cst)}")
        return cst

    def _run(self, data, value=None):
        cst = self._process(value)
        try:
            res = np.full(tuple(data), cst)
        except TypeError as e:
            raise RuntimeError(
                f"Unable to create a constant of shape "
                f"{data!r} with value {cst!r} "
                f"(raw value={value!r})."
            ) from e
        return (res,)
