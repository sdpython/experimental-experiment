import numpy as np
from onnx.reference.op_run import OpRun


class AddAdd(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, x, y, z):
        return (x + y + z,)


class MulMul(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, x, y, z):
        return (x * y * z,)


class AddMul(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, x, y, z, transposeMiddle=None):
        res = (x + y) * z
        if transposeMiddle:
            res = np.transpose(res, axes=[0, 2, 1, 3])
        return (res,)


class MulAdd(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, x, y, z, transposeMiddle=None):
        res = (x * y) + z
        if transposeMiddle:
            res = np.transpose(res, axes=[0, 2, 1, 3])
        return (res,)


class SubMul(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, x, y, z, negative=None):
        if negative:
            return ((y - x) * z,)
        return ((x - y) * z,)


class MulSub(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, x, y, z, negative=None):
        if negative:
            return (z - (x * y),)
        return ((x * y) - z,)


class AddSharedInput(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, x, y, z):
        return (x + y, x + z)


class MulSharedInput(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, x, y, z):
        return (x * y, x * z)
