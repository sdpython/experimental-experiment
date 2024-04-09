from onnx.reference.op_run import OpRun


class AddAdd(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, x, y, z):
        return (x + y + z,)


class MulMul(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, x, y, z):
        return (x * y * z,)
