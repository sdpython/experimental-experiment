from onnx.reference.op_run import OpRun


class NegXplus1(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, X):
        return ((1 - X).astype(X.dtype),)
