from onnx.reference.op_run import OpRun


class ReplaceZero(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, X, by=None, equal=None):
        x2 = X.copy().flatten()
        if equal:
            x2[x2 == 0] = by
        else:
            x2[x2 != 0] = by
        return (x2.reshape(X.shape),)
