from onnx.reference.op_run import OpRun


class SimplifiedLayerNormalization(OpRun):
    def _run(self, x, scale, bias=None, axis=None, epsilon=None, stash_type=None):
        xm = (x**2).mean(axis=axis, keepdims=1) + epsilon
        xq = xm ** (-0.5)
        return (x * xq, xq)
