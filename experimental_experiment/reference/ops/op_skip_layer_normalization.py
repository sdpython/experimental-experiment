from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_layer_normalization import _layer_normalization


class SkipLayerNormalization(OpRun):
    op_domain = "com.microsoft"

    def _run(self, x, skip, gamma=None, beta=None, bias=None, epsilon=None):
        add = x + skip
        if bias is not None:
            add = add + bias
        res = _layer_normalization(add, gamma, beta, axis=-1, epsilon=epsilon)
        return (*res, add)
