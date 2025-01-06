import numpy as np


class QuantizedTensor:
    """
    Quantizes a vector in range [0, 255].

    :param tensor: original tensor
    """

    def __init__(self, tensor):
        _min = tensor.min()
        _max = tensor.max()
        _min = min(_min, 0)
        _max = max(_max, 0)
        qmin = 0
        qmax = 255

        self.scale_ = np.array((_max - _min) / (qmax - qmin), dtype=tensor.dtype)
        initial_zero_point = qmin - _min / self.scale_
        self.zero_point_ = np.array(
            int(max(qmin, min(qmax, initial_zero_point))), dtype=np.uint8
        )
        self.quantized_ = np.maximum(
            0, np.minimum(qmax, (tensor / self.scale_).astype(int) + self.zero_point_)
        ).astype(self.zero_point_.dtype)

    @property
    def shape(self):
        "accessor"
        return self.quantized_.shape

    @property
    def scale(self):
        "accessor"
        return self.scale_

    @property
    def zero_point(self):
        "accessor"
        return self.zero_point_

    @property
    def qtensor(self):
        "accessor"
        return self.quantized_
