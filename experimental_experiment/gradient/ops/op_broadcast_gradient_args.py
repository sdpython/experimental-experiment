import numpy
from onnx.reference.op_run import OpRun


class BroadcastGradientArgs(OpRun):
    op_domain = "com.microsoft"

    def _run(self, a_shape, b_shape):
        A_dims = a_shape
        B_dims = b_shape
        a_size = len(a_shape)
        b_size = len(b_shape)

        ndim = max(a_size, b_size)

        i = a_size - 1
        j = b_size - 1
        k = ndim - 1

        a_axes = []
        b_axes = []

        while i >= 0 and j >= 0:
            A_dim = A_dims[i]
            B_dim = B_dims[j]

            if A_dim != B_dim:
                if A_dim == 1:
                    a_axes.append(k)
                elif B_dim == 1:
                    b_axes.append(k)
                else:
                    a = A_dims[:a_size]
                    b = B_dims[:b_size]
                    raise RuntimeError(
                        f"Broadcast is not possible between inputs of shapes: {a} and {b}."
                    )
            i -= 1
            j -= 1
            k -= 1

        if i < 0:
            while k >= 0:
                a_axes.append(k)
                k -= 1
        else:
            while k >= 0:
                b_axes.append(k)
                k -= 1

        return (
            numpy.array(a_axes, dtype=numpy.int64),
            numpy.array(b_axes, dtype=numpy.int64),
        )
