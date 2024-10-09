import numpy as np
from onnx.reference.op_run import OpRun


class AveragePoolGrad(OpRun):
    def _run(
        self,
        out,
        auto_pad=None,
        ceil_mode=None,
        count_include_pad=None,
        kernel_shape=None,
        pads=None,
        strides=None,
    ):
        assert auto_pad is not None, "auto_pad is None"
        assert ceil_mode is not None, "ceil_mode is None"
        assert count_include_pad is not None, "count_include_pad is None"
        assert kernel_shape is not None, "kernel_shape is None"
        assert pads is not None, "pads is None"
        assert strides is not None, "strides is None"

        assert auto_pad == "NOTSET", f"Not implemented for autopad={auto_pad!r}"
        assert ceil_mode == 0, f"Not implemented for ceil_mode={ceil_mode!r}"
        assert (
            count_include_pad == 1
        ), f"Not implemented for count_include_pad={count_include_pad!r}"

        grad_shape = list(out.shape[:2])
        for i in range(len(kernel_shape)):
            d = (
                out.shape[i + 2] * strides[i]
                + kernel_shape[i]
                - 1
                + sum(pads[i * 2 : i * 2 + 2])
            )
            grad_shape.append(d)

        grad = np.zeros(tuple(grad_shape), dtype=out.dtype)
        scale = (1.0 / np.prod(kernel_shape)).astype(out.dtype)
        if len(grad_shape) == 4:
            # 2D
            for batch in range(grad.shape[0]):
                for channel in range(grad.shape[1]):
                    for i in range(out.shape[2]):
                        t = max(i * strides[0] - pads[0], 0)
                        b = min(i * strides[0] - pads[0] + kernel_shape[0], grad.shape[2])
                        for j in range(out.shape[3]):
                            le = max(j * strides[1] - pads[2], 0)
                            ri = min(
                                j * strides[1] - pads[2] + kernel_shape[1],
                                grad.shape[3],
                            )

                            grad[batch, channel, t:b, le:ri] += (
                                out[batch, channel, i, j] * scale
                            )
        else:
            raise NotImplementedError(
                f"AveragePoolGrad is not implemented for shape={out.shape}."
            )

        return (grad.astype(out.dtype),)
