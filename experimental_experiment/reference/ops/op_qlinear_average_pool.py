from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_average_pool import AveragePool_19 as AveragePool
from onnx.reference.ops.op_dequantize_linear import DequantizeLinear_19 as DequantizeLinear
from onnx.reference.ops.op_quantize_linear import QuantizeLinear_19 as QuantizeLinear


class QLinearAveragePool(OpRun):
    op_domain = "com.microsoft"

    def _run(
        self,
        x,
        x_scale,
        x_zero_point,
        y_scale,
        y_zero_point,
        auto_pad=None,
        ceil_mode=None,
        channels_last=None,
        count_include_pad=None,
        kernel_shape=None,
        pads=None,
        strides=None,
    ):
        assert channels_last in (
            None,
            0,
        ), f"QLinearAveragePool not implemented if channels_last={channels_last}"
        dqx = DequantizeLinear.eval(x, x_scale, x_zero_point)
        y = AveragePool.eval(
            dqx,
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )
        qy = QuantizeLinear.eval(y, y_scale, y_zero_point)
        return (qy,)
