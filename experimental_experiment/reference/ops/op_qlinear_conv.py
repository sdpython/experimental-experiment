from typing import Tuple
from onnx.defs import OpSchema
from onnx.helper import make_attribute
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_conv import Conv
from onnx.reference.ops.op_dequantize_linear import DequantizeLinear_19 as DequantizeLinear
from onnx.reference.ops.op_quantize_linear import QuantizeLinear_19 as QuantizeLinear


def _switch_dims_nchw_nhwc(dims: Tuple[int, ...], from_nchw_to_nhwc: bool):
    if len(dims) == 4:
        if from_nchw_to_nhwc:
            return (dims[0], *dims[2:], dims[1])
        return (dims[0], dims[-1], *dims[1:-1])
    if len(dims) == 3:
        if from_nchw_to_nhwc:
            return (*dims[1:], dims[0])
        return (dims[-1], *dims[:-1])
    raise NotImplementedError(f"Unable to process shape={dims}")


class QLinearConv(OpRun):
    op_domain = "com.microsoft"

    op_schema = OpSchema(
        "QLinearConv",
        "com.microsoft",
        1,
        inputs=[
            OpSchema.FormalParameter("x", "T"),
            OpSchema.FormalParameter("x_scale", "T"),
            OpSchema.FormalParameter("x_zero_point", "T1"),
            OpSchema.FormalParameter("w", "T"),
            OpSchema.FormalParameter("w_scale", "T"),
            OpSchema.FormalParameter("w_zero_point", "T2"),
            OpSchema.FormalParameter("y_scale", "T"),
            OpSchema.FormalParameter("y_zero_point", "T3"),
            OpSchema.FormalParameter(
                "B", "T3", param_option=OpSchema.FormalParameterOption.Optional
            ),
        ],
        outputs=[OpSchema.FormalParameter("y", "T3")],
        type_constraints=[
            ("T", ["tensor(float)"], ""),
            ("T1", ["tensor(int8)", "tensor(uint8)"], ""),
            ("T2", ["tensor(int8)", "tensor(uint8)"], ""),
            ("T3", ["tensor(int8)", "tensor(uint8)"], ""),
        ],
        attributes=[
            OpSchema.Attribute("auto_pad", make_attribute("auto_pad", "NOTSET"), ""),
            OpSchema.Attribute("kernel_shape", OpSchema.AttrType.INTS, "", required=False),
            OpSchema.Attribute("dilations", OpSchema.AttrType.INTS, "", required=False),
            OpSchema.Attribute("strides", OpSchema.AttrType.INTS, "", required=False),
            OpSchema.Attribute("pads", OpSchema.AttrType.INTS, "", required=False),
            OpSchema.Attribute("group", make_attribute("group", 1), ""),
            OpSchema.Attribute("channels_last", make_attribute("channels_last", 0), ""),
        ],
    )

    def _run(
        self,
        x,
        x_scale,
        x_zero_point,
        w,
        w_scale,
        w_zero_point,
        y_scale,
        y_zero_point,
        B=None,
        auto_pad=None,
        channels_last=None,
        dilations=None,
        group=None,
        kernel_shape=None,
        pads=None,
        strides=None,
    ):
        dqx = DequantizeLinear.eval(x, x_scale, x_zero_point)
        dqw = DequantizeLinear.eval(w, w_scale, w_zero_point)
        if channels_last:
            dqx = dqx.reshape(_switch_dims_nchw_nhwc(x.shape, False))
        dqb = (
            DequantizeLinear.eval(B, x_scale * w_scale, 0).astype(dqx.dtype)
            if B is not None
            else None
        )
        y = Conv.eval(
            dqx,
            dqw,
            dqb,
            auto_pad=auto_pad,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )
        if channels_last:
            y = y.reshape(_switch_dims_nchw_nhwc(y.shape, True))
        qy = QuantizeLinear.eval(y, y_scale, y_zero_point)
        return (qy,)
