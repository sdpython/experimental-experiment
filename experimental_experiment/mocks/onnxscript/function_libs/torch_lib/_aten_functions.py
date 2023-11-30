from typing import Sequence
from onnx.defs import OpSchema
from ...onnx_opset import Opset
from ...values import ParamSchema, TypeConstraint, OnnxFunction

TENSOR = "TENSOR"


def aten_convolution(
    op: Opset,
    input: TENSOR,
    weight: TENSOR,
    bias: TENSOR = None,
    stride: Sequence[int] = (1,),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    transposed: bool = False,
    output_padding: Sequence[int] = (0,),
    groups: int = 1,
) -> TENSOR:
    if not isinstance(padding, Sequence):
        padding = (padding, padding)
    pads = [*padding, *padding]

    if not isinstance(dilation, Sequence):
        dilation = (dilation, dilation)
    dilations = list(dilation)

    if not isinstance(stride, Sequence):
        stride = (stride, stride)
    strides = list(stride)

    if bias is None:
        weight_dim_0 = op.Shape(weight, start=0, end=1)
        bias_shape = op.Expand(weight_dim_0, op.Constant(value_ints=[1]))
        zero = op.CastLike(0.0, input)
        bias = op.Expand(zero, bias_shape)

    # if Rank(input) != Rank(weight):
    #    input = op.Unsqueeze(input, op.Constant(value_ints=[0]))

    return op.Conv(
        input,
        weight,
        bias,
        strides=strides,
        pads=pads,
        group=groups,
        dilations=dilations,
    )
