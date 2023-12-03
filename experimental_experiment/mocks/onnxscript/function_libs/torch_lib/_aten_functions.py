from typing import Sequence

TENSOR = "TENSOR"


def aten_convolution(
    g: "GraphBuilder",  # noqa: F821
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
    if not hasattr(g.op, "domain") or not hasattr(g.op, "version"):
        raise TypeError(f"Unexpected type {type(g.op)} for op.")
    if transposed:
        raise NotImplementedError(
            f"aten_convolution does not support transposed={transposed}."
        )
    if output_padding and (min(output_padding) != 0 or max(output_padding) != 0):
        raise NotImplementedError(
            f"aten_convolution does not support output_padding={output_padding}."
        )
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
        weight_dim_0 = g.op.Shape(g, weight, start=0, end=1)
        bias_shape = g.op.Expand(weight_dim_0, g.op.Constant(g, value_ints=[1]))
        zero = g.op.CastLike(g, 0.0, input)
        bias = g.op.Expand(g, zero, bias_shape)

    # if Rank(input) != Rank(weight):
    #    input = op.Unsqueeze(input, op.Constant(value_ints=[0]))

    return g.op.Conv(
        g,
        input,
        weight,
        bias,
        strides=strides,
        pads=pads,
        group=groups,
        dilations=dilations,
    )
