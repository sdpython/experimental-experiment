from typing import List, Sequence
import numpy as np
from .graph_builder import GraphBuilder

T = str


def aten_abs(g: GraphBuilder, outputs: List[str], x: T) -> T:
    return g.make_node("Abs", [x], outputs)


def aten_convolution(
    g: GraphBuilder,
    outputs: List[str],
    input: T,
    weight: T,
    bias: T = None,
    stride: Sequence[int] = (1,),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    transposed: bool = False,
    output_padding: Sequence[int] = (0,),
    groups: int = 1,
) -> T:
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
        weight_dim_0 = g.make_node("Shape", [weight], start=0, end=1)
        bias_shape = g.make_node(
            "Expand", [weight_dim_0], g.make_node("Constant", value_ints=[1])
        )
        zero = g.make_node("CastLike", [np.array([0.0]), input])
        bias = g.make_node("Expand", [zero, bias_shape])

    # if Rank(input) != Rank(weight):
    #    input = op.Unsqueeze(input, op.Constant(value_ints=[0]))

    return g.make_node(
        "Conv",
        [input, weight, bias],
        outputs,
        strides=strides,
        pads=pads,
        group=groups,
        dilations=dilations,
    )


def aten_relu(g: GraphBuilder, outputs: List[str], x: T) -> T:
    return g.op.Relu(x, outputs)
