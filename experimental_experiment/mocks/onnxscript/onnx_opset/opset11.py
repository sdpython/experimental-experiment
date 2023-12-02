from typing import Optional, Sequence
from ..values import Opset, TENSOR
from ..graph_builder import GraphBuilder
from .opset10 import Opset10


class Opset11(Opset10):
    def __new__(cls):
        return Opset.__new__(cls, "", 11)

    def __init__(self):
        pass

    def Conv(
        self,
        g: GraphBuilder,
        X: TENSOR,
        W: TENSOR,
        B: Optional[TENSOR] = None,
        *,
        auto_pad: str = "NOTSET",
        dilations: Optional[Sequence[int]] = None,
        group: int = 1,
        kernel_shape: Optional[Sequence[int]] = None,
        pads: Optional[Sequence[int]] = None,
        strides: Optional[Sequence[int]] = None,
    ) -> TENSOR:
        schema = None
        return g.make_node(
            "Conv",
            g._prepare_inputs(schema, X, W, B),
            auto_pad=auto_pad,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )
