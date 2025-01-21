from typing import Any, Dict, List, Optional
from onnx import TensorProto
from ..xbuilder.graph_builder import GraphBuilder


T = str


def math_ceil(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name="math_ceil",
) -> T:
    "math_ceil"
    itype = g.get_type_known(outputs[0])
    if itype is None:
        itype = TensorProto.INT64
    return g.op.Cast(g.op.Ceil(x, name=name), to=itype, name=name, outputs=outputs)
