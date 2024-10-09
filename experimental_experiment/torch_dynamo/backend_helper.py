from typing import List, Optional, Tuple
from onnx import ModelProto


def get_dimensions(
    onx: ModelProto,
) -> Tuple[List[Tuple[bool, int, str, int]], List[Tuple[bool, int, Optional[str], int]]]:
    is_dimension_in = []
    for o in onx.graph.input:
        b = "_dim_" in o.name
        rk = len(o.type.tensor_type.shape.dim)
        dt = o.type.tensor_type.elem_type
        is_dimension_in.append((b, rk, o.name, dt))

    is_dimension_out = []
    for o in onx.graph.output:
        b = "_dim_" in o.name
        rk = len(o.type.tensor_type.shape.dim)
        dt = o.type.tensor_type.elem_type
        is_dimension_out.append((b, rk, None if "_NONE_" in o.name else o.name, dt))
    return is_dimension_in, is_dimension_out
