from typing import Any, Dict, List, Optional
import numpy as np
import onnx
from ..xbuilder.graph_builder import GraphBuilder
from ..xshape._shape_helper import is_static_shape
from ..xshape.shape_type_compute import set_type_shape_unary_op

T = str


def transformers_grouped_mm_fallback(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    mat_a: T,
    mat_b: T,
    offs: T,
    name: str = "transformers_grouped_mm_fallback",
):
    "transformers.integrations.moe._grouped_mm_fallback"
    assert g.has_rank(mat_a), f"not type for {mat_a=}{g.get_debug_msg()}"
    assert g.has_rank(mat_b), f"not type for {mat_b=}{g.get_debug_msg()}"
    dtype_a, dtype_b = g.get_type(mat_a), g.get_type(mat_b)
    assert dtype_a == dtype_b, (
        f"not implemented when {dtype_a=} != {dtype_b=} "
        f"for mat_a={mat_a!r} and mat_b={mat_b!r}{g.get_debug_msg()}"
    )
    assert dtype_a in {
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
    }, f"unexpected type for mat_a={mat_a!r}: {dtype_a=}{g.get_debug_msg()}"
    assert g.has_shape(offs) and is_static_shape(g.get_shape(offs)), (
        f"This is not implemented for offs={offs!r} and shape={g.get_shape(offs)}"
        f"{g.get_debug_msg()}"
    )
    assert g.has_shape(offs) and is_static_shape(
        g.get_shape(offs)
    ), f"Wrong shape for {offs=}, expecting a static one{g.get_debug_msg()}"

    # def _grouped_mm_fallback(input, weight, offs):
    # output = torch.zeros(input.size(0), weight.size(2))  # (S, output_dim)
    # start = 0
    # for i, end in enumerate(offs.tolist()):
    #     if start == end:
    #         continue
    #     torch.mm(input[start:end], weight[i], out=output[start:end])
    #     start = end
    # return output

    cats = []
    size = g.get_shape(offs)[0]
    start = g.ZERO
    for i in range(size):
        end = g.op.Cast(
            g.op.Gather(offs, np.array([i], dtype=np.int64), name=name),
            to=onnx.TensorProto.INT64,
            name=name,
        )
        mat = g.op.Slice(mat_a, start, end, g.ZERO, name=name)
        weight = g.op.Gather(mat_b, np.array(i, dtype=np.int64), name=name)
        mm = g.op.MatMul(mat, weight, name=name)
        cats.append(mm)
        start = end

    res = g.op.Concat(*cats, axis=0, name=name, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, res, mat_a)
    return res
