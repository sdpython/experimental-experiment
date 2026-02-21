from typing import Any, Dict, List, Optional
import numpy as np
import onnx
from ..helpers import tensor_dtype_to_np_dtype
from ..xbuilder.graph_builder import GraphBuilder
from ..xshape._shape_helper import is_static_shape

T = str


def transformers_grouped_mm_fallback(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    mat_a: T,
    mat_b: T,
    offs: Optional[T] = None,
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
    assert (
        g.get_rank(offs) == 1
    ), f"Unexpected rank for {offs=}, rank={g.get_rank(offs)}{g.get_debug_msg()}"

    # def _grouped_mm_fallback(input, weight, offs):
    # output = torch.zeros(input.size(0), weight.size(2))  # (S, output_dim)
    # start = 0
    # for i, end in enumerate(offs.tolist()):
    #     if start == end:
    #         continue
    #     torch.mm(input[start:end], weight[i], out=output[start:end])
    #     start = end
    # return output

    loop_size = g.get_shape(offs)[0]
    total_size = g.op.Shape(mat_a, start=-1, name=name)
    last_split = g.op.Slice(
        offs, g.MINUS_ONE, np.array([loop_size], dtype=np.int64), g.ZERO, name=name
    )

    if dtype_a != onnx.TensorProto.FLOAT:
        ma = g.op.Cast(mat_a, to=onnx.TensorProto.FLOAT, name=name)
        mb = g.op.Cast(mat_b, to=onnx.TensorProto.FLOAT, name=name)
    else:
        ma, mb = mat_a, mat_b

    starts = g.op.Concat(
        np.array([0], dtype=tensor_dtype_to_np_dtype(g.get_type(offs))),
        g.op.Slice(offs, g.ZERO, g.MINUS_ONE, g.ZERO, name=name),
        axis=0,
        name=name,
    )
    ends = offs
    g.set_shape(starts, (loop_size,))
    g.set_shape(ends, (loop_size,))
    split_size_miss = g.op.Cast(
        g.op.Sub(ends, starts, name=name), to=onnx.TensorProto.INT64, name=name
    )
    cast_last_split = g.op.Cast(last_split, to=onnx.TensorProto.INT64, name=name)
    g.set_shape(cast_last_split, (1,))
    left = g.op.Sub(total_size, cast_last_split, name=name)
    split_size = g.op.Concat(split_size_miss, left, axis=0, name=name)

    assert (
        g.get_rank(mat_a) == 2
    ), f"Unexpected rank {g.get_rank(mat_a)} for {mat_a!r}{g.get_debug_msg()}"
    split_a = [g.unique_name(f"gma#{i}") for i in range(loop_size + 1)]
    g.make_node("Split", [ma, split_size], split_a, axis=0, name=name)

    # setting shapes
    if g.has_shape(mat_a):
        shape_ma = list(g.get_shape(mat_a))
        for i in range(len(split_a)):
            shape_ma[0] = g.unique_dimension_name(f"NEWDIM_mm_split{i}_dim")
            g.set_shape(split_a[i], tuple(shape_ma))
    else:
        shape_ma = None

    split_b = [g.unique_name(f"gmb#{i}") for i in range(loop_size)]
    g.make_node("Split", [mb], split_b, axis=0, name=name, num_outputs=loop_size)

    cats = []
    for msa, msb in zip(split_a[:-1], split_b):
        mm = g.op.MatMul(msa, g.op.SqueezeAnyOpset(msb, g.ZERO, name=name), name=name)
        cats.append(mm)

    concatenation = g.op.Concat(*cats, axis=0, name=name)
    if dtype_a != onnx.TensorProto.FLOAT:
        concatenation = g.op.Cast(concatenation, to=dtype_a, name=name)
    res = g.op.Identity(concatenation, name=name, outputs=outputs)
    if not sts:
        g.set_type(res, dtype_a)
        if g.has_shape(mat_a) and g.has_shape(mat_b):
            g.set_shape(res, (*g.get_shape(mat_a)[:-1], g.get_shape(mat_b)[-1]))
        else:
            g.set_rank(res, g.get_rank(mat_a))
    return res
