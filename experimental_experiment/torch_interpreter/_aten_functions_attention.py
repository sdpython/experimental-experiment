"""
See https://pytorch.org/docs/stable/torch.compiler_ir.html
for the full list of aten functions.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from onnx import TensorProto
from onnx.helper import make_tensor
from ..helpers import tensor_dtype_to_np_dtype, from_array_extended
from ..xbuilder.graph_builder import GraphBuilder
from ..xbuilder.shape_type_compute import set_type_shape_unary_op, set_type_shape_binary_op


T = str


def _attention_scale(g: GraphBuilder, query: T, name: str = "_attention_scale") -> T:
    if g.has_shape(query):
        shape = g.get_shape(query)
        last = shape[-1]
        if isinstance(last, int):
            scale = 1.0 / (float(last) ** 0.5)
            return np.array([scale], dtype=tensor_dtype_to_np_dtype(g.get_type(query)))

    shape = g.op.Shape(query, name=name)
    last = g.op.Gather(shape, g.MINUS_ONE, name=name)
    itype = g.get_type(query)
    clast = g.op.Cast(last, to=itype, name=name)
    return g.op.Reciprocal(g.op.Sqrt(clast, name=name), name=name)


def _causal_attention_mask(
    g: GraphBuilder, query: T, key: T, name: str = "_causal_attention_mask"
) -> T:
    itype = g.get_type(query)
    dtype = tensor_dtype_to_np_dtype(itype)
    attn_mask = None
    if g.has_shape(query) and g.has_shape(key):
        shape_query, shape_key = g.get_shape(query), g.get_shape(key)
        if isinstance(shape_query[-2], int) and isinstance(shape_key[-2], int):
            shape = (shape_query[-2], shape_key[-2])
            attn_mask = g.op.ConstantOfShape(
                np.array(shape, dtype=np.int64),
                value=from_array_extended(np.array([1], dtype=dtype)),
                name=name,
            )

    if attn_mask is None:
        # dynamic path
        shape_query = g.op.Shape(query, name=name)
        shape_key = g.op.Shape(key, name=name)
        dquery = g.op.Gather(shape_query, np.array([-2], dtype=np.int64), name=name)
        g.set_type(dquery, g.get_type(shape_query))
        dkey = g.op.Gather(shape_key, np.array([-2], dtype=np.int64), name=name)
        g.set_type(dkey, g.get_type(shape_key))
        size = g.op.Concat(dquery, dkey, axis=0, name=name)
        g.set_type(size, g.get_type(dkey))
        attn_mask = g.op.ConstantOfShape(
            size, value=from_array_extended(np.array([1], dtype=dtype)), name=name
        )
        g.set_type(attn_mask, itype)

    tri_attn_mask = g.op.Trilu(attn_mask, upper=0, name=name)
    set_type_shape_unary_op(g, tri_attn_mask, attn_mask)

    new_attn_mask = g.op.Where(
        g.op.Equal(tri_attn_mask, np.array([0], dtype=dtype), name=name),
        np.array([-float("inf")], dtype=dtype),
        np.array([0], dtype=dtype),
        name=name,
    )
    set_type_shape_unary_op(g, new_attn_mask, tri_attn_mask, itype=itype)
    return new_attn_mask


def aten_scaled_dot_product_attention(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    query: T,
    key: T,
    value: T,
    attn_mask: Optional[T] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    name: str = "aten_scaled_dot_product_attention",
):
    """
    scaled_dot_product_attention

    See `torch.nn.functional.scaled_dot_product_attention
    <https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>`_.

    Equivalent to the PyTorch code::

        scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
        attn_mask = (
            torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            if is_causal
            else attn_mask
        )
        attn_mask = (
            attn_mask.masked_fill(not attn_mask, -float('inf'))
            if attn_mask.dtype==torch.bool
            else attn_mask
        )
        attn_weight = torch.softmax(
            (Q @ K.transpose(-2, -1) * scale_factor) + attn_mask, dim=-1
        )
        attn_weight = torch.dropout(attn_weight, dropout_p)
        return attn_weight @ V

    where *Q*, *K*, *V* are the query, key, and value tensors, respectively.
    *L* is the target sequence length,
    *S* is the source sequence length, and E is the embedding size.
    """
    assert not enable_gqa, f"not implemented if enable_gqa={enable_gqa}"
    assert (not is_causal) or (is_causal and attn_mask is None), (
        f"is_causal and attn_mask cannot be set at the same time"
        f"is_causal={is_causal}, attn_mask={attn_mask}{g.get_debug_msg()}"
    )
    if g.main_opset >= 23:
        if dropout_p == 0:
            assert g.has_rank(query), f"missing shape for {query!r}{g.get_debug_msg()}"
            assert g.has_rank(key), f"missing shape for {key!r}{g.get_debug_msg()}"
            assert g.has_rank(value), f"missing shape for {value!r}{g.get_debug_msg()}"
            ranks = {g.get_rank(query), g.get_rank(key), g.get_rank(value)}
            assert len(ranks) == 1, (
                f"The converter is only implemented when all inputs have the same rank "
                f"but rank(query)={g.get_rank(query)}, rank(key)={g.get_rank(key)}, "
                f"rank(value)={g.get_rank(value)}{g.get_debug_msg()}"
            )
            rk = ranks.pop()
            assert rk in {4}, (
                f"The converter is only implemented "
                f"when all inputs have the same rank 4 "
                f"but rank(query)={g.get_rank(query)}, rank(key)={g.get_rank(key)}, "
                f"rank(value)={g.get_rank(value)}{g.get_debug_msg()}"
            )
            Y = g.op.Attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                scale=scale,
                # q_num_heads=q1,
                # kv_num_heads=k1,
                is_causal=is_causal,
                outputs=outputs,
                name=name,
            )
            return Y

    if scale is None:
        tscale = _attention_scale(g, query, name=name)
    elif isinstance(scale, (float, int)):
        assert g.has_type(query), f"Input {query!r} must have a type{g.get_debug_msg()}"
        itype = g.get_type(query)
        dtype = tensor_dtype_to_np_dtype(itype)
        tscale = np.array([scale], dtype=dtype)
    else:
        raise AssertionError(f"Unexpected type {type(scale)} for scale{g.get_debug_msg()}")

    if is_causal:
        attn_mask = _causal_attention_mask(g, query, key, name=name)

    key_transposed_axes = list(range(g.get_rank(key)))
    key_transposed_axes[-1], key_transposed_axes[-2] = (
        key_transposed_axes[-2],
        key_transposed_axes[-1],
    )
    key_transposed = g.op.Transpose(key, perm=key_transposed_axes, name=name)

    sc = g.op.Sqrt(tscale, name=name)
    if isinstance(scale, str):
        set_type_shape_unary_op(g, sc, tscale)

    query_scaled = g.op.Mul(query, sc, name=name)
    key_transposed_scaled = g.op.Mul(key_transposed, sc, name=name)
    mul_qk = g.op.MatMul(query_scaled, key_transposed_scaled, name=name)

    itype = g.get_type(query)
    dtype = tensor_dtype_to_np_dtype(itype)

    if attn_mask is None:
        mul_qk_add = mul_qk
    elif g.get_type(attn_mask) == TensorProto.BOOL:
        _attn_mask = g.op.Where(
            attn_mask,
            np.array([0.0], dtype=dtype),
            np.array([-float("inf")], dtype=dtype),
            name=name,
        )
        set_type_shape_unary_op(g, _attn_mask, attn_mask, itype=itype)
        attn_mask = _attn_mask
        mul_qk_add = g.op.Add(mul_qk, attn_mask, name=name)
        set_type_shape_binary_op(g, mul_qk_add, mul_qk, attn_mask)
    else:
        mul_qk_add = g.op.Add(mul_qk, attn_mask, name=name)
        set_type_shape_binary_op(g, mul_qk_add, mul_qk, attn_mask)

    attn_weight = g.op.Softmax(mul_qk_add, axis=-1, name=name)
    set_type_shape_unary_op(g, attn_weight, mul_qk_add)

    if dropout_p != 0:
        _attn_weight = g.op.Dropout(attn_weight, np.array(dropout_p, dtype=dtype), name=name)[
            0
        ]
        set_type_shape_unary_op(g, _attn_weight, attn_weight)
        attn_weight = _attn_weight

    res = g.op.MatMul(attn_weight, value, name=name, outputs=outputs)
    if not sts:
        g.set_type(res, g.get_type(attn_weight))
        if g.has_rank(query):
            g.set_rank(res, g.get_rank(query))
        elif g.has_rank(value):
            g.set_rank(res, g.get_rank(value))
    return res


def _aten__scaled_dot_product_flash_attention_fillin_empty_outputs(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    query: T,
    name: str = "_scaled_dot_product_flash_attention_fillin_empty_outputs",
) -> Tuple[T, T, T, T]:
    query_first_three_dims = g.op.Slice(
        g.op.Shape(query, name=name),
        g.op.Constant(value_ints=[0], name=name),
        g.op.Constant(value_ints=[3], name=name),
        name=name,
    )
    logsumexp = g.op.Expand(
        np.array([0], dtype=tensor_dtype_to_np_dtype(g.get_type(query))),
        query_first_three_dims,
        name=name,
        outputs=[outputs[0]],
    )

    empty_tensor_int = g.op.Cast(
        g.op.ConstantOfShape(
            g.op.Constant(
                value=make_tensor("Empty_INTS", TensorProto.INT64, [1], [0]), name=name
            ),
            name=name,
        ),
        to=TensorProto.INT64,
        name=name,
        outputs=None if not outputs[1] else [outputs[1]],
    )
    empty_tensor_float = g.op.ConstantOfShape(
        g.op.Constant(
            value=make_tensor("Empty_FLOATS", TensorProto.INT64, [1], [0]), name=name
        ),
        name=name,
        outputs=None if not outputs[2] else [outputs[2]],
    )
    empty_int = g.op.Constant(
        value_int=0, name=name, outputs=None if not outputs[3] else [outputs[3]]
    )
    return logsumexp, empty_tensor_int, empty_int, empty_tensor_float


def aten__scaled_dot_product_flash_attention(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    query: T,
    key: T,
    value: T,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: Optional[float] = None,
    name: str = "_scaled_dot_product_flash_attention",
) -> Tuple[T, T, T, T, T, T, T, T, T]:
    """_scaled_dot_product_flash_attention"""
    assert not return_debug_mask, "Not implemented when return_debug_mask is false."
    assert len(outputs) == 9, (
        f"Unexpected number of outputs {len(outputs)}, "
        f"outputs={outputs}({len(outputs)}){g.get_debug_msg()}"
    )

    result = aten_scaled_dot_product_attention(
        g,
        sts,
        [outputs[0]],
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        name=name,
    )

    # The followings are not comsumed by the graph.
    (
        logsumexp,
        empty_tensor_int,
        empty_int,
        empty_tensor_float,
    ) = _aten__scaled_dot_product_flash_attention_fillin_empty_outputs(
        g, sts, [outputs[1], outputs[3], outputs[8], outputs[4]], query, name=name
    )

    empty_tensor_int2 = g.op.Identity(empty_tensor_int, name=name)
    empty_tensor_int3 = g.op.Identity(empty_tensor_int, name=name, outputs=[outputs[6]])
    empty_tensor_int4 = g.op.Identity(empty_tensor_int, name=name, outputs=[outputs[7]])
    empty_int2 = g.op.Identity(empty_int, name=name, outputs=[outputs[5]])

    return (
        result,  # 0
        logsumexp,  # 1
        empty_tensor_int,  # 2
        empty_tensor_int2,  # 3
        empty_int,  # 4
        empty_int2,  # 5
        empty_tensor_int3,  # 6
        empty_tensor_int4,  # 7
        empty_tensor_float,  # 8
    )


def aten__scaled_dot_product_flash_attention_for_cpu(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    query: T,
    key: T,
    value: T,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    attn_mask: Optional[T] = None,
    scale: Optional[float] = None,
    return_debug_mask: bool = False,
    name: str = "_scaled_dot_product_flash_attention_for_cpu_default",
) -> Tuple[T, T, T, T, T, T, T, T, T]:
    """_scaled_dot_product_flash_attention"""
    assert not return_debug_mask, "Not implemented when return_debug_mask is false."
    result = aten_scaled_dot_product_attention(
        g,
        sts,
        [outputs[0]],
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        name=name,
    )
    assert isinstance(result, str), f"Unexpected type {type(result)}{g.get_debug_msg()}"

    # The followings are not comsumed by the graph on llama 3 at least.
    if len(outputs) == 2:
        # only need 2
        query_first_three_dims = g.op.Slice(
            g.op.Shape(query, name=name),
            g.op.Constant(value_ints=[0], name=name),
            g.op.Constant(value_ints=[3], name=name),
            name=name,
        )
        logsumexp = g.op.Expand(
            np.array([0], dtype=tensor_dtype_to_np_dtype(g.get_type(query))),
            query_first_three_dims,
            name=name,
            outputs=[outputs[1]],
        )
        return result, logsumexp

    assert len(outputs) == 8, (
        f"Unexpected number of outputs {len(outputs)}, "
        f"outputs={outputs}{g.get_debug_msg()}"
    )
    (
        logsumexp,
        empty_tensor_int,
        empty_int,
        empty_tensor_float,
    ) = _aten__scaled_dot_product_flash_attention_fillin_empty_outputs(
        g, sts, [outputs[1], outputs[3], outputs[4], outputs[8]], query, name=name
    )

    empty_tensor_int2 = g.op.Identity(empty_tensor_int, name=name)
    empty_int2 = g.op.Identity(empty_int, name=name)
    empty_tensor_int2 = g.op.Identity(empty_tensor_int, name=name)

    return (
        result,  # 0
        logsumexp,  # 1
        empty_tensor_int,  # 2
        empty_tensor_int2,  # 3
        empty_int,  # 4
        empty_int2,  # 5
        empty_tensor_int,  # 6
        empty_tensor_int2,  # 7
        empty_tensor_float,  # 8
    )


def aten__scaled_dot_product_efficient_attention(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    query: T,
    key: T,
    value: T,
    attn_bias: Optional[T],
    compute_log_sumexp: bool,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    name: str = "_scaled_dot_product_efficient_attention",
) -> Tuple[T, T, T, T]:
    """_scaled_dot_product_efficient_attention (cuda)"""
    result = aten_scaled_dot_product_attention(
        g,
        sts,
        [outputs[0]],
        query,
        key,
        value,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        name=name,
    )
    assert isinstance(result, str), f"Unexpected type {type(result)}{g.get_debug_msg()}"

    # only need 2
    query_first_three_dims = g.op.Slice(
        g.op.Shape(query, name=name),
        g.op.Constant(value_ints=[0], name=name),
        g.op.Constant(value_ints=[3], name=name),
        name=name,
    )
    logsumexp = g.op.Expand(
        np.array([0], dtype=tensor_dtype_to_np_dtype(g.get_type(query))),
        query_first_three_dims,
        name=name,
        outputs=[outputs[1]],
    )

    # The followings are not comsumed by the graph on llama 3 at least.
    if len(outputs) == 2:
        return result, logsumexp

    assert len(outputs) == 4, (
        f"Unexpected number of outputs {len(outputs)}, "
        f"outputs={outputs}{g.get_debug_msg()}"
    )

    empty_tensor_int = g.op.Cast(
        g.op.ConstantOfShape(
            g.op.Constant(
                value=make_tensor("Empty_INTS", TensorProto.INT64, [1], [0]), name=name
            ),
            name=name,
        ),
        to=TensorProto.INT64,
        name=name,
        outputs=[outputs[2]],
    )
    empty_tensor_int2 = g.op.Cast(
        g.op.ConstantOfShape(
            g.op.Constant(
                value=make_tensor("Empty_INTS", TensorProto.INT64, [1], [0]), name=name
            ),
            name=name,
        ),
        to=TensorProto.INT64,
        name=name,
        outputs=[outputs[3]],
    )

    return (
        result,  # 0
        logsumexp,  # 1
        empty_tensor_int,  # 2
        empty_tensor_int2,  # 3
    )
