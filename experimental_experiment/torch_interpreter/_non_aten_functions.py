from typing import Any, Dict, List, Optional, Tuple
from ..xbuilder.graph_builder import GraphBuilder

T = str


def onnx_symbolic__symbolic_default(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    args: T,
    op_name: str,
    dtype: int,
    shape: Optional[Tuple[Any, ...]] = None,
    attr_keys: Optional[List[str]] = None,
    attr_types: Optional[List[int]] = None,
    attr_pos: Optional[List[int]] = None,
    attr_ints: Optional[List[int]] = None,
    attr_floats: Optional[List[float]] = None,
    attr_strs: Optional[List[str]] = None,
    metadata_props_keys: Optional[List[str]] = None,
    metadata_props_values: Optional[List[str]] = None,
    domain: str = "",
    version: int = 1,
    name: str = "onnx_symbolic",
):
    "onnx_symbolic::_symbolic"
    assert not attr_keys, f"not supported with attr_keys={attr_keys}{g.get_debug_msg()}"
    assert not attr_types, f"not supported with attr_types={attr_types}{g.get_debug_msg()}"
    assert not attr_pos, f"not supported with attr_pos={attr_pos}{g.get_debug_msg()}"
    assert not attr_ints, f"not supported with attr_ints={attr_ints}{g.get_debug_msg()}"
    assert not attr_floats, f"not supported with attr_floats={attr_floats}{g.get_debug_msg()}"
    assert not attr_strs, f"not supported with attr_strs={attr_strs}{g.get_debug_msg()}"
    assert (
        not metadata_props_keys
    ), f"not supported with metadata_props_keys={metadata_props_keys}{g.get_debug_msg()}"
    assert (
        not metadata_props_values
    ), f"not supported with metadata_props_values={metadata_props_values}{g.get_debug_msg()}"
    assert all(
        isinstance(a, str) for a in args
    ), f"not implemented with args={args!r}{g.get_debug_msg()}"

    g.add_domain(domain, version)
    res = g.make_node(op_name, args, outputs, domain=domain, name=f"onnx_symbolic_{op_name}")
    if not sts:
        if isinstance(dtype, int):
            g.get_type(res, dtype)
        if shape is not None:
            assert all(isinstance(s, (int, str)) for s in shape), (
                f"unexpected shape={shape!r}, types are {[type(s) for s in shape]}"
                f"{g.get_debug_msg()}"
            )
            g.set_shape(res, shape)
    return res
