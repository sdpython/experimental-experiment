import copy
from typing import Any, List, Union
import numpy as np
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic
from torch.fx.experimental.sym_node import SymNode
from torch._dynamo.source import ConstantSource
from ..helpers import string_type


def create_input_names(
    graph_module: "torch.fx.GraphModule",  # noqa: F821
    args: List[Union["torch.Tensor", "torch.SymInt"]],  # noqa: F821
):
    res = []
    for i, a in enumerate(args):
        if isinstance(a, (torch.SymInt, torch.SymFloat)):
            res.append(f"input_dim_{i}")
        else:
            res.append(f"input{i}")
    return res


def create_symtype(cls, pytype, shape_env, val):
    symbol = shape_env.create_symbol(
        val,
        source=ConstantSource(__file__),
        dynamic_dim=DimDynamic.DUCK,
        constraint_dim=None,
    )
    return cls(
        SymNode(
            symbol,
            shape_env,
            pytype,
            hint=val,
        )
    )


def create_symint(i: int, shape_env=None) -> "torch.SymInt":
    return create_symtype(torch.SymInt, int, shape_env or ShapeEnv(), i)


def make_copy(obj: Any) -> Any:
    """Makes a copy of the objects."""
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, tuple):
        return tuple(make_copy(_) for _ in obj)
    if isinstance(obj, list):
        return [make_copy(_) for _ in obj]
    if isinstance(obj, dict):
        return {k: make_copy(v) for k, v in obj.items()}
    if hasattr(obj, "clone"):
        return obj.clone()
    if obj.__class__.__name__ in ("DynamicCache", "patched_DynamicCache"):
        cache = obj.__class__()
        if hasattr(obj, "_seen_tokens"):
            cache._seen_tokens = obj._seen_tokens
        cache.key_cache = make_copy(obj.key_cache)
        cache.value_cache = make_copy(obj.value_cache)
        return cache
    try:
        return copy.deepcopy(obj)
    except RuntimeError as e:
        raise RuntimeError(
            f"deepcopy did not work on type {type(obj)}: {string_type(obj)}"
        ) from e
