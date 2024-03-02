from typing import List, Union
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic
from torch.fx.experimental.sym_node import SymNode
from torch._dynamo.source import ConstantSource


def create_input_names(
    graph_module: "torch.fx.GraphModule",  # noqa: F821
    args: List[Union["torch.Tensor", "torch.SymInt"]],  # noqa: F821
):
    res = []
    for i, a in enumerate(args):
        if isinstance(a, torch.SymInt):
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
