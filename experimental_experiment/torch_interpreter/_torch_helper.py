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


def _tune_thresholds_histc(
    tensor: torch.Tensor, bins: int, fmin: float, fmax: float
) -> torch.Tensor:
    """
    Adjusts tensor threshold for function :func:`torch.histc`.
    """
    if tensor.dtype not in {torch.float16, torch.bfloat16}:
        # Nothing to do.
        return tensor
    assert tensor.ndim == 1, f"tensor should be 1D not {tensor.ndim}D."
    minf = torch.tensor(-torch.inf, dtype=tensor.dtype)
    pinf = torch.tensor(torch.inf, dtype=tensor.dtype)
    new_tensor = tensor.clone()
    buffer = torch.empty((1,), dtype=tensor.dtype)
    for i in range(tensor.numel() - 1):
        th = tensor[i]
        buffer[0] = th
        n = int(max(10, 1.0 / max(torch.abs(th), 1e-4)))
        for _ in range(n):
            buffer = torch.nextafter(buffer, minf)

        it = 0
        while it < 2 * n:
            buffer = torch.nextafter(buffer, pinf)
            res = torch.histc(buffer, bins, fmin, fmax)
            index = torch.argmax(res)
            if i == index:
                new_tensor[i] = buffer[0]
                break
            it += 1
    return new_tensor
