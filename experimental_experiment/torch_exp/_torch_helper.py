from typing import List, Union
import torch


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
