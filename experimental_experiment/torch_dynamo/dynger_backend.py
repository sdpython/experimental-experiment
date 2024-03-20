from typing import Callable, List, Tuple, Union
import torch


def dynger_backend(
    graph_module: "torch.fx.GraphModule",  # noqa: F821
    args: List[Union["torch.Tensor", "torch.SymInt"]],  # noqa: F821
    optimize: bool = True,
    verbose: Union[int, Tuple[int, int]] = 0,
) -> Callable:
    """
    Eager backend for dynamo.

    :param graph_module: graph to export
    :param args: arguments
    :param optimize: optimize or not, those optimization would be done
        on the graph module itself
    :param verbose: adjust verbosity, if tuple, if gives different verbosity level
        to the exporter and the runtime
    :return: Callable
    """

    def run(*inputs, gm=graph_module):
        return gm(*inputs)

    return run
