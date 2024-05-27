from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def dynger_backend(
    graph_module: "torch.fx.GraphModule",  # noqa: F821
    args: List[Union["torch.Tensor", "torch.SymInt"]],  # noqa: F821
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
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
    if isinstance(graph_module, torch.fx.GraphModule):
        if verbose > 0:
            print(f"[dynger_backend] use existing {type(graph_module)}")
        exported_mod = graph_module
    else:
        exported_mod = torch.export.export(
            graph_module, tuple(args), dynamic_shapes=dynamic_shapes
        )

    if verbose >= 10:

        def _identity(target, inputs, name, *args, **kwargs):
            print(f"{target}({inputs}) -> {name}")
            res = target(*args, **kwargs)
            if isinstance(res, torch.Tensor):
                assert isinstance(
                    name, str
                ), f"One name is expexted for one result but name={name!r}"
                if np.prod(res.shape) <= 8:
                    v = ",".join(map(str, res.ravel().detach().cpu().numpy().tolist()))
                else:
                    v = (
                        ",".join(
                            map(str, res.ravel().detach().cpu().numpy().tolist()[:5])
                        )
                        + "..."
                    )
                print(f"  + {name}: {res.dtype}:{res.shape}:{v}")
            else:
                raise AssertionError(f"Not implemented when type(res)={type(res)}")
            return res

        for i, node in enumerate(exported_mod.graph.nodes):
            if node.op != "call_function":
                continue
            if isinstance(node.target, str):
                continue
            node.target = lambda *args, __=node.target, _args=node.args, _name=node.name, **kwargs: _identity(
                __, _args, _name, *args, **kwargs
            )
        exported_mod.graph.lint()
        exported_mod.recompile()

    def run(*inputs, gm=exported_mod):
        if verbose:
            print("[dynger_backend] begin execution")
            res = gm(*inputs)
            print("[dynger_backend] done")
            return res
        return gm(*inputs)

    return run
