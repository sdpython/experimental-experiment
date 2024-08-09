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

    Next examples shows how to display intermediate results
    while executing the graph produced by torch dynamo.

    .. runpython::
        :showcode:

        import torch
        from experimental_experiment.torch_dynamo import dynger_backend


        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(10, 32),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.layers(x)


        x = torch.randn(3, 10, dtype=torch.float32)

        mlp = MLP()
        expected = mlp(x)

        compiled_model = torch.compile(
            mlp,
            backend=lambda *args, **kwargs: dynger_backend(*args, verbose=10, **kwargs),
            dynamic=False,
            fullgraph=True,
        )

        got = compiled_model(x)
        diff = (expected - got).max()
        print(f"discrepancies: {diff}")
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

        def _identity(
            target: str, inputs: List[str], name: str, *args: Any, **kwargs: Any
        ) -> Any:
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

        class _identity_graph:
            def __init__(
                self,
                graph: "torch.fx.graph.Graph",
                inputs: List[str],
                name: str,
                f: Callable,
            ):
                self._graph = graph
                self._inputs = inputs
                self._name = name
                self._f = f
                assert isinstance(
                    name, str
                ), f"One name is expexted for one result but name={name!r}"

            def __call__(self, *args, **kwargs):
                print(
                    f"{self._graph.__class__.__name__}({self._inputs}) -> {self._name}"
                )
                res = self._f(*args, **kwargs)
                if isinstance(res, torch.Tensor):
                    if np.prod(res.shape) <= 8:
                        v = ",".join(
                            map(str, res.ravel().detach().cpu().numpy().tolist())
                        )
                    else:
                        v = (
                            ",".join(
                                map(
                                    str, res.ravel().detach().cpu().numpy().tolist()[:5]
                                )
                            )
                            + "..."
                        )
                    print(f"  + {self._name}: {res.dtype}:{res.shape}:{v}")
                else:
                    raise AssertionError(f"Not implemented when type(res)={type(res)}")
                return res

        for node in exported_mod.graph.nodes:
            if node.op in ("call_function", "call_method"):
                node.target = lambda *args, __=node.target, _args=node.args, _name=node.name, **kwargs: _identity(  # noqa: E501
                    __, _args, _name, *args, **kwargs
                )
                continue
            if node.op == "call_module":
                sub_module = node.graph.owning_module.get_submodule(node.target)
                sub_module.forward = _identity_graph(
                    sub_module, node.args, node.target, f=sub_module.forward
                )
                continue
            if node.op in {"get_attr"}:
                raise AssertionError(
                    f"Not implemented for node.op={node.op!r}, node.__dict__={node.__dict__}"
                )
        exported_mod.graph.lint()
        exported_mod.recompile()

    def run(*inputs, gm=exported_mod):
        if verbose:
            print(
                f"[dynger_backend] begin execution with "
                f"{len(exported_mod.graph.nodes)} nodes"
            )
            res = gm(*inputs)
            print("[dynger_backend] done")
            return res
        return gm(*inputs)

    return run
