from typing import Any, Dict, Iterator, Optional, Tuple, Union
import onnx
import onnx.numpy_helper as onh
import torch
from ..helpers import string_type, string_diff, max_diff


def run_fx_node(
    node: torch.fx.Node, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[Any, ...]:
    """
    Executes a node

    :param node: runs a node
    :param args: unnamed inputs to the node
    :param kwargs: named inputs to the node
    :return: results
    """
    if node.op == "output":
        assert (
            len(args) == 1 and not kwargs
        ), f"Unexpected inputs: args={string_type(args)} kwargs={string_type(kwargs)}"
        return args
    if node.op == "call_function":
        return node.target(*args, **kwargs)
    raise NotImplementedError(
        f"node.op={node.op!r} is not implemented, node.name={node.name!r}"
    )


def run_aligned(
    ep: torch.export.ExportedProgram,
    onx: Union[onnx.ModelProto, onnx.FunctionProto],
    args: Tuple[torch.Tensor, ...],
    check_conversion_cls: Union[Dict[str, Any], type],
    verbose: int = 0,
) -> Iterator[Dict[str, Any]]:
    """
    Runs both the exported program and the onnx proto and looks for discrepancies.

    :param ep: exported program
    :param onx: model or function proto
    :param args: input args
    :param check_conversion_cls: defines the runtime to use for this task
    :return: a list of dictionaries containing the results.
    """
    cls, atol, rtol = (
        (
            check_conversion_cls["cls"],
            check_conversion_cls["atol"],
            check_conversion_cls["rtol"],
        )
        if isinstance(check_conversion_cls, dict)
        else (check_conversion_cls, None, None)
    )

    # retreive the positions
    positions = {}
    for i, node in enumerate(ep.graph.nodes):
        if isinstance(node.name, str):
            positions[node.name] = dict(fx=i)
        else:
            for n in node.name:
                positions[n] = dict(fx=i)

    for i, node in enumerate(onx.graph.node):
        for n in node.output:
            if n in positions:
                positions[n]["onnx"] = i
            else:
                positions[n] = dict(onnx=i)

    onnx_results = {}
    for init in onx.graph.initializer:
        positions[init.name] = -1
        onnx_results[init.name] = onh.to_array(init)

    torch_results = {k: torch.from_numpy(v) for k, v in onnx_results.items()}
    last_position = 0

    for inp, v in zip(onx.graph.input, args):
        onnx_results[inp.name] = v.numpy()

    if verbose:
        for k, v in torch_results.items():
            print(
                f"[run_aligned] +torch-cst: {k}: "
                f"{string_type(v, with_shape=True, with_min_max=True)}"
            )
        for k, v in onnx_results.items():
            print(
                f"[run_aligned] +onnx-init: {k}: "
                f"{string_type(v, with_shape=True, with_min_max=True)}"
            )

    for i, node in enumerate(ep.graph.nodes):
        if verbose:
            print(f"[run_aligned] run ep.graph.nodes[{i}]: {node.op} -> {node.name!r}")
        if node.op == "placeholder":
            if node.name in onnx_results:
                torch_results[node.name] = torch.from_numpy(onnx_results[node.name])
                if verbose:
                    t = torch_results[node.name]
                    print(
                        f"[run_aligned] +torch {node.name}="
                        f"{string_type(t, with_shape=True, with_min_max=True)}"
                    )
                continue
            raise AssertionError(f"unable to process node {node.op} -> {node.name!r}")
        outputs = [node.name] if isinstance(node.name, str) else list(node.name)
        args = tuple(
            (torch_results[n.name] if isinstance(n, torch.fx.Node) else n) for n in node.args
        )
        kwargs = {
            k: (onnx_results[n] if isinstance(n, str) and n in onnx_results else n)
            for k, v in node.kwargs
        }
        new_outputs = run_fx_node(node, args, kwargs)
        if isinstance(new_outputs, torch.Tensor):
            new_outputs = (new_outputs,)

        for k, v in zip(outputs, new_outputs):
            torch_results[k] = v
        if verbose:
            for k, v in zip(outputs, new_outputs):
                print(
                    f"[run_aligned] +torch {k}="
                    f"{string_type(v, with_shape=True, with_min_max=True)}"
                )

        max_pos = -2
        for n in outputs:
            if n in positions and "onnx" in positions[n]:
                max_pos = max(max_pos, positions[n]["onnx"])
        if max_pos == -2:
            # we skip.
            continue

        for i_onnx in range(last_position, max_pos + 1):
            node = onx.graph.node[i_onnx]
            if verbose:
                print(
                    f"[run_aligned] run onx.graph.node[{i_onnx}]: "
                    f"{node.op_type}({', '.join(node.input)}) -> {', '.join(node.output)}"
                )
            ref = cls(node)
            feeds = {k: onnx_results[k] for k in node.input}
            res = ref.run(None, feeds)
            for o, r in zip(node.output, res):
                onnx_results[o] = r
                if verbose:
                    print(
                        f"[run_aligned] +onnx {o}="
                        f"{string_type(r, with_shape=True, with_min_max=True)}"
                    )
                if o in torch_results:
                    d = max_diff(torch_results[o], r)
                    if verbose:
                        print(f"[run_aligned] common results {o}: {string_diff(d)}")
                        if not (
                            atol is None
                            or rtol is None
                            or (d["abs"] <= atol and d["rel"] <= rtol)
                        ):
                            raise ValueError(
                                f"discrepancies detected for results {r!r}: {string_diff(d)}"
                            )
                    yield (i, i_onnx, o, d)

        last_position = max_pos + 1
