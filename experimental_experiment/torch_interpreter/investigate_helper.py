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


def _pick_result(torch_results: Dict[str, Any], ref: Any) -> Any:
    if isinstance(ref, torch.fx.Node):
        return torch_results[ref.name]
    if isinstance(ref, list):
        return [_pick_result(torch_results, n) for n in ref]
    if isinstance(ref, tuple):
        return tuple(_pick_result(torch_results, n) for n in ref)
    if isinstance(ref, dict):
        return {k: _pick_result(torch_results, v) for k, v in ref.items()}
    if isinstance(ref, (bool, int, float, str, torch.device, torch.dtype)):
        return ref
    raise NotImplementedError(f"Unable to process args type {type(ref)}")


def prepare_args_kwargs(
    torch_results: Dict[str, Any], node: torch.fx.Node
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Prepares args and kwargs before executing a fx node.

    :param torch_results: existing results
    :param node: node to execute
    :return: new args and kwargs
    """
    new_args = _pick_result(torch_results, node.args)
    new_kwargs = _pick_result(torch_results, node.kwargs)
    return new_args, new_kwargs


def run_aligned(
    ep: torch.export.ExportedProgram,
    onx: Union[onnx.ModelProto, onnx.FunctionProto],
    args: Tuple[torch.Tensor, ...],
    check_conversion_cls: Union[Dict[str, Any], type],
    kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
) -> Iterator[Tuple[Any, ...]]:
    """
    Runs both the exported program and the onnx proto and looks for discrepancies.

    :param ep: exported program
    :param onx: model or function proto
    :param args: input args
    :param check_conversion_cls: defines the runtime to use for this task
    :param kwargs: input kwargs
    :param verbose: verbosity level
    :return: a list of tuples containing the results.
    """
    assert not kwargs, f"Not implemented when kwargs={string_type(kwargs,with_shape=True)}"
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

    torch_results = {
        k: torch.from_numpy(v) for k, v in onnx_results.items() if not k.startswith("init")
    }
    last_position = 0
    torch_output_names = None
    for node in ep.graph.nodes:
        if node.op == "output":
            torch_output_names = [n.name for n in node.args[0]]
    onnx_outputs_names = [o.name for o in onx.graph.output]
    assert len(torch_output_names) == len(onnx_outputs_names), (
        f"Unexpected number of outputs, torch_output_names={torch_output_names}, "
        f"onnx_outputs_names={onnx_outputs_names}"
    )
    mapping_onnx_to_torch = dict(zip(onnx_outputs_names, torch_output_names))

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

    for inp, v in zip(onx.graph.input, args):
        onnx_results[inp.name] = v.numpy()
        if verbose:
            print(
                f"[run_aligned] +onnx-input: {inp.name}: "
                f"{string_type(v, with_shape=True, with_min_max=True)}"
            )

    for i, node in enumerate(ep.graph.nodes):
        if verbose:
            if node.op == "call_function":
                print(
                    f"[run_aligned] run ep.graph.nodes[{i}]: "
                    f"{node.op}[{node.target}] -> {node.name!r}"
                )
            else:
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
        args, kwargs = prepare_args_kwargs(torch_results, node)
        new_outputs = run_fx_node(node, args, kwargs)
        if isinstance(new_outputs, (torch.Tensor, int, float)):
            new_outputs = (new_outputs,)

        if new_outputs is None:
            # Probably an assert.
            continue

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

                to = mapping_onnx_to_torch.get(o, o)
                if to in torch_results:
                    d = max_diff(torch_results[to], r)
                    if verbose:
                        if o == to:
                            print(f"[run_aligned] common results {to}: {string_diff(d)}")
                        else:
                            print(f"[run_aligned] common results {to}/{o}: {string_diff(d)}")
                        if not (
                            atol is None
                            or rtol is None
                            or (d["abs"] <= atol and d["rel"] <= rtol)
                        ):
                            skw = dict(with_shape=True, with_min_max=True)
                            raise ValueError(
                                f"discrepancies detected for results [{to}/{o}]: "
                                f"{string_diff(d)}"
                                f"\n-- torch_results: {string_type(torch_results[to], **skw)}"
                                f"\n-- onnx_results: {string_type(r, **skw)}"
                                f"\n-- torch\n{torch_results[to]}\n-- onnx\n{r}"
                            )
                    yield (i, i_onnx, o, to, d)

        last_position = max_pos + 1

    # complete the execution of the onnx graph
    for i_onnx in range(last_position, len(onx.graph.node)):
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

            to = mapping_onnx_to_torch.get(o, o)
            if to in torch_results:
                d = max_diff(torch_results[to], r)
                if verbose:
                    if o == to:
                        print(f"[run_aligned] common results* {to}: {string_diff(d)}")
                    else:
                        print(f"[run_aligned] common results* {to}/{o}: {string_diff(d)}")
                    if not (
                        atol is None or rtol is None or (d["abs"] <= atol and d["rel"] <= rtol)
                    ):
                        skw = dict(with_shape=True, with_min_max=True)
                        raise ValueError(
                            f"discrepancies detected for results* [{to}/{o}]: {string_diff(d)}"
                            f"\n-- torch_results: {string_type(torch_results[to], **skw)}"
                            f"\n-- onnx_results: {string_type(r, **skw)}"
                            f"\n-- torch\n{torch_results[to]}\n-- onnx\n{r}"
                        )
                yield (i, i_onnx, o, to, d)
