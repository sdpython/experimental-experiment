import json
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import onnx.numpy_helper as onh
from onnx import (
    ModelProto,
    load,
    TensorProto,
    AttributeProto,
    FunctionProto,
    GraphProto,
    NodeProto,
)
from .helpers import tensor_dtype_to_np_dtype, from_array_extended, np_dtype_to_tensor_dtype


def _make_stat(init: TensorProto) -> Dict[str, float]:
    """
    Produces statistics.

    :param init: tensor
    :return statistics
    """
    ar = onh.to_array(init)
    return dict(
        mean=float(ar.mean()),
        std=float(ar.std()),
        shape=ar.shape,
        itype=np_dtype_to_tensor_dtype(ar.dtype),
        min=float(ar.min()),
        max=float(ar.max()),
    )


def onnx_lighten(
    onx: Union[str, ModelProto],
    verbose: int = 0,
) -> Tuple[ModelProto, Dict[str, Dict[str, float]]]:
    """
    Creates a model without big initializers but stores statistics
    into dictionaries. The function can be reversed with
    :func:`experimental_experiment.onnx_tools.onnx_unlighten`.
    The model is modified inplace.

    :param onx: model
    :param verbose: verbosity
    :return: new model, statistics
    """
    if isinstance(onx, str):
        if verbose:
            print(f"[onnx_lighten] load {onx!r}")
        model = load(onx)
    else:
        assert isinstance(onx, ModelProto), f"Unexpected type {type(onx)}"
        model = onx

    keep = []
    stats = []
    for init in model.graph.initializer:
        shape = init.dims
        size = np.prod(shape)
        if size > 2**12:
            stat = _make_stat(init)
            stats.append((init.name, stat))
            if verbose:
                print(f"[onnx_lighten] remove initializer {init.name!r} stat={stat}")
        else:
            keep.append(init)

    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return model, dict(stats)


def _get_tensor(min=None, max=None, mean=None, std=None, shape=None, itype=None):
    assert itype is not None, "itype must be specified."
    assert shape is not None, "shape must be specified."
    dtype = tensor_dtype_to_np_dtype(itype)
    if (mean is None or std is None) or (
        min is not None and max is not None and abs(max - min - 1) < 0.01
    ):
        if min is None:
            min = 0
        if max is None:
            max = 0
        return (np.random.random(shape) * (max - min) + min).astype(dtype)
    assert std is not None and mean is not None, f"mean={mean} or std={std} is None"
    t = np.random.randn(*shape).astype(dtype)
    return t


def onnx_unlighten(
    onx: Union[str, ModelProto],
    stats: Optional[Dict[str, Dict[str, float]]] = None,
    verbose: int = 0,
) -> ModelProto:
    """
    Function fixing the model produced by function
    :func:`experimental_experiment.onnx_tools.onnx_lighten`.
    The model is modified inplace.

    :param onx: model
    :param stats: statics, can be None if onx is a file,
        then it loads the file ``<filename>.stats``,
        it assumes it is json format
    :param verbose: verbosity
    :return: new model, statistics
    """
    if isinstance(onx, str):
        if stats is None:
            fstats = f"{onx}.stats"
            assert os.path.exists(fstats), f"File {fstats!r} is missing."
            if verbose:
                print(f"[onnx_unlighten] load {fstats!r}")
            with open(fstats, "r") as f:
                stats = json.load(f)
        if verbose:
            print(f"[onnx_unlighten] load {onx!r}")
        model = load(onx)
    else:
        assert isinstance(onx, ModelProto), f"Unexpected type {type(onx)}"
        model = onx
        assert stats is not None, "stats is missing"

    keep = []
    for name, stat in stats.items():
        t = _get_tensor(**stat)
        init = from_array_extended(t, name=name)
        keep.append(init)

    model.graph.initializer.extend(keep)
    return model


def _validate_graph(
    g: GraphProto,
    existing: Set[str],
    verbose: int = 0,
    watch: Optional[Set[str]] = None,
    path: Optional[Sequence[str]] = None,
):
    found = []
    path = path or ["root"]
    set_init = set(i.name for i in g.initializer)
    set_input = set(i.name for i in g.input)
    existing |= set_init | set_input
    if watch and set_init & watch:
        if verbose:
            print(f"-- found init {set_init & watch} in {path}")
        found.extend([i for i in g.initializer if i.name in set_init & watch])
    if watch and set_input & watch:
        if verbose:
            print(f"-- found input {set_input & watch} in {path}")
        found.extend([i for i in g.input if i.name in set_input & watch])
    try:
        import tqdm

        loop = tqdm.tqdm(g.node) if verbose else g.node
    except ImportError:
        loop = g.node

    for node in loop:
        ins = set(node.input) & existing
        if ins != set(node.input):
            raise AssertionError(
                f"One input is missing from node.input={node.input}, "
                f"existing={ins}, path={'/'.join(path)}, "
                f"node: {node.op_type}[{node.name}]"
            )
        if watch and ins & watch:
            if verbose:
                print(
                    f"-- found input {ins & watch} in "
                    f"{'/'.join(path)}/{node.op_type}[{node.name}]"
                )
            found.append(node)
        for att in node.attribute:
            if att.type == AttributeProto.GRAPH:
                found.extend(
                    _validate_graph(
                        att.g,
                        existing.copy(),
                        watch=watch,
                        path=[*path, f"{node.op_type}[{node.name}]"],
                        verbose=verbose,
                    )
                )
        existing |= set(node.output)
        if watch and set(node.output) & watch:
            if verbose:
                print(
                    f"-- found output {set(node.output) & watch} "
                    f"in {'/'.join(path)}/{node.op_type}[{node.name}]"
                )
            found.append(node)
    out = set(o.name for o in g.output)
    ins = out & existing
    if ins != out:
        raise AssertionError(
            f"One output is missing, out={node.input}, existing={ins}, path={path}"
        )
    return found


def _validate_function(g: FunctionProto, verbose: int = 0, watch: Optional[Set[str]] = None):
    existing = set(g.input)
    found = []
    for node in g.node:
        ins = set(node.input) & existing
        if ins != set(node.input):
            raise AssertionError(
                f"One input is missing from node.input={node.input}, existing={ins}"
            )
        if watch and ins & watch:
            if verbose:
                print(f"-- found input {ins & watch} in {node.op_type}[{node.name}]")
            found.append(node)
        for att in node.attribute:
            if att.type == AttributeProto.GRAPH:
                found.extend(
                    _validate_graph(g, existing.copy(), path=[g.name], verbose=verbose)
                )
        existing |= set(node.output)
        if watch and set(node.output) & watch:
            if verbose:
                print(
                    f"-- found output {set(node.output) & watch} "
                    f"in {node.op_type}[{node.name}]"
                )
    out = set(g.output)
    ins = out & existing
    if ins != out:
        raise AssertionError(
            f"One output is missing, out={node.input}, existing={ins}, path={g.name}"
        )
    return found


def onnx_find(
    onx: Union[str, ModelProto], verbose: int = 0, watch: Optional[Set[str]] = None
) -> List[Union[NodeProto, TensorProto]]:
    """
    Looks for node producing or consuming some results.

    :param onx: model
    :param verbose: verbosity
    :param watch: names to search for
    :return: list of nodes
    """

    if isinstance(onx, str):
        onx = load(onx, load_external_data=False)
    found = []
    found.extend(_validate_graph(onx.graph, set(), verbose=verbose, watch=watch))
    for f in onx.functions:
        found.extend(_validate_function(f, watch=watch, verbose=verbose))
    if verbose and found:
        print(f"-- found {len(found)} nodes")
    return found
