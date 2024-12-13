import json
import os
from typing import Dict, Optional, Tuple, Union
import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import ModelProto, load, TensorProto
from .helpers import tensor_dtype_to_np_dtype


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
        itype=oh.np_dtype_to_tensor_dtype(ar.dtype),
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
        init = onh.from_array(t, name=name)
        keep.append(init)

    model.graph.initializer.extend(keep)
    return model
