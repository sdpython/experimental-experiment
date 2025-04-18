import contextlib
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import ModelProto, load
from onnx.numpy_helper import to_array
from ..helpers import tensor_dtype_to_np_dtype


@contextlib.contextmanager
def dump_onnx(prefix: str, folder: Optional[str] = None, clean: bool = False):
    """
    context enabling the dump of models generated by
    :epkg:`onnxrt backend`.

    :param prefix: prefix for all files
    :param folder: sub folder (created if it does not exist)
    :param clean: if True, cleans the folder

    See :ref:`l-plot-onnxrt-diff` for an example.
    """
    if folder:
        if not os.path.exists(folder):
            os.makedirs(folder)
        if clean:
            for f in os.listdir(folder):
                ff = os.path.join(folder, f)
                if os.path.isfile(ff):
                    os.remove(ff)
    else:
        assert not clean, "cleaning can only happen if folder is specified"

    value = os.environ.get("ONNXRT_DUMP_PATH", None)
    os.environ["ONNXRT_DUMP_PATH"] = os.path.join(folder, f"{prefix}_")

    try:
        yield
    finally:
        os.environ["ONNXRT_DUMP_PATH"] = value or ""


def assert_all_close(
    v1: Any,
    v2: Any,
    atol: Union[float, Tuple[float, float]] = 1e-5,
    rtol: float = 1e-5,
    msg: str = "",
):
    """
    Checks that the expected outputs and new outputs are the same.

    :param v1: tensor or tuple of tensors
    :param v2: tensor or tuple of tensors
    :param atol: absolute error or (absolute error, quantile), if quantile is specified,
        the function checks the error is < atol for quantile %
    :param rtol: relative error
    :param msg: more complex message

    See :ref:`l-plot-onnxrt-diff` for an example.
    """
    if msg:
        try:
            assert_all_close(v1, v2, atol=atol, rtol=rtol)
        except AssertionError as e:
            raise AssertionError(f"ERROR: {msg}") from e
        return

    import torch

    aatol = atol
    if isinstance(atol, tuple):
        atol, quantile = atol
    else:
        atol, quantile = atol, None

    if isinstance(v1, torch.Tensor):
        assert isinstance(v2, torch.Tensor), f"v2 is not a tensor but {type(v2)}"
        assert_all_close(
            v1.detach().cpu().numpy(), v2.detach().cpu().numpy(), atol=aatol, rtol=rtol
        )
        # assert torch.allclose(v1.cpu(), v2.cpu(), atol=atol, rtol=rtol, equal_nan=True)
    elif isinstance(v1, np.ndarray):
        assert isinstance(v2, np.ndarray), f"v2 is not an array but {type(v2)}"
        try:
            # desired is the second input
            np.testing.assert_allclose(v2, v1, atol=atol, rtol=rtol, verbose=True)
        except AssertionError as e:
            if quantile is None:
                if v1.size <= 10:
                    raise AssertionError(
                        f"Discrepancies between\nv1={v1}\nv2={v2}\nratio={v2/v1}"
                    ) from e
                raise
            maxdiff = np.abs(v1 - v2)
            th = np.quantile(maxdiff, quantile)
            ind = maxdiff <= th
            r = maxdiff[ind]
            rmax = r.max()
            if rmax > atol:
                li = r.ravel().tolist()
                li.sort()
                msg = (
                    f"quantile={quantile} th={th} rmax={rmax} atol={atol} "
                    f"dtypes={v1.dtype} {v2.dtype}, shapes={v1.shape} {v2.shape}, "
                    f"means={v1.mean()} median={np.median(v1)} {np.median(v2)}, "
                    f"{v2.mean()}, min={v1.min()} {v2.min()}, "
                    f"max={v1.max()} {v2.max()}, ..... {li[:10]}  ..... {li[-10:]}"
                )
                raise AssertionError(msg) from e
    elif isinstance(v1, (tuple, list)):
        assert isinstance(v2, type(v1)), f"v2 is not a {type(v1)} but {type(v2)}"
        v1 = tuple(_ for _ in v1 if _ is not None)
        v2 = tuple(_ for _ in v2 if _ is not None)
        assert len(v1) == len(v2), f"tuple have different lengths {len(v1)} != {len(v2)}"
        for a, b in zip(v1, v2):
            assert_all_close(a, b, atol=aatol, rtol=rtol)
    elif isinstance(v1, int):
        assert isinstance(v2, type(v1)), f"v2 is not a {type(v1)} but {type(v2)}"
        assert v1 == v2
    else:
        raise AssertionError(f"Unexpected type for v1 and v2 {type(v1)}, {type(v2)}")


def reorder_functions_in_proto(proto: Union[str, ModelProto]) -> Union[str, ModelProto]:
    """
    The reference implementation expects function to be defined.
    So rank function has to be placed in the first position

    :param proto: a model
    :return: modified model inplace

    See :ref:`l-plot-onnxrt-diff` for an example.
    """
    if isinstance(proto, str):
        p = load(proto)
        p2 = reorder_functions_in_proto(p)
        with open(proto, "wb") as f:
            f.write(p2.SerializeToString())
        return proto

    def _order(name):
        if name == "Rank":
            return 0
        if name == "IsScalar":
            return 1
        return 10

    names = [(_order(f.name), f.name, f) for f in proto.functions]
    names.sort()
    del proto.functions[:]
    proto.functions.extend([_[-1] for _ in names])
    return proto


def inputs_from_onnx_model(
    model: Union[str, ModelProto], init: bool = False
) -> List[Tuple[str, int, Tuple[int, ...]]]:
    """
    Returns the inputs for a model.

    :param model: model or filename
    :param init: include the initializer as well
    :return: list of inputs and initializers

    See :ref:`l-plot-onnxrt-diff` for an example.
    """
    if isinstance(model, str):
        proto = load(model)
    else:
        proto = model
    res = []
    for i in proto.graph.input:
        res.append(
            (
                "INPUT",
                i.name,
                i.type.tensor_type.elem_type,
                tuple(
                    d.dim_param if d.dim_param else d.dim_value
                    for d in i.type.tensor_type.shape.dim
                ),
            )
        )
    if init:
        for i in proto.graph.initializer:
            res.append(("INIT", i.name, i.data_type, tuple(i.dims)))
    return res


def build_matching_inputs(
    model1: Union[str, ModelProto],
    feeds: Dict[str, Any],
    model2: Union[str, ModelProto],
) -> Dict[str, Any]:
    """
    Builds a list of inputs for a model based on the inputs made for another.
    We assume they both needs the same inputs.

    :param model1: first model
    :param feeds: inputs for the first model
    :param model2: second model, the one we need the inputs for
    :return: new inputs

    See :ref:`l-plot-onnxrt-diff` for an example.
    """
    if isinstance(model1, str):
        return build_matching_inputs(load(model1), feeds, model2)
    if isinstance(model2, str):
        return build_matching_inputs(model1, feeds, load(model2))
    feeds_rev = {}
    for k, v in feeds.items():
        if hasattr(v, "detach"):
            v = v.deatch().numpy()
        key = v.dtype, v.shape
        if key not in feeds_rev:
            feeds_rev[key] = []
        feeds_rev[key].append((k, v))
    for i in model1.graph.initializer:
        if i.name in feeds:
            continue
        dt = tensor_dtype_to_np_dtype(i.data_type)
        shape = tuple(i.dims)
        key = dt, shape
        if key not in feeds_rev:
            feeds_rev[key] = []
        feeds_rev[key].append((i.name, to_array(i)))

    # inputs2
    inputs = inputs_from_onnx_model(model2)
    feeds2 = {}
    for _kind, name, dt, shape in inputs:
        dt = tensor_dtype_to_np_dtype(dt)
        key = dt, shape
        if key in feeds_rev:
            if feeds_rev[key]:
                feeds2[name] = feeds_rev[key][0][1]
                del feeds_rev[key][0]
                continue
        raise RuntimeError(f"Unable to find key={key} among {list(feeds_rev)}")

    return feeds2


def results_to_string(results: Any, indent: str = "") -> str:
    """
    Builds a string showing the type and shape of every tensor in it.
    """
    import torch

    if isinstance(results, torch.Tensor):
        return f"{indent}{results.dtype} {tuple(results.shape)} [sum={results.sum():1.3g}]"
    if isinstance(results, tuple):
        return f"{indent}{len(results)} results\n" + "\n".join(
            results_to_string(r, indent=indent + "  ") for r in results
        )
    raise RuntimeError(f"Unexpected type {type(results)} for results")
