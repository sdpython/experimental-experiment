import contextlib
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import ModelProto, load
from onnx.helper import tensor_dtype_to_np_dtype
from onnx.numpy_helper import to_array


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


def assert_all_close(v1: Any, v2: Any, atol=1e-5, rtol=1e-5):
    """
    Checks that the expected outputs and new outputs are the same.

    :param v1: tensor or tuple of tensors
    :param v2: tensor or tuple of tensors

    See :ref:`l-plot-onnxrt-diff` for an example.
    """
    import torch

    if isinstance(v1, torch.Tensor):
        assert isinstance(v2, torch.Tensor), f"v2 is not a tensor but {type(v2)}"
        assert torch.allclose(v1, v2, atol=atol, rtol=rtol)
    elif isinstance(v1, np.ndarray):
        assert isinstance(v2, np.ndarray), f"v2 is not an array but {type(v2)}"
        np.testing.assert_all_close(v1, v2, atol=atol, rtol=rtol)
    elif isinstance(v1, (tuple, list)):
        assert isinstance(v2, type(v1)), f"v2 is not a {type(v1)} but {type(v2)}"
        v1 = tuple(_ for _ in v1 if _ is not None)
        v2 = tuple(_ for _ in v2 if _ is not None)
        assert len(v1) == len(
            v2
        ), f"tuple have different lengths {len(v1)} != {len(v2)}"
        for a, b in zip(v1, v2):
            assert_all_close(a, b, atol=atol, rtol=rtol)
    else:
        raise AssertionError(f"Unexpected type for v1 and v2 {type(v1)}, {type(v2)}")


def _get_session(
    onx: ModelProto, impl: str = "ref", exc: bool = True
) -> Union["ReferenceEvaluator", "InferenceSession"]:  # noqa: F821
    if exc:
        try:
            return _get_session(onx, impl, exc=False)
        except Exception as e:
            from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

            raise AssertionError(
                f"Unable to build session ({str(e)})\n{onnx_simple_text_plot(onx)}"
            ) from e

    if impl == "ref":
        from onnx.reference import ReferenceEvaluator

        return ReferenceEvaluator(onx, verbose=10)
    else:
        import onnxruntime

        return onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )


def onnx_debug_backend(
    graph_module: "torch.fx.GraphModule",  # noqa: F821
    args: List["torch.Tensor"],  # noqa: F821
    target_opset: Optional[int] = None,
    backend: str = "ort",
    verbose: Union[int, Tuple[int, int]] = 0,
    dump_prefix: Optional[None] = str,
    providers: Optional[Tuple[str]] = None,
    raise_exc: bool = True,
    storage: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Custom backend to export torch models into onnx.
    This backend is not meant to be efficient, it more to check
    the conversion is ok.

    :param graph_module: graph to export
    :param args: arguments
    :param target_opset: opset to use for the conversion
    :param backend: after the conversion, the model is executed with a runtime,
        :epkg:`onnxruntime` or the reference implementation,
        it must be a value among `'ort'`, `'ref'` or a class
    :param verbose: adjust verbosity, if tuple, if gives different verbosity level
        to the exporter and the runtime
    :param dump_prefix
    :param providers: where to run the model, by default
    :param raise_exc: raise an exception whenever something goes wrong
    :param storage: to store any interesting objects during the process,
        including this inputs or anything else
    :return: Callable

    See :ref:`l-plot-onnxrt-diff` for an example.
    """
    import torch
    from ..torch_exp.onnx_export import to_onnx

    input_names = (
        ["input"] if len(args) == 1 else [f"input{i}" for i in range(len(args))]
    )

    verbose_onnx, verbose_backend = (
        verbose if isinstance(verbose, tuple) else (verbose, verbose)
    )

    onx, builder = to_onnx(
        graph_module,
        tuple(args),
        input_names=input_names,
        remove_unused=True,
        constant_folding=False,
        verbose=verbose_onnx,
        target_opset=target_opset,
        return_builder=True,
    )

    if dump_prefix:
        counter = 0
        name = f"{dump_prefix}_{counter}.onnx"
        while os.path.exists(name):
            counter += 1
            name = f"{dump_prefix}_{counter}.onnx"

        with open(name, "wb") as f:
            f.write(onx.SerializeToString())
        name = f"{dump_prefix}_{counter}.txt"
        with open(name, "w") as f:
            f.write(str(graph_module.graph))
            f.write("\n")

    sess = _get_session(onx, backend, exc=raise_exc)

    names = [i.name for i in onx.graph.input]

    _dtype = {
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("bool"): torch.bool,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.bool_: torch.bool,
    }

    if storage is not None:
        stor = {}
        if "instance" in storage:
            storage["instance"].append(stor)
        else:
            storage["instance"] = [stor]
        stor["graph_module"] = graph_module
        stor["onnx"] = onx
        stor["builder"] = builder
        stor["sess"] = sess
        stor["inputs"] = []
        stor["outputs"] = []

    def run(*inputs, sess=sess, names=names):
        xnp = [x.detach().numpy() for x in inputs]
        feeds = dict(zip(names, xnp))
        results = sess.run(None, feeds)
        res = tuple(torch.Tensor(y).to(_dtype[y.dtype]) for y in results)
        if storage:
            stor["inputs"].append(feeds)
            stor["outputs"].append(res)
        return res

    return run


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
    for kind, name, dt, shape in inputs:
        dt = tensor_dtype_to_np_dtype(dt)
        key = dt, shape
        if key in feeds_rev:
            if feeds_rev[key]:
                feeds2[name] = feeds_rev[key][0][1]
                del feeds_rev[key][0]
                continue
        raise RuntimeError(f"Unable to find key={key} among {list(feeds_rev)}")

    return feeds2
