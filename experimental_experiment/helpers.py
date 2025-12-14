import enum
import functools
import inspect
import sys
import textwrap
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union
import numpy as np
import ml_dtypes
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    ValueInfoProto,
    load as onnx_load,
)
from onnx.helper import (
    np_dtype_to_tensor_dtype as onnx_np_dtype_to_tensor_dtype,
    tensor_dtype_to_np_dtype as onnx_tensor_dtype_to_np_dtype,
)
from onnx.numpy_helper import from_array as onnx_from_array


def is_integer_type(itype: int) -> bool:
    return itype in {
        TensorProto.INT64,
        TensorProto.UINT64,
        TensorProto.INT16,
        TensorProto.UINT16,
        TensorProto.INT32,
        TensorProto.UINT32,
        TensorProto.INT8,
        TensorProto.UINT8,
        TensorProto.INT4,
    }


def is_float_type(itype: int) -> bool:
    return itype in {
        TensorProto.DOUBLE,
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
    }


def size_type(dtype: Any) -> int:
    """Returns the element size for an element type."""
    if isinstance(dtype, int):
        # It is a TensorProto.DATATYPE
        if dtype in {
            TensorProto.DOUBLE,
            TensorProto.INT64,
            TensorProto.UINT64,
            TensorProto.COMPLEX64,
        }:
            return 8
        if dtype in {TensorProto.FLOAT, TensorProto.INT32, TensorProto.UINT32}:
            return 4
        if dtype in {
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
            TensorProto.INT16,
            TensorProto.UINT16,
        }:
            return 2
        if dtype in {TensorProto.INT8, TensorProto.UINT8, TensorProto.BOOL}:
            return 1
        if dtype in {TensorProto.COMPLEX128}:
            return 16
        raise AssertionError(f"Unable to return the element size for type {dtype}")

    if dtype == np.float64 or dtype == np.int64:
        return 8
    if dtype == np.float32 or dtype == np.float32 or dtype == np.int32:
        return 4
    if dtype == np.float16 or dtype == np.int16 or dtype == ml_dtypes.bfloat16:
        return 2
    if dtype == np.int8 or dtype == np.uint8:
        return 1
    if hasattr(np, "uint64"):
        # it fails on mac
        if dtype == np.uint64:
            return 8
        if dtype == np.uint32:
            return 4
        if dtype == np.uint16:
            return 2

    import torch

    if dtype in {torch.float64, torch.int64}:
        return 8
    if dtype in {torch.float32, torch.int32}:
        return 4
    if dtype in {torch.float16, torch.int16, torch.bfloat16}:
        return 2
    if dtype in {torch.int8, torch.uint8, torch.bool}:
        return 1
    if hasattr(torch, "uint64"):
        # it fails on mac
        if dtype in {torch.uint64}:
            return 8
        if dtype in {torch.uint32}:
            return 4
        if dtype in {torch.uint16}:
            return 2
    raise AssertionError(f"Unexpected dtype={dtype}")


def tensor_dtype_to_np_dtype(tensor_dtype: int) -> np.dtype:
    """
    Converts a TensorProto's data_type to corresponding numpy dtype.
    It can be used while making tensor.

    :param tensor_dtype: TensorProto's data_type
    :return: numpy's data_type
    """
    if tensor_dtype >= 16:
        try:
            import ml_dtypes  # noqa: F401
        except ImportError as e:
            raise ValueError(
                f"Unsupported value for tensor_dtype, "
                f"numpy does not support onnx type {tensor_dtype}. "
                f"ml_dtypes can be used."
            ) from e

        mapping = {
            TensorProto.BFLOAT16: ml_dtypes.bfloat16,
            TensorProto.FLOAT8E4M3FN: ml_dtypes.float8_e4m3fn,
            TensorProto.FLOAT8E4M3FNUZ: ml_dtypes.float8_e4m3fnuz,
            TensorProto.FLOAT8E5M2: ml_dtypes.float8_e5m2,
            TensorProto.FLOAT8E5M2FNUZ: ml_dtypes.float8_e5m2fnuz,
        }
        assert (
            tensor_dtype in mapping
        ), f"Unable to find tensor_dtype={tensor_dtype!r} in mapping={mapping}"
        return mapping[tensor_dtype]

    return onnx_tensor_dtype_to_np_dtype(tensor_dtype)


def string_type(
    obj: Any,
    with_shape: bool = False,
    with_min_max: bool = False,
    with_device: bool = False,
    ignore: bool = False,
    limit: int = 10,
) -> str:
    """
    Displays the types of an object as a string.

    :param obj: any
    :param with_shape: displays shapes as well
    :param with_min_max: displays information about the values
    :param with_device: display the device
    :param ignore: if True, just prints the type for unknown types
    :return: str

    .. runpython::
        :showcode:

        from experimental_experiment.helpers import string_type
        print(string_type((1, ["r", 6.6])))
    """
    from onnx_diagnostic.helpers import string_type as string_type2

    return string_type2(
        obj,
        with_shape=with_shape,
        with_min_max=with_min_max,
        with_device=with_device,
        ignore=ignore,
        limit=limit,
    )


def string_signature(sig: Any) -> str:
    """Displays the signature of a functions."""

    def _k(p, kind):
        for name in dir(p):
            if getattr(p, name) == kind:
                return name
        return repr(kind)

    text = [" __call__ ("]
    for p in sig.parameters:
        pp = sig.parameters[p]
        kind = repr(pp.kind)
        t = f"{p}: {pp.annotation}" if pp.annotation is not inspect._empty else p
        if pp.default is not inspect._empty:
            t = f"{t} = {pp.default!r}"
        if kind == pp.VAR_POSITIONAL:
            t = f"*{t}"
        le = (30 - len(t)) * " "
        text.append(f"    {t}{le}|{_k(pp,kind)}")
        text.append(
            f") -> {sig.return_annotation}"
            if sig.return_annotation is not inspect._empty
            else ")"
        )
        return "\n".join(text)


def get_sig_kwargs(f: Callable, kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Returns updated attributes."""
    if hasattr(f, "__init__") and kwargs is None:
        fct = f.__init__
        kwargs = f.__dict__
    else:
        fct = f

    if kwargs is None:
        kwargs = {}
    res = {}
    sig = inspect.signature(fct)
    for p in sig.parameters:
        pp = sig.parameters[p]
        d = pp.default
        if d is inspect._empty:
            if p in kwargs:
                v = kwargs[p]
                res[p] = v
            continue
    return res


def string_sig(f: Callable, kwargs: Optional[Dict[str, Any]] = None) -> str:
    """
    Displays the signature of a function. Parameters are displayed
    if the given value is different from the default one.
    """
    if hasattr(f, "__init__") and kwargs is None:
        fct = f.__init__
        kwargs = f.__dict__
        name = f.__class__.__name__
    else:
        fct = f
        name = f.__name__

    if kwargs is None:
        kwargs = {}
    rows = []
    sig = inspect.signature(fct)
    for p in sig.parameters:
        pp = sig.parameters[p]
        d = pp.default
        if d is inspect._empty:
            if p in kwargs:
                v = kwargs[p]
                rows.append(f"{p}={v!r}" if not isinstance(v, enum.IntEnum) else f"{p}={v.name}")
            continue
        v = kwargs.get(p, d)
        if d != v:
            rows.append(f"{p}={v!r}" if not isinstance(v, enum.IntEnum) else f"{p}={v.name}")
            continue
    atts = ", ".join(rows)
    return f"{name}({atts})"


@functools.cache
def onnx_dtype_name(itype: int) -> str:
    """Returns the ONNX name for a specific element type."""
    for k in dir(TensorProto):
        v = getattr(TensorProto, k)
        if v == itype:
            return k
    raise ValueError(f"Unexpected value itype: {itype}")


def pretty_onnx(
    onx: Union[FunctionProto, GraphProto, ModelProto, ValueInfoProto, str],
    with_attributes: bool = False,
    highlight: Optional[Set[str]] = None,
) -> str:
    """
    Displays an onnx prot in a better way.

    :param with_attributes: displays attributes as well, if only a node is printed
    :param highlight: to highlight some names
    :return: text
    """
    assert onx is not None, "onx cannot be None"
    if isinstance(onx, str):
        onx = onnx_load(onx, load_external_data=False)
    assert onx is not None, "onx cannot be None"

    if isinstance(onx, ValueInfoProto):
        name = onx.name
        itype = onx.type.tensor_type.elem_type
        shape = tuple((d.dim_param or d.dim_value) for d in onx.type.tensor_type.shape.dim)
        shape_str = ",".join(map(str, shape))
        return f"{onnx_dtype_name(itype)}[{shape_str}] {name}"

    if isinstance(onx, AttributeProto):
        att = onx
        if att.type == AttributeProto.INT:
            return f"{att.name}={att.i}"
        if att.type == AttributeProto.INTS:
            return f"{att.name}={att.ints}"
        if att.type == AttributeProto.FLOAT:
            return f"{att.name}={att.f}"
        if att.type == AttributeProto.FLOATS:
            return f"{att.name}={att.floats}"
        if att.type == AttributeProto.STRING:
            return f"{att.name}={att.s!r}"
        if att.type == AttributeProto.GRAPH:
            return f"{att.name}\n{textwrap.indent(pretty_onnx(att.g), '    ')}\n"
        if att.type == AttributeProto.TENSOR:
            from .reference import to_array_extended

            v = to_array_extended(att.t)
            vf = v.reshape((-1,))
            if vf.size < 10:
                tt = f"[{', '.join(map(str, vf))}]"
            else:
                tt = f"[{', '.join(map(str, vf[:10]))}, ...]"
            if len(v.shape) != 1:
                return f"{att.name}=tensor({tt}, dtype={v.dtype}).reshape({v.shape})"
            return f"{att.name}=tensor({tt}, dtype={v.dtype})"
        raise NotImplementedError(f"pretty_onnx not implemented yet for AttributeProto={att!r}")

    if isinstance(onx, NodeProto):

        def _high(n):
            if highlight and n in highlight:
                return f"**{n}**"
            return n

        text = (
            f"{onx.op_type}({', '.join(map(_high, onx.input))})"
            f" -> {', '.join(map(_high, onx.output))}"
        )
        if onx.domain:
            text = f"{onx.domain}.{text}"
        if not with_attributes or not onx.attribute:
            return text
        rows = []
        for att in onx.attribute:
            rows.append(pretty_onnx(att))
        if len(rows) > 1:
            suffix = "\n".join(f"    {s}" for s in rows)
            return f"{text}\n{suffix}"
        return f"{text}  ---  {rows[0]}"

    if isinstance(onx, TensorProto):
        shape = "x".join(map(str, onx.dims))
        return f"TensorProto:{onx.data_type}:{shape}:{onx.name}"

    try:
        from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

        if isinstance(onx, FunctionProto):
            return (
                f"function: {onx.name}[{onx.domain}]\n"
                f"{onnx_simple_text_plot(onx, recursive=True)}"
            )
        return onnx_simple_text_plot(onx, recursive=True)
    except ImportError:
        from onnx.printer import to_text

        return to_text(onx)


def make_hash(obj: Any) -> str:
    """Returns a simple hash of ``id(obj)`` in four letters."""
    aa = id(obj) % (26**3)
    return f"{chr(65 + aa // 26 ** 2)}{chr(65 + (aa // 26) % 26)}{chr(65 + aa % 26)}"


def get_onnx_signature(model: ModelProto) -> Tuple[Tuple[str, Any], ...]:
    """
    Produces a tuple of tuples correspinding to the signatures.

    :param model: model
    :return: signature
    """
    sig = []
    for i in model.graph.input:
        dt = i.type
        if dt.HasField("sequence_type"):
            dst = dt.sequence_type.elem_type
            tdt = dst.tensor_type
            el = tdt.elem_type
            shape = tuple(d.dim_param or d.dim_value for d in tdt.shape.dim)
            sig.append((i.name, [(i.name, el, shape)]))
        elif dt.HasField("tensor_type"):
            el = dt.tensor_type.elem_type
            shape = tuple(d.dim_param or d.dim_value for d in dt.tensor_type.shape.dim)
            sig.append((i.name, el, shape))
        else:
            raise AssertionError(f"Unable to interpret dt={dt!r} in {i!r}")
    return tuple(sig)


def convert_endian(tensor: TensorProto) -> None:
    """
    Calls to convert endianness of raw data in tensor.

    Args:
        tensor: TensorProto to be converted.
    """
    tensor_dtype = tensor.data_type
    np_dtype = tensor_dtype_to_np_dtype(tensor_dtype)
    tensor.raw_data = np.frombuffer(tensor.raw_data, dtype=np_dtype).byteswap().tobytes()


def from_array_ml_dtypes(arr: np.ndarray, name: Optional[str] = None) -> TensorProto:
    """
    Converts a numpy array to a tensor def assuming the dtype
    is defined in ml_dtypes.

    Args:
        arr: a numpy array.
        name: (optional) the name of the tensor.

    Returns:
        TensorProto: the converted tensor def.
    """
    import ml_dtypes

    assert isinstance(arr, np.ndarray), f"arr must be of type np.ndarray, got {type(arr)}"

    tensor = TensorProto()
    tensor.dims.extend(arr.shape)
    if name:
        tensor.name = name

    if arr.dtype == ml_dtypes.bfloat16:
        dtype = TensorProto.BFLOAT16
    elif arr.dtype == ml_dtypes.float8_e4m3fn:
        dtype = TensorProto.FLOAT8E4M3FN
    elif arr.dtype == ml_dtypes.float8_e4m3fnuz:
        dtype = TensorProto.FLOAT8E4M3FNUZ
    elif arr.dtype == ml_dtypes.float8_e5m2:
        dtype = TensorProto.FLOAT8E5M2
    elif arr.dtype == ml_dtypes.float8_e5m2fnuz:
        dtype = TensorProto.FLOAT8E5M2FNUZ
    else:
        raise NotImplementedError(f"No conversion from {arr.dtype}")
    tensor.data_type = dtype
    tensor.raw_data = arr.tobytes()  # note: tobytes() is only after 1.9.
    if sys.byteorder == "big":
        convert_endian(tensor)
    return tensor


def from_array_extended(tensor: np.ndarray, name: Optional[str] = None) -> TensorProto:
    """
    Converts an array into a TensorProto.

    :param tensor: numpy array
    :param name: name
    :return: TensorProto
    """
    try:
        from onnx.reference.ops.op_cast import (
            bfloat16,
            float8e4m3fn,
            float8e4m3fnuz,
            float8e5m2,
            float8e5m2fnuz,
        )
    except ImportError:
        return onnx_from_array(tensor, name)

    dt = tensor.dtype
    if dt == float8e4m3fn and dt.descr[0][0] == "e4m3fn":
        to = TensorProto.FLOAT8E4M3FN
        dt_to = np.uint8
    elif dt == float8e4m3fnuz and dt.descr[0][0] == "e4m3fnuz":
        to = TensorProto.FLOAT8E4M3FNUZ
        dt_to = np.uint8
    elif dt == float8e5m2 and dt.descr[0][0] == "e5m2":
        to = TensorProto.FLOAT8E5M2
        dt_to = np.uint8
    elif dt == float8e5m2fnuz and dt.descr[0][0] == "e5m2fnuz":
        to = TensorProto.FLOAT8E5M2FNUZ
        dt_to = np.uint8
    elif dt == bfloat16 and dt.descr[0][0] == "bfloat16":
        to = TensorProto.BFLOAT16
        dt_to = np.uint16
    else:
        try:
            import ml_dtypes
        except ImportError:
            ml_dtypes = None
        if ml_dtypes is not None and (
            tensor.dtype == ml_dtypes.bfloat16
            or tensor.dtype == ml_dtypes.float8_e4m3fn
            or tensor.dtype == ml_dtypes.float8_e4m3fnuz
            or tensor.dtype == ml_dtypes.float8_e5m2
            or tensor.dtype == ml_dtypes.float8_e5m2fnuz
        ):
            return from_array_ml_dtypes(tensor, name)
        return onnx_from_array(tensor, name)

    t = onnx_from_array(tensor.astype(dt_to), name)
    t.data_type = to
    return t


def onnx_dtype_to_torch_dtype(itype: int) -> "torch.dtype":  # noqa: F821
    """
    Converts an onnx type into a torch dtype.

    :param to: onnx dtype
    :return: torch dtype
    """
    import torch

    if itype == TensorProto.FLOAT:
        return torch.float32
    if itype == TensorProto.FLOAT16:
        return torch.float16
    if itype == TensorProto.BFLOAT16:
        return torch.bfloat16
    if itype == TensorProto.DOUBLE:
        return torch.float64
    if itype == TensorProto.INT32:
        return torch.int32
    if itype == TensorProto.INT64:
        return torch.int64
    if itype == TensorProto.UINT32:
        return torch.uint32
    if itype == TensorProto.UINT64:
        return torch.uint64
    if itype == TensorProto.BOOL:
        return torch.bool
    if itype == TensorProto.INT16:
        return torch.int16
    if itype == TensorProto.UINT16:
        return torch.uint16
    if itype == TensorProto.INT8:
        return torch.int16
    if itype == TensorProto.UINT8:
        return torch.uint16
    if itype == TensorProto.COMPLEX64:
        return torch.complex64
    if itype == TensorProto.COMPLEX128:
        return torch.complex128
    raise NotImplementedError(f"Unable to convert onnx type {itype} to torch.type.")


def torch_dtype_to_onnx_dtype(to: "torch.dtype") -> int:  # noqa: F821
    """
    Converts a torch dtype into a onnx element type.

    :param to: torch dtype
    :return: onnx type
    """
    import torch

    if to == torch.float32:
        return TensorProto.FLOAT
    if to == torch.float16:
        return TensorProto.FLOAT16
    if to == torch.bfloat16:
        return TensorProto.BFLOAT16
    if to == torch.float64:
        return TensorProto.DOUBLE
    if to == torch.int64:
        return TensorProto.INT64
    if to == torch.int32:
        return TensorProto.INT32
    if to == torch.bool:
        return TensorProto.BOOL
    if to == torch.SymInt:
        return TensorProto.INT64
    if to == torch.SymFloat:
        return TensorProto.FLOAT
    if to == torch.complex64:
        return TensorProto.COMPLEX64
    if to == torch.complex128:
        return TensorProto.COMPLEX128
    raise NotImplementedError(
        f"Unable to convert torch dtype {to!r}, type(to)={type(to)} to onnx dtype."
    )


def dtype_to_tensor_dtype(dt: "dtype") -> int:  # noqa: F821
    """
    Converts a torch dtype or numpy dtype into a onnx element type.

    :param to: dtype
    :return: onnx type
    """
    try:
        return np_dtype_to_tensor_dtype(dt)
    except (KeyError, TypeError, ValueError):
        pass
    return torch_dtype_to_onnx_dtype(dt)


def np_dtype_to_tensor_dtype(dt: "dtype") -> int:  # noqa: F821
    """
    Converts a numpy dtype into a onnx element type.

    :param to: dtype
    :return: onnx type
    """
    try:
        return onnx_np_dtype_to_tensor_dtype(dt)
    except ValueError:
        try:
            import ml_dtypes
        except ImportError:
            ml_dtypes = None
        if ml_dtypes is not None:
            if dt == ml_dtypes.bfloat16:
                return TensorProto.BFLOAT16
            if dt == ml_dtypes.float8_e4m3fn:
                return TensorProto.FLOAT8E4M3FN
            if dt == ml_dtypes.float8_e4m3fnuz:
                return TensorProto.FLOAT8E4M3FNUZ
            if dt == ml_dtypes.float8_e5m2:
                return TensorProto.FLOAT8E5M2
            if dt == ml_dtypes.float8_e5m2fnuz:
                return TensorProto.FLOAT8E5M2FNUZ
    raise ValueError(f"Unable to convert type {dt}")


def flatten_object(x: Any, drop_keys: bool = False) -> List[Any]:
    """
    Flattens the object.
    It accepts some common classes used in deep learning.

    :param x: any object
    :param drop_keys: drop the keys if a dictionary is flattened.
        Keeps the order defined by the dictionary if False, sort them if True.
    :return: flattened object
    """
    if x is None:
        return x
    if isinstance(x, (list, tuple)):
        res = []
        for i in x:
            if i is None or hasattr(i, "shape") or isinstance(i, (int, float, str)):
                res.append(i)
            else:
                res.extend(flatten_object(i, drop_keys=drop_keys))
        return tuple(res) if isinstance(x, tuple) else res
    if isinstance(x, dict):
        # We flatten the keys.
        if drop_keys:
            return flatten_object(list(x.values()), drop_keys=drop_keys)
        return flatten_object(list(x.items()), drop_keys=drop_keys)

    if x.__class__.__name__ == "DynamicCache":
        from onnx_diagnostic.helpers.cache_helper import CacheKeyValue

        x = CacheKeyValue(x)
        res = flatten_object(x.key_cache) + flatten_object(x.value_cache)
        return tuple(res)
    if x.__class__.__name__ == "EncoderDecoderCache":
        res = flatten_object(x.self_attention_cache) + flatten_object(x.cross_attention_cache)
        return tuple(res)
    if x.__class__.__name__ == "MambaCache":
        if isinstance(x.conv_states, list):
            res = flatten_object(x.conv_states) + flatten_object(x.ssm_states)
            return tuple(res)
        return tuple(x.conv_states, x.ssm_states)
    if hasattr(x, "to_tuple"):
        return flatten_object(x.to_tuple(), drop_keys=drop_keys)
    if hasattr(x, "shape"):
        # A tensor. Nothing to do.
        return x
    raise TypeError(
        f"Unexpected type {type(x)} for x, drop_keys={drop_keys}, "
        f"content is {string_type(x, with_shape=True)}"
    )


def string_diff(diff: Dict[str, Any]) -> str:
    """
    Renders discrepancies return by :func:`onnx_diagnostic.helpers.max_diff`
    into one string.
    """
    # dict(abs=, rel=, sum=, n=n_diff, dnan=)
    if diff.get("dnan", None):
        if diff["abs"] == 0 or diff["rel"] == 0:
            return f"abs={diff['abs']}, rel={diff['rel']}, dnan={diff['dnan']}"
        return f"abs={diff['abs']}, rel={diff['rel']}, n={diff['n']}, dnan={diff['dnan']}"
    if diff["abs"] == 0 or diff["rel"] == 0:
        return f"abs={diff['abs']}, rel={diff['rel']}"
    return f"abs={diff['abs']}, rel={diff['rel']}, n={diff['n']}"


def type_info(itype: int, att: str):
    """
    Returns the minimum or maximum value for a type.

    :param itype: onnx type
    :param att: 'min' or 'max'
    :return: value
    """
    if itype in {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE}:
        dtype = tensor_dtype_to_np_dtype(itype)
        fi = np.finfo(dtype)
    elif itype == TensorProto.BFLOAT16:
        import ml_dtypes

        dtype = tensor_dtype_to_np_dtype(itype)
        fi = ml_dtypes.finfo(dtype)
    else:
        dtype = tensor_dtype_to_np_dtype(itype)
        fi = np.iinfo(dtype)
    if att == "min":
        return fi.min
    if att == "max":
        return fi.max
    raise ValueError(f"Unexpected value {att!r}")


def make_idn(node: NodeProto) -> str:
    """
    Creates a unique id for a node hoping collision cannot happen.
    onnx may reuse sometimes the nodes, ``id(node)`` may not be enough sometimes.
    """
    # return f"{id(node)}-{node.op_type}-{node.output[0]}-{node.name}"
    # id(node) should be enough and faster.
    return id(node)


def make_idg(g: GraphProto) -> str:
    """
    Creates a unique id for a graph hoping collision cannot happen.
    onnx may reuse sometimes the nodes, ``id(node)`` may not be enough sometimes.
    """
    # return f"{id(g)}-{len(g.node)}-{len(g.input)}-{len(g.output)}-{g.name}"
    # id(g) should be enough and faster.
    return id(g)


def enumerate_nodes(graph: GraphProto) -> Iterator[NodeProto]:
    """
    Enumerates all inputs from a node including all the hidden inputs
    from subgraphs.
    """
    for node in graph.node:
        yield node
        if node.op_type[0] in "LSI" and node.op_type in {"Loop", "Scan", "If", "SequenceMap"}:
            for att in node.attribute:
                if att.type == AttributeProto.GRAPH:
                    yield from enumerate_nodes(att.g)
