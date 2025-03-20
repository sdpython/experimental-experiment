import ast
import enum
import functools
import inspect
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
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
    if dtype == np.float32 or dtype == np.float32:
        return 4
    if dtype == np.float16 or dtype == np.int16:
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
    if obj is None:
        return "None"
    if isinstance(obj, tuple):
        if len(obj) == 1:
            s = string_type(
                obj[0],
                with_shape=with_shape,
                with_min_max=with_min_max,
                with_device=with_device,
                ignore=ignore,
                limit=limit,
            )
            return f"({s},)"
        if len(obj) < limit:
            js = ",".join(
                string_type(
                    o,
                    with_shape=with_shape,
                    with_min_max=with_min_max,
                    with_device=with_device,
                    ignore=ignore,
                    limit=limit,
                )
                for o in obj
            )
            return f"({js})"
        tt = string_type(
            obj[0],
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
        )
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            return f"({tt},...)#{len(obj)}[{mini},{maxi}:A[{avg}]]"
        return f"({tt},...)#{len(obj)}" if with_shape else f"({tt},...)"
    if isinstance(obj, list):
        if len(obj) < limit:
            js = ",".join(
                string_type(
                    o,
                    with_shape=with_shape,
                    with_min_max=with_min_max,
                    with_device=with_device,
                    ignore=ignore,
                    limit=limit,
                )
                for o in obj
            )
            return f"#{len(obj)}[{js}]"
        tt = string_type(
            obj[0],
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
        )
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            return f"[{tt},...]#{len(obj)}[{mini},{maxi}:{avg}]"
        return f"[{tt},...]#{len(obj)}" if with_shape else f"[{tt},...]"
    if isinstance(obj, set):
        if len(obj) < 10:
            js = ",".join(
                string_type(
                    o,
                    with_shape=with_shape,
                    with_min_max=with_min_max,
                    with_device=with_device,
                    ignore=ignore,
                    limit=limit,
                )
                for o in obj
            )
            return f"{{{js}}}"
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            return f"{{...}}#{len(obj)}[{mini},{maxi}:A{avg}]"
        return f"{{...}}#{len(obj)}" if with_shape else "{...}"
    if isinstance(obj, dict):
        if len(obj) == 0:
            return "{}"
        kws = dict(
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
        )
        s = ",".join(f"{kv[0]}:{string_type(kv[1],**kws)}" for kv in obj.items())
        return f"dict({s})"
    if isinstance(obj, np.ndarray):
        if with_min_max:
            s = string_type(obj, with_shape=with_shape)
            if len(obj.shape) == 0:
                return f"{s}={obj}"
            if obj.size == 0:
                return f"{s}[empty]"
            n_nan = np.isnan(obj.reshape((-1,))).astype(int).sum()
            if n_nan > 0:
                nob = obj.ravel()
                nob = nob[~np.isnan(nob)]
                if nob.size == 0:
                    return f"{s}[N{n_nan}nans]"
                return f"{s}[{nob.min()},{nob.max()}:A{nob.astype(float).mean()}N{n_nan}nans]"
            return f"{s}[{obj.min()},{obj.max()}:A{obj.astype(float).mean()}]"
        i = np_dtype_to_tensor_dtype(obj.dtype)
        if not with_shape:
            return f"A{i}r{len(obj.shape)}"
        return f"A{i}s{'x'.join(map(str, obj.shape))}"

    import torch

    if isinstance(obj, torch.export.dynamic_shapes._DerivedDim):
        return "DerivedDim"
    if isinstance(obj, torch.export.dynamic_shapes._Dim):
        return "Dim"
    if isinstance(obj, torch.SymInt):
        return "SymInt"
    if isinstance(obj, torch.SymFloat):
        return "SymFloat"
    if isinstance(obj, torch._subclasses.fake_tensor.FakeTensor):
        i = torch_dtype_to_onnx_dtype(obj.dtype)
        prefix = ("G" if obj.get_device() >= 0 else "C") if with_device else ""
        if not with_shape:
            return f"{prefix}F{i}r{len(obj.shape)}"
        return f"{prefix}F{i}s{'x'.join(map(str, obj.shape))}"
    if isinstance(obj, torch.Tensor):
        if with_min_max:
            s = string_type(obj, with_shape=with_shape, with_device=with_device)
            if len(obj.shape) == 0:
                return f"{s}={obj}"
            if obj.numel() == 0:
                return f"{s}[empty]"
            n_nan = obj.reshape((-1,)).isnan().to(int).sum()
            if n_nan > 0:
                nob = obj.reshape((-1,))
                nob = nob[~nob.isnan()]
                if obj.dtype in {torch.complex64, torch.complex128}:
                    return (
                        f"{s}[{nob.abs().min()},{nob.abs().max():A{nob.mean()}N{n_nan}nans}]"
                    )
                return f"{s}[{obj.min()},{obj.max()}:A{obj.to(float).mean()}N{n_nan}nans]"
            if obj.dtype in {torch.complex64, torch.complex128}:
                return f"{s}[{obj.abs().min()},{obj.abs().max()}:A{obj.abs().mean()}]"
            return f"{s}[{obj.min()},{obj.max()}:A{obj.to(float).mean()}]"
        i = torch_dtype_to_onnx_dtype(obj.dtype)
        prefix = ("G" if obj.get_device() >= 0 else "C") if with_device else ""
        if not with_shape:
            return f"{prefix}T{i}r{len(obj.shape)}"
        return f"{prefix}T{i}s{'x'.join(map(str, obj.shape))}"
    if isinstance(obj, bool):
        if with_min_max:
            return f"bool={obj}"
        return "bool"
    if isinstance(obj, int):
        if with_min_max:
            return f"int={obj}"
        return "int"
    if isinstance(obj, float):
        if with_min_max:
            return f"float={obj}"
        return "float"
    if isinstance(obj, str):
        return "str"
    if isinstance(obj, slice):
        return "slice"

    # others classes

    if type(obj).__name__ == "MambaCache":
        c = string_type(
            obj.conv_states,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        d = string_type(
            obj.ssm_states,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"MambaCache(conv_states={c}, ssm_states={d})"
    if type(obj).__name__ == "Node" and hasattr(obj, "meta"):
        # torch.fx.node.Node
        return f"%{obj.target}"
    if type(obj).__name__ == "ValueInfoProto":
        return f"OT{obj.type.tensor_type.elem_type}"

    if obj.__class__.__name__ in ("DynamicCache", "patched_DynamicCache"):
        kc = string_type(
            obj.key_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        vc = string_type(
            obj.value_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"{obj.__class__.__name__}(key_cache={kc}, value_cache={vc})"

    if obj.__class__.__name__ == "BatchFeature":
        s = string_type(
            obj.data,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"BatchFeature(data={s})"

    if obj.__class__.__name__ == "BatchEncoding":
        s = string_type(
            obj.data,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"BatchEncoding(data={s})"

    if obj.__class__.__name__ == "VirtualTensor":
        return (
            f"{obj.__class__.__name__}(name={obj.name!r}, "
            f"dtype={obj.dtype}, shape={obj.shape})"
        )

    if obj.__class__.__name__ == "_DimHint":
        return str(obj)

    if isinstance(obj, torch.nn.Module):
        return f"{obj.__class__.__name__}(...)"

    if isinstance(obj, torch.dtype):
        return f"{obj.__class__.__name__}({obj})"

    if isinstance(obj, torch.memory_format):
        return f"{obj.__class__.__name__}({obj})"

    if isinstance(obj, torch.utils._pytree.TreeSpec):
        return repr(obj).replace(" ", "").replace("\n", " ")

    if ignore:
        return f"{obj.__class__.__name__}(...)"
    raise AssertionError(f"Unsupported type {type(obj).__name__!r} - {type(obj)}")


def string_signature(sig: Any) -> str:
    """
    Displays the signature of a functions.
    """

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


def string_sig(f: Callable, kwargs: Optional[Dict[str, Any]] = None) -> str:
    """
    Displays the signature of a functions if the default
    if the given value is different from
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
                rows.append(
                    f"{p}={v!r}" if not isinstance(v, enum.IntEnum) else f"{p}={v.name}"
                )
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
        raise NotImplementedError(
            f"pretty_onnx not implemented yet for AttributeProto={att!r}"
        )

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
    """
    Returns a simple hash of ``id(obj)`` in four letter.
    """
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
    """Call to convert endianness of raw data in tensor.

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
    from onnx.reference.ops.op_cast import (
        bfloat16,
        float8e4m3fn,
        float8e4m3fnuz,
        float8e5m2,
        float8e5m2fnuz,
    )

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
    raise NotImplementedError(f"Unable to convert torch dtype {to!r} to onnx dtype.")


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
    Converts a tnumpy dtype into a onnx element type.

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


def rename_dynamic_dimensions(
    constraints: Dict[str, Set[str]], original: Set[str], ban_prefix: str = "DYN"
) -> Dict[str, str]:
    """
    Renames dynamic shapes as requested by the user. :func:`torch.export.export` uses
    many names for dynamic dimensions. When building the onnx model,
    some of them are redundant and can be replaced by the name provided by the user.

    :param constraints: exhaustive list of used name and all the values equal to it
    :param original: the names to use if possible
    :param ban_prefix: avoid any rewriting by a constant starting with this prefix
    :return: replacement dictionary
    """
    replacements = {s: s for s in original}
    all_values = set(constraints) | original

    not_done = set(constraints)
    max_iter = len(replacements)
    while not_done and max_iter > 0:
        max_iter -= 1
        for k, v in constraints.items():
            common = v & original
            if not common:
                continue
            common = sorted(common)
            by = common[0]
            if ban_prefix and by.startswith(ban_prefix):
                continue
            replacements[k] = by
            for vv in v:
                if vv not in replacements:
                    replacements[vv] = by
        not_done = all_values - set(replacements)
    return replacements


def rename_dynamic_expression(expression: str, replacements: Dict[str, str]):
    """
    Renames variables of an expression.

    :param expression: something like ``s15 + seq_length``
    :param replacements: replacements to make
    :return: new string
    """

    class RenameVariable(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in replacements:
                node.id = replacements[node.id]
            return node

    tree = ast.parse(expression)
    transformer = RenameVariable()
    new_tree = transformer.visit(tree)
    return ast.unparse(new_tree)


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
        res = flatten_object(x.key_cache) + flatten_object(x.value_cache)
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


def max_diff(
    expected: Any,
    got: Any,
    verbose: int = 0,
    level: int = 0,
    flatten: bool = False,
    debug_info: Optional[List[str]] = None,
    begin: int = 0,
    end: int = -1,
    _index: int = 0,
    allow_unique_tensor_with_list_of_one_element: bool = True,
) -> Dict[str, float]:
    """
    Returns the maximum discrepancy.

    :param expected: expected values
    :param got: values
    :param verbose: verbosity level
    :param level: for embedded outputs, used for debug purpposes
    :param flatten: flatten outputs
    :param debug_info: debug information
    :param begin: first output to considered
    :param end: last output to considered (-1 for the last one)
    :param _index: used with begin and end
    :param allow_unique_tensor_with_list_of_one_element:
        allow a comparison between a single tensor and a list of one tensor
    :return: dictionary with many values

    * abs: max abolute error
    * rel: max relative error
    * sum: sum of the errors
    * n: number of outputs values, if there is one
        output, this number will be the number of elements
        of this output
    * dnan: difference in the number of nan

    You may use :func:`string_diff` to display the discrepancies in one string.
    """
    if expected is None and got is None:
        return dict(abs=0, rel=0, sum=0, n=0, dnan=0)
    if allow_unique_tensor_with_list_of_one_element:
        if hasattr(expected, "shape") and isinstance(got, (list, tuple)) and len(got) == 1:
            return max_diff(
                expected,
                got[0],
                verbose=verbose,
                level=level,
                flatten=False,
                debug_info=debug_info,
                allow_unique_tensor_with_list_of_one_element=False,
            )
        return max_diff(
            expected,
            got,
            verbose=verbose,
            level=level,
            flatten=flatten,
            debug_info=debug_info,
            begin=begin,
            end=end,
            _index=_index,
            allow_unique_tensor_with_list_of_one_element=False,
        )
    if hasattr(expected, "to_tuple"):
        if verbose >= 6:
            print(f"[max_diff] to_tuple1: {string_type(expected)} ? {string_type(got)}")
        return max_diff(
            expected.to_tuple(),
            got,
            verbose=verbose,
            level=level + 1,
            debug_info=(
                [*(debug_info if debug_info else []), f"{' ' * level}to_tupleA"]
                if verbose > 5
                else None
            ),
            begin=begin,
            end=end,
            _index=_index,
            flatten=flatten,
        )

    if hasattr(got, "to_tuple"):
        if verbose >= 6:
            print(f"[max_diff] to_tuple2: {string_type(expected)} ? {string_type(got)}")
        return max_diff(
            expected,
            got.to_tuple(),
            verbose=verbose,
            level=level + 1,
            debug_info=(
                [*(debug_info if debug_info else []), f"{' ' * level}to_tupleB"]
                if verbose > 5
                else None
            ),
            begin=begin,
            end=end,
            _index=_index,
            flatten=flatten,
        )

        if isinstance(got, (list, tuple)):
            if len(got) != 1:
                if verbose >= 6:
                    print(
                        f"[max_diff] list,tuple,2: {string_type(expected)} "
                        f"? {string_type(got)}"
                    )
                if verbose > 2:
                    import torch

                    print(
                        f"[max_diff] (a) inf because len(expected)={len(expected)}!=1, "
                        f"len(got)={len(got)}, level={level}, _index={_index}"
                    )
                    for i, (a, b) in enumerate(zip(expected, got)):
                        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                            print(
                                f"    i={i} expected {a.dtype}:{a.shape}, "
                                f"has {b.dtype}:{b.shape}, _index={_index}"
                            )
                        else:
                            print(
                                f"    i={i} a is {type(a)}, "
                                f"b is {type(b)}, _index={_index}"
                            )
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            if verbose >= 6:
                print(f"[max_diff] list,tuple,1: {string_type(expected)} ? {string_type(got)}")
            return max_diff(
                expected,
                got[0],
                verbose=verbose,
                level=level + 1,
                begin=begin,
                end=end,
                _index=_index,
                debug_info=debug_info,
                flatten=flatten,
            )

    if isinstance(expected, (tuple, list)):
        if verbose >= 6:
            print(f"[max_diff] list,tuple,0: {string_type(expected)} ? {string_type(got)}")
        if len(expected) == 1 and not isinstance(got, type(expected)):
            if verbose >= 6:
                print(f"[max_diff] list,tuple,3: {string_type(expected)} ? {string_type(got)}")
            return max_diff(
                expected[0],
                got,
                verbose=verbose,
                level=level + 1,
                begin=begin,
                end=end,
                _index=_index,
                debug_info=debug_info,
                flatten=flatten,
            )
        if not isinstance(got, (tuple, list)):
            if verbose >= 6:
                print(f"[max_diff] list,tuple,4: {string_type(expected)} ? {string_type(got)}")
            if verbose > 2:
                print(
                    f"[max_diff] inf because type(expected)={type(expected)}, "
                    f"type(got)={type(got)}, level={level}, _index={_index}"
                )
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)

        if len(got) != len(expected):
            if flatten:
                if verbose >= 6:
                    print(
                        f"[max_diff] list,tuple,5: {string_type(expected)} "
                        f"? {string_type(got)}"
                    )
                # Let's flatten.
                if verbose > 2:
                    print(
                        f"[max_diff] flattening because of length mismatch, "
                        f"expected is {string_type(expected)} and got is {string_type(got)}"
                    )
                return max_diff(
                    flatten_object(expected, drop_keys=True),
                    flatten_object(got, drop_keys=True),
                    verbose=verbose,
                    level=level,
                    begin=begin,
                    end=end,
                    _index=_index,
                    debug_info=(
                        [
                            *(debug_info if debug_info else []),
                            (
                                f"{' ' * level}flatten["
                                f"{string_type(expected)},{string_type(got)}]"
                            ),
                        ]
                        if verbose > 5
                        else None
                    ),
                    flatten=flatten,
                )

            if verbose > 2:
                import torch

                print(
                    f"[max_diff] (b) inf because len(expected)={len(expected)}, "
                    f"len(got)={len(got)}, level={level}, _index={_index}"
                )
                for i, (a, b) in enumerate(zip(expected, got)):
                    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                        print(
                            f"    i={i} expected {a.dtype}:{a.shape}, "
                            f"has {b.dtype}:{b.shape}, _index={_index}"
                        )
                    else:
                        print(f"    i={i} a is {type(a)}, b is {type(b)}")
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)

        if verbose >= 6:
            print(f"[max_diff] list,tuple,6: {string_type(expected)} ? {string_type(got)}")
        am, rm, sm, n, dn = 0, 0, 0.0, 0.0, 0
        for ip, (e, g) in enumerate(zip(expected, got)):
            d = max_diff(
                e,
                g,
                verbose=verbose,
                level=level + 1,
                debug_info=(
                    [
                        *(debug_info if debug_info else []),
                        f"{' ' * level}[{ip}] so far abs {am} - rel {rm}",
                    ]
                    if verbose > 5
                    else None
                ),
                begin=begin,
                end=end,
                _index=_index + ip,
                flatten=flatten,
            )
            am = max(am, d["abs"])
            dn = max(dn, d["dnan"])
            rm = max(rm, d["rel"])
            sm += d["sum"]
            n += d["n"]
        return dict(abs=am, rel=rm, sum=sm, n=n, dnan=dn)

    if isinstance(expected, dict):
        if verbose >= 6:
            print(f"[max_diff] dict: {string_type(expected)} ? {string_type(got)}")
        assert (
            begin == 0 and end == -1
        ), f"begin={begin}, end={end} not compatible with dictionaries"
        if isinstance(got, dict):
            if len(expected) != len(got):
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            if set(expected) != set(got):
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            keys = sorted(expected)
            return max_diff(
                [expected[k] for k in keys],
                [got[k] for k in keys],
                level=level,
                flatten=flatten,
                debug_info=debug_info,
                begin=begin,
                end=end,
                _index=_index,
                verbose=verbose,
            )

        if not isinstance(got, (tuple, list)):
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        if len(expected) != len(got):
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        return max_diff(
            list(expected.values()),
            got,
            level=level,
            flatten=flatten,
            debug_info=debug_info,
            begin=begin,
            end=end,
            _index=_index,
            verbose=verbose,
        )

    import torch

    if isinstance(expected, np.ndarray) or isinstance(got, np.ndarray):
        if isinstance(expected, torch.Tensor):
            expected = expected.detach().cpu().numpy()
        if isinstance(got, torch.Tensor):
            got = got.detach().cpu().numpy()

        if verbose >= 6:
            print(f"[max_diff] tensor: {string_type(expected)} ? {string_type(got)}")

        if _index < begin or (end != -1 and _index >= end):
            # out of boundary
            return dict(abs=0.0, rel=0.0, sum=0.0, n=0.0, dnan=0)
        if isinstance(expected, (int, float)):
            if isinstance(got, np.ndarray) and len(got.shape) == 0:
                got = float(got)
            if isinstance(got, (int, float)):
                if expected == got:
                    return dict(abs=0.0, rel=0.0, sum=0.0, n=0.0, dnan=0)
                return dict(
                    abs=abs(expected - got),
                    rel=abs(expected - got) / (abs(expected) + 1e-5),
                    sum=abs(expected - got),
                    n=1,
                    dnan=0,
                )
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        if expected.dtype in (np.complex64, np.complex128):
            if got.dtype == expected.dtype:
                got = np.real(got)
            elif got.dtype not in (np.float32, np.float64):
                if verbose >= 10:
                    # To understand the value it comes from.
                    if debug_info:
                        print("\n".join(debug_info))
                    print(
                        f"[max_diff-c] expected.dtype={expected.dtype}, "
                        f"got.dtype={got.dtype}"
                    )
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            expected = np.real(expected)

        if expected.shape != got.shape:
            if verbose >= 10:
                # To understand the value it comes from.
                if debug_info:
                    print("\n".join(debug_info))
                print(f"[max_diff-s] expected.shape={expected.shape}, got.shape={got.shape}")
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        # nan are replace by 1e10, any discrepancies in that order of magnitude
        # is likely caused by nans
        exp_cpu = np.nan_to_num(expected.astype(np.float64), nan=1e10)
        got_cpu = np.nan_to_num(got.astype(np.float64), nan=1e10)
        diff = np.abs(got_cpu - exp_cpu)
        ndiff = np.abs(np.isnan(expected).astype(int) - np.isnan(got).astype(int))
        rdiff = diff / (np.abs(exp_cpu) + 1e-3)
        if diff.size == 0:
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (
                (0, 0, 0, 0, 0)
                if exp_cpu.size == got_cpu.size
                else (np.inf, np.inf, np.inf, 0, np.inf)
            )
        else:
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (
                float(diff.max()),
                float(rdiff.max()),
                float(diff.sum()),
                float(diff.size),
                float(ndiff.sum()),
            )
        if verbose >= 10 and (abs_diff >= 10 or rel_diff >= 10):
            # To understand the value it comes from.
            if debug_info:
                print("\n".join(debug_info))
            print(
                f"[max_diff-1] abs_diff={abs_diff}, rel_diff={rel_diff}, "
                f"nan_diff={nan_diff}, dtype={expected.dtype}, "
                f"shape={expected.shape}, level={level}, _index={_index}"
            )
            if abs_diff >= 10:
                idiff = np.argmax(diff.reshape((-1,)))
                x = expected.reshape((-1,))[idiff]
                y = got.reshape((-1,))[idiff]
                print(
                    f"   [max_diff-2] abs diff={abs_diff}, "
                    f"x={x}, y={y}, level={level}, "
                    f"_index={_index}"
                )
                print(y)

            if rel_diff >= 10:
                idiff = np.argmax(rdiff.reshape((-1,)))
                x = expected.reshape((-1,))[idiff]
                y = got.reshape((-1,))[idiff]
                print(
                    f"   [max_diff-3] rel diff={rel_diff}, "
                    f"x={x}, y={y}, level={level}, "
                    f"_index={_index}"
                )

        return dict(abs=abs_diff, rel=rel_diff, sum=sum_diff, n=n_diff, dnan=nan_diff)

    if isinstance(expected, torch.Tensor) and isinstance(got, torch.Tensor):
        if verbose >= 6:
            print(f"[max_diff] tensor: {string_type(expected)} ? {string_type(got)}")
        if _index < begin or (end != -1 and _index >= end):
            # out of boundary
            return dict(abs=0.0, rel=0.0, sum=0.0, n=0.0, dnan=0)
        if expected.dtype in (torch.complex64, torch.complex128):
            if got.dtype == expected.dtype:
                got = torch.view_as_real(got)
            elif got.dtype not in (torch.float32, torch.float64):
                if verbose >= 10:
                    # To understand the value it comes from.
                    if debug_info:
                        print("\n".join(debug_info))
                    print(
                        f"[max_diff-c] expected.dtype={expected.dtype}, "
                        f"got.dtype={got.dtype}"
                    )
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            expected = torch.view_as_real(expected)

        if expected.shape != got.shape:
            if verbose >= 10:
                # To understand the value it comes from.
                if debug_info:
                    print("\n".join(debug_info))
                print(f"[max_diff-s] expected.shape={expected.shape}, got.shape={got.shape}")
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        # nan are replace by 1e10, any discrepancies in that order of magnitude
        # is likely caused by nans
        exp_cpu = expected.to(torch.float64).cpu().nan_to_num(1e10)
        got_cpu = got.to(torch.float64).cpu().nan_to_num(1e10)
        diff = (got_cpu - exp_cpu).abs()
        ndiff = (expected.isnan().cpu().to(int) - got.isnan().cpu().to(int)).abs()
        rdiff = diff / (exp_cpu.abs() + 1e-3)
        if diff.numel() > 0:
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (
                float(diff.max()),
                float(rdiff.max()),
                float(diff.sum()),
                float(diff.numel()),
                float(ndiff.sum()),
            )
        elif got_cpu.numel() == exp_cpu.numel():
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
            )

        if verbose >= 10 and (abs_diff >= 10 or rel_diff >= 10):
            # To understand the value it comes from.
            if debug_info:
                print("\n".join(debug_info))
            print(
                f"[max_diff-1] abs_diff={abs_diff}, rel_diff={rel_diff}, "
                f"nan_diff={nan_diff}, dtype={expected.dtype}, "
                f"shape={expected.shape}, level={level}, _index={_index}"
            )
            if abs_diff >= 10:
                idiff = torch.argmax(diff.reshape((-1,)))
                x = expected.reshape((-1,))[idiff]
                y = got.reshape((-1,))[idiff]
                print(
                    f"   [max_diff-2] abs diff={abs_diff}, "
                    f"x={x}, y={y}, level={level}, "
                    f"_index={_index}"
                )
                print(y)

            if rel_diff >= 10:
                idiff = torch.argmax(rdiff.reshape((-1,)))
                x = expected.reshape((-1,))[idiff]
                y = got.reshape((-1,))[idiff]
                print(
                    f"   [max_diff-3] rel diff={rel_diff}, "
                    f"x={x}, y={y}, level={level}, "
                    f"_index={_index}"
                )

        return dict(abs=abs_diff, rel=rel_diff, sum=sum_diff, n=n_diff, dnan=nan_diff)

    if "SquashedNormal" in expected.__class__.__name__:
        if verbose >= 6:
            print(f"[max_diff] SquashedNormal: {string_type(expected)} ? {string_type(got)}")
        values = (
            expected.mean.detach().to("cpu"),
            expected.scale.detach().to("cpu"),
        )
        return max_diff(
            values,
            got,
            verbose=verbose,
            level=level + 1,
            begin=begin,
            end=end,
            _index=_index,
            flatten=flatten,
        )

    if expected.__class__.__name__ in ("DynamicCache", "patched_DynamicCache"):
        if got.__class__.__name__ in ("DynamicCache", "patched_DynamicCache"):
            if verbose >= 6:
                print(f"[max_diff] DynamicCache: {string_type(expected)} ? {string_type(got)}")
            return max_diff(
                [expected.key_cache, expected.value_cache],
                [got.key_cache, got.value_cache],
                verbose=verbose,
            )
        if isinstance(got, tuple) and len(got) == 2:
            return max_diff(
                [expected.key_cache, expected.value_cache],
                [got[0], got[1]],
                verbose=verbose,
            )
        raise AssertionError(
            f"DynamicCache not fully implemented with classes "
            f"{expected.__class__.__name__!r} and {got.__class__.__name__!r}, "
            f"and expected={string_type(expected)}, got={string_type(got)},\n"
            f"level={level}"
        )

    if expected.__class__.__name__ in ("transformers.cache_utils.MambaCache", "MambaCache"):
        if verbose >= 6:
            print(f"[max_diff] MambaCache: {string_type(expected)} ? {string_type(got)}")
        if got.__class__.__name__ != expected.__class__.__name__:
            # This case happens with onnx where the outputs are flattened.
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        atts = []
        for k in ["conv_states", "ssm_states"]:
            if hasattr(expected, k) and not hasattr(got, k):
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            atts.append(k)

        return max_diff(
            [getattr(expected, k) for k in atts],
            [getattr(got, k) for k in atts],
            level=level,
            flatten=flatten,
            debug_info=debug_info,
            begin=begin,
            end=end,
            _index=_index,
            verbose=verbose,
        )

    raise AssertionError(
        f"Not implemented with implemented with expected="
        f"{string_type(expected)}, got={string_type(got)},\n"
        f"level={level}"
    )


def string_diff(diff: Dict[str, Any]) -> str:
    """Renders discrepancies return by :func:`max_diff` into one string."""
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
