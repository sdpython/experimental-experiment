import ast
import inspect
import sys
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
import numpy as np
from onnx import FunctionProto, GraphProto, ModelProto, TensorProto, load as onnx_load
from onnx.helper import (
    np_dtype_to_tensor_dtype as onnx_np_dtype_to_tensor_dtype,
    tensor_dtype_to_np_dtype as onnx_tensor_dtype_to_np_dtype,
)
from onnx.numpy_helper import from_array as onnx_from_array


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


def string_type(obj: Any, with_shape: bool = False, with_min_max: bool = False) -> str:
    """
    Displays the types of an object as a string.

    :param obj: any
    :param with_shape: displays shapes as well
    :param with_min_max: displays information about the values
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
            s = string_type(obj[0], with_shape=with_shape, with_min_max=with_min_max)
            return f"({s},)"
        if len(obj) < 10:
            js = ",".join(
                string_type(o, with_shape=with_shape, with_min_max=with_min_max) for o in obj
            )
            return f"({js})"
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi = min(obj), max(obj)
            return f"(...)#{len(obj)}[{mini},{maxi}]"
        return f"(...)#{len(obj)}" if with_shape else "(...)"
    if isinstance(obj, list):
        if len(obj) < 10:
            js = ",".join(
                string_type(o, with_shape=with_shape, with_min_max=with_min_max) for o in obj
            )
            return f"#{len(obj)}[{js}]"
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi = min(obj), max(obj)
            return f"[...]#{len(obj)}[{mini},{maxi}]"
        return f"[...]#{len(obj)}" if with_shape else "[...]"
    if isinstance(obj, set):
        if len(obj) < 10:
            js = ",".join(
                string_type(o, with_shape=with_shape, with_min_max=with_min_max) for o in obj
            )
            return f"{{{js}}}"
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi = min(obj), max(obj)
            return f"{{...}}#{len(obj)}[{mini},{maxi}]"
        return f"{{...}}#{len(obj)}" if with_shape else "{...}"
    if isinstance(obj, dict):
        s = ",".join(
            f"{kv[0]}:{string_type(kv[1],with_shape=with_shape,with_min_max=with_min_max)}"
            for kv in obj.items()
        )
        return f"dict({s})"
    if isinstance(obj, np.ndarray):
        if with_min_max:
            s = string_type(obj, with_shape=with_shape)
            n_nan = np.isnan(obj.reshape((-1,))).astype(int).sum()
            if n_nan > 0:
                return f"{s}[{obj.min()}:{obj.max()}:{n_nan}nans]"
            return f"{s}[{obj.min()}:{obj.max()}]"
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
    if isinstance(obj, torch.Tensor):
        if with_min_max:
            s = string_type(obj, with_shape=with_shape)
            if obj.numel() == 0:
                return f"{s}[empty]"
            n_nan = obj.reshape((-1,)).isnan().to(int).sum()
            if n_nan > 0:
                if obj.dtype in {torch.complex64, torch.complex128}:
                    return f"{s}[{obj.abs().min()}:{obj.abs().max():{n_nan}nans}]"
                return f"{s}[{obj.min()}:{obj.max()}:{n_nan}nans]"
            if obj.dtype in {torch.complex64, torch.complex128}:
                return f"{s}[{obj.abs().min()}:{obj.abs().max()}]"
            return f"{s}[{obj.min()}:{obj.max()}]"
        i = torch_dtype_to_onnx_dtype(obj.dtype)
        if not with_shape:
            return f"T{i}r{len(obj.shape)}"
        return f"T{i}s{'x'.join(map(str, obj.shape))}"
    if isinstance(obj, int):
        if with_min_max:
            return f"int[{obj}]"
        return "int"
    if isinstance(obj, float):
        if with_min_max:
            return f"float[{obj}]"
        return "float"
    if isinstance(obj, str):
        return "str"
    if isinstance(obj, slice):
        return "slice"

    # others classes

    if type(obj).__name__ == "MambaCache":
        c = string_type(obj.conv_states, with_shape=with_shape, with_min_max=with_min_max)
        d = string_type(obj.ssm_states, with_shape=with_shape, with_min_max=with_min_max)
        return f"MambaCache(conv_states={c}, ssm_states={d})"
    if type(obj).__name__ == "Node" and hasattr(obj, "meta"):
        # torch.fx.node.Node
        return f"%{obj.target}"
    if type(obj).__name__ == "ValueInfoProto":
        return f"OT{obj.type.tensor_type.elem_type}"

    if obj.__class__.__name__ in ("DynamicCache", "patched_DynamicCache"):
        kc = string_type(obj.key_cache, with_shape=with_shape, with_min_max=with_min_max)
        vc = string_type(obj.value_cache, with_shape=with_shape, with_min_max=with_min_max)
        return f"{obj.__class__.__name__}(key_cache={kc}, value_cache={vc})"

    if obj.__class__.__name__ == "BatchFeature":
        s = string_type(obj.data, with_shape=with_shape, with_min_max=with_min_max)
        return f"BatchFeature(data={s})"

    if obj.__class__.__name__ == "BatchEncoding":
        s = string_type(obj.data, with_shape=with_shape, with_min_max=with_min_max)
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
                rows.append(f"{p}={v!r}")
            continue
        v = kwargs.get(p, d)
        if d != v:
            rows.append(f"{p}={v!r}")
            continue
    atts = ", ".join(rows)
    return f"{name}({atts})"


def pretty_onnx(onx: Union[FunctionProto, GraphProto, ModelProto, str]) -> str:
    """
    Displays an onnx prot in a better way.
    """
    assert onx is not None, "onx cannot be None"
    if isinstance(onx, str):
        onx = onnx_load(onx, load_external_data=False)
    assert onx is not None, "onx cannot be None"
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
    constraints: Dict[str, Set[str]], original: Set[str]
) -> Dict[str, str]:
    """
    Renames dynamic shapes as requested by the user. :func:`torch.export.export` uses
    many names for dynamic dimensions. When building the onnx model,
    some of them are redundant and can be replaced by the name provided by the user.

    :param constraints: exhaustive list of used name and all the values equal to it
    :param original: the names to use if possible
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
