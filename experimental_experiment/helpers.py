import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
from onnx import FunctionProto, GraphProto, ModelProto, load as onnx_load
from onnx.helper import np_dtype_to_tensor_dtype


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
        js = ",".join(
            string_type(o, with_shape=with_shape, with_min_max=with_min_max) for o in obj
        )
        return f"({js})"
    if isinstance(obj, list):
        js = ",".join(
            string_type(o, with_shape=with_shape, with_min_max=with_min_max) for o in obj
        )
        return f"#{len(obj)}[{js}]"
    if isinstance(obj, set):
        js = ",".join(
            string_type(o, with_shape=with_shape, with_min_max=with_min_max) for o in obj
        )
        return f"{{{js}}}"
    if isinstance(obj, dict):
        s = ",".join(
            f"{kv[0]}:{string_type(kv[1],with_shape=with_shape,with_min_max=with_min_max)}"
            for kv in obj.items()
        )
        return f"dict({s})"
    if isinstance(obj, np.ndarray):
        if with_min_max:
            s = string_type(obj, with_shape=with_shape)
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
            return f"{s}[{obj.min()}:{obj.max()}]"
        from .xbuilder._dtype_helper import torch_dtype_to_onnx_dtype

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
    if type(obj).__name__ == "MambaCache":
        return "MambaCache"
    if type(obj).__name__ == "Node" and hasattr(obj, "meta"):
        # torch.fx.node.Node
        return f"%{obj.target}"
    if type(obj).__name__ == "ValueInfoProto":
        return f"OT{obj.type.tensor_type.elem_type}"

    if obj.__class__.__name__ == "DynamicCache":
        kc = string_type(obj.key_cache, with_shape=with_shape, with_min_max=with_min_max)
        vc = string_type(obj.value_cache, with_shape=with_shape, with_min_max=with_min_max)
        return f"DynamicCache(key_cache={kc}, DynamicCache(value_cache={vc})"

    if obj.__class__.__name__ == "BatchFeature":
        s = string_type(obj.data, with_shape=with_shape, with_min_max=with_min_max)
        return f"BatchFeature(data={s})"

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
    if isinstance(onx, str):
        onx = onnx_load(onx, load_external_data=False)
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
            sig.append((i.name, [dt.sequence_type.elem_type]))
        elif dt.HasField("tensor_type"):
            el = dt.tensor_type.elem_type
            shape = tuple(d.dim_param or d.dim_value for d in dt.tensor_type.shape.dim)
            sig.append((i.name, el, shape))
    return tuple(sig)
