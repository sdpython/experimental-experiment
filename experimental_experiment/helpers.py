import inspect
from typing import Any, Callable, Dict, Optional, Union
import numpy as np
from onnx import FunctionProto, GraphProto, ModelProto, load as onnx_load


def string_type(obj: Any) -> str:
    """
    Displays the types of an object as a string.

    :param obj: any
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
            return f"({string_type(obj[0])},)"
        return f"({','.join(map(string_type, obj))})"
    if isinstance(obj, list):
        return f"[{','.join(map(string_type, obj))}]"
    if isinstance(obj, dict):
        s = ",".join(f"{kv[0]}:{string_type(kv[1])}" for kv in obj.items())
        return f"dict({s})"
    if isinstance(obj, np.ndarray):
        return f"A{len(obj.shape)}"

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
        return f"T{len(obj.shape)}"
    if isinstance(obj, int):
        return "int"
    if isinstance(obj, float):
        return "float"
    if isinstance(obj, str):
        return "str"
    if type(obj).__name__ == "MambaCache":
        return "MambaCache"
    if type(obj).__name__ == "Node" and hasattr(obj, "meta"):
        # torch.fx.node.Node
        return f"%{obj.target}"
    if type(obj).__name__ == "ValueInfoProto":
        return f"OT{obj.type.tensor_type.elem_type}"

    raise AssertionError(f"Unsupported type {type(obj).__name__!r} - {type(obj)}")


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
            return f"function: {onx.name}[{onx.domain}]\n{onnx_simple_text_plot(onx)}"
        return onnx_simple_text_plot(onx)
    except ImportError:
        from onnx.printer import to_text

        return to_text(onx)
