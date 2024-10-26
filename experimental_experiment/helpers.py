from typing import Any
import numpy as np


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

    raise AssertionError(f"Unsupported type {type(obj).__name__!r} - {type(obj)}")
