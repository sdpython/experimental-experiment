from typing import Any, Callable, Dict, List, Optional, Tuple
from . import _aten_functions, _prims_functions


def _enumerate_aten_functions():
    for k, v in _aten_functions.__dict__.items():
        if not k.startswith("aten_") or not callable(v):
            continue
        assert v.__doc__, f"doc missing for {k!r} ({v})"
        yield k, v


def _enumerate_prims_functions():
    for k, v in _prims_functions.__dict__.items():
        if not k.startswith("prims_") or not callable(v):
            continue
        assert v.__doc__, f"doc missing for {k!r} ({v})"
        yield k, v


def _register() -> Dict[str, Callable]:
    res = {}
    for k, v in _enumerate_aten_functions():
        other_key = "::".join(k.split("_", maxsplit=1))
        options = {k: v, other_key: v}
        for c in options:
            if c in res:
                raise RuntimeError(
                    f"Alias {c!r} for function {v} is already taken by {res[k]}."
                )
        res.update(options)
    for k, v in _enumerate_prims_functions():
        other_key = "::".join(k.split("_", maxsplit=1))
        options = {k: v, other_key: v}
        for c in options:
            if c in res:
                raise RuntimeError(
                    f"Alias {c!r} for function {v} is already taken by {res[k]}."
                )
        res.update(options)
    return res


registered_functions = _register()


def find_function(name: Any) -> Tuple[Optional[Callable], List[str], List[str]]:
    if isinstance(name, str):
        if name not in registered_functions:
            return None, [name], []
        return registered_functions[name], [name], []

    lookup = []
    if isinstance(name, type(abs)):
        # example: conv2d or _VariableFunctionsClass.conv2d
        new_name = f"aten_{name.__name__.replace('.', '_')}"
        lookup.append(new_name)
        if new_name in registered_functions:
            return registered_functions[new_name], lookup, []

    lookup_names = ["__qualname__", "__name__"]
    for att in lookup_names:
        if hasattr(name, att):
            v = getattr(name, att).replace(".", "_")
            lookup.append(v)
            if v in registered_functions:
                return registered_functions[v], lookup, lookup_names

    return None, lookup, lookup_names
