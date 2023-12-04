from typing import Any, Callable, Dict
from . import _aten_functions


def _register() -> Dict[str, Callable]:
    res = {}
    for k, v in _aten_functions.__dict__.items():
        if k.startswith("aten_"):
            options = {
                k: v,
                k.replace("_", "::"): v,
            }
            for c in options:
                if c in res:
                    raise RuntimeError(
                        f"Alias {c!r} for function {v} is already taken by {res[k]}."
                    )
            res.update(options)
    return res


registered_functions = _register()


def find_function(name: Any) -> Callable:
    if isinstance(name, str):
        if name not in registered_functions:
            raise RuntimeError(
                f"Unable find function {name!r} among "
                f"{', '.join(sorted(registered_functions))}."
            )
        return registered_functions[name]
    for att in ["__qualname__", "__name__"]:
        if hasattr(name, att):
            v = getattr(name, att)
            if v in registered_functions:
                return registered_functions[v]
    raise RuntimeError(f"Unable to interpret type {type(name)}: {name!r}.")
