from typing import Any, Callable, Dict, Optional
from . import _aten_functions


def _register() -> Dict[str, Callable]:
    res = {}
    for k, v in _aten_functions.__dict__.items():
        if k.startswith("aten_"):
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


def find_function(
    name: Any,
    args: Optional[Any] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    graph_builder: Optional["GraphBuilder"] = None,  # noqa: F821
) -> Callable:
    if isinstance(name, str):
        if name not in registered_functions:
            raise RuntimeError(
                f"Unable find function {name!r}, "
                f"args={args}, kwargs={kwargs}"
                f"{'' if graph_builder is None else graph_builder.get_debug_msg()}"
            )
        return registered_functions[name]

    lookup = []
    if isinstance(name, type(abs)):
        # example: conv2d or _VariableFunctionsClass.conv2d
        new_name = f"aten_{name.__name__.replace('.', '_')}"
        if new_name in registered_functions:
            return registered_functions[new_name]
        lookup.append(new_name)

    lookup_names = ["__qualname__", "__name__"]
    for att in lookup_names:
        if hasattr(name, att):
            v = getattr(name, att).replace(".", "_")
            lookup.append(v)
            if v in registered_functions:
                return registered_functions[v]
    raise RuntimeError(
        f"Unable to interpret function {type(name)}: {name!r}, searched for "
        f"{lookup} and attributes {lookup_names}, "
        f"args={args}, kwargs={kwargs}"
        f"{'' if graph_builder is None else graph_builder.get_debug_msg()}"
    )
