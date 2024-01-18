from typing import Any, Callable, Dict, Optional
from . import _aten_methods


def _register() -> Dict[str, Callable]:
    res = {}
    for k, v in _aten_methods.__dict__.items():
        if k.startswith("aten_meth_"):
            options = {k: v}
            for c in options:
                if c in res:
                    raise RuntimeError(
                        f"Alias {c!r} for method {v} is already taken by {res[k]}."
                    )
            res.update(options)
    return res


registered_methods = _register()


def find_method(
    name: Any,
    args: Optional[Any] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    graph_builder: Optional["GraphBuilder"] = None,  # noqa: F821
) -> Callable:
    if isinstance(name, str):
        if name not in registered_methods:
            raise RuntimeError(
                f"Unable find method {name!r}, "
                f"args={args}, kwargs={kwargs}"
                f"{'' if graph_builder is None else graph_builder.get_debug_msg()}"
            )
        return registered_methods[name]

    raise RuntimeError(
        f"Unable to interpret method {type(name)}: {name!r}, "
        f"args={args}, kwargs={kwargs}"
        f"{'' if graph_builder is None else graph_builder.get_debug_msg()}"
    )
