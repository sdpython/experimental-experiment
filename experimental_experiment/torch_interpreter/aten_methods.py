from typing import Any, Callable, Dict, Optional
from . import _aten_methods


def _register() -> Dict[str, Callable]:
    res = {}
    for k, v in _aten_methods.__dict__.items():
        if k.startswith("aten_meth_"):
            assert v.__doc__, f"doc missing for {k!r} ({v})"
            options = {k: v}
            for c in options:
                if c in res:
                    raise RuntimeError(
                        f"Alias {c!r} for method {v} is already taken by {res[k]}."
                    )
            res.update(options)
    return res


registered_methods = _register()


def find_method(name: Any) -> Optional[Callable]:
    if isinstance(name, str):
        if name not in registered_methods:
            return None
        return registered_methods[name]

    return None
