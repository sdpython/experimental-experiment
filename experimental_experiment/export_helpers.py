import torch

_torch_guard_or = torch.fx.experimental.symbolic_shapes._guard_or


def _guard_or(a: "BoolLikeType", default: bool) -> bool:  # noqa: F821
    if not isinstance(a, torch.fx.experimental.symbolic_shapes.SymBool):
        assert isinstance(a, bool)
        return a

    result = torch.fx.experimental.symbolic_shapes._static_eval_sym_bool(a)
    return result if result is not None else default


def torch_export(*args, backed_size_oblivious: bool = False, **kwargs):
    """Wrapper around :func:`torch.export.export`."""

    if backed_size_oblivious:
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = torch.export.export(*args, **kwargs)
        return ep

    value = torch.fx.experimental.symbolic_shapes.ShapeEnv._init.__kwdefaults__[
        "specialize_zero_one"
    ]
    torch.fx.experimental.symbolic_shapes.ShapeEnv._init.__kwdefaults__["specialize_zero_one"] = (
        False
    )
    torch.fx.experimental.symbolic_shapes._guard_or = _guard_or

    with torch.fx.experimental._config.patch(backed_size_oblivious=True):
        ep = torch.export.export(*args, **kwargs)

    torch.fx.experimental.symbolic_shapes._guard_or = _torch_guard_or
    torch.fx.experimental.symbolic_shapes.ShapeEnv._init.__kwdefaults__["specialize_zero_one"] = (
        value
    )

    return ep
