from typing import Union
import torch
import torch.fx.experimental.symbolic_shapes as _tds

_torch_guard_or = _tds._guard_or


def _guard_or(a: "BoolLikeType", default: bool) -> bool:  # noqa: F821
    if not isinstance(a, _tds.SymBool):
        assert isinstance(a, bool)
        return a

    result = _tds._static_eval_sym_bool(a)
    return result if result is not None else default


def torch_export(
    *args,
    backed_size_oblivious: Union[bool, str] = False,
    oblivious: Union[bool, str] = False,
    **kwargs,
):
    """
    Wrapper around :func:`torch.export.export`.
    ``backed_size_oblivious`` can be boolean then it calls
    ``torch.fx.experimental._config.patch(backed_size_oblivious=True)``
    or not. It can be ``'auto'`` to let select automatically the best
    mode. It can be ``'half'`` to disable some non oblivious exceptions.
    """

    if backed_size_oblivious == "half":
        value = _tds.ShapeEnv._init.__kwdefaults__["specialize_zero_one"]
        _tds.ShapeEnv._init.__kwdefaults__["specialize_zero_one"] = False
        _tds._guard_or = _guard_or

        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = torch.export.export(*args, **kwargs)

        _tds._guard_or = _torch_guard_or
        _tds.ShapeEnv._init.__kwdefaults__["specialize_zero_one"] = value
        return ep

    if backed_size_oblivious == "auto":
        kws = kwargs.get("kwargs", None)
        ds = kwargs.get("dynamic_shapes", None)
        if not ds:
            # Unable to predict, calling the second recursively
            # to let the stacktrace keep a trace of it.
            return torch_export(*args, backed_size_oblivious=False, **kwargs)
        if isinstance(ds, tuple):
            inputs = args[1] if len(args) > 1 else kwargs.get("args", None)
            if not inputs:
                # Unable to predict, calling the second recursively
                # to let the stacktrace keep a trace of it.
                return torch_export(*args, backed_size_oblivious=False, **kwargs)
            assert not kws, (
                f"args and kwargs are specified for this call and dynamic_shapes are {ds}, "
                f"this is not implemented yet."
            )
            ars = inputs
        else:
            assert not kws
            ars = tuple()

        from onnx_diagnostic.export.dynamic_shapes import CoupleInputsDynamicShapes

        cpl = CoupleInputsDynamicShapes(ars, kws, ds)
        backed_size_oblivious = cpl.invalid_dimensions_for_export()

    if backed_size_oblivious:
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = torch.export.export(*args, **kwargs)
        return ep

    return torch.export.export(*args, **kwargs)
