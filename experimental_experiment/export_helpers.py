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
    verbose: int = 0,
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
        if verbose:
            print(f"[torch_export] backed_size_oblivious={backed_size_oblivious!r}")
        value = _tds.ShapeEnv._init.__kwdefaults__["specialize_zero_one"]
        _tds.ShapeEnv._init.__kwdefaults__["specialize_zero_one"] = False
        _tds._guard_or = _guard_or

        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = torch.export.export(*args, **kwargs)

        _tds._guard_or = _torch_guard_or
        _tds.ShapeEnv._init.__kwdefaults__["specialize_zero_one"] = value
        return ep

    if backed_size_oblivious == "auto":
        if verbose:
            print(f"[torch_export] backed_size_oblivious={backed_size_oblivious!r}")
        ds = kwargs.get("dynamic_shapes", {})
        if not ds:
            # Unable to predict, calling the second recursively
            # to let the stacktrace keep a trace of it.
            if verbose:
                print("[torch_export] no dynamic shapes, back to default behaviour")
            return torch_export(*args, backed_size_oblivious=False, verbose=verbose, **kwargs)
        kws = args[2] if len(args) > 2 else kwargs.get("kwargs", {})
        ags = args[1] if len(args) > 1 else kwargs.get("args", tuple())
        if isinstance(ds, tuple):
            if not ags:
                # Unable to predict, calling the second recursively
                # to let the stacktrace keep a trace of it.
                if verbose:
                    print(
                        f"[torch_export] dynamic_shapes={ds}, "
                        f"args is empty, back to default behaviour"
                    )
                return torch_export(*args, backed_size_oblivious=False, verbose=verbose, **kwargs)
            assert not kws, (
                f"args and kwargs are specified for this call and dynamic_shapes are {ds}, "
                f"this is not implemented yet."
            )

        from onnx_diagnostic.export.dynamic_shapes import CoupleInputsDynamicShapes

        if not kws and len(ags) != len(ds) and len(ds) == 1 and len(ds[0]) == len(ags):
            ds = ds[0]

        cpl = CoupleInputsDynamicShapes(ags, kws, ds)
        backed_size_oblivious = cpl.invalid_dimensions_for_export()
        if verbose:
            print(f"[torch_export] inferred backed_size_oblivious={backed_size_oblivious!r}")

    if backed_size_oblivious:
        if verbose:
            print(
                f"[torch_export] export starts with backed_size_oblivious={backed_size_oblivious}"
            )
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = torch.export.export(*args, **kwargs)
        return ep

    if verbose:
        print(f"[torch_export] export starts with backed_size_oblivious={backed_size_oblivious}")
    return torch.export.export(*args, **kwargs)
