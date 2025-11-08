from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import torch
import torch.fx.experimental.symbolic_shapes as _tds
from onnx_diagnostic.helpers import flatten_object

_torch_guard_or = _tds._guard_or


def _guard_or(a: "BoolLikeType", default: bool) -> bool:  # noqa: F821
    if not isinstance(a, _tds.SymBool):
        assert isinstance(a, bool)
        return a

    result = _tds._static_eval_sym_bool(a)
    return result if result is not None else default


def torch_export(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any, ...], List[Any]]] = None,
    strict: bool = False,
    preserve_module_call_signature: Tuple[str, ...] = (),
    # prefer_deferred_runtime_asserts_over_guards: bool = False,  # torch==2.9
    backed_size_oblivious: Union[bool, str] = False,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
    verbose: int = 0,
    **other_kwargs,
):
    """
    Wrapper around :func:`torch.export.export`.
    ``backed_size_oblivious`` can be boolean then it calls
    ``torch.fx.experimental._config.patch(backed_size_oblivious=True)``
    or not. It can be ``'auto'`` to let select automatically the best
    mode. It can be ``'half'`` to disable some non oblivious exceptions.
    """
    export_kwargs = {}
    if prefer_deferred_runtime_asserts_over_guards:
        export_kwargs["prefer_deferred_runtime_asserts_over_guards"] = (
            prefer_deferred_runtime_asserts_over_guards
        )
    if preserve_module_call_signature:
        export_kwargs["preserve_module_call_signature"] = preserve_module_call_signature

    export_kwargs.update(other_kwargs)
    if backed_size_oblivious == "half":
        if verbose:
            print(f"[torch_export] backed_size_oblivious={backed_size_oblivious!r}")
        value = _tds.ShapeEnv._init.__kwdefaults__["specialize_zero_one"]
        _tds.ShapeEnv._init.__kwdefaults__["specialize_zero_one"] = False
        _tds._guard_or = _guard_or

        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = torch.export.export(
                mod, args, kwargs, dynamic_shapes=dynamic_shapes, strict=strict, **export_kwargs
            )

        _tds._guard_or = _torch_guard_or
        _tds.ShapeEnv._init.__kwdefaults__["specialize_zero_one"] = value
        return ep

    if backed_size_oblivious == "auto":
        if verbose:
            print(f"[torch_export] backed_size_oblivious={backed_size_oblivious!r}")

        if not dynamic_shapes:
            # Unable to predict, calling the second recursively
            # to let the stacktrace keep a trace of it.
            if verbose:
                print("[torch_export] no dynamic shapes, back to default behaviour")
            return torch_export(
                mod,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
                backed_size_oblivious=False,
                verbose=verbose,
                **export_kwargs,
            )

        if isinstance(dynamic_shapes, tuple):
            if not args:
                # Unable to predict, calling the second recursively
                # to let the stacktrace keep a trace of it.
                if verbose:
                    print(
                        f"[torch_export] dynamic_shapes={dynamic_shapes}, "
                        f"args is empty, back to default behaviour"
                    )
                return torch_export(
                    mod,
                    args,
                    kwargs,
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                    backed_size_oblivious=False,
                    verbose=verbose,
                    **export_kwargs,
                )
            assert not kwargs, (
                f"args and kwargs are specified for this call and dynamic_shapes "
                f"are {dynamic_shapes}, this is not implemented yet."
            )

        from onnx_diagnostic.export.dynamic_shapes import CoupleInputsDynamicShapes

        ags, kws, ds = args, kwargs, dynamic_shapes

        if (
            ags
            and isinstance(ds, tuple)
            and len(ds) == 1
            and len(ds[0]) == len(ags)
            and isinstance(ds[0], tuple)
        ):
            ds = ds[0]

        if not ds or (args and None in ags):
            backed_size_oblivious = False
        else:
            from torch._subclasses.fake_tensor import FakeTensor

            if not any(
                isinstance(f, FakeTensor) for f in flatten_object([ags, kws], drop_keys=True)
            ):
                try:
                    cpl = CoupleInputsDynamicShapes(ags, kws, ds)
                    backed_size_oblivious = cpl.invalid_dimensions_for_export()
                except AssertionError as e:
                    from onnx_diagnostic.helpers import string_type

                    raise AssertionError(
                        f"Unable to guess backed_size_oblivious with "
                        f"args={string_type(ags,with_shape=True)}, "
                        f"kwargs={string_type(kws,with_shape=True)}, "
                        f"dynamic_shapes={string_type(ds,with_shape=True)}"
                    ) from e

        if verbose:
            print(f"[torch_export] inferred backed_size_oblivious={backed_size_oblivious!r}")

    if backed_size_oblivious:
        if verbose:
            print(
                f"[torch_export] export starts with backed_size_oblivious={backed_size_oblivious}"
            )
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = torch.export.export(
                mod, args, kwargs, dynamic_shapes=dynamic_shapes, strict=strict, **export_kwargs
            )
        ep._computed_backed_size_oblivious = backed_size_oblivious
        return ep

    if verbose:
        print(f"[torch_export] export starts with backed_size_oblivious={backed_size_oblivious}")
    if strict:
        return torch.export.export(
            mod, args, kwargs, dynamic_shapes=dynamic_shapes, strict=strict, **export_kwargs
        )
    try:
        return torch.export.export(
            mod, args, kwargs, dynamic_shapes=dynamic_shapes, strict=strict, **export_kwargs
        )
    except RuntimeError as e:
        # This happens when tensor.data_ptr() is accessed.
        if "Cannot access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor)" in str(e):
            return torch.export.export(
                mod, args, kwargs, dynamic_shapes=dynamic_shapes, strict=True, **export_kwargs
            )
        raise
