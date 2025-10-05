from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
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
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any, ...], List[Any]]] = None,
    strict: bool = False,
    preserve_module_call_signature: Tuple[str, ...] = (),
    prefer_deferred_runtime_asserts_over_guards: bool = False,
    backed_size_oblivious: Union[bool, str] = False,
    verbose: int = 0,
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
            ep = torch.export.export(
                mod,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
                preserve_module_call_signature=preserve_module_call_signature,
                prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
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
                preserve_module_call_signature=preserve_module_call_signature,
                prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
                backed_size_oblivious=False,
                verbose=verbose,
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
                    preserve_module_call_signature=preserve_module_call_signature,
                    prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
                    backed_size_oblivious=False,
                    verbose=verbose,
                )
            assert not kwargs, (
                f"args and kwargs are specified for this call and dynamic_shapes "
                f"are {dynamic_shapes}, this is not implemented yet."
            )

        from onnx_diagnostic.export.dynamic_shapes import CoupleInputsDynamicShapes

        if (
            not kwargs
            and isinstance(dynamic_shapes, tuple)
            and len(dynamic_shapes) == 1
            and len(dynamic_shapes[0]) == len(args)
        ):
            dynamic_shapes = dynamic_shapes[0]
        if (
            not kwargs
            and isinstance(dynamic_shapes, dict)
            and len(args) == 1
            and all(isinstance(k, int) for k in dynamic_shapes)
        ):
            dynamic_shapes = (dynamic_shapes,)

        if not dynamic_shapes or (args and None in args):
            backed_size_oblivious = False
        else:
            try:
                cpl = CoupleInputsDynamicShapes(args, kwargs, dynamic_shapes)
                backed_size_oblivious = cpl.invalid_dimensions_for_export()
            except AssertionError as e:
                from onnx_diagnostic.helpers import string_type

                raise AssertionError(
                    f"Unable to guess backed_size_oblivious with "
                    f"args={string_type(args,with_shape=True)}, "
                    f"kwargs={string_type(kwargs,with_shape=True)}, "
                    f"dynamic_shapes={string_type(dynamic_shapes,with_shape=True)}"
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
                mod,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
                preserve_module_call_signature=preserve_module_call_signature,
                prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
            )
        return ep

    if verbose:
        print(f"[torch_export] export starts with backed_size_oblivious={backed_size_oblivious}")
    return torch.export.export(
        mod,
        args,
        kwargs,
        dynamic_shapes=dynamic_shapes,
        strict=strict,
        preserve_module_call_signature=preserve_module_call_signature,
        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
    )
