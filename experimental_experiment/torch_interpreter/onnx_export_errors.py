import contextlib
import inspect
from typing import Any, Callable, Dict, List, Tuple, Union


def flatten_mamba_cache(
    mamba_cache: "MambaCache",  # noqa: F821
) -> Tuple[List[Any], "torch.utils._pytree.Context"]:  # noqa: F821
    flat = [
        (k, getattr(mamba_cache, k))
        for k in [
            "max_batch_size",  # new in transformers 4.47
            "intermediate_size",
            "ssm_state_size",
            "conv_kernel_size",
            "conv_states",
            "ssm_states",
        ]
        if hasattr(mamba_cache, k)
    ]
    return [f[1] for f in flat], [f[0] for f in flat]


def unflatten_mamba_cache(
    values: List[Any],
    context: "torch.utils._pytree.Context",  # noqa: F821
    output_type=None,
) -> "MambaCache":  # noqa: F821

    class _config:
        def __init__(self):
            self.intermediate_size = 16
            self.state_size = 16
            self.conv_kernel = 16
            self.num_hidden_layers = 16

    from transformers.cache_utils import MambaCache

    cache = MambaCache(_config(), batch_size=1, dtype=values[-1].dtype)
    values = dict(zip(context, values))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache


def flatten_with_keys_mamba_cache(d: Dict[Any, Any]) -> Tuple[
    List[Tuple["torch.utils._pytree.KeyEntry", Any]],  # noqa: F821
    "torch.utils._pytree.Context",  # noqa: F821
]:
    import torch

    values, context = flatten_mamba_cache(d)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def flatten_dynamic_cache(
    dynamic_cache: "DynamicCache",  # noqa: F821
) -> Tuple[List[Any], "torch.utils._pytree.Context"]:  # noqa: F821
    flat = [
        (k, getattr(dynamic_cache, k))
        for k in ["key_cache", "value_cache"]
        if hasattr(dynamic_cache, k)
    ]
    return [f[1] for f in flat], [f[0] for f in flat]


def unflatten_dynamic_cache(
    values: List[Any],
    context: "torch.utils._pytree.Context",  # noqa: F821
    output_type=None,
) -> "DynamicCache":  # noqa: F821

    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    values = dict(zip(context, values))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache


def unflatten_pached_dynamic_cache(
    values: List[Any],
    context: "torch.utils._pytree.Context",  # noqa: F821
    output_type=None,
) -> "DynamicCache":  # noqa: F821

    from .patches.patch_transformers import patched_DynamicCache

    cache = patched_DynamicCache()
    values = dict(zip(context, values))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache


def flatten_with_keys_dynamic_cache(d: Dict[Any, Any]) -> Tuple[
    List[Tuple["torch.utils._pytree.KeyEntry", Any]],  # noqa: F821
    "torch.utils._pytree.Context",  # noqa: F821
]:
    import torch

    values, context = flatten_dynamic_cache(d)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def _catch_produce_guards_and_solve_constraints(
    previous_function: Callable,
    fake_mode: "FakeTensorMode",  # noqa: F821
    gm: "torch.fx.GraphModule",  # noqa: F821
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any], None],
    equalities_inputs: "EqualityConstraint",  # noqa: F821
    original_signature: inspect.Signature,
    _is_torch_jit_trace: bool = False,
    verbose: int = 0,
):
    try:
        return previous_function(
            fake_mode=fake_mode,
            gm=gm,
            dynamic_shapes=dynamic_shapes,
            equalities_inputs=equalities_inputs,
            original_signature=original_signature,
            _is_torch_jit_trace=_is_torch_jit_trace,
        )
    except Exception as e:
        if verbose:
            print(
                f"[_catch_produce_guards_and_solve_constraints] ERROR"
                f"produce_guards_and_solve_constraints failed\n"
                f"fake_mode={fake_mode}\n"
                f"dynamic_shapes={dynamic_shapes}\n"
                f"equalities_inputs={equalities_inputs}\n"
                f"original_signature={original_signature}\n"
                f"_is_torch_jit_trace={_is_torch_jit_trace}\n"
                f"exc={e}\ngm={gm}"
            )


@contextlib.contextmanager
def bypass_export_some_errors(
    patch_transformers: bool = False, replace_dynamic_cache: bool = False, verbose: int = 0
):
    """
    Tries to bypass some functions :func:`torch.export.export` does not
    support:

    * `torch.jit.isinstance`
    * `torch._dynamo.mark_static_address`
    * Serialialization of `MambaCache` (in :epkg:`transformers`)
    * Serialialization of `DynamicCache` (in :epkg:`transformers`)
    * reduce errors due to shape inference

    :param patch_transformers: patches transformers with supported implementation
    :param replace_dynamic_cache: replaces DynamicCache by a patched class
        avoiding issues with the dynamic shapes inferences,
        it should be True with LLM using that class and only during the export

    * ``AttentionMaskConverter._make_causal_mask``

    Serialization issues happen when a module takes one input or output
    has a type :func:`torch.export.export` cannot serialize.
    """
    import torch
    import torch.jit
    import torch._export.non_strict_utils  # produce_guards_and_solve_constraints

    if verbose:
        print(
            "[bypass_export_some_errors] replace torch.jit.isinstance, "
            "torch._dynamo.mark_static_address"
        )

    ###############
    # patch pytorch
    ###############
    if verbose:
        print("[bypass_export_some_errors] patch pytorch")

    # torch.jit.isinstance
    f_jit_isinstance = torch.jit.isinstance
    torch.jit.isinstance = isinstance

    # torch._dynamo.mark_static_address
    f_mark_static_address = torch._dynamo.mark_static_address
    torch._dynamo.mark_static_address = lambda *_, **y_: None

    # torch._export.non_strict_utils.produce_guards_and_solve_constraints
    f_produce_guards_and_solve_constraints = (
        torch._export.non_strict_utils.produce_guards_and_solve_constraints
    )
    torch._export.non_strict_utils.produce_guards_and_solve_constraints = (
        lambda *args, **kwargs: _catch_produce_guards_and_solve_constraints(
            f_produce_guards_and_solve_constraints, *args, verbose=verbose, **kwargs
        )
    )

    ####################
    # patch transformers
    ####################

    if patch_transformers:
        import transformers
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter
        from .patches.patch_transformers import (
            patched_AttentionMaskConverter,
            patched_DynamicCache,
        )

        def raise_assert():
            raise AssertionError("One replacement of DynamicCache was not patched.")

        if verbose:
            print("[bypass_export_some_errors] patch transformers")
        keep__make_causal_mask = AttentionMaskConverter._make_causal_mask

    if replace_dynamic_cache:
        if verbose:
            print("[bypass_export_some_errors] replace DynamicCache")
        AttentionMaskConverter._make_causal_mask = (
            patched_AttentionMaskConverter._make_causal_mask
        )

        keep_DynamicCache = transformers.cache_utils.DynamicCache
        keep_DynamicCache_init = keep_DynamicCache.__init__
        keep_DynamicCache.__init__ = lambda *args, **kwargs: raise_assert()
        transformers.cache_utils.DynamicCache = patched_DynamicCache
        transformers.models.phi.modeling_phi.DynamicCache = patched_DynamicCache

    #####################
    # Cache Serialization
    #####################

    # Cache serialization: to be moved into appropriate packages
    try:
        from transformers.cache_utils import MambaCache, DynamicCache
    except ImportError:
        MambaCache = None
        DynamicCache = None

    # MambaCache
    unregistered_mamba_cache = True
    if MambaCache is not None and MambaCache in torch.utils._pytree.SUPPORTED_NODES:
        # It is already registered because bypass_export_some_errors was called
        # within a section already calling bypass_export_some_errors or transformers
        # has updated its code to do it.
        # No need to register and unregister then.
        unregistered_mamba_cache = False
    else:
        if verbose:
            print("[bypass_export_some_errors] register MambaCache")
        torch.utils._pytree.register_pytree_node(
            MambaCache,
            flatten_mamba_cache,
            unflatten_mamba_cache,
            serialized_type_name=f"{MambaCache.__module__}.{MambaCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_mamba_cache,
        )

    # DynamicCache
    unregistered_dynamic_cache = True
    if DynamicCache is not None and DynamicCache in torch.utils._pytree.SUPPORTED_NODES:
        unregistered_dynamic_cache = False
    else:
        from .patches.patch_transformers import patched_DynamicCache

        if not patch_transformers:
            if verbose:
                print("[bypass_export_some_errors] register DynamicCache")
            torch.utils._pytree.register_pytree_node(
                DynamicCache,
                flatten_dynamic_cache,
                unflatten_dynamic_cache,
                serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
                flatten_with_keys_fn=flatten_with_keys_dynamic_cache,
            )
            torch.fx._pytree.register_pytree_flatten_spec(
                DynamicCache, lambda x, _: [x.key_cache, x.value_cache]
            )

        if verbose:
            print("[bypass_export_some_errors] register patched_DynamicCache")
        torch.utils._pytree.register_pytree_node(
            patched_DynamicCache,
            flatten_dynamic_cache,
            unflatten_pached_dynamic_cache,
            serialized_type_name=f"{patched_DynamicCache.__module__}.{patched_DynamicCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_dynamic_cache,
        )
        torch.fx._pytree.register_pytree_flatten_spec(
            patched_DynamicCache, lambda x, _: [x.key_cache, x.value_cache]
        )

    # export

    try:
        yield replacement_before_exporting
    finally:

        # restores everything

        # torch
        torch.jit.isinstance = f_jit_isinstance
        torch._dynamo.mark_static_address = f_mark_static_address
        torch._export.non_strict_utils.produce_guards_and_solve_constraints = (
            f_produce_guards_and_solve_constraints
        )

        if verbose:
            print("[bypass_export_some_errors] restored pytorch functions")

        if patch_transformers:
            AttentionMaskConverter._make_causal_mask = keep__make_causal_mask
            if verbose:
                print("[bypass_export_some_errors] restored transformer")

        if replace_dynamic_cache:
            keep_DynamicCache.__init__ = keep_DynamicCache_init
            transformers.cache_utils.DynamicCache = keep_DynamicCache
            transformers.models.phi.modeling_phi.DynamicCache = keep_DynamicCache
            if verbose:
                print("[bypass_export_some_errors] restored DynamicCache")

        if unregistered_mamba_cache and MambaCache is not None:
            torch.utils._pytree.SUPPORTED_NODES.pop(MambaCache)
            if verbose:
                print("[bypass_export_some_errors] unregistered MambaCache")

        if unregistered_dynamic_cache and DynamicCache is not None:
            from .patches.patch_transformers import patched_DynamicCache

            if not patch_transformers:
                torch.utils._pytree.SUPPORTED_NODES.pop(DynamicCache)
                torch.fx._pytree.SUPPORTED_NODES.pop(DynamicCache)
                torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH.pop(DynamicCache)
                if verbose:
                    print("[bypass_export_some_errors] unregistered DynamicCache")
            torch.utils._pytree.SUPPORTED_NODES.pop(patched_DynamicCache)
            torch.fx._pytree.SUPPORTED_NODES.pop(patched_DynamicCache)
            torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH.pop(patched_DynamicCache)
            if verbose:
                print("[bypass_export_some_errors] unregistered patched_DynamicCache")


def replacement_before_exporting(args: Any) -> Any:
    """
    Does replacements on the given inputs such replacing
    :class:`transformers.cache_utils.DynamicCache` by
    :class:`experimental_experiment.torch_interpreter.patches.
    patched_transformers.patched_DynamicCache`.
    """
    if args is None:
        return None
    if isinstance(args, (int, float)):
        return args
    if isinstance(args, dict):
        return {k: replacement_before_exporting(v) for k, v in args.items()}
    if isinstance(args, tuple):
        return tuple(replacement_before_exporting(v) for v in args)
    if isinstance(args, list):
        return [replacement_before_exporting(v) for v in args]

    if args.__class__.__name__ == "DynamicCache":
        # Do not use isinstance, the class may have been replaced.
        from .patches.patch_transformers import patched_DynamicCache

        patched = patched_DynamicCache()
        for k in ["_seen_tokens", "key_cache", "value_cache"]:
            setattr(patched, k, getattr(args, k))
        return patched

    return args
