import contextlib
from typing import Any, Dict, List, Tuple


def flatten_mamba_cache(
    mamba_cache: "MambaCache",  # noqa: F821
) -> Tuple[List[Any], "torch.utils._pytree.Context"]:  # noqa: F821
    flat = [
        (k, getattr(mamba_cache, k))
        for k in [
            "batch_size",
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


def flatten_with_keys_dynamic_cache(d: Dict[Any, Any]) -> Tuple[
    List[Tuple["torch.utils._pytree.KeyEntry", Any]],  # noqa: F821
    "torch.utils._pytree.Context",  # noqa: F821
]:
    import torch

    values, context = flatten_dynamic_cache(d)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


@contextlib.contextmanager
def bypass_export_some_errors(patch_transformers: bool = False, verbose: int = 0):
    """
    Tries to bypass some functions :func:`torch.export.export` does not
    support:

    * `torch.jit.isinstance`
    * `torch._dynamo.mark_static_address`
    * Serialialization of `MambaCache` (in :epkg:`transformers`)
    * Serialialization of `DynamicCache` (in :epkg:`transformers`)

    :param patch_transformers: patches transformers

    * ``AttentionMaskConverter._make_causal_mask``

    Serialization issues happen when a module takes one input or output
    has a type :func:`torch.export.export` cannot serialize.
    """
    import torch
    import torch.jit

    if verbose:
        print(
            "[bypass_export_some_errors] replace torch.jit.isinstance, "
            "torch._dynamo.mark_static_address"
        )
    f = torch.jit.isinstance
    torch.jit.isinstance = isinstance
    f2 = torch._dynamo.mark_static_address
    torch._dynamo.mark_static_address = lambda *_, **y_: None

    try:
        from transformers.cache_utils import MambaCache, DynamicCache
    except ImportError:
        MambaCache = None
        DynamicCache = None

    unregistered = True
    if MambaCache is not None and DynamicCache is not None:
        if MambaCache in torch.utils._pytree.SUPPORTED_NODES:
            # It is already registered because bypass_export_some_errors was called
            # within a section already calling bypass_export_some_errors or transformers
            # has updated its code to do it.
            # No need to register and unregister then.
            unregistered = False
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

        if DynamicCache in torch.utils._pytree.SUPPORTED_NODES:
            unregistered = False
        else:
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

    if patch_transformers:
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter
        from .patches.patch_transformers import patched_AttentionMaskConverter

        if verbose:
            print("[bypass_export_some_errors] patch AttentionMaskConverter._make_causal_mask")
        keep__make_causal_mask = AttentionMaskConverter._make_causal_mask
        AttentionMaskConverter._make_causal_mask = (
            patched_AttentionMaskConverter._make_causal_mask
        )

    try:
        yield
    finally:
        torch.jit.isinstance = f
        torch._dynamo.mark_static_address = f2
        if verbose:
            print(
                "[bypass_export_some_errors] restored torch.jit.isinstance, "
                "torch._dynamo.mark_static_address"
            )
        if unregistered and MambaCache is not None:
            torch.utils._pytree.SUPPORTED_NODES.pop(MambaCache)
            if verbose:
                print("[bypass_export_some_errors] unregistered MambaCache")
        if unregistered and DynamicCache is not None:
            torch.utils._pytree.SUPPORTED_NODES.pop(DynamicCache)
            torch.fx._pytree.SUPPORTED_NODES.pop(DynamicCache)
            torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH.pop(DynamicCache)
            if verbose:
                print("[bypass_export_some_errors] unregistered DynamicCache")
        if patch_transformers:
            AttentionMaskConverter._make_causal_mask = keep__make_causal_mask
            if verbose:
                print(
                    "[bypass_export_some_errors] restored "
                    "AttentionMaskConverter._make_causal_mask"
                )
