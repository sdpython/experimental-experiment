from typing import Any, Dict, List, Tuple

############
# MambaCache
############


# self.conv_states: torch.Tensor = torch.zeros(
#     config.num_hidden_layers,
#     self.max_batch_size,
#     self.intermediate_size,
#     self.conv_kernel_size,
#     device=device,
#     dtype=dtype,
# )
# self.ssm_states: torch.Tensor = torch.zeros(
#     config.num_hidden_layers,
#     self.max_batch_size,
#     self.intermediate_size,
#     self.ssm_state_size,
#     device=device,
#     dtype=dtype,
# )
def flatten_mamba_cache(
    mamba_cache: "MambaCache",  # noqa: F821
) -> Tuple[List[Any], "torch.utils._pytree.Context"]:  # noqa: F821
    """Serializes a :class:`transformers.cache_utils.MambaCache` with python objects."""
    flat = [
        (k, getattr(mamba_cache, k))
        for k in [
            # "max_batch_size",  # new in transformers==4.47
            # "intermediate_size",
            # "ssm_state_size",
            # "conv_kernel_size",
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
    """Restores a :class:`transformers.cache_utils.MambaCache` from python objects."""

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
    """Serializes a :class:`transformers.cache_utils.MambaCache` with python objects."""
    import torch

    values, context = flatten_mamba_cache(d)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


##############
# DynamicCache
##############


def flatten_dynamic_cache(
    dynamic_cache: "DynamicCache",  # noqa: F821
) -> Tuple[List[Any], "torch.utils._pytree.Context"]:  # noqa: F821
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    flat = [
        (k, getattr(dynamic_cache, k))
        for k in ["key_cache", "value_cache"]
        if hasattr(dynamic_cache, k)
    ]
    return [f[1] for f in flat], [f[0] for f in flat]


def flatten_with_keys_dynamic_cache(d: Dict[Any, Any]) -> Tuple[
    List[Tuple["torch.utils._pytree.KeyEntry", Any]],  # noqa: F821
    "torch.utils._pytree.Context",  # noqa: F821
]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    import torch

    values, context = flatten_dynamic_cache(d)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def unflatten_dynamic_cache(
    values: List[Any],
    context: "torch.utils._pytree.Context",  # noqa: F821
    output_type=None,
) -> "DynamicCache":  # noqa: F821
    """Restores a :class:`transformers.cache_utils.DynamicCache` from python objects."""
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
    """Restores a :class:`patched_DynamicCache` from python objects."""

    from .patches.patch_transformers import patched_DynamicCache

    cache = patched_DynamicCache()
    values = dict(zip(context, values))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache
