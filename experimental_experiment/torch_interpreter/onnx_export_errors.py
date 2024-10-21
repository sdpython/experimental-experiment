import contextlib
from typing import Any, List, Tuple


def flatten_mamba_cache(
    mamba_cache: "MambaCache",  # noqa: F821
) -> Tuple[List[Any], "_torch_pytree.Context"]:  # noqa: F821
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
    context: "_torch_pytree.Context",  # noqa: F821
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


@contextlib.contextmanager
def bypass_export_some_errors():
    """
    Tries to bypass some functions torch.export.export does not
    support such as ``torch.jit.isinstance``.
    """
    import torch.jit
    import torch.utils._pytree as _torch_pytree

    f = torch.jit.isinstance
    torch.jit.isinstance = isinstance
    f2 = torch._dynamo.mark_static_address
    torch._dynamo.mark_static_address = lambda *_, **y_: None

    try:
        from transformers.cache_utils import MambaCache
    except ImportError:
        MambaCache = None

    unregistered = True
    if MambaCache is not None:
        if MambaCache in _torch_pytree.SUPPORTED_NODES:
            # It is already registered because bypass_export_some_errors was called
            # within a section already calling bypass_export_some_errors or transformers
            # has updated its code to do it.
            # No need to register and unregister then.
            unregistered = False
        else:
            _torch_pytree.register_pytree_node(
                MambaCache,
                flatten_mamba_cache,
                unflatten_mamba_cache,
                serialized_type_name=f"{MambaCache.__module__}.{MambaCache.__name__}",
            )

    try:
        yield
    finally:
        torch.jit.isinstance = f
        torch._dynamo.mark_static_address = f2
        if unregistered and MambaCache is not None:
            _torch_pytree.SUPPORTED_NODES.pop(MambaCache)
