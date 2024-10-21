from typing import Any, Callable, Tuple, Optional
import torch
from . import CACHE


def load_model(
    verbose: int = 0,
    cache: str = CACHE,
    device: str = "cuda",
    dtype: Optional[str] = None,
) -> Optional["model"]:  # noqa: F821
    """
    See https://github.com/pytorch/pytorch/issues/138195.

    :param verbose: verbosity
    :param device: where to put the model
    :param dtype: which type to use
    :return: model
    """
    from diffusers.models import FluxTransformer2DModel

    model_name = "black-forest-labs/FLUX.1-de"
    if verbose:
        print(f"[load_model] load {model_name!r}")
    tensor_dtype = getattr(torch, dtype, dtype)
    model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        cache_dir=cache,
        torch_dtype=tensor_dtype,
    ).to(device)
    if verbose:
        print("[load_model] done")
    return model


def get_model_inputs(
    verbose: int = 0,
    cache: str = CACHE,
    device: str = "cuda",
    dtype: Optional[str] = None,
) -> Tuple[Callable, Tuple[Any, ...]]:
    """Returns a model and its inputs."""

    tensor_dtype = getattr(torch, dtype, dtype)
    batch_size = 1
    text_maxlen = 4096
    latent_height, latent_width = 1024 // 8, 1024 // 8
    config = {"in_channels": 64, "joint_attention_dim": 4096, "pooled_projection_dim": 768}
    inputs = {
        "hidden_states": torch.randn(
            batch_size,
            (latent_height // 2) * (latent_width // 2),
            config["in_channels"],
            dtype=tensor_dtype,
            device=device,
        ),
        "encoder_hidden_states": torch.randn(
            batch_size,
            text_maxlen,
            config["joint_attention_dim"],
            dtype=tensor_dtype,
            device=device,
        ),
        "pooled_projections": torch.randn(
            batch_size, config["pooled_projection_dim"], dtype=tensor_dtype, device=device
        ),
        "timestep": torch.tensor([1.0] * batch_size, dtype=tensor_dtype, device=device),
        "img_ids": torch.randn(
            batch_size,
            (latent_height // 2) * (latent_width // 2),
            3,
            dtype=tensor_dtype,
            device=device,
        ),
        "txt_ids": torch.randn(batch_size, text_maxlen, 3, dtype=tensor_dtype, device=device),
        "guidance": torch.tensor([1.0] * batch_size, dtype=tensor_dtype, device=device),
    }

    return (
        lambda: load_model(verbose=verbose, device=device, dtype=dtype, cache=cache)
    ), inputs
