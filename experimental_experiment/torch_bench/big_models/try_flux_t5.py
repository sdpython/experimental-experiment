from typing import Any, Callable, Tuple, Optional
import numpy as np
import torch
from . import CACHE


def load_model(
    verbose: int = 0,
    cache: str = CACHE,
    device: str = "cuda",
    dtype: Optional[str] = None,
) -> Optional["model"]:  # noqa: F821
    """
    See https://github.com/pytorch/pytorch/issues/138196.

    :param verbose: verbosity
    :param device: where to put the model
    :param dtype: which type to use
    :return: model
    """
    from transformers import T5EncoderModel

    model_name = "black-forest-labs/FLUX.1-de"
    if verbose:
        print(f"[load_model] load {model_name!r}")
    model = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="text_encoder_2",
        cache_dir=cache,
        torch_dtype=dtype,
    ).to(device)
    if verbose:
        print("[load_model] done")
    return model


def ids_tensor(shape, vocab_size):
    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(np.random.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


def get_model_inputs(
    verbose: int = 0,
    cache: str = CACHE,
    device: str = "cuda",
    dtype: Optional[str] = None,
) -> Tuple[Callable, Tuple[Any, ...]]:
    """Returns a model and its inputs."""

    input_ids = ids_tensor((1, 512), 23688).to(device)

    return (
        lambda: load_model(verbose=verbose, device=device, dtype=dtype, cache=cache)
    ), dict(input_ids=input_ids)
