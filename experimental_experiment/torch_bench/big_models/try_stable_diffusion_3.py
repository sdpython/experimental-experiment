import os
from typing import Any, Callable, Tuple, Optional
import numpy as np
import torch
from . import str_dtype

CACHE = os.environ.get("BASH_BENCH_CACHE", "_bigcache")


def load_model(
    verbose: int = 0,
    cache: str = CACHE,
    device: str = "cuda",
    dtype: Optional[str] = None,
) -> Optional["model"]:  # noqa: F821
    """
    Loads `stable-diffusion-3-medium
    <https://huggingface.co/stabilityai/stable-diffusion-3-medium>`_.

    :param verbose: verbosity
    :param device: where to put the model
    :param dtype: which type to use
    :return: tokenizer, model
    """
    from diffusers import StableDiffusion3Pipeline

    assert isinstance(dtype, str), f"Unexpected type for dtype={dtype!r}"
    stype = str_dtype(dtype) if dtype is not None else ""

    if os.path.exists(os.path.join(cache, f"StableDiffusion3Medium{stype}")):
        if verbose:
            print("[load_model] loads cached codellama model")
        model = StableDiffusion3Pipeline.from_pretrained(
            os.path.join(cache, f"StableDiffusion3Medium{stype}")
        )
    else:
        if verbose:
            print("[load_model] retrieves codellama model")
        model = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=dtype
        )
        if verbose:
            print("[load_model] cache codellama model")
        model.save_pretrained(os.path.join(cache, f"StableDiffusion3Medium{stype}"))

    if verbose:
        print(f"[load_model] converts to {device!r}")

    model = model.to(device)
    # if dtype:
    #     model = model.to(dtype)
    if verbose:
        print("[load_model] done-stable-diffusion-3-medium")
    return model


def demo_model(
    model: "model",  # noqa: F821
    inputs: Optional[Any] = None,
    verbose: int = 0,
    num_inference_steps: int = 28,
    guidance_scale: float = 0.7,
) -> Any:
    """
    Demonstrates the models.
    This example is usually taken from the documentation.

    :param tokenizer: tokenizer
    :param model: model
    :param inputs: a string in this case
    :param verbose: verbosity
    :return: results (a string in this case)
    """
    prompt = inputs or """Which animal for the president?"""
    if isinstance(prompt, tuple):
        prompt, negative_prompt = prompt
    else:
        negative_prompt = ""
    with torch.no_grad():
        if verbose:
            print("[demo_model] run the model")
        generated_images = model(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        if verbose:
            print("[demo_model] interpret the answer")
        if verbose:
            print(f"[demo model] len(generated_images.images)={len(generated_images.images)}")
            print("[demo_model] done")
            generated_images.images[0].save("try_stable_diffusion_3.demo_model.png")
        return generated_images.images[0]


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
    """Returns a codellama model and its inputs."""

    input_ids = ids_tensor((1, 128), 32016).to(device)

    return (lambda: load_model(verbose=verbose, device=device, dtype=dtype, cache=cache)), (
        input_ids,
    )
