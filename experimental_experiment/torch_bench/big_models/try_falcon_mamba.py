from typing import Any, Callable, Tuple, Optional
import numpy as np
import torch
from . import CACHE, load_llm_model


def load_model(
    verbose: int = 0,
    load_tokenizer: bool = False,
    cache: str = CACHE,
    device: str = "cuda",
    dtype: Optional[str] = None,
) -> Tuple[Optional["tokenizer"], "model"]:  # noqa: F821
    """
    Loads `Falcon Mamba 7b
    <https://huggingface.co/tiiuae/falcon-mamba-7b>`_.

    :param verbose: verbosity
    :param load_tokenizer: loads the tokenizer as well
    :param device: where to put the model
    :param dtype: which type to use, it should be a string such as 'float16'
        to avoid the following error
        ``TypeError: Object of type dtype is not JSON serializable``
    :return: tokenizer, model

    There are several paths for the implementation.
    One faster one requires `causal_conv1d
    <https://github.com/Dao-AILab/causal-conv1d>`_ to be installed.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    return load_llm_model(
        "FalconMamba-7b",
        "tiiuae/falcon-mamba-7b",
        AutoTokenizer,
        AutoModelForCausalLM,
        cache=cache,
        verbose=verbose,
        device=device,
        dtype=dtype,
        load_tokenizer=load_tokenizer,
    )


def demo_model(
    tokenizer: "tokenizer",  # noqa: F821
    model: "model",  # noqa: F821
    inputs: Optional[Any] = None,
    verbose: int = 0,
    device: str = "cuda",
    max_new_tokens: int = 128,
    skip_special_tokens: bool = True,
    model_name: str = "FalconMamba-7b",
) -> Any:
    """
    Demonstrates the models.
    This example is usually taken from the documentation.

    :param tokenizer: tokenizer
    :param model: model
    :param inputs: a string in this case
    :param verbose: verbosity
    :param model_name: only used for verbosity
    :return: results (a string in this case)
    """
    prompt = inputs or "Question: How many hours in one day? Answer: "
    with torch.no_grad():
        if verbose:
            print(f"[demo_model] tokenize the input for {model_name}")
            print(prompt)
            print("--")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        if verbose:
            print(
                f"[demo_model] input_ids: shape={input_ids.shape}, "
                f"dtype={input_ids.dtype}, min={input_ids.min()}, max={input_ids.max()}, "
                f"avg={input_ids.to(float).mean()}"
            )
            print(f"[demo_model] generates the token for {model_name}")
        generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
        output = tokenizer.decode(generated_ids[0])
        if verbose:
            print(output)
            print("[demo_model] done")
        return output


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

    input_ids = ids_tensor((1, 12), 23688).to(device)

    return (
        lambda: load_model(
            load_tokenizer=False, verbose=verbose, device=device, dtype=dtype, cache=cache
        )[1]
    ), (input_ids,)
