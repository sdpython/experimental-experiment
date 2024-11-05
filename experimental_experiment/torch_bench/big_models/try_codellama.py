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
    Loads `code_llama
    <https://huggingface.co/docs/transformers/en/model_doc/code_llama>`_.

    :param verbose: verbosity
    :param load_tokenizer: loads the tokenizer as well
    :param device: where to put the model
    :param dtype: which type to use, it should be a string such as 'float16'
        to avoid the following error
        ``TypeError: Object of type dtype is not JSON serializable``
    :return: tokenizer, model
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    return load_llm_model(
        "CodeLlama-7b",
        "codellama/CodeLlama-7b-hf",
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
    prompt = (
        inputs
        or '''
    def remove_non_ascii_characters_in_a_file(
        filename:str,
    ) -> str:
        """ <FILL_ME> """
    '''
    )
    with torch.no_grad():
        if verbose:
            print("[demo_model] tokenize the input for codellama")
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(device)
        if verbose:
            print("[demo_model] generates the token (codellama)")
        generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
        if verbose:
            print("[demo_model] interpret the answer")
        filling = tokenizer.batch_decode(
            generated_ids[:, input_ids.shape[1] :], skip_special_tokens=skip_special_tokens
        )[0]
        output = prompt.replace("<FILL_ME>", filling)
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

    input_ids = ids_tensor((1, 128), 32016).to(device)

    return (
        lambda: load_model(
            load_tokenizer=False, verbose=verbose, device=device, dtype=dtype, cache=cache
        )[1]
    ), (input_ids,)
