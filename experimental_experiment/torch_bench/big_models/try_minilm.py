from typing import Any, Callable, Dict, List, Tuple, Optional
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
    Loads `all-MiniLM-L6-v1
    <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v1>`_.

    :param verbose: verbosity
    :param load_tokenizer: loads the tokenizer as well
    :param device: where to put the model
    :param dtype: which type to use, it should be a string such as 'float16'
        to avoid the following error
        ``TypeError: Object of type dtype is not JSON serializable``
    :return: tokenizer, model
    """
    from transformers import AutoTokenizer, AutoModel

    return load_llm_model(
        "all-MiniLM-L6-v1",
        "sentence-transformers/all-MiniLM-L6-v1",
        AutoTokenizer,
        AutoModel,
        cache=cache,
        verbose=verbose,
        device=device,
        dtype=dtype,
        load_tokenizer=load_tokenizer,
    )


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def demo_model(
    tokenizer: "tokenizer",  # noqa: F821
    model: "model",  # noqa: F821
    inputs: Optional[List[str]] = None,
    verbose: int = 0,
    device: str = "cuda",
    max_new_tokens: int = 128,
    skip_special_tokens: bool = True,
    model_name: str = "all-MiniLM-L6-v1",
) -> Any:
    """
    Demonstrates the models.
    This example is usually taken from the documentation.

    :param tokenizer: tokenizer
    :param model: model
    :param inputs: a list of strings in this case
    :param verbose: verbosity
    :param model_name: only used for verbosity
    :return: results (a string in this case)
    """
    prompt = inputs or [
        "This is an example sentence very simple as you can see",
        "Each sentence is converted as you can see as well",
    ]
    with torch.no_grad():
        if verbose:
            print(f"[demo_model] tokenize the input for {model_name}")
            print(prompt)
            print("--")
        encoded_input = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        if verbose:
            print(f"[demo_model] encoded_input: {sorted(encoded_input)}")
            for k in sorted(encoded_input):
                v = encoded_input[k]
                print(
                    f"[demo_model]   {k!r}: shape={v.shape}, dtype={v.dtype}, "
                    f"min={v.min()}, max={v.max()}"
                )
            print(encoded_input)
            print(f"[demo_model] generates the token for {model_name}")
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        output = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        if verbose:
            print(f"[demo_model] output.shape={output.shape}, dtype={output.dtype}")
            print("[demo_model] done")
        return output


def get_model_inputs(
    verbose: int = 0,
    cache: str = CACHE,
    device: str = "cuda",
    dtype: Optional[str] = None,
) -> Tuple[Callable, Dict[str, torch.Tensor]]:
    """Returns a model and its inputs."""

    inputs = {
        "input_ids": torch.Tensor(
            [
                [101, 2023, 2003, 2019, 2742, 6251, 2200, 3722, 2004, 2017, 2064, 2156, 102],
                [101, 2169, 6251, 2003, 4991, 2004, 2017, 2064, 2156, 2004, 2092, 102, 0],
            ]
        )
        .to(device)
        .to(torch.int64),
        "token_type_ids": torch.Tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        .to(device)
        .to(torch.int64),
        "attention_mask": torch.Tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            ]
        )
        .to(device)
        .to(torch.int64),
    }

    return (
        lambda: load_model(
            load_tokenizer=False, verbose=verbose, device=device, dtype=dtype, cache=cache
        )[1]
    ), inputs
