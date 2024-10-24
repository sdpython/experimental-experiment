from typing import Any, Callable, Tuple, Optional
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
    Loads `Phi-3.5-mini-instruct
    <https://huggingface.co/microsoft/Phi-3.5-mini-instruct>`_.

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
        "phi_3_5_mini_instruct",
        "microsoft/Phi-3.5-mini-instruct",
        AutoTokenizer,
        AutoModelForCausalLM,
        cache=cache,
        verbose=verbose,
        device=device,
        dtype=dtype,
        load_tokenizer=load_tokenizer,
    )


def get_model_inputs(
    verbose: int = 0,
    cache: str = CACHE,
    device: str = "cuda",
    dtype: Optional[str] = None,
) -> Tuple[Callable, Tuple[Any, ...]]:
    """Returns a codellama model and its inputs."""

    # model.model.layers = model.model.layers[:1]
    # model.config.num_hidden_layers = 1

    dim = (1, 30)
    input_ids = torch.randint(0, 32064, dim).to(device)  # Batch size 1, sequence length 30
    attention_masks = torch.ones(*dim, dtype=torch.int64).to(device)

    # Prepare the inputs for the model
    # inputs = {'input_ids': input_ids, 'attention_mask':attention_masks}

    return (
        lambda: load_model(
            load_tokenizer=False, verbose=verbose, device=device, dtype=dtype, cache=cache
        )[1]
    ), (input_ids, attention_masks)
