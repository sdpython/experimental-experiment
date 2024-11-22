import os
from typing import Optional, Tuple

CACHE = f"{os.environ.get('HOME', 'HOME')}/.cache/exporter_benchmark"


def str_dtype(dtype: "torch.dtype") -> str:  # noqa: F821
    """Converts a dtype into a string."""
    return str(dtype).replace("torch.", "")


def load_llm_model(
    model_name: str,
    model_id: str,
    cls_tokenizer: type,
    cls_model: type,
    verbose: int = 0,
    load_tokenizer: bool = False,
    cache: str = CACHE,
    device: str = "cuda",
    dtype: Optional[str] = None,
) -> Tuple[Optional["tokenizer"], "model"]:  # noqa: F821
    """
    Loads `code_llama
    <https://huggingface.co/docs/transformers/en/model_doc/code_llama>`_.

    :param model_name: model name to store it
    :param model_id: model id to download it
    :param cls_tokenizer: tokenizer class
    :param cls_model: model class
    :param verbose: verbosity
    :param load_tokenizer: loads the tokenizer as well
    :param device: where to put the model
    :param dtype: which type to use, it should be a string such as 'float16'
        to avoid the following error
        ``TypeError: Object of type dtype is not JSON serializable``
    :return: tokenizer, model
    """
    import torch

    assert isinstance(dtype, str), f"Unexpected type for dtype={dtype!r}"
    if dtype in (None, "None"):
        dtype = "auto"
    stype = str_dtype(dtype) if dtype is not None else ""
    dtype = getattr(torch, dtype, dtype)
    if load_tokenizer:
        if verbose:
            print(f"[load_model] retrieves tokenizer for {model_name}, dtype={stype}")
        tokenizer = cls_tokenizer.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache)
    else:
        tokenizer = None

    if verbose:
        print(f"[load_model] retrieves model {model_name}, dtype={stype}")
    model = cls_model.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache)

    if verbose:
        print(f"[load_model] converts to {device!r} and dtype={dtype!r}")

    model = model.to(device)
    if verbose:
        print(f"[load_model] done {model_name}")
    return tokenizer, model
