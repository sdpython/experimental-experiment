import os
from typing import Any, Dict, List, Tuple, Union


def create_configuration_for_benchmark(
    model: str = "llama",
    config: str = "small",
    repeat: int = 5,
    warmup: int = 3,
    num_hidden_layers: int = 1,
    implementation: str = "eager",
) -> Dict[str, Union[str, int, List[Tuple[int, int]]]]:
    """
    Creates a model based on the given configuration.

    :param model: model name
    :param config: size of the model (small, medium, large)
    :param warmup: number of warmup steps
    :param repeat: number of repetition
    :param num_hidden_layers: number of hidden layers
    :param implementation: implementation
    :return: dictionary
    """
    assert model == "llama", "not implemented yet for any other model than llama"

    if config == "small":
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=16,
            num_hidden_layers=num_hidden_layers,
            vocab_size=1024,
            intermediate_size=16,
            max_position_embeddings=1024,
            num_attention_heads=2,
            _attn_implementation=implementation,
        )
    if config == "medium":
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=1024,
            num_hidden_layers=num_hidden_layers,
            vocab_size=1024,
            intermediate_size=1024,
            max_position_embeddings=1024,
            num_attention_heads=2,
            _attn_implementation=implementation,
        )
    if config in ("large", "default"):
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=4096,
            num_hidden_layers=num_hidden_layers,
            vocab_size=32000,
            intermediate_size=11008,
            max_position_embeddings=2048,
            num_attention_heads=32,
            _attn_implementation=implementation,
        )
    raise ValueError(f"Unexpected value for config={config!r}.")


def create_compiled_model(
    model: Any,
    backend: str,
    use_dynamic: bool,
    target_opset: int,
    verbose: int,
    enable_pattern: str,
    disable_pattern: str,
) -> Any:
    """
    Creates the compilrf model.

    :param model: module
    :param backend: kind of backend
    :param use_dynamic: use dynamic shape
    :param verbose: verbosity
    :param enable_pattern: to enable optimization pattern
    :param disable_pattern: to disable optimization pattern
    :return: compiled model
    """
    import torch
    from torch._dynamo.backends.common import aot_autograd
    from experimental_experiment.torch_helper.training_helper import make_aot_ort
    from experimental_experiment.torch_dynamo import (
        get_decomposition_table,
        onnx_custom_backend,
        onnx_debug_backend,
    )

    if backend == "ort":
        local_aot_ort, local_ort = make_aot_ort(
            dynamic=use_dynamic, rewrite=True, verbose=verbose
        )
        return torch.compile(model, backend=local_ort)

    if backend == "plug":
        os.environ["ONNXRT_CHANGE_REWRITER"] = "1"

        local_aot_ort, local_ort = make_aot_ort(
            dynamic=use_dynamic, rewrite=False, verbose=verbose
        )
        return torch.compile(model, backend=local_ort)

    if backend == "inductor":
        return torch.compile(model, backend="inductor", dynamic=use_dynamic)

    if backend == "eager":
        return model

    if backend == "custom":
        target_opset = target_opset
        aot_compiler = aot_autograd(
            fw_compiler=lambda *args, **kwargs: onnx_custom_backend(
                *args,
                target_opset=target_opset,
                verbose=verbose,
                enable_pattern=enable_pattern,
                disable_pattern=disable_pattern,
                **kwargs,
            ),
            decompositions=get_decomposition_table(),
        )
        return torch.compile(
            model, backend=aot_compiler, fullgraph=True, dynamic=use_dynamic
        )

    if backend == "debug":
        target_opset = target_opset
        aot_compiler = aot_autograd(
            fw_compiler=lambda *args, **kwargs: onnx_debug_backend(
                *args,
                target_opset=target_opset,
                backend="ref",
                enable_pattern=enable_pattern,
                disable_pattern=disable_pattern,
                **kwargs,
            ),
            decompositions=get_decomposition_table(),
        )
        return torch.compile(
            model, backend=aot_compiler, fullgraph=True, dynamic=use_dynamic
        )

    raise ValueError(f"Unexpected backend={backend!r}.")
