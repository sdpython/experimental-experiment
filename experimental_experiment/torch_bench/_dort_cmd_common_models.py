from typing import Dict, List, Tuple, Union


def _create_configuration_for_benchmark_llama(
    config: str = "small",
    repeat: int = 5,
    warmup: int = 3,
    num_hidden_layers: int = 1,
    implementation: str = "eager",
) -> Dict[str, Union[str, int, List[Tuple[int, int]]]]:
    """
    Creates a model based on the given configuration.

    :param config: size of the model (small, medium, large)
    :param warmup: number of warmup steps
    :param repeat: number of repetition
    :param num_hidden_layers: number of hidden layers
    :param implementation: implementation
    :return: dictionary
    """
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


def _create_configuration_for_benchmark_mistral(
    config: str = "small",
    repeat: int = 5,
    warmup: int = 3,
    num_hidden_layers: int = 1,
    implementation: str = "eager",
) -> Dict[str, Union[str, int, List[Tuple[int, int]]]]:
    """
    Creates a model based on the given configuration.

    :param config: size of the model (small, medium, large)
    :param warmup: number of warmup steps
    :param repeat: number of repetition
    :param num_hidden_layers: number of hidden layers
    :param implementation: implementation
    :return: dictionary
    """
    if config == "small":
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=32,
            num_hidden_layers=num_hidden_layers,
            vocab_size=99,
            intermediate_size=16,
            max_position_embeddings=512,
            num_attention_heads=2,
            num_key_value_heads=2,
            _attn_implementation=implementation,
        )
    if config == "medium":
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=1024,
            num_hidden_layers=num_hidden_layers,
            vocab_size=1024,
            intermediate_size=1024,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=1024,
            sliding_window=4096,
            _attn_implementation=implementation,
        )
    if config in ("large", "default"):
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=4096,
            num_hidden_layers=num_hidden_layers,
            vocab_size=32000,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            sliding_window=4096,
            _attn_implementation=implementation,
        )
    raise ValueError(f"Unexpected value for config={config!r}.")


def _create_configuration_for_benchmark_phi(
    config: str = "small",
    repeat: int = 5,
    warmup: int = 3,
    num_hidden_layers: int = 1,
    implementation: str = "eager",
) -> Dict[str, Union[str, int, List[Tuple[int, int]]]]:
    """
    Creates a model based on the given configuration.

    :param config: size of the model (small, medium, large)
    :param warmup: number of warmup steps
    :param repeat: number of repetition
    :param num_hidden_layers: number of hidden layers
    :param implementation: implementation
    :return: dictionary
    """
    if config == "small":
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=32,
            num_hidden_layers=num_hidden_layers,
            vocab_size=99,
            intermediate_size=16,
            max_position_embeddings=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            _attn_implementation=implementation,
        )
    if config == "medium":
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=1024,
            num_hidden_layers=num_hidden_layers,
            vocab_size=1024,
            intermediate_size=1024,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=1024,
            _attn_implementation=implementation,
        )
    if config in ("large", "default"):
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=2048,
            num_hidden_layers=num_hidden_layers,
            vocab_size=51200,
            intermediate_size=8192,
            num_attention_heads=32,
            num_key_value_heads=None,
            max_position_embeddings=2048,
            _attn_implementation=implementation,
        )
    raise ValueError(f"Unexpected value for config={config!r}.")
