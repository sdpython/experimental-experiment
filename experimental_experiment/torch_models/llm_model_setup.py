import enum
from typing import Any, Dict, Optional


class LLMInputKind(enum.IntEnum):
    """
    Defines the dummy inputs which can be generated for a LLM vision model.

    Example::

        K = LLMInputKind.

        K.input_ids
        K.input_ids | K.position_ids | K.attention_mask
        K.input_ids | K.position_ids | K.attention_mask | K.images | K.past_key_values

    Remarks, for Phi 3.5:

    * images means two new inputs pixel_value Ix5x3x336x336 where I is the number of images
      and image_size Ix2 which contains the image sizes
    * min(LLMInputKind.input_ids) = -I where I is still the number of images.
    * the number of caches is equal to the number of hidden kayers

    What does batch size means? Multiple prompts? The image embedding does not seem
    to support that.
    """

    # possible scenario for iteration 0
    input_ids = 4  # input_dis
    position_ids = 8  # position_ids
    attention_mask = 16  # attention_mask
    images = 32  # pixels_values, image_size
    # possible values for iteration 1
    past_key_values = 64  # caches
    # everyyhing checked
    ALL = 255


def get_input_cache(
    num_hidden_layers: int,
    batch_size: int,
    num_key_value_heads: int,
    sequence_length: int,
    cache_last_dim: int,
    device: str,
    input_cache_class: Optional[type] = None,
):
    """
    Creates a random cache.
    """
    import torch
    import transformers
    from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache

    if input_cache_class is None or input_cache_class is transformers.cache_utils.DynamicCache:
        cache = make_dynamic_cache(
            [
                (
                    torch.randn(
                        batch_size, num_key_value_heads, sequence_length, cache_last_dim
                    ).to(device),
                    torch.randn(
                        batch_size, num_key_value_heads, sequence_length, cache_last_dim
                    ).to(device),
                )
                for i in range(num_hidden_layers)
            ]
        )
        return cache

    if input_cache_class is transformers.cache_utils.MambaCache:

        class _config:
            def __init__(self):
                self.intermediate_size = 8192
                self.state_size = 16
                self.conv_kernel = 4
                self.num_hidden_layers = num_hidden_layers
                self.dtype = torch.float32

        # self.conv_states: torch.Tensor = torch.zeros(
        #     config.num_hidden_layers,     1
        #     self.max_batch_size,          2
        #     self.intermediate_size,       8192
        #     self.conv_kernel_size,        4
        #     device=device,
        #     dtype=dtype,
        # )
        # self.ssm_states: torch.Tensor = torch.zeros(
        #     config.num_hidden_layers,     1
        #     self.max_batch_size,          2
        #     self.intermediate_size,       8192
        #     self.ssm_state_size,          16
        #     device=device,
        #     dtype=dtype,
        # )

        cache = transformers.cache_utils.MambaCache(
            _config(), batch_size=batch_size, device=device
        )
        if isinstance(cache.conv_states, list):
            cache.conv_states = [
                torch.randn(t.shape).to(torch.float32) for t in cache.conv_states
            ]
            cache.ssm_states = [
                torch.randn(t.shape).to(torch.float32) for t in cache.ssm_states
            ]
        else:
            cache.conv_states = torch.randn(cache.conv_states.shape).to(torch.float32)
            cache.ssm_states = torch.randn(cache.ssm_states.shape).to(torch.float32)
        return cache

    raise NotImplementedError(
        f"get_input_cache not implemented for input_cache_class={input_cache_class}"
    )


def finalize_llm_setup(
    model: Any,
    batch_size: int,
    max_token_id: int = 50285,
    cache_last_dim: int = 80,
    common_dynamic_shapes: bool = True,
    inputs_as_tuple: bool = False,
    num_hidden_layers: int = 2,
    num_key_value_heads: int = 32,
    input_cache: bool = True,
    device: str = "cpu",
    seq_length_multiple: int = 1,
    input_cache_class: Optional[type] = None,
    sequence_length: int = 30,
    sequence_inc: int = 1,
    sequence_length2: int = 3,
) -> Dict[str, Any]:
    """
    Creates dummy inputs for a model ran as if it were the second iteration.
    Inputs contains cache.
    """
    import torch

    batch = torch.export.Dim("batch", min=1, max=1024)
    seq_length = torch.export.Dim("seq_length", min=1, max=4096)
    if seq_length_multiple > 1:
        seq_length = seq_length * seq_length_multiple

    shapes = {}

    if seq_length_multiple > 1:
        sequence_length = (
            (sequence_length + seq_length_multiple)
            // seq_length_multiple
            * seq_length_multiple
        )
        sequence_inc = seq_length_multiple
        sequence_length2 = seq_length_multiple

    if not input_cache:
        dim = (batch_size, sequence_length)
        inputs = dict(
            input_ids=torch.randint(0, max_token_id, dim).to(torch.int64).to(device),
            attention_mask=torch.ones(*dim, dtype=torch.int64).to(device),
        )
        dim = (batch_size + 1, sequence_length + sequence_inc)
        inputs2 = dict(
            input_ids=torch.randint(0, max_token_id, dim).to(torch.int64).to(device),
            attention_mask=torch.ones(*dim, dtype=torch.int64).to(device),
        )
        shapes.update(
            {
                "input_ids": {0: batch, 1: seq_length},
                "attention_mask": {0: batch, 1: seq_length},
            }
        )
    else:
        cache = get_input_cache(
            num_hidden_layers,
            batch_size,
            num_key_value_heads,
            sequence_length,
            cache_last_dim,
            device=device,
            input_cache_class=input_cache_class,
        )
        cache2 = get_input_cache(
            num_hidden_layers,
            batch_size + 1,
            num_key_value_heads,
            sequence_length + sequence_inc,
            cache_last_dim,
            device=device,
            input_cache_class=input_cache_class,
        )

        inputs = dict(
            input_ids=torch.randint(0, max_token_id, (batch_size, sequence_length2))
            .to(torch.int64)
            .to(device),
            attention_mask=torch.ones((batch_size, sequence_length + sequence_length2))
            .to(torch.int64)
            .to(device),
        )
        inputs2 = dict(
            input_ids=torch.randint(
                0, max_token_id, (batch_size + 1, sequence_length2 + sequence_inc)
            ).to(torch.int64),
            attention_mask=torch.ones(
                (
                    batch_size + 1,
                    sequence_length + sequence_inc + sequence_length2 + sequence_inc,
                )
            ).to(torch.int64),
        )
        shapes.update(
            {
                "input_ids": {0: batch, 1: seq_length},
                "attention_mask": {
                    0: batch,
                    1: torch.export.Dim.DYNAMIC,  # cache_length + seq_length
                },
            }
        )
        if input_cache_class is None or input_cache_class.__name__ == "DynamicCache":
            n = len(cache.key_cache)
            cache_length = torch.export.Dim("cache_length", min=1, max=4096)
            shapes.update(
                {
                    "past_key_values": [
                        [{0: batch, 2: cache_length} for _ in range(n)],
                        [{0: batch, 2: cache_length} for _ in range(n)],
                    ],
                }
            )
            inputs["past_key_values"] = cache
            inputs2["past_key_values"] = cache2
        elif input_cache_class.__name__ == "MambaCache":
            n = len(cache.conv_states)
            shapes.update(
                {
                    "cache_params": [
                        [{1: batch} for _ in range(n)],
                        [{1: batch} for _ in range(n)],
                    ]
                }
            )
            inputs["cache_params"] = cache
            inputs2["cache_params"] = cache2
        else:
            raise AssertionError(f"Unexpected cache class {input_cache_class}")

    if inputs_as_tuple:
        inputs = tuple(inputs.values())
        shapes = tuple(shapes.values())

    if common_dynamic_shapes:
        return dict(inputs=inputs, model=model, dynamic_shapes=shapes, inputs2=inputs2)
    return dict(inputs=inputs, model=model)


def finalize_llm_vision_setup(
    model: Any,
    input_kind: LLMInputKind,
    batch_size: int,
    max_token_id: int = 50285,
    cache_last_dim: int = 80,
    common_dynamic_shapes: bool = True,
    inputs_as_tuple: bool = False,
    num_hidden_layers: int = 2,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Creates dummy inputs for a model ran as if it were the second iteration.
    Inputs contains cache.
    """
    import torch

    if input_kind == LLMInputKind.input_ids:
        dim = (1, 30)
        inputs = dict(input_ids=torch.randint(0, max_token_id, dim).to(torch.int64))
        shapes = {
            "input_ids": {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq_length")}
        }
    elif input_kind == LLMInputKind.input_ids | LLMInputKind.attention_mask:
        dim = (1, 30)
        inputs = dict(
            input_ids=torch.randint(0, max_token_id, dim).to(torch.int64),
            attention_mask=torch.ones(*dim, dtype=torch.int64),
        )
        batch = torch.export.Dim("batch")
        seq_length = torch.export.Dim("seq_length")
        shapes = {
            "input_ids": {0: batch, 1: seq_length},
            "attention_mask": {0: batch, 1: seq_length},
        }
    else:
        from .dummy_inputs.llm_dummy_inputs import (
            restore_dummy_inputs_for_phi35_vision_instruct,
        )

        input_cache = input_kind & LLMInputKind.past_key_values
        data = restore_dummy_inputs_for_phi35_vision_instruct(
            num_hidden_layers=num_hidden_layers,
            n_iteration=1 if input_cache else 0,
            with_images=input_kind & LLMInputKind.images,
            device=device,
        )
        args, kwargs = data
        inputs = {}
        shapes = {}

        batch = torch.export.Dim("batch", min=1, max=1024)
        seq_length = torch.export.Dim("seq_length", min=1, max=4096)
        cache_length = torch.export.Dim("cache_length", min=1, max=4096)
        if input_kind & LLMInputKind.input_ids:
            inputs["input_ids"] = kwargs["input_ids"]
            shapes["input_ids"] = {0: batch, 1: seq_length} if not input_cache else {0: batch}
        if input_kind & LLMInputKind.position_ids:
            inputs["position_ids"] = kwargs["position_ids"]
            shapes["position_ids"] = {0: batch, 1: seq_length}
        if input_kind & LLMInputKind.attention_mask:
            inputs["attention_mask"] = kwargs["attention_mask"]
            shapes["attention_mask"] = {0: batch, 1: cache_length + 1}
        if input_kind & LLMInputKind.past_key_values:
            inputs["past_key_values"] = kwargs["past_key_values"]
            n = len(data[1]["past_key_values"].key_cache)
            shapes["past_key_values"] = [
                [{0: batch, 2: cache_length} for _ in range(n)],
                [{0: batch, 2: cache_length} for _ in range(n)],
            ]
        if input_kind & LLMInputKind.images:
            inputs["pixel_values"] = kwargs["pixel_values"]
            inputs["image_sizes"] = kwargs["image_sizes"]
            n_images = torch.export.Dim("n_images", min=0, max=1024)
            shapes["pixel_values"] = shapes["image_sizes"] = {0: n_images}

    if inputs_as_tuple:
        inputs = tuple(inputs.values())
        shapes = tuple(shapes.values())

    if common_dynamic_shapes:
        return dict(model=model, inputs=inputs, dynamic_shapes=shapes)
    return dict(model=model, inputs=inputs)
