from typing import Any, Dict, Tuple, Union
from . import assert_found


def get_stable_diffusion_2_unet(
    inputs_as_dict: bool = False,
    overwrite: bool = False,
    **kwargs,
) -> Tuple[Any, Union[Tuple[Any, ...], Dict[str, Any]]]:
    """
    Gets a non initialized model.

    :param inputs_as_dict: returns dummy inputs as a dictionary or not
    :param overwrite: do not consider the config from the true model
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :return: model, inputs

    See `StableDiffusion2Unet
    <https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/unet/config.json>`_.
    """
    import torch
    from diffusers import UNet2DConditionModel

    config = {
        "_class_name": "UNet2DConditionModel",
        "_diffusers_version": "0.8.0",
        "_name_or_path": "hf-models/stable-diffusion-v2-768x768/unet",
        "act_fn": "silu",
        "attention_head_dim": [5, 10, 20, 20],
        "block_out_channels": [320, 640, 1280, 1280],
        "center_input_sample": False,
        "cross_attention_dim": 1024,
        "down_block_types": [
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ],
        "downsample_padding": 1,
        "dual_cross_attention": False,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "in_channels": 4,
        "layers_per_block": 2,
        "mid_block_scale_factor": 1,
        "norm_eps": 1e-05,
        "norm_num_groups": 32,
        "out_channels": 4,
        "sample_size": 96,
        "up_block_types": [
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ],
        "use_linear_projection": True,
    }
    if overwrite:
        config = kwargs
    else:
        assert_found(kwargs, config)
        config.update(**kwargs)
    model = UNet2DConditionModel(**config)
    model.eval()

    inputs = dict(
        sample=torch.randn(1, 4, 128, 128),
        timestep=torch.tensor([1.0]),
        encoder_hidden_states=torch.randn(1, 1, 32 if overwrite else 1024),
    )

    if inputs_as_dict:
        inputs = tuple(inputs.values())

    return model, inputs
