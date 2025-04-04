from typing import Any, Dict, Tuple, Optional, Union
import numpy as np
from . import assert_found
from .llm_model_setup import LLMInputKind, finalize_llm_setup, finalize_llm_vision_setup


######
# Bert
######


def get_all_mini_ml_l6_v1(
    inputs_as_tuple: bool = False,
    input_cache: bool = True,
    batch_size: int = 1,
    common_dynamic_shapes: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Gets a non initialized model.

    :param inputs_as_tuple: returns dummy inputs as a dictionary or not
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :param batch_size: batch size
    :param common_dynamic_shapes: if True returns dynamic shapes as well
    :return: dicionary

    See `all-MiniLM-L6-v1
    <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v1/blob/main/config.json>`_.

    Model forward signature:

    ::

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

    """
    import transformers

    config = {
        "_name_or_path": "nreimers/MiniLM-L6-H384-uncased",
        "architectures": ["BertModel"],
        "attention_probs_dropout_prob": 0.1,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "initializer_range": 0.02,
        "intermediate_size": 1536,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.8.2",
        "type_vocab_size": 2,
        "use_cache": True,
        "vocab_size": 30522,
    }
    assert_found(kwargs, config)
    config.update(**kwargs)
    conf = transformers.BertConfig(**config)
    model = transformers.BertModel(conf)
    model.eval()

    res = finalize_llm_setup(
        model,
        batch_size,
        max_token_id=30522,
        cache_last_dim=32,
        num_key_value_heads=12,
        common_dynamic_shapes=common_dynamic_shapes,
        inputs_as_tuple=inputs_as_tuple,
        num_hidden_layers=config["num_hidden_layers"],
        input_cache=input_cache,
    )
    if not input_cache:
        return res

    # We flatten the cache.
    for k in ["inputs", "inputs2"]:
        if k not in res:
            continue
        kv = res[k]["past_key_values"]
        kv = [
            (kv.key_cache[i], kv.value_cache[i], kv.key_cache[i], kv.value_cache[i])
            for i in range(len(kv.key_cache))
        ]
        res[k]["past_key_values"] = kv

    if common_dynamic_shapes:
        sh = res["dynamic_shapes"]["past_key_values"]
        if sh:
            sh1 = sh[0][0]
            res["dynamic_shapes"]["past_key_values"] = [
                tuple(sh1 for _ in range(4)) for s in range(config["num_hidden_layers"])
            ]
    return res


############
# Phi Series
############


def get_phi2(
    inputs_as_tuple: bool = False,
    input_cache: bool = True,
    batch_size: int = 1,
    common_dynamic_shapes: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Gets a non initialized model.

    :param inputs_as_tuple: returns dummy inputs as a dictionary or not
    :param input_cache: generate data for this iteration with or without cache
    :param batch_size: batch size
    :param common_dynamic_shapes: if True returns dynamic shapes as well
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :return: dict

    See `Phi-2/config.json
    <https://huggingface.co/microsoft/phi-2/blob/main/config.json>`_.
    """
    import transformers

    config = {
        "_name_or_path": "microsoft/phi-2",
        "architectures": ["PhiForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 50256,
        "embd_pdrop": 0.0,
        "eos_token_id": 50256,
        "hidden_act": "gelu_new",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 10240,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 2048,
        "model_type": "phi",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "partial_rotary_factor": 0.4,
        "qk_layernorm": False,
        "resid_pdrop": 0.1,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.37.0",
        "use_cache": True,
        "vocab_size": 51200,
    }
    assert_found(kwargs, config)
    config.update(**kwargs)
    conf = transformers.PhiConfig(**config)
    model = transformers.PhiForCausalLM(conf)
    model.eval()
    return finalize_llm_setup(
        model,
        batch_size,
        max_token_id=50285,
        cache_last_dim=80,
        common_dynamic_shapes=common_dynamic_shapes,
        inputs_as_tuple=inputs_as_tuple,
        num_hidden_layers=config["num_hidden_layers"],
        input_cache=input_cache,
    )


def get_phi35_mini_instruct(
    inputs_as_tuple: bool = False,
    input_cache: bool = True,
    batch_size: int = 1,
    common_dynamic_shapes: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Gets a non initialized model.

    :param inputs_as_tuple: returns dummy inputs as a dictionary or not
    :param batch_size: batch size
    :param input_cache: generate data for this iteration with or without cache
    :param common_dynamic_shapes: if True returns dynamic shapes as well
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :return: dictionary

    See `Phi-3.5-mini-instruct/config.json
    <https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/config.json>`_.
    """
    import transformers

    config = {
        "_name_or_path": "Phi-3.5-mini-instruct",
        "architectures": ["Phi3ForCausalLM"],
        "attention_dropout": 0.0,
        "auto_map": {
            "AutoConfig": "configuration_phi3.Phi3Config",
            "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM",
        },
        "bos_token_id": 1,
        "embd_pdrop": 0.0,
        "eos_token_id": 32000,
        "hidden_act": "silu",
        "hidden_size": 3072,
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "max_position_embeddings": 131072,
        "model_type": "phi3",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "original_max_position_embeddings": 4096,
        "pad_token_id": 32000,
        "resid_pdrop": 0.0,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "long_factor": [
                1.0800000429153442,
                1.1100000143051147,
                1.1399999856948853,
                1.340000033378601,
                1.5899999141693115,
                1.600000023841858,
                1.6200000047683716,
                2.620000123977661,
                3.2300000190734863,
                3.2300000190734863,
                4.789999961853027,
                7.400000095367432,
                7.700000286102295,
                9.09000015258789,
                12.199999809265137,
                17.670000076293945,
                24.46000099182129,
                28.57000160217285,
                30.420001983642578,
                30.840002059936523,
                32.590003967285156,
                32.93000411987305,
                42.320003509521484,
                44.96000289916992,
                50.340003967285156,
                50.45000457763672,
                57.55000305175781,
                57.93000411987305,
                58.21000289916992,
                60.1400032043457,
                62.61000442504883,
                62.62000274658203,
                62.71000289916992,
                63.1400032043457,
                63.1400032043457,
                63.77000427246094,
                63.93000411987305,
                63.96000289916992,
                63.970001220703125,
                64.02999877929688,
                64.06999969482422,
                64.08000183105469,
                64.12000274658203,
                64.41000366210938,
                64.4800033569336,
                64.51000213623047,
                64.52999877929688,
                64.83999633789062,
            ],
            "short_factor": [
                1.0,
                1.0199999809265137,
                1.0299999713897705,
                1.0299999713897705,
                1.0499999523162842,
                1.0499999523162842,
                1.0499999523162842,
                1.0499999523162842,
                1.0499999523162842,
                1.0699999332427979,
                1.0999999046325684,
                1.1099998950958252,
                1.1599998474121094,
                1.1599998474121094,
                1.1699998378753662,
                1.2899998426437378,
                1.339999794960022,
                1.679999828338623,
                1.7899998426437378,
                1.8199998140335083,
                1.8499997854232788,
                1.8799997568130493,
                1.9099997282028198,
                1.9399996995925903,
                1.9899996519088745,
                2.0199997425079346,
                2.0199997425079346,
                2.0199997425079346,
                2.0199997425079346,
                2.0199997425079346,
                2.0199997425079346,
                2.0299997329711914,
                2.0299997329711914,
                2.0299997329711914,
                2.0299997329711914,
                2.0299997329711914,
                2.0299997329711914,
                2.0299997329711914,
                2.0299997329711914,
                2.0299997329711914,
                2.0799996852874756,
                2.0899996757507324,
                2.189999580383301,
                2.2199995517730713,
                2.5899994373321533,
                2.729999542236328,
                2.749999523162842,
                2.8399994373321533,
            ],
            "type": "longrope",
        },
        "rope_theta": 10000.0,
        "sliding_window": 262144,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "attention_bias": False,
        "vocab_size": 32064,
    }
    assert_found(kwargs, config)
    config.update(**kwargs)
    conf = transformers.Phi3Config(**config)
    model = transformers.Phi3ForCausalLM(conf)
    model.eval()
    return finalize_llm_setup(
        model,
        batch_size,
        max_token_id=32064,
        cache_last_dim=96,
        common_dynamic_shapes=common_dynamic_shapes,
        inputs_as_tuple=inputs_as_tuple,
        num_hidden_layers=config["num_hidden_layers"],
        input_cache=input_cache,
    )


def get_phi35_vision_instruct(
    inputs_as_tuple: bool = False,
    batch_size: int = 2,
    input_kind: LLMInputKind = LLMInputKind.input_ids,
    device: str = "cpu",
    common_dynamic_shapes: bool = False,
    **kwargs,
) -> Tuple[Any, Union[Tuple[Any, ...], Dict[str, Any]], Optional[Any]]:
    """
    Gets a non initialized model.

    :param batch_size: batch size to use
    :param inputs_as_tuple: returns dummy inputs as a dictionary or not
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :param common_dynamic_shapes: if True returns dynamic shapes as well
    :return: model, inputs, dynamic shapes
    """
    from .fromhub.configuration_phi3_v import Phi3VConfig
    from .fromhub.modeling_phi3_v import Phi3VForCausalLM

    config = {
        "_name_or_path": "Phi-3.5-vision-instruct",
        "architectures": ["Phi3VForCausalLM"],
        "attention_dropout": 0.0,
        "auto_map": {
            "AutoConfig": "configuration_phi3_v.Phi3VConfig",
            "AutoModelForCausalLM": "modeling_phi3_v.Phi3VForCausalLM",
        },
        "bos_token_id": 1,
        "embd_layer": {
            "embedding_cls": "image",
            "hd_transform_order": "sub_glb",
            "projection_cls": "mlp",
            "use_hd_transform": True,
            "with_learnable_separator": True,
        },
        "embd_pdrop": 0.0,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 3072,
        "img_processor": {
            "image_dim_out": 1024,
            "model_name": "openai/clip-vit-large-patch14-336",
            "name": "clip_vision_model",
            "num_img_tokens": 144,
        },
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "max_position_embeddings": 131072,
        "model_type": "phi3_v",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "original_max_position_embeddings": 4096,
        "pad_token_id": 32000,
        "resid_pdrop": 0.0,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "long_factor": [
                1.0800000429153442,
                1.1100000143051147,
                1.1399999856948853,
                1.340000033378601,
                1.5899999141693115,
                1.600000023841858,
                1.6200000047683716,
                2.620000123977661,
                3.2300000190734863,
                3.2300000190734863,
                4.789999961853027,
                7.400000095367432,
                7.700000286102295,
                9.09000015258789,
                12.199999809265137,
                17.670000076293945,
                24.46000099182129,
                28.57000160217285,
                30.420001983642578,
                30.840002059936523,
                32.590003967285156,
                32.93000411987305,
                42.320003509521484,
                44.96000289916992,
                50.340003967285156,
                50.45000457763672,
                57.55000305175781,
                57.93000411987305,
                58.21000289916992,
                60.1400032043457,
                62.61000442504883,
                62.62000274658203,
                62.71000289916992,
                63.1400032043457,
                63.1400032043457,
                63.77000427246094,
                63.93000411987305,
                63.96000289916992,
                63.970001220703125,
                64.02999877929688,
                64.06999969482422,
                64.08000183105469,
                64.12000274658203,
                64.41000366210938,
                64.4800033569336,
                64.51000213623047,
                64.52999877929688,
                64.83999633789062,
            ],
            "short_factor": [
                1.08,
                1.1,
                1.1300000000000001,
                1.2800000000000002,
                1.3100000000000003,
                1.4500000000000004,
                1.4500000000000004,
                1.9500000000000008,
                2.030000000000001,
                2.4299999999999926,
                2.5699999999999896,
                2.9499999999999815,
                3.729999999999965,
                3.869999999999962,
                4.189999999999955,
                4.43999999999995,
                4.6399999999999455,
                4.979999999999938,
                5.159999999999934,
                5.279999999999932,
                5.759999999999922,
                5.889999999999919,
                5.889999999999919,
                5.969999999999917,
                6.089999999999915,
                6.2799999999999105,
                6.7699999999999,
                6.8899999999998975,
                7.109999999999893,
                7.129999999999892,
                7.179999999999891,
                7.289999999999889,
                7.339999999999888,
                7.559999999999883,
                7.619999999999882,
                7.69999999999988,
                7.879999999999876,
                7.879999999999876,
                7.879999999999876,
                7.939999999999875,
                7.949999999999875,
                7.979999999999874,
                8.19999999999987,
                8.439999999999864,
                8.469999999999864,
                8.589999999999861,
                8.809999999999857,
                8.999999999999853,
            ],
            "type": "su",
        },
        "rope_theta": 10000.0,
        "sliding_window": 262144,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.38.1",
        "use_cache": True,
        "vocab_size": 32064,
        # "_attn_implementation": "flash_attention_2",
        "_attn_implementation": "eager",
    }

    assert_found(kwargs, config)
    config.update(**kwargs)
    conf = Phi3VConfig(**config)
    model = Phi3VForCausalLM(conf)
    model.eval().to(device)

    if (
        input_kind
        == LLMInputKind.input_ids | LLMInputKind.attention_mask | LLMInputKind.past_key_values
    ):
        return finalize_llm_setup(
            model,
            batch_size=batch_size,
            max_token_id=32064,
            cache_last_dim=96,
            common_dynamic_shapes=common_dynamic_shapes,
            inputs_as_tuple=inputs_as_tuple,
            num_hidden_layers=config["num_hidden_layers"],
            input_cache=True,
            device=device,
        )
    return finalize_llm_vision_setup(
        model,
        input_kind,
        batch_size=batch_size,
        max_token_id=32064,
        cache_last_dim=96,
        common_dynamic_shapes=common_dynamic_shapes,
        inputs_as_tuple=inputs_as_tuple,
        num_hidden_layers=config["num_hidden_layers"],
        device=device,
    )


def get_phi3_vision_128k_instruct(
    inputs_as_tuple: bool = False,
    batch_size: int = 2,
    input_kind: LLMInputKind = LLMInputKind.input_ids,
    device: str = "cpu",
    common_dynamic_shapes: bool = False,
    **kwargs,
) -> Tuple[Any, Union[Tuple[Any, ...], Dict[str, Any]], Optional[Any]]:
    """
    Gets a non initialized model.

    :param batch_size: batch size to use
    :param inputs_as_tuple: returns dummy inputs as a dictionary or not
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :param common_dynamic_shapes: if True returns dynamic shapes as well
    :return: model, inputs, dynamic shapes

    See `Phi-3-vision-128k-instruct/config.json
    <https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/config.json>`_.
    """
    from .configuration_phi3_v import Phi3VConfig
    from .modeling_phi3_v import Phi3VForCausalLM

    config = {
        "_name_or_path": "Phi-3-vision-128k-instruct",
        "architectures": ["Phi3VForCausalLM"],
        "attention_dropout": 0.0,
        "auto_map": {
            "AutoConfig": "configuration_phi3_v.Phi3VConfig",
            "AutoModelForCausalLM": "modeling_phi3_v.Phi3VForCausalLM",
        },
        "bos_token_id": 1,
        "embd_layer": {
            "embedding_cls": "image",
            "hd_transform_order": "sub_glb",
            "projection_cls": "mlp",
            "use_hd_transform": True,
            "with_learnable_separator": True,
        },
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 3072,
        "img_processor": {
            "image_dim_out": 1024,
            "model_name": "openai/clip-vit-large-patch14-336",
            "name": "clip_vision_model",
            "num_img_tokens": 144,
        },
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "max_position_embeddings": 131072,
        "model_type": "phi3_v",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "original_max_position_embeddings": 4096,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "long_factor": [
                1.0299999713897705,
                1.0499999523162842,
                1.0499999523162842,
                1.0799999237060547,
                1.2299998998641968,
                1.2299998998641968,
                1.2999999523162842,
                1.4499999284744263,
                1.5999999046325684,
                1.6499998569488525,
                1.8999998569488525,
                2.859999895095825,
                3.68999981880188,
                5.419999599456787,
                5.489999771118164,
                5.489999771118164,
                9.09000015258789,
                11.579999923706055,
                15.65999984741211,
                15.769999504089355,
                15.789999961853027,
                18.360000610351562,
                21.989999771118164,
                23.079999923706055,
                30.009998321533203,
                32.35000228881836,
                32.590003967285156,
                35.56000518798828,
                39.95000457763672,
                53.840003967285156,
                56.20000457763672,
                57.95000457763672,
                59.29000473022461,
                59.77000427246094,
                59.920005798339844,
                61.190006256103516,
                61.96000671386719,
                62.50000762939453,
                63.3700065612793,
                63.48000717163086,
                63.48000717163086,
                63.66000747680664,
                63.850006103515625,
                64.08000946044922,
                64.760009765625,
                64.80001068115234,
                64.81001281738281,
                64.81001281738281,
            ],
            "short_factor": [
                1.05,
                1.05,
                1.05,
                1.1,
                1.1,
                1.1,
                1.2500000000000002,
                1.2500000000000002,
                1.4000000000000004,
                1.4500000000000004,
                1.5500000000000005,
                1.8500000000000008,
                1.9000000000000008,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.1000000000000005,
                2.1000000000000005,
                2.2,
                2.3499999999999996,
                2.3499999999999996,
                2.3499999999999996,
                2.3499999999999996,
                2.3999999999999995,
                2.3999999999999995,
                2.6499999999999986,
                2.6999999999999984,
                2.8999999999999977,
                2.9499999999999975,
                3.049999999999997,
                3.049999999999997,
                3.049999999999997,
            ],
            "type": "su",
        },
        "rope_theta": 10000.0,
        "sliding_window": 131072,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.38.1",
        "use_cache": True,
        "vocab_size": 32064,
        "_attn_implementation": "flash_attention_2",
    }

    assert_found(kwargs, config)
    config.update(**kwargs)
    conf = Phi3VConfig(**config)
    model = Phi3VForCausalLM(conf)
    model.eval()

    if (
        input_kind
        == LLMInputKind.input_ids | LLMInputKind.attention_mask | LLMInputKind.past_key_values
    ):
        return finalize_llm_setup(
            model,
            batch_size=batch_size,
            max_token_id=32064,
            cache_last_dim=96,
            common_dynamic_shapes=common_dynamic_shapes,
            inputs_as_tuple=inputs_as_tuple,
            num_hidden_layers=config["num_hidden_layers"],
            input_cache=True,
            device=device,
        )
    return finalize_llm_vision_setup(
        model,
        input_kind,
        batch_size=batch_size,
        max_token_id=32064,
        cache_last_dim=96,
        common_dynamic_shapes=common_dynamic_shapes,
        inputs_as_tuple=inputs_as_tuple,
        num_hidden_layers=config["num_hidden_layers"],
        device=device,
    )


def get_phi4(
    inputs_as_tuple: bool = False,
    input_cache: bool = True,
    batch_size: int = 1,
    common_dynamic_shapes: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Gets a non initialized model.

    :param inputs_as_tuple: returns dummy inputs as a dictionary or not
    :param batch_size: batch size
    :param input_cache: generate data for this iteration with or without cache
    :param common_dynamic_shapes: if True returns dynamic shapes as well
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :return: dictionary

    See `Phi-4/config.json
    <https://huggingface.co/microsoft/phi-4/blob/main/config.json>`_.
    """
    import transformers

    config = {
        "_name_or_path": "microsoft/phi-4",
        "architectures": ["Phi3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "auto_map": {},
        "bos_token_id": 100257,
        "embd_pdrop": 0.0,
        "eos_token_id": 100257,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 17920,
        "max_position_embeddings": 16384,
        "model_type": "phi3",
        "num_attention_heads": 40,
        "num_hidden_layers": 40,
        "num_key_value_heads": 10,
        "original_max_position_embeddings": 16384,
        "pad_token_id": 100257,
        "resid_pdrop": 0.0,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 250000,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.47.0",
        "use_cache": True,
        "vocab_size": 100352,
    }
    assert_found(kwargs, config)
    config.update(**kwargs)
    conf = transformers.Phi3Config(**config)
    model = transformers.Phi3ForCausalLM(conf)
    model.eval()
    return finalize_llm_setup(
        model,
        batch_size,
        max_token_id=32064,
        num_key_value_heads=config["num_key_value_heads"],
        cache_last_dim=128,
        common_dynamic_shapes=common_dynamic_shapes,
        inputs_as_tuple=inputs_as_tuple,
        num_hidden_layers=config["num_hidden_layers"],
        input_cache=input_cache,
    )


##############
# Jamba, Mamba
##############


def get_ai21_jamba_15_mini(
    inputs_as_tuple: bool = False,
    input_cache: bool = True,
    batch_size: int = 1,
    common_dynamic_shapes: bool = False,
    device: str = "cpu",
    **kwargs,
) -> Dict[str, Any]:
    """
    Gets a non initialized model.

    :param inputs_as_tuple: returns dummy inputs as a dictionary or not
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :param batch_size: batch size
    :param common_dynamic_shapes: if True returns dynamic shapes as well
    :param device: device
    :return: dictionary

    See `ai21labs/AI21-Jamba-1.5-Mini/config.json
    <https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini/blob/main/config.json>`_.
    """
    import torch
    import transformers

    config = {
        "architectures": ["JambaForCausalLM"],
        "attention_dropout": 0.0,
        "attn_layer_offset": 4,
        "attn_layer_period": 8,
        "bos_token_id": 1,
        "eos_token_id": [2, 518],
        "expert_layer_offset": 1,
        "expert_layer_period": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "mamba_conv_bias": True,
        "mamba_d_conv": 4,
        "mamba_d_state": 16,
        "mamba_dt_rank": 256,
        "mamba_expand": 2,
        "mamba_proj_bias": False,
        "max_position_embeddings": 262144,
        "model_type": "jamba",
        "num_attention_heads": 32,
        "num_experts": 16,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "num_logits_to_keep": 1,
        "output_router_logits": False,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-06,
        "router_aux_loss_coef": 0.001,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.2",
        "use_cache": True,
        "use_mamba_kernels": False,  # maybe another test to add
        "vocab_size": 65536,
    }
    config.update(
        {
            "_from_model_config": True,
            "bos_token_id": 1,
            "eos_token_id": [2, 518],
            "pad_token_id": 0,
            "transformers_version": "4.40.2",
        }
    )
    assert_found(kwargs, config)
    config.update(**kwargs)
    conf = transformers.JambaConfig(**config)
    model = transformers.JambaForCausalLM(conf)
    model.eval()

    res = finalize_llm_setup(
        model,
        batch_size,
        max_token_id=63028,
        cache_last_dim=80,
        common_dynamic_shapes=common_dynamic_shapes,
        inputs_as_tuple=inputs_as_tuple,
        num_hidden_layers=config["num_hidden_layers"],
        input_cache=input_cache,
        device=device,
    )
    res["inputs"]["attention_mask"] = torch.ones(
        res["inputs"]["input_ids"].shape, dtype=torch.float32, device=device
    )
    if "inputs2" in res:
        res["inputs2"]["attention_mask"] = torch.ones(
            res["inputs2"]["input_ids"].shape, dtype=torch.float32, device=device
        )
    return res


def get_falcon_mamba_7b(
    batch_size: int = 2,
    input_cache: bool = True,
    inputs_as_tuple: bool = False,
    common_dynamic_shapes: bool = False,
    device: str = "cpu",
    **kwargs,
) -> Tuple[Any, Union[Tuple[Any, ...], Dict[str, Any]]]:
    """
    Gets a non initialized model.

    :param inputs_as_tuple: returns dummy inputs as a dictionary or not
    :param batch_size: batch size
    :param input_cache: generate data for this iteration with or without cache
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``:
    :param device: device
    :param common_dynamic_shapes: if True returns dynamic shapes as well
    :return: dictionary

    See `flacon-mamba-7b/config.json
    <https://huggingface.co/tiiuae/falcon-mamba-7b/blob/main/config.json>`_.
    """
    import torch
    import transformers

    config = {
        "_name_or_path": "./",
        "architectures": ["FalconMambaForCausalLM"],
        "bos_token_id": 0,
        "conv_kernel": 4,
        "eos_token_id": 11,
        "expand": 16,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.1,
        "intermediate_size": 8192,
        "layer_norm_epsilon": 1e-05,
        "model_type": "falcon_mamba",
        "num_hidden_layers": 64,
        "pad_token_id": 11,
        "rescale_prenorm_residual": False,
        "residual_in_fp32": True,
        "state_size": 16,
        "tie_word_embeddings": False,
        "time_step_floor": 0.0001,
        "time_step_init_scheme": "random",
        "time_step_max": 0.1,
        "time_step_min": 0.001,
        "time_step_rank": 256,
        "time_step_scale": 1.0,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.43.0.dev0",
        "use_bias": False,
        "use_cache": True,
        "use_conv_bias": True,
        "vocab_size": 65024,
    }
    config.update(
        {
            "_from_model_config": True,
            "bos_token_id": 0,
            "eos_token_id": 11,
            "pad_token_id": 11,
            "transformers_version": "4.43.0.dev0",
        }
    )
    assert_found(kwargs, config)
    config.update(**kwargs)
    conf = transformers.FalconMambaConfig(**config)
    model = transformers.FalconMambaForCausalLM(conf)
    model.eval()

    res = finalize_llm_setup(
        model,
        batch_size,
        max_token_id=65024,
        cache_last_dim=64,
        common_dynamic_shapes=common_dynamic_shapes,
        inputs_as_tuple=inputs_as_tuple,
        num_hidden_layers=config["num_hidden_layers"],
        input_cache=input_cache,
        seq_length_multiple=8,
        input_cache_class=transformers.cache_utils.MambaCache,
        device=device,
    )
    res["inputs"]["attention_mask"] = torch.ones(
        res["inputs"]["input_ids"].shape, dtype=torch.float32, device=device
    )
    res["inputs"]["cache_position"] = torch.arange(0, config["conv_kernel"], device=device)
    if "inputs2" in res:
        res["inputs2"]["attention_mask"] = torch.ones(
            res["inputs2"]["input_ids"].shape, dtype=torch.float32, device=device
        )
        res["inputs2"]["cache_position"] = torch.arange(
            0, config["conv_kernel"], device=device
        )
    return res


#######
# Llama
#######


def get_smollm_1_7b(
    batch_size: int = 2,
    input_cache: bool = True,
    inputs_as_tuple: bool = False,
    common_dynamic_shapes: bool = False,
    **kwargs,
) -> Tuple[Any, Union[Tuple[Any, ...], Dict[str, Any]]]:
    """
    Gets a non initialized model.

    :param inputs_as_tuple: returns dummy inputs as a dictionary or not
    :param batch_size: batch size
    :param input_cache: generate data for this iteration with or without cache
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :param common_dynamic_shapes: if True returns dynamic shapes as well
    :return: dictionary

    See `SmolLM-1.7B
    <https://huggingface.co/HuggingFaceTB/SmolLM-1.7B/blob/main/config.json>`_.
    """
    import transformers

    config = {
        "_name_or_path": "/fsx/loubna/checkpoints/cosmo2_1T/500000",
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 24,
        "num_key_value_heads": 32,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "transformers_version": "4.39.3",
        "use_cache": True,
        "vocab_size": 49152,
        "_attn_implementation": "eager",
    }
    config.update(
        {
            "_from_model_config": True,
            "bos_token_id": 0,
            "eos_token_id": 0,
            "transformers_version": "4.39.3",
        }
    )
    assert_found(kwargs, config)
    config.update(**kwargs)
    conf = transformers.LlamaConfig(**config)
    model = transformers.LlamaForCausalLM(conf)
    model.eval()

    return finalize_llm_setup(
        model,
        batch_size,
        max_token_id=49152,
        cache_last_dim=64,
        common_dynamic_shapes=common_dynamic_shapes,
        inputs_as_tuple=inputs_as_tuple,
        num_hidden_layers=config["num_hidden_layers"],
        input_cache=input_cache,
        seq_length_multiple=8,
    )


def get_llama32_9b_vision(
    inputs_as_tuple: bool = False, common_dynamic_shapes: bool = False, **kwargs
) -> Tuple[Any, Union[Tuple[Any, ...], Dict[str, Any]]]:
    """
    Gets a non initialized model.

    :param inputs_as_tuple: returns dummy inputs as a dictionary or not
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :param common_dynamic_shapes: if True returns dynamic shapes as well
    :return: model, inputs

    See `MLlama
    <https://huggingface.co/docs/transformers/main/en/model_doc/mllama>`_.
    """
    import torch
    import transformers

    assert not common_dynamic_shapes, "dynamic shapes are not implemented"

    config = {}
    config.update(**kwargs)

    vision_config = transformers.MllamaVisionConfig(**config)
    text_config = transformers.MllamaTextConfig(**config)
    configuration = transformers.MllamaConfig(vision_config, text_config)
    model = transformers.MllamaForConditionalGeneration(configuration)
    model.eval()

    dim = (1, 30)
    inputs = dict(
        input_ids=torch.randint(0, 49152, dim).to(torch.int64),
        pixel_values=torch.rand((1, 1, 1, 3, 512, 1080)).to(torch.float16),
        aspect_ratio_mask=None,
        aspect_ratio_ids=torch.from_numpy(np.array([[2]], dtype=np.int32)),
        attention_mask=torch.ones(*dim, dtype=torch.int64),
    )

    if inputs_as_tuple:
        inputs = tuple(inputs.values())

    return model, inputs


def get_tiny_llm(
    batch_size: int = 2,
    input_cache: bool = True,
    inputs_as_tuple: bool = False,
    common_dynamic_shapes: bool = False,
    dynamic_rope: bool = False,
    **kwargs,
) -> Tuple[Any, Union[Tuple[Any, ...], Dict[str, Any]]]:
    """
    Gets a non initialized model.

    :param inputs_as_tuple: returns dummy inputs as a dictionary or not
    :param batch_size: batch size
    :param input_cache: generate data for this iteration with or without cache
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :param common_dynamic_shapes: if True returns dynamic shapes as well
    :param dynamic_rope: use dynamic rope (see :class:`transformers.LlamaConfig`)
    :return: dictionary

    See `arnir0/Tiny-LLM
    <https://huggingface.co/arnir0/Tiny-LLM>`_.
    """
    import transformers

    config = {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 192,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 1024,
        "model_type": "llama",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {"rope_type": "dynamic", "factor": 10.0} if dynamic_rope else None,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "transformers_version": "4.31.0.dev0",
        "use_cache": True,
        "vocab_size": 32000,
    }

    assert_found(kwargs, config)
    config.update(**kwargs)
    conf = transformers.LlamaConfig(**config)
    model = transformers.LlamaForCausalLM(conf)
    model.eval()

    return finalize_llm_setup(
        model,
        batch_size,
        max_token_id=31999,
        common_dynamic_shapes=common_dynamic_shapes,
        inputs_as_tuple=inputs_as_tuple,
        input_cache=input_cache,
        cache_last_dim=96,
        num_key_value_heads=1,
    )
