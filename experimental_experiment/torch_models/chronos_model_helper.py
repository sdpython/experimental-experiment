from typing import Any, Dict
from . import assert_found


#########
# chronos
#########


def get_chronos_t5_tiny(
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

    See `chronos_t5_tiny
    <https://huggingface.co/amazon/chronos-t5-tiny>`_.

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
    import torch
    import transformers
    import chronos

    config = {
        "architectures": ["T5ForConditionalGeneration"],
        "classifier_dropout": 0.0,
        "d_ff": 1024,
        "d_kv": 64,
        "d_model": 256,
        "decoder_start_token_id": 0,
        "dense_act_fn": "relu",
        "dropout_rate": 0.1,
        "eos_token_id": 1,
        "feed_forward_proj": "relu",
        "initializer_factor": 0.05,
        "is_encoder_decoder": True,
        "is_gated_act": False,
        "layer_norm_epsilon": 1e-6,
        "model_type": "t5",
        "n_positions": 512,
        "num_decoder_layers": 4,
        "num_heads": 4,
        "num_layers": 4,
        "pad_token_id": 0,
        "relative_attention_max_distance": 128,
        "relative_attention_num_buckets": 32,
        "torch_dtype": "float32",
        "transformers_version": "4.37.2",
        "use_cache": True,
        "vocab_size": 4096,
        "chronos_config": {
            "tokenizer_class": "MeanScaleUniformBins",
            "tokenizer_kwargs": {"low_limit": -15.0, "high_limit": 15.0},
            "n_tokens": 4096,
            "n_special_tokens": 2,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "use_eos_token": True,
            "model_type": "seq2seq",
            "context_length": 512,
            "prediction_length": 64,
            "num_samples": 20,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
        },
    }
    assert_found(kwargs, config)
    config.update(**kwargs)

    t5conf = transformers.T5Config(**config)
    inner_model = transformers.T5ForConditionalGeneration(t5conf)
    conf = chronos.ChronosConfig(**config["chronos_config"])
    model = chronos.ChronosModel(conf, inner_model)
    model.eval()

    if common_dynamic_shapes:
        dbatch_size = torch.export.Dim("batch_size", min=1, max=1024)
        dseq_length = torch.export.Dim("seq_length", min=1, max=128)
        res = dict(
            model=model,
            inputs=(
                torch.randint(low=1, high=2352, size=(batch_size, 50), dtype=torch.int64),
                torch.full((batch_size, 50), fill_value=True),
                torch.tensor(12, dtype=torch.int64),
            ),
            inputs2=(
                torch.randint(low=1, high=2352, size=(batch_size + 1, 55), dtype=torch.int64),
                torch.full((batch_size + 1, 55), fill_value=True),
                torch.tensor(16, dtype=torch.int64),
            ),
            dynamic_shapes={
                "input_ids": {0: dbatch_size, 1: dseq_length},
                "attention_mask": {0: dbatch_size, 1: dseq_length},
                "prediction_length": {},
            },
        )
    else:
        res = dict(
            model=model,
            inputs=(
                torch.randint(low=1, high=2352, size=(batch_size, 50), dtype=torch.int64),
                torch.full((batch_size, 50), fill_value=True),
                torch.tensor(12, dtype=torch.int64),
            ),
        )
    if not inputs_as_tuple:
        for k in ["inputs", "inputs2"]:
            if k not in res:
                continue
            res[k] = dict(zip(["input_ids", "attention_mask", "prediction_length"], res[k]))
    return res
