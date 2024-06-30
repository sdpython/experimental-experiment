import importlib
import os
import re
import torch
from torch._dynamo.testing import collect_results
from torch._dynamo.utils import clone_inputs

# from .common import BenchmarkRunner, download_retry_decorator, main, reset_rng_state

imports = [
    "AlbertForPreTraining",
    "AutoConfig",
    "AutoModelForCausalLM",
    "AutoModelForMaskedLM",
    "AutoModelForSeq2SeqLM",
    "BigBirdConfig",
    "BlenderbotForConditionalGeneration",
    "BlenderbotModel",
    "BlenderbotSmallForConditionalGeneration",
    "BlenderbotSmallModel",
    "CLIPModel",
    "CLIPVisionModel",
    "ElectraForPreTraining",
    "GPT2ForSequenceClassification",
    "GPTJForSequenceClassification",
    "GPTNeoForSequenceClassification",
    "HubertForSequenceClassification",
    "LxmertForPreTraining",
    "LxmertForQuestionAnswering",
    "MarianForCausalLM",
    "MarianModel",
    "MarianMTModel",
    "PegasusForConditionalGeneration",
    "PegasusModel",
    "ReformerConfig",
    "ViTForImageClassification",
    "ViTForMaskedImageModeling",
    "ViTModel",
]

MODELS_FILENAME = """
AlbertForMaskedLM,8
AlbertForQuestionAnswering,8
AllenaiLongformerBase,8
BartForCausalLM,8
BartForConditionalGeneration,4
BertForMaskedLM,32
BertForQuestionAnswering,32
BlenderbotForCausalLM,32
BlenderbotForConditionalGeneration,16
BlenderbotSmallForCausalLM,256
BlenderbotSmallForConditionalGeneration,128
CamemBert,32
DebertaForMaskedLM,32
DebertaForQuestionAnswering,32
DebertaV2ForMaskedLM,8
DebertaV2ForQuestionAnswering,8
DistilBertForMaskedLM,256
DistilBertForQuestionAnswering,512
DistillGPT2,32
ElectraForCausalLM,64
ElectraForQuestionAnswering,128
GPT2ForSequenceClassification,8
GPTJForCausalLM,1
GPTJForQuestionAnswering,1
GPTNeoForCausalLM,32
GPTNeoForSequenceClassification,32
GoogleFnet,32
LayoutLMForMaskedLM,32
LayoutLMForSequenceClassification,32
M2M100ForConditionalGeneration,64
MBartForCausalLM,8
MBartForConditionalGeneration,4
MT5ForConditionalGeneration,32
MegatronBertForCausalLM,16
MegatronBertForQuestionAnswering,16
MobileBertForMaskedLM,256
MobileBertForQuestionAnswering,256
OPTForCausalLM,4
PLBartForCausalLM,16
PLBartForConditionalGeneration,8
PegasusForCausalLM,128
PegasusForConditionalGeneration,64
RobertaForCausalLM,32
RobertaForQuestionAnswering,32
Speech2Text2ForCausalLM,1024
T5ForConditionalGeneration,8
T5Small,8
TrOCRForCausalLM,64
XGLMForCausalLM,32
XLNetLMHeadModel,16
YituTechConvBert,32
"""

BATCH_SIZE_KNOWN_MODELS = dict()

SKIP = {
    # Difficult to setup accuracy test because .eval() not supported
    # "Reformer",
    # Fails deepcopy
    # "BlenderbotForConditionalGeneration",
    # "GPTNeoForCausalLM",
    # "GPTNeoForSequenceClassification",
    # Fails with even batch size = 1
    # "GPTJForCausalLM",
    # "GPTJForQuestionAnswering",
}

BATCH_SIZE_DIVISORS = {
    "AlbertForMaskedLM": 2,
    "AlbertForQuestionAnswering": 2,
    "AllenaiLongformerBase": 2,
    "BartForCausalLM": 2,
    "BartForConditionalGeneration": 2,
    "BertForMaskedLM": 2,
    "BertForQuestionAnswering": 2,
    "BlenderbotForCausalLM": 8,
    # "BlenderbotForConditionalGeneration" : 16,
    "BlenderbotSmallForCausalLM": 4,
    "BlenderbotSmallForConditionalGeneration": 2,
    "CamemBert": 2,
    "DebertaForMaskedLM": 4,
    "DebertaForQuestionAnswering": 2,
    "DebertaV2ForMaskedLM": 4,
    "DebertaV2ForQuestionAnswering": 8,
    "DistilBertForMaskedLM": 2,
    "DistilBertForQuestionAnswering": 2,
    "DistillGPT2": 2,
    "ElectraForCausalLM": 2,
    "ElectraForQuestionAnswering": 2,
    "GPT2ForSequenceClassification": 2,
    # "GPTJForCausalLM" : 2,
    # "GPTJForQuestionAnswering" : 2,
    # "GPTNeoForCausalLM" : 32,
    # "GPTNeoForSequenceClassification" : 2,
    "GoogleFnet": 2,
    "LayoutLMForMaskedLM": 2,
    "LayoutLMForSequenceClassification": 2,
    "M2M100ForConditionalGeneration": 4,
    "MBartForCausalLM": 2,
    "MBartForConditionalGeneration": 2,
    "MT5ForConditionalGeneration": 2,
    "MegatronBertForCausalLM": 4,
    "MegatronBertForQuestionAnswering": 2,
    "MobileBertForMaskedLM": 2,
    "MobileBertForQuestionAnswering": 2,
    "OPTForCausalLM": 2,
    "PLBartForCausalLM": 2,
    "PLBartForConditionalGeneration": 2,
    "PegasusForCausalLM": 4,
    "PegasusForConditionalGeneration": 2,
    "RobertaForCausalLM": 2,
    "RobertaForQuestionAnswering": 2,
    "Speech2Text2ForCausalLM": 4,
    "T5ForConditionalGeneration": 2,
    "T5Small": 2,
    "TrOCRForCausalLM": 2,
    "XGLMForCausalLM": 4,
    "XLNetLMHeadModel": 2,
    "YituTechConvBert": 2,
}

EXTRA_MODELS = {}


def initialize():
    """
    Steps to run before running the benchmark.
    """
    import transformers

    if "TORCHINDUCTOR_FX_GRAPH_CACHE" not in os.environ:
        torch._inductor.config.fx_graph_cache = True

    for cls in imports:
        if not hasattr(transformers, cls):
            raise ModuleNotFoundError(f"{cls!r} is here, update transformers")

    lines = MODELS_FILENAME.split("\n")
    lines = [line.rstrip() for line in lines]
    for line in lines:
        model_name, batch_size = line.split(",")
        batch_size = int(batch_size)
        BATCH_SIZE_KNOWN_MODELS[model_name] = batch_size

    EXTRA_MODELS.update(
        {
            "AllenaiLongformerBase": (
                transformers.AutoConfig.from_pretrained("allenai/longformer-base-4096"),
                transformers.AutoModelForMaskedLM,
            ),
            "Reformer": (
                transformers.ReformerConfig(),
                transformers.AutoModelForMaskedLM,
            ),
            "T5Small": (
                transformers.AutoConfig.from_pretrained("t5-small"),
                transformers.AutoModelForSeq2SeqLM,
            ),
            # "BigBird": (
            #     BigBirdConfig(attention_type="block_sparse"),
            #     AutoModelForMaskedLM,
            # ),
            "DistillGPT2": (
                transformers.AutoConfig.from_pretrained("distilgpt2"),
                transformers.AutoModelForCausalLM,
            ),
            "GoogleFnet": (
                transformers.AutoConfig.from_pretrained("google/fnet-base"),
                transformers.AutoModelForMaskedLM,
            ),
            "YituTechConvBert": (
                transformers.AutoConfig.from_pretrained("YituTech/conv-bert-base"),
                transformers.AutoModelForMaskedLM,
            ),
            "CamemBert": (
                transformers.AutoConfig.from_pretrained("camembert-base"),
                transformers.AutoModelForMaskedLM,
            ),
        }
    )


def _get_module_cls_by_model_name(model_cls_name):
    _module_by_model_name = {
        "Speech2Text2Decoder": "transformers.models.speech_to_text_2.modeling_speech_to_text_2",
        "TrOCRDecoder": "transformers.models.trocr.modeling_trocr",
    }
    module_name = _module_by_model_name.get(model_cls_name, "transformers")
    module = importlib.import_module(module_name)
    return getattr(module, model_cls_name)


def _get_sequence_length(model_cls, model_name):
    if model_name.startswith(("Blenderbot",)):
        seq_length = 128
    elif model_name.startswith(("GPT2", "Bart", "T5", "PLBart", "MBart")):
        seq_length = 1024
    elif model_name in ("AllenaiLongformerBase", "BigBird"):
        seq_length = 1024
    elif model_name.startswith("OPT"):
        seq_length = 2048
    elif "Reformer" in model_name:
        seq_length = 4096
    elif model_name.startswith(
        (
            "Albert",
            "Deberta",
            "Layout",
            "Electra",
            "XLNet",
            "MegatronBert",
            "Bert",
            "Roberta",
        )
    ) or model_name in {"DistillGPT2", "GoogleFnet", "YituTechConvBert", "CamemBert"}:
        seq_length = 512
    elif model_name in ["TrOCRForCausalLM"]:
        seq_length = 256
    elif model_name.startswith("MobileBert"):
        seq_length = 128
    elif model_name.startswith("Wav2Vec2"):
        seq_length = 2**16
    else:
        seq_length = 128
    return seq_length


def _generate_inputs_for_model(
    model_cls, model, model_name, bs, device, include_loss_args=False
):
    import transformers

    num_choices = 3
    num_visual_features = 42
    seq_length = _get_sequence_length(model_cls, model_name)
    vocab_size = model.config.vocab_size

    if model_name.startswith("Wav2Vec2"):
        # TODO: If we add more input_values style models, try to work this
        # into the overall control flow
        target_length = 100
        return {
            "input_values": torch.randn((bs, seq_length), device=device),
            # Added because that's what the example training script has
            "attention_mask": _rand_int_tensor(device, 0, 2, (bs, seq_length)),
            "labels": _rand_int_tensor(device, 0, vocab_size, (bs, target_length)),
        }

    if model_name.endswith("MultipleChoice"):
        inputt = _rand_int_tensor(device, 0, vocab_size, (bs, num_choices, seq_length))
    elif model_name.startswith("Roberta"):
        inputt = _rand_int_tensor(device, 0, 1, (bs, seq_length))
    else:
        inputt = _rand_int_tensor(device, 0, vocab_size, (bs, seq_length))

    if "Bart" in model_name:
        inputt[:, -1] = model.config.eos_token_id

    input_dict = {"input_ids": inputt}

    if (
        model_name.startswith("T5")
        or model_name.startswith("M2M100")
        or model_name.startswith("MT5")
        or model_cls
        in {
            transformers.BlenderbotModel,
            transformers.BlenderbotSmallModel,
            transformers.BlenderbotForConditionalGeneration,
            transformers.BlenderbotSmallForConditionalGeneration,
            transformers.PegasusModel,
            transformers.PegasusForConditionalGeneration,
            transformers.MarianModel,
            transformers.MarianMTModel,
        }
    ):
        input_dict["decoder_input_ids"] = inputt

    if model_name.startswith("Lxmert"):
        visual_feat_dim, visual_pos_dim = (
            model.config.visual_feat_dim,
            model.config.visual_pos_dim,
        )
        input_dict["visual_feats"] = torch.randn(
            bs, num_visual_features, visual_feat_dim
        )
        input_dict["visual_pos"] = torch.randn(bs, num_visual_features, visual_pos_dim)

    if include_loss_args:
        if model_name.endswith("PreTraining"):
            if model_cls in [
                transformers.ElectraForPreTraining,
                transformers.LxmertForPreTraining,
            ]:
                input_dict["labels"] = _rand_int_tensor(device, 0, 1, (bs, seq_length))
            else:
                label_name = (
                    "sentence_order_label"
                    if model_cls in [transformers.AlbertForPreTraining]
                    else "next_sentence_label"
                )
                input_dict["labels"] = (
                    _rand_int_tensor(device, 0, vocab_size, (bs, seq_length)),
                )
                input_dict[label_name] = _rand_int_tensor(device, 0, 1, (bs,))
        elif model_name.endswith("QuestionAnswering"):
            input_dict["start_positions"] = _rand_int_tensor(
                device, 0, seq_length, (bs,)
            )
            input_dict["end_positions"] = _rand_int_tensor(device, 0, seq_length, (bs,))
        elif (
            model_name.endswith("MaskedLM")
            or model_name.endswith("HeadModel")
            or model_name.endswith("CausalLM")
            or model_name.endswith("DoubleHeadsModel")
        ):
            input_dict["labels"] = _rand_int_tensor(
                device, 0, vocab_size, (bs, seq_length)
            )
        elif model_name.endswith("TokenClassification"):
            input_dict["labels"] = _rand_int_tensor(
                device, 0, model.config.num_labels - 1, (bs, seq_length)
            )
        elif model_name.endswith("MultipleChoice"):
            input_dict["labels"] = _rand_int_tensor(device, 0, num_choices, (bs,))
        elif model_name.endswith("SequenceClassification"):
            input_dict["labels"] = _rand_int_tensor(
                device, 0, model.config.num_labels - 1, (bs,)
            )
        elif model_name.endswith("NextSentencePrediction"):
            input_dict["labels"] = _rand_int_tensor(device, 0, 1, (bs,))
        elif model_name.endswith("ForConditionalGeneration"):
            input_dict["labels"] = _rand_int_tensor(
                device, 0, vocab_size - 1, (bs, seq_length)
            )
        elif model_name in EXTRA_MODELS:
            input_dict["labels"] = _rand_int_tensor(
                device, 0, vocab_size, (bs, seq_length)
            )
        else:
            raise NotImplementedError(
                f"Class {model_name!r} unsupported for training test "
            )

    return input_dict


def _rand_int_tensor(device, low, high, shape):
    return torch.randint(
        low,
        high,
        shape,
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )


class HuggingfaceRunner(BenchmarkRunner):
    def __init__(self):
        super().__init__()
        self.suite_name = "huggingface"

    def _get_model_cls_and_config(self, model_name):
        if model_name not in EXTRA_MODELS:
            import transformers

            model_cls = _get_module_cls_by_model_name(model_name)
            config_cls = model_cls.config_class
            config = config_cls()

            # NB: some models need a pad token defined to handle BS > 1
            if (
                model_cls
                in {
                    transformers.GPT2ForSequenceClassification,
                    transformers.GPTNeoForSequenceClassification,
                    transformers.GPTJForSequenceClassification,
                }
                or model_cls.__name__.startswith("Roberta")
                or model_cls.__name__.startswith("Marian")
            ):
                config.pad_token_id = 0

        else:
            config, model_cls = EXTRA_MODELS[model_name]

        return model_cls, config

    @download_retry_decorator
    def _download_model(self, model_name):
        model_cls, config = self._get_model_cls_and_config(model_name)
        if "auto" in model_cls.__module__:
            # Handle auto classes
            model = model_cls.from_config(config)
        else:
            model = model_cls(config)
        return model

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
        extra_args=None,
    ):
        is_training = self.args.training
        use_eval_mode = self.args.use_eval_mode
        dtype = torch.float32
        reset_rng_state()
        model_cls, config = self._get_model_cls_and_config(model_name)
        model = self._download_model(model_name)
        model = model.to(device, dtype=dtype)
        if self.args.enable_activation_checkpointing:
            model.gradient_checkpointing_enable()
        if model_name in BATCH_SIZE_KNOWN_MODELS:
            batch_size_default = BATCH_SIZE_KNOWN_MODELS[model_name]
        elif batch_size is None:
            batch_size_default = 16

        if batch_size is None:
            batch_size = batch_size_default
            if model_name in BATCH_SIZE_DIVISORS:
                batch_size = max(int(batch_size / BATCH_SIZE_DIVISORS[model_name]), 1)

        example_inputs = _generate_inputs_for_model(
            model_cls, model, model_name, batch_size, device, include_loss_args=True
        )

        for attr in dir(config):
            if "drop" in attr and isinstance(getattr(config, attr), float):
                setattr(config, attr, 1e-30)

        if is_training and not use_eval_mode:
            model.train()
        else:
            model.eval()

        self.validate_model(model, example_inputs)
        return device, model_name, model, example_inputs, batch_size

    def iter_model_names(self, args):
        model_names = list(BATCH_SIZE_KNOWN_MODELS.keys()) + list(EXTRA_MODELS.keys())
        model_names = set(model_names)
        model_names = sorted(model_names)

        start, end = self.get_benchmark_indices(len(model_names))
        for index, model_name in enumerate(model_names):
            if index < start or index >= end:
                continue
            if (
                not re.search("|".join(args.filter), model_name, re.I)
                or re.search("|".join(args.exclude), model_name, re.I)
                or model_name in args.exclude_exact
                or model_name in SKIP
            ):
                continue
            yield model_name

    @property
    def get_output_amp_train_process_func(self):
        return {}

    def pick_grad(self, name, is_training):
        if is_training:
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        cosine = self.args.cosine
        return 1e-3, cosine

    def compute_loss(self, pred):
        return pred[0]

    def forward_pass(self, mod, inputs, collect_outputs=True):
        with self.autocast(**self.autocast_arg):
            return mod(**inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        cloned_inputs = clone_inputs(inputs)
        self.optimizer_zero_grad(mod)
        with self.autocast(**self.autocast_arg):
            pred = mod(**cloned_inputs)
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self.optimizer_step()
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        return None


def huggingface_main():
    initialize()
    main(HuggingfaceRunner())


if __name__ == "__main__":
    huggingface_main()
