"""
Benchmark exporters
===================

Benchmarks many models from the `HuggingFace <https://huggingface.co/models>`_.

::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --help
    
    
::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model ""
    
::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model dummy,dummy16 --verbose=1
    
"""

import importlib
import pprint
import textwrap
from typing import Any, Optional, Set, Tuple, List
import torch
from torch._dynamo.testing import collect_results, reset_rng_state
from torch._dynamo.utils import clone_inputs
from experimental_experiment.torch_bench._bash_bench_common import (
    download_retry_decorator,
    _rand_int_tensor,
    BenchmarkRunner,
    ModelRunner,
    MakeConfig,
)


class HuggingfaceRunner(BenchmarkRunner):

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

    MODELS_FILENAME = textwrap.dedent(
        """
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
    )

    BATCH_SIZE_KNOWN_MODELS = dict()

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

    @classmethod
    def initialize(container):
        """
        Steps to run before running the benchmark.
        """
        import transformers

        for cls in container.imports:
            assert hasattr(
                transformers, cls
            ), f"{cls!r} not found, update transformers."

        lines = container.MODELS_FILENAME.split("\n")
        lines = [line.rstrip() for line in lines]
        for line in lines:
            if not line or len(line) < 2:
                continue
            model_name, batch_size = line.split(",")
            batch_size = int(batch_size)
            container.BATCH_SIZE_KNOWN_MODELS[model_name] = batch_size

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

            def _get_random_inputs(self, device: str):
                return (torch.randn(1, 5).to(device),)

            config = MakeConfig(download=False)

        class Neuron16(Neuron):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets, dtype=torch.float16)
                assert self.linear.weight.dtype == torch.float16
                assert self.linear.bias.dtype == torch.float16

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

            def _get_random_inputs(self, device: str):
                return (torch.randn(1, 5).to(torch.float16).to(device),)

        container.EXTRA_MODELS.update(
            {
                "dummy": (
                    Neuron.config,
                    Neuron,
                ),
                "dummy16": (
                    Neuron16.config,
                    Neuron16,
                ),
                "AllenaiLongformerBase": (
                    transformers.AutoConfig.from_pretrained(
                        "allenai/longformer-base-4096"
                    ),
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

    @classmethod
    def _get_module_cls_by_model_name(container, model_cls_name):
        _module_by_model_name = {
            "Speech2Text2Decoder": "transformers.models.speech_to_text_2.modeling_speech_to_text_2",
            "TrOCRDecoder": "transformers.models.trocr.modeling_trocr",
        }
        module_name = _module_by_model_name.get(model_cls_name, "transformers")
        module = importlib.import_module(module_name)
        return getattr(module, model_cls_name)

    @classmethod
    def _get_sequence_length(container, model_cls, model_name):
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
        ) or model_name in {
            "DistillGPT2",
            "GoogleFnet",
            "YituTechConvBert",
            "CamemBert",
        }:
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

    @classmethod
    def _generate_inputs_for_model(
        container, model_cls, model, model_name, bs, device, include_loss_args=False
    ):
        if hasattr(model, "_get_random_inputs"):
            return model._get_random_inputs(device)

        import transformers

        num_choices = 3
        num_visual_features = 42
        seq_length = container._get_sequence_length(model_cls, model_name)
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
            inputt = _rand_int_tensor(
                device, 0, vocab_size, (bs, num_choices, seq_length)
            )
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
            input_dict["visual_pos"] = torch.randn(
                bs, num_visual_features, visual_pos_dim
            )

        if include_loss_args:
            if model_name.endswith("PreTraining"):
                if model_cls in [
                    transformers.ElectraForPreTraining,
                    transformers.LxmertForPreTraining,
                ]:
                    input_dict["labels"] = _rand_int_tensor(
                        device, 0, 1, (bs, seq_length)
                    )
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
                input_dict["end_positions"] = _rand_int_tensor(
                    device, 0, seq_length, (bs,)
                )
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
            elif model_name in container.EXTRA_MODELS:
                input_dict["labels"] = _rand_int_tensor(
                    device, 0, vocab_size, (bs, seq_length)
                )
            else:
                raise NotImplementedError(
                    f"Class {model_name!r} unsupported for training test "
                )

        return input_dict

    def __init__(
        self,
        device: str,
        partition_id: int = 0,
        total_partitions: int = 1,
        include_model_names: Optional[Set[str]] = None,
        exclude_model_names: Optional[Set[str]] = None,
        verbose: int = 0,
        target_opset: int = 18,
    ):
        super().__init__(
            "huggingface",
            device=device,
            partition_id=partition_id,
            total_partitions=total_partitions,
            include_model_names=include_model_names,
            exclude_model_names=exclude_model_names,
            verbose=verbose,
            target_opset=target_opset,
        )
        if not self.EXTRA_MODELS:
            self.initialize()

    def _get_model_cls_and_config(self, model_name: str) -> Tuple[type, Any]:
        if model_name not in self.EXTRA_MODELS:
            import transformers

            model_cls = self._get_module_cls_by_model_name(model_name)
            config_cls = model_cls.config_class
            config = config_cls() if config_cls else None

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
            config, model_cls = self.EXTRA_MODELS[model_name]

        assert config is not None, f"Config cannot be None for model {model_name!r}."
        return model_cls, config

    @download_retry_decorator(retry=5)
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
        model_name: str,
        batch_size: Optional[int] = None,
    ) -> ModelRunner:
        is_training = self.training
        use_eval_mode = self.use_eval_mode
        dtype = self.dtype
        reset_rng_state()
        model_cls, config = self._get_model_cls_and_config(model_name)
        if getattr(config, "download", True):
            model = self._download_model(model_name)
        else:
            model = model_cls()
        if dtype is None:
            model = model.to(self.device)
        else:
            model = model.to(self.device, dtype=dtype)
        if self.enable_activation_checkpointing:
            model.gradient_checkpointing_enable()
        if model_name in self.BATCH_SIZE_KNOWN_MODELS:
            batch_size_default = self.BATCH_SIZE_KNOWN_MODELS[model_name]
        elif batch_size is None:
            batch_size_default = 16

        if batch_size is None:
            batch_size = batch_size_default
            if model_name in self.BATCH_SIZE_DIVISORS:
                batch_size = max(
                    int(batch_size / self.BATCH_SIZE_DIVISORS[model_name]), 1
                )

        example_inputs = self._generate_inputs_for_model(
            model_cls,
            model,
            model_name,
            batch_size,
            self.device,
            include_loss_args=True,
        )

        for attr in dir(config):
            if "drop" in attr and isinstance(getattr(config, attr), float):
                setattr(config, attr, 1e-30)

        if is_training and not use_eval_mode:
            model.train()
        else:
            model.eval()

        return ModelRunner(
            model,
            example_inputs,
            device=self.device,
            dtype=self.dtype,
            warmup=self.warmup,
            repeat=self.repeat,
        )

    def iter_model_names(self):
        model_names = list(self.BATCH_SIZE_KNOWN_MODELS.keys()) + list(
            self.EXTRA_MODELS.keys()
        )
        model_names = set(model_names)
        model_names = sorted(model_names)

        start, end = self.get_benchmark_indices(len(model_names))
        assert (
            start < end
        ), f"Empty partition (start={start}, end={end}, model_names={model_names!r})"
        for index, model_name in enumerate(model_names):
            if index < start or index >= end:
                continue
            if (
                self.include_model_names and model_name not in self.include_model_names
            ) or model_name in self.exclude_model_names:
                continue
            yield model_name

    def forward_pass(self, mod, inputs, collect_outputs=True):
        return mod(**inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        cloned_inputs = clone_inputs(inputs)
        self.optimizer_zero_grad(mod)
        pred = mod(**cloned_inputs)
        loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self.optimizer_step()
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        return None


def parse_args(new_args: Optional[List[str]] = None):
    from experimental_experiment.args import get_parsed_args

    args = get_parsed_args(
        "experimental_experiment.torch_bench.bash_bench_huggingface",
        description=__doc__,
        model=(
            "dummy",
            "if empty, prints the list of models, "
            "all for all models, a list of indices works as well",
        ),
        exporter=(
            "custom",
            "custom, dynamo, dynamo2, script",
        ),
        process=("0", "run every run in a separate process"),
        device=("cpu", "'cpu' or 'cuda'"),
        dynamic=("0", "use dynamic shapes"),
        target_opset=("18", "opset to convert into, use with backend=custom"),
        verbose=("0", "verbosity"),
        disable_pattern=("", "a list of optimization patterns to disable"),
        enable_pattern=("default", "list of optimization patterns to enable"),
        dump_folder=("dump_bash_bench", "where to dump the exported model"),
        output_data=(
            "output_data_huggingface.csv",
            "when running multiple configuration, save the results in that file",
        ),
        new_args=new_args,
    )
    return args


def main(args: Optional[List[str]] = None):
    args = parse_args(new_args=args)

    from experimental_experiment.bench_run import (
        multi_run,
        make_configs,
        make_dataframe_from_benchmark_data,
        run_benchmark,
    )

    runner = HuggingfaceRunner(device=args.device)
    names = runner.get_model_name_list()

    if not args.model:
        # prints the list of models.
        print(f"list of models for device={args.device}")
        print("--")
        print("\n".join([f"{i+1: 3d} - {n}" for i, n in enumerate(names)]))
        print("--")

    else:
        if args.model == "all":
            args.model = ",".join(names)

        if multi_run(args):
            configs = make_configs(args)
            data = run_benchmark(
                "experimental_experiment.torch_bench.bash_bench_huggingface",
                configs,
                args.verbose,
                stop_if_exception=False,
            )
            if args.verbose > 2:
                pprint.pprint(data if args.verbose > 3 else data[:2])
            if args.output_data:
                df = make_dataframe_from_benchmark_data(data, detailed=False)
                print("Prints the results into file {args.output_data!r}")
                df.to_csv(args.output_data, index=False)
                df.to_excel(args.output_data + ".xlsx", index=False)
                if args.verbose:
                    print(df)

        else:
            try:
                indice = int(args.model)
                name = names[indice]
            except (TypeError, ValueError):
                name = args.model

            runner = HuggingfaceRunner(
                include_model_names={name},
                verbose=args.verbose,
                device=args.device,
                target_opset=args.target_opset,
            )
            data = list(
                runner.enumerate_test_models(
                    process=args.process in ("1", 1, "True", True),
                    exporter=args.exporter,
                )
            )
            if len(data) == 1:
                for k, v in data[0].items():
                    print(f":{k},{v};")
            else:
                print(f"::model_name,{name};")
                print(f":device,{args.device};")
                print(f":ERROR,unexpected number of data {len(data)};")


if __name__ == "__main__":
    main()