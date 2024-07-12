import io
import importlib
import textwrap
from typing import Any, Optional, Set, Tuple
import torch
from torch._dynamo.testing import collect_results
from torch._dynamo.utils import clone_inputs
from ._bash_bench_model_runner import (
    download_retry_decorator,
    _rand_int_tensor,
    ModelRunner,
    MakeConfig,
)
from ._bash_bench_benchmark_runner import BenchmarkRunner


class TimmRunner(BenchmarkRunner):

    MODELS_FILENAME = textwrap.dedent(
        """
        adv_inception_v3 128
        beit_base_patch16_224 128
        botnet26t_256 128
        cait_m36_384 4
        coat_lite_mini 128
        convit_base 128
        convmixer_768_32 64
        convnext_base 128
        crossvit_9_240 256
        cspdarknet53 128
        deit_base_distilled_patch16_224 128
        dla102 128
        dm_nfnet_f0 128
        dpn107 64
        eca_botnext26ts_256 128
        eca_halonext26ts 128
        ese_vovnet19b_dw 256
        fbnetc_100 512
        fbnetv3_b 256
        gernet_l 128
        ghostnet_100 512
        gluon_inception_v3 256
        gmixer_24_224 128
        gmlp_s16_224 128
        hrnet_w18 128
        inception_v3 128
        jx_nest_base 128
        lcnet_050 256
        levit_128 1024
        mixer_b16_224 128
        mixnet_l 128
        mnasnet_100 512
        mobilenetv2_100 128
        mobilenetv3_large_100 512
        mobilevit_s 128
        nfnet_l0 128
        pit_b_224 64
        pnasnet5large 32
        poolformer_m36 128
        regnety_002 1024
        repvgg_a2 128
        res2net101_26w_4s 128
        res2net50_14w_8s 128
        res2next50 128
        resmlp_12_224 128
        resnest101e 128
        rexnet_100 256
        sebotnet33ts_256 64
        selecsls42b 128
        spnasnet_100 128
        swin_base_patch4_window7_224 128
        swsl_resnext101_32x16d 64
        tf_efficientnet_b0 128
        tf_mixnet_l 128
        tinynet_a 128
        tnt_s_patch16_224 128
        twins_pcpvt_base 128
        visformer_small 128
        vit_base_patch16_224 128
        volo_d1_224 128
        xcit_large_24_p8_224 16
        """
    )

    BATCH_SIZE_DIVISORS = {
        "beit_base_patch16_224": 2,
        "convit_base": 2,
        "convmixer_768_32": 2,
        "convnext_base": 2,
        "cspdarknet53": 2,
        "deit_base_distilled_patch16_224": 2,
        "gluon_xception65": 2,
        "mobilevit_s": 2,
        "pnasnet5large": 2,
        "poolformer_m36": 2,
        "resnest101e": 2,
        "swin_base_patch4_window7_224": 2,
        "swsl_resnext101_32x16d": 2,
        "vit_base_patch16_224": 2,
        "volo_d1_224": 2,
        "jx_nest_base": 4,
    }

    REQUIRE_HIGHER_TOLERANCE = {
        "fbnetv3_b",
        "gmixer_24_224",
        "hrnet_w18",
        "inception_v3",
        "mixer_b16_224",
        "mobilenetv3_large_100",
        "sebotnet33ts_256",
        "selecsls42b",
        "cspdarknet53",
    }

    REQUIRE_EVEN_HIGHER_TOLERANCE = {
        "levit_128",
        "sebotnet33ts_256",
        "beit_base_patch16_224",
    }

    # These models need higher tolerance in MaxAutotune mode
    REQUIRE_EVEN_HIGHER_TOLERANCE_MAX_AUTOTUNE = {
        "gluon_inception_v3",
    }

    REQUIRE_HIGHER_TOLERANCE_FOR_FREEZING = {
        "adv_inception_v3",
        "botnet26t_256",
        "gluon_inception_v3",
        "selecsls42b",
        "swsl_resnext101_32x16d",
    }

    SCALED_COMPUTE_LOSS = {
        "ese_vovnet19b_dw",
        "fbnetc_100",
        "mnasnet_100",
        "mobilevit_s",
        "sebotnet33ts_256",
    }

    FORCE_AMP_FOR_FP16_BF16_MODELS = {
        "convit_base",
        "xcit_large_24_p8_224",
    }

    REQUIRE_LARGER_MULTIPLIER_FOR_SMALLER_TENSOR = {
        "mobilenetv3_large_100",
    }

    TIMM_MODELS = {}

    @classmethod
    def refresh_model_names(cls):
        import timm

        def get_family_name(name):
            known_families = [
                "darknet",
                "densenet",
                "dla",
                "dpn",
                "ecaresnet",
                "halo",
                "regnet",
                "efficientnet",
                "deit",
                "mobilevit",
                "mnasnet",
                "convnext",
                "resnet",
                "resnest",
                "resnext",
                "selecsls",
                "vgg",
                "xception",
            ]

            for known_family in known_families:
                if known_family in name:
                    return known_family

            if name.startswith("gluon_"):
                return "gluon_" + name.split("_")[1]
            return name.split("_")[0]

        def populate_family(models):
            family = {}
            for model_name in models:
                family_name = get_family_name(model_name)
                if family_name not in family:
                    family[family_name] = []
                family[family_name].append(model_name)
            return family

        all_models = timm.list_models(pretrained=True, exclude_filters=["*in21k"])
        all_models_family = populate_family(all_models)
        chosen_models = set()
        chosen_models.update(value[0] for key, value in all_models_family.items())
        return [c.split(".")[0] for c in chosen_models]

    @classmethod
    def initialize(container):
        """
        Steps to run before running the benchmark.
        """
        import timm

        models = set(timm.list_models())

        container._config = {"done": True}
        with io.StringIO(container.MODELS_FILENAME) as fh:
            lines = fh.readlines()
            lines = [line.rstrip() for line in lines]
            for line in lines:
                if len(line) < 5:
                    continue
                model_name, batch_size = line.split(" ")
                if model_name not in models:
                    continue
                container.TIMM_MODELS[model_name] = int(batch_size)

    @classmethod
    def _get_module_cls_by_model_name(container, model_cls_name):
        _module_by_model_name = {}
        module_name = _module_by_model_name.get(model_cls_name, "torchbenchmark")
        module = importlib.import_module(module_name)
        return getattr(module, model_cls_name)

    @classmethod
    def _get_sequence_length(container, model_cls, model_name):
        if model_name.startswith(("Blenderbot",)):
            seq_length = 128
        elif model_name.startswith(("GPT2", "Bart", "T5", "PLBart", "MBart")):
            seq_length = 1024
        elif model_name in {"AllenaiLongformerBase", "BigBird", "Phi2"}:
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
        warmup: int = 10,
        repeat: int = 30,
        fake_tensor: bool = False,
        no_grad: bool = True,
        target_opset: int = 18,
        dtype: Optional[Any] = None,
        nvtx: bool = False,
        dump_ort: bool = False,
    ):
        super().__init__(
            "torchbench",
            device=device,
            partition_id=partition_id,
            total_partitions=total_partitions,
            include_model_names=include_model_names,
            exclude_model_names=exclude_model_names,
            verbose=verbose,
            target_opset=target_opset,
            warmup=warmup,
            repeat=repeat,
            fake_tensor=fake_tensor,
            no_grad=no_grad,
            dtype=dtype,
            dump_ort=dump_ort,
            nvtx=nvtx,
        )
        if not hasattr(self, "_config") or not self._config:
            self.initialize()
        assert self._config

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

    @classmethod
    def _gen_target(self, batch_size, device, num_classes):
        return torch.empty((batch_size,) + (), device=device, dtype=torch.long).random_(
            num_classes
        )

    @download_retry_decorator(retry=5)
    def _download_model(self, model_name):
        from timm.models import create_model

        model = create_model(
            model_name,
            in_chans=3,
            scriptable=False,
            num_classes=None,
            drop_rate=0.0,
            drop_path_rate=None,
            drop_block_rate=None,
            pretrained=True,
        )
        if hasattr(model, "config"):
            model.config.to_tuple = False
        else:
            model.config = MakeConfig(to_tuple=False)
        return model

    def load_model(
        self,
        model_name: str,
        batch_size: Optional[int] = None,
    ):
        if self.enable_activation_checkpointing:
            raise NotImplementedError(
                "Activation checkpointing not implemented for Timm models"
            )

        is_training = self.training
        use_eval_mode = self.use_eval_mode
        channels_last = False

        # channels_last = self.channels_last
        model = self._download_model(model_name)

        if model is None:
            raise RuntimeError(f"Failed to load model '{model_name}'")
        model.to(
            device=self.device,
            # memory_format=torch.channels_last if channels_last else None,
        )

        self.num_classes = model.num_classes

        from timm.data import resolve_data_config

        # see https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/config.py
        data_config = resolve_data_config(
            args=None,
            model=model,
            use_test_size=not is_training,
            verbose=self.verbose,
        )
        input_size = data_config["input_size"]
        recorded_batch_size = self.TIMM_MODELS[model_name]

        if model_name in self.BATCH_SIZE_DIVISORS:
            recorded_batch_size = max(
                int(recorded_batch_size / self.BATCH_SIZE_DIVISORS[model_name]), 1
            )
        batch_size = batch_size or recorded_batch_size

        torch.manual_seed(1337)
        input_tensor = torch.randint(
            256, size=(batch_size,) + input_size, device=self.device
        ).to(dtype=torch.float32)
        mean = torch.mean(input_tensor)
        std_dev = torch.std(input_tensor)
        example_inputs = (input_tensor - mean) / std_dev

        if channels_last:
            example_inputs = example_inputs.contiguous(
                memory_format=torch.channels_last
            )
        example_inputs = [
            example_inputs,
        ]
        self.target = self._gen_target(batch_size, self.device, model.num_classes)

        self.loss = torch.nn.CrossEntropyLoss().to(self.device)

        if model_name in self.SCALED_COMPUTE_LOSS:
            self.compute_loss = self.scaled_compute_loss

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
            suite="Timm",
        )

    def iter_model_names(self):
        # for model_name in list_models(pretrained=True, exclude_filters=["*in21k"]):
        model_names = sorted(self.TIMM_MODELS.keys())
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
