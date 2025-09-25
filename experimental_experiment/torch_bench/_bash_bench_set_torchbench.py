import gc
import importlib
import inspect
import io
import os
import pprint
import textwrap
import warnings
from typing import Any, NamedTuple, Optional, Set, Tuple

import torch
from torch._dynamo.testing import reset_rng_state
from ..torch_interpreter import DEFAULT_TARGET_OPSET
from ._bash_bench_benchmark_runner import BenchmarkRunner
from ._bash_bench_model_runner import (
    MakeConfig,
    ModelRunner,
    _rand_int_tensor,
    download_retry_decorator,
)


class TorchBenchRunner(BenchmarkRunner):
    SUITE = "TorchBench"

    YAML = textwrap.dedent(
        """
        # Some models have large dataset that doesn't fit in memory. Lower the batch
        # size to test the accuracy.
        batch_size:
            training:
                demucs: 4
                dlrm: 1024
                densenet121: 4
                hf_Reformer: 4
                hf_T5_base: 4
                timm_efficientdet: 1
                llama_v2_7b_16h: 1
                # reduced from 16 due to cudagraphs OOM in TorchInductor dashboard
                yolov3: 8

            inference:
                timm_efficientdet: 32

        dont_change_batch_size:
            - demucs
            - pytorch_struct
            - pyhpc_turbulent_kinetic_energy
            # https://github.com/pytorch/benchmark/pull/1656
            - vision_maskrcnn

        tolerance:
        # Need lower tolerance on GPU. GPU kernels
        # have non deterministic kernels for these models.
        higher:
            - alexnet
            - attention_is_all_you_need_pytorch
            - densenet121
            - hf_Albert
            - vgg16
            - mobilenet_v3_large
            - nvidia_deeprecommender
            - timm_efficientdet

        # These models need >1e-3 tolerance
        even_higher:
            - soft_actor_critic
            - tacotron2
            - yolov3
            - timm_efficientdet
            - squeezenet1_1

        higher_fp16:
            - doctr_reco_predictor
            - drq
            - hf_Whisper

        higher_bf16:
            - doctr_reco_predictor
            - drq
            - hf_Whisper

        cosine: []

        require_larger_multiplier_for_smaller_tensor:
            - yolov3

        # These benchmarks took >600s on an i9-11900K CPU
        very_slow: &VERY_SLOW_MODELS
            # 3339s
            - hf_BigBird
            # 3062s
            - hf_Longformer
            # 930s
            - hf_T5

        # These benchmarks took >60s on an i9-11900K CPU
        slow:
            - *VERY_SLOW_MODELS
            # 137s
            - BERT_pytorch
            # 116s
            - demucs
            # 242s
            - fastNLP_Bert
            # 221s
            - hf_Albert
            # 400s
            - hf_Bart
            # 334s
            - hf_Bert
            # 187s
            - hf_DistilBert
            # 470s
            - hf_GPT2
            # 141s
            - hf_Reformer
            # 317s
            - speech_transformer
            # 99s
            - vision_maskrcnn

        non_deterministic:
            # https://github.com/pytorch/pytorch/issues/98355
            - mobilenet_v3_large
            - sam_fast

        dtype:
            force_amp_for_fp16_bf16_models:
                - DALLE2_pytorch
                - doctr_det_predictor
                - doctr_reco_predictor
                - Super_SloMo
                - tts_angular
                - pyhpc_turbulent_kinetic_energy
                - detectron2_fcos_r_50_fpn

        force_fp16_for_bf16_models:
            - vision_maskrcnn

        # models in canary_models that we should run anyway
        canary_models:
            - DALLE2_pytorch
            - torchrec_dlrm
            # ado
            - codellama
            - hf_mixtral
            - hf_Yi
            - mistral_7b_instruct
            - orca_2
            - phi_1_5
            - phi_2
            - stable_diffusion_xl
            - llama_v31_8b

        detectron2_models: &DETECTRON2_MODELS
            - detectron2_fasterrcnn_r_101_c4
            - detectron2_fasterrcnn_r_101_dc5
            - detectron2_fasterrcnn_r_101_fpn
            - detectron2_fasterrcnn_r_50_c4
            - detectron2_fasterrcnn_r_50_dc5
            - detectron2_fasterrcnn_r_50_fpn
            - detectron2_maskrcnn_r_101_c4
            - detectron2_maskrcnn_r_101_fpn
            - detectron2_maskrcnn_r_50_fpn

        # These models support only train mode. So accuracy checking can't be done in
        # eval mode.
        only_training:
            - *DETECTRON2_MODELS
            - tts_angular
            - tacotron2
            - demucs
            - hf_Reformer
            - pytorch_struct
            - yolov3

        trt_not_yet_working:
            - alexnet
            - resnet18
            - resnet50
            - mobilenet_v2
            - mnasnet1_0
            - squeezenet1_1
            - shufflenetv2_x1_0
            - vgg16
            - resnext50_32x4d

        skip:
            all:
                # OOMs (A100 40G)
                - detectron2_maskrcnn
                # TIMEOUT
                - tacotron2
                # Failing in eager mode
                - hf_clip
                # multi gpu not always available in benchmark runners
                - simple_gpt_tp_manual

        device:
            cpu:
                # OOMs
                - hf_T5_generate
                # model is CUDA only
                - cm3leon_generate
                # timeout
                - nanogpt
                # timeout
                - sam
                # model is CUDA only
                - sam_fast
                # model is CUDA only
                - llama_v2_7b_16h
                # flaky
                - stable_diffusion
                # requires FBGEMM, CUDA only
                - torchrec_dlrm
                - simple_gpt
                # works on cuda, accuracy failure on cpu
                - hf_Whisper
                - stable_diffusion_text_encoder
                - llava

            cuda: []

        test:
            training:
                - *DETECTRON2_MODELS
                # not designed for training
                - pyhpc_equation_of_state
                - pyhpc_isoneutral_mixing
                - pyhpc_turbulent_kinetic_energy
                - maml
                - llama
                - llama_v2_7b_16h
                - simple_gpt
                - sam_fast
                # Model's DEFAULT_TRAIN_BSIZE is not implemented
                - cm3leon_generate
                - hf_T5_generate
                - doctr_det_predictor
                - doctr_reco_predictor
                - moondream
                # doesnt fit in memory
                - phi_1_5
                - detectron2_fcos_r_50_fpn

        control_flow:
            - cm3leon_generate
            - detectron2_fcos_r_50_fpn
            - fastNLP_Bert
            - hf_Longformer
            - hf_Reformer
            - hf_T5_generate
            - opacus_cifar10
            - speech_transformer

        # Models that should only run in --multiprocess mode
        multiprocess:
            - simple_gpt

        # for these models, conv-batchnorm fusing causes big numerical churn.
        # Skip them
        freezing:
            - mnasnet1_0
            - moco
            - shufflenet_v2_x1_0

        accuracy:
            skip:
                large_models:
                # Models too large to have eager, dynamo and fp64_numbers simultaneosuly
                # even for 40 GB machine. We have tested accuracy for smaller version of
                # these models
                - hf_GPT2_large
                - hf_T5_large
                - timm_vision_transformer_large
                # accuracy https://github.com/pytorch/pytorch/issues/93847
                - maml
                - llama_v2_7b_16h
                - Background_Matting
                - stable_diffusion_unet
                eager_not_deterministic:
                # Models that deterministic algorithms can not be turned on for eager mode.
                - Background_Matting
                - pytorch_unet

        max_batch_size:
            hf_GPT2: 2
            pytorch_unet: 2
        """
    )

    MODELS_FILENAME = textwrap.dedent(
        """
        BERT_pytorch,128
        Background_Matting,1
        LearningToPaint,1024
        alexnet,1024
        dcgan,1024
        densenet121,64
        hf_Albert,32
        hf_Bart,16
        hf_Bert,16
        hf_GPT2,16
        hf_T5,4
        mnasnet1_0,256
        mobilenet_v2,128
        mobilenet_v3_large,256
        nvidia_deeprecommender,1024
        pytorch_unet,8
        resnet18,512
        resnet50,128
        resnext50_32x4d,128
        shufflenet_v2_x1_0,512
        squeezenet1_1,512
        timm_nfnet,256
        timm_efficientnet,128
        timm_regnet,128
        timm_resnest,256
        timm_vision_transformer,256
        timm_vovnet,128
        vgg16,128
        """
    )

    EXPECTED_MODELS = textwrap.dedent(
        """
        BERT_pytorch
        Background_Matting
        DALLE2_pytorch
        LearningToPaint
        Super_SloMo
        alexnet
        basic_gnn_edgecnn
        basic_gnn_gcn
        basic_gnn_gin
        basic_gnn_sage
        cm3leon_generate
        dcgan
        demucs
        densenet121
        detectron2_fasterrcnn_r_101_c4
        detectron2_fasterrcnn_r_101_dc5
        detectron2_fasterrcnn_r_101_fpn
        detectron2_fasterrcnn_r_50_c4
        detectron2_fasterrcnn_r_50_dc5
        detectron2_fasterrcnn_r_50_fpn
        detectron2_fcos_r_50_fpn
        detectron2_maskrcnn_r_101_c4
        detectron2_maskrcnn_r_101_fpn
        detectron2_maskrcnn_r_50_c4
        detectron2_maskrcnn_r_50_fpn
        dlrm
        doctr_det_predictor
        doctr_reco_predictor
        drq
        fastNLP_Bert
        functorch_dp_cifar10
        functorch_maml_omniglot
        hf_Albert
        hf_Bart
        hf_Bert
        hf_Bert_large
        hf_BigBird
        hf_DistilBert
        hf_GPT2
        hf_GPT2_large
        hf_Longformer
        hf_Reformer
        hf_T5
        hf_T5_base
        hf_T5_generate
        hf_T5_large
        hf_Whisper
        lennard_jones
        llama
        maml
        maml_omniglot
        mnasnet1_0
        mobilenet_v2
        mobilenet_v2_quantized_qat
        mobilenet_v3_large
        moco
        nanogpt
        nvidia_deeprecommender
        opacus_cifar10
        phlippe_densenet
        phlippe_resnet
        pyhpc_equation_of_state
        pyhpc_isoneutral_mixing
        pyhpc_turbulent_kinetic_energy
        pytorch_stargan
        pytorch_unet
        resnet152
        resnet18
        resnet50
        resnet50_quantized_qat
        resnext50_32x4d
        sam
        sam_fast
        shufflenet_v2_x1_0
        soft_actor_critic
        speech_transformer
        squeezenet1_1
        timm_efficientnet
        timm_nfnet
        timm_regnet
        timm_resnest
        timm_vision_transformer
        timm_vision_transformer_large
        timm_vovnet
        tts_angular
        vgg16
        vision_maskrcnn
        yolov3
        """
    )

    BATCH_SIZE_KNOWN_MODELS = dict()

    BATCH_SIZE_DIVISORS = {}

    EXTRA_MODELS = {}

    _config = None

    @classmethod
    def load_yaml_file(cls):
        import yaml

        with io.StringIO(cls.YAML) as f:
            data = yaml.safe_load(f)

        def flatten(lst):
            for item in lst:
                if isinstance(item, list):
                    yield from flatten(item)
                else:
                    yield item

        def maybe_list_to_set(obj):
            if isinstance(obj, dict):
                return {k: maybe_list_to_set(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return set(flatten(obj))
            return obj

        config = maybe_list_to_set(data)
        return config

    def initialize(self):
        """Steps to run before running the benchmark."""
        try:
            import torch

            _ = torch.ops.fbgemm.asynchronous_complete_cumsum
        except (AttributeError, ImportError) as e:
            warnings.warn(f"Something wrong in the installation because of {e}.", stacklevel=2)
        self._config = self.load_yaml_file()
        assert "batch_size" in self._config, f"config wrong {self._config}"
        assert self._config["batch_size"] is not None, f"config wrong {self._config}"
        assert "inference" in self._config["batch_size"], f"config wrong {self._config}"
        lines = self.MODELS_FILENAME.split("\n")
        lines = [line.rstrip() for line in lines]
        expected_models = {_.strip() for _ in self.EXPECTED_MODELS.split("\n") if _}
        for line in lines:
            if not line or len(line) < 2:
                continue
            model_name, batch_size = line.split(",")
            if model_name not in expected_models:
                continue
            batch_size = int(batch_size)
            if "batch_size" not in self._config or self._config["batch_size"] is None:
                self._config["batch_size"] = {}
            if (
                "inference" not in self._config["batch_size"]
                or self._config["batch_size"]["inference"] is None
            ):
                self._config["batch_size"]["inference"] = {}
            assert "inference" in self._config["batch_size"]
            self._config["batch_size"]["inference"][model_name] = batch_size
        for o in expected_models:
            model_name = o.strip()
            if len(model_name) < 3:
                continue
            if model_name not in self._config["batch_size"]["inference"]:
                self._config["batch_size"]["inference"][model_name] = 1

    @classmethod
    def _get_module_cls_by_model_name(cls, model_cls_name):
        _module_by_model_name = {}
        module_name = _module_by_model_name.get(model_cls_name, "torchbenchmark")
        module = importlib.import_module(module_name)
        return getattr(module, model_cls_name)

    @classmethod
    def _get_sequence_length(cls, model_cls, model_name):
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
        cls, model_cls, model, model_name, bs, device, include_loss_args=False
    ):
        if hasattr(model, "_get_random_inputs"):
            return model._get_random_inputs(device)

        import transformers

        num_choices = 3
        num_visual_features = 42
        seq_length = cls._get_sequence_length(model_cls, model_name)
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

        if model_name.startswith(("T5", "M2M100", "MT5")) or model_cls in {
            transformers.BlenderbotModel,
            transformers.BlenderbotSmallModel,
            transformers.BlenderbotForConditionalGeneration,
            transformers.BlenderbotSmallForConditionalGeneration,
            transformers.PegasusModel,
            transformers.PegasusForConditionalGeneration,
            transformers.MarianModel,
            transformers.MarianMTModel,
        }:
            input_dict["decoder_input_ids"] = inputt

        if model_name.startswith("Lxmert"):
            visual_feat_dim, visual_pos_dim = (
                model.config.visual_feat_dim,
                model.config.visual_pos_dim,
            )
            input_dict["visual_feats"] = torch.randn(bs, num_visual_features, visual_feat_dim)
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
                input_dict["start_positions"] = _rand_int_tensor(device, 0, seq_length, (bs,))
                input_dict["end_positions"] = _rand_int_tensor(device, 0, seq_length, (bs,))
            elif model_name.endswith(("MaskedLM", "HeadModel", "CausalLM", "DoubleHeadsModel")):
                input_dict["labels"] = _rand_int_tensor(device, 0, vocab_size, (bs, seq_length))
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
            elif model_name in cls.EXTRA_MODELS:
                input_dict["labels"] = _rand_int_tensor(device, 0, vocab_size, (bs, seq_length))
            else:
                raise NotImplementedError(f"Class {model_name!r} unsupported for training test ")

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
        target_opset: int = DEFAULT_TARGET_OPSET,
        dtype: Optional[Any] = None,
        nvtx: bool = False,
        dump_ort: bool = False,
        attn_impl: str = "eager",
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
            attn_impl=attn_impl,
        )
        if not self._config:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.initialize()
        assert self._config
        assert "inference" in self._config["batch_size"]
        assert "inference" in self._batch_size

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
    def _reassign_parameters(cls, model):
        # torch_geometric models register parameter as tensors due to
        # https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/dense/linear.py#L158-L168
        # Since it is unusual thing to do, we just reassign them to parameters
        def state_dict_hook(module, destination, prefix, local_metadata):
            for name, _param in module.named_parameters():
                if (
                    name in destination
                    and isinstance(destination[name], torch.Tensor)
                    and not isinstance(destination[name], torch.nn.Parameter)
                ):
                    destination[name] = torch.nn.Parameter(destination[name])

        model._register_state_dict_hook(state_dict_hook)

    @download_retry_decorator(retry=5)
    def _download_model(self, model_name):
        model_cls, config = self._get_model_cls_and_config(model_name)
        if "auto" in model_cls.__module__:
            # Handle auto classes
            model = model_cls.from_config(config)
        else:
            model = model_cls(config)
        if hasattr(model, "config"):
            model.config.to_tuple = False
        else:
            model.config = MakeConfig(to_tuple=False)
        return model

    @property
    def _batch_size(self):
        return self._config["batch_size"]

    def load_model(
        self,
        model_name: str,
        batch_size: Optional[int] = None,
    ) -> ModelRunner:
        import torchbenchmark

        torchbenchmark.TORCH_DEPS[:] = ["numpy", "torch", "torchaudio"]

        if self.verbose:
            print(
                f"[{self.__class__.__name__}.load_model] setup {model_name!r} "
                f"with batch_size={batch_size} set TORCHBENCHSETUP=0 to skip it"
            )

        if os.environ.get("TORCHBENCHSETUP", "1") in ("1", "true", "True", 1, True):
            status = torchbenchmark.setup(
                models=[model_name],
                verbose=self.verbose,
                continue_on_fail=False,
                allow_canary=False,  # TODO: should we allow canary models?
            )
            assert status, f"Could not setup model {model_name!r}, status={status!r}"

        is_training = self.training
        use_eval_mode = self.use_eval_mode
        dtype = self.dtype
        reset_rng_state()

        candidates = [
            f"torchbenchmark.models.{model_name}",
            f"torchbenchmark.canary_models.{model_name}",
            f"torchbenchmark.models.fb.{model_name}",
        ]
        for c in candidates:
            try:
                module = importlib.import_module(c)
                break
            except ModuleNotFoundError as e:
                if e.name != c:
                    raise ModuleNotFoundError(
                        f"Unable to find {c!r}, model_name={model_name!r}, "
                        f"e.name={e.name!r}, candidates={candidates}"
                    ) from e
        else:
            raise ImportError(f"could not import any of {candidates}")
        benchmark_cls = getattr(module, "Model", None)
        if benchmark_cls is None:
            raise NotImplementedError(f"{model_name}.Model is None")

        if not hasattr(benchmark_cls, "name"):
            benchmark_cls.name = model_name

        cant_change_batch_size = (
            not getattr(benchmark_cls, "ALLOW_CUSTOMIZE_BSIZE", True)
            or model_name in self._config["dont_change_batch_size"]
        )
        if cant_change_batch_size:
            batch_size = None
        if batch_size is None and is_training and model_name in self._batch_size["training"]:
            batch_size = self._batch_size["training"][model_name]
        elif (
            batch_size is None and not is_training and model_name in self._batch_size["inference"]
        ):
            batch_size = self._batch_size["inference"][model_name]

        # workaround "RuntimeError: not allowed to set torch.backends.cudnn flags"
        torch.backends.__allow_nonbracketed_mutation_flag = True
        if self.verbose:
            print(f"[{self.__class__.__name__}.load_model] start loading")

        if model_name == "vision_maskrcnn" and is_training:
            # Output of vision_maskrcnn model is a list of bounding boxes,
            # sorted on the basis of their scores. This makes accuracy
            # comparison hard with torch.compile. torch.compile can cause minor
            # divergences in the output because of how fusion works for amp in
            # TorchInductor compared to eager.  Therefore, instead of looking at
            # all the bounding boxes, we compare only top 4.
            model_kwargs = {"box_detections_per_img": 4}
            benchmark = benchmark_cls(
                test="train",
                device=self.device,
                batch_size=batch_size,
                model_kwargs=model_kwargs,
            )
            use_eval_mode = True
        elif is_training:
            benchmark = benchmark_cls(
                test="train",
                device=self.device,
                batch_size=batch_size,
            )
        else:
            try:
                benchmark = benchmark_cls(
                    test="eval",
                    device=self.device,
                    batch_size=batch_size,
                )
            except Exception as e:
                raise AssertionError(
                    f"Unable to create class {benchmark_cls}, "
                    f"device={self.device}, batch_size={batch_size}, "
                    f"signature={inspect.signature(benchmark_cls).parameters}, "
                    f"DEFAULT_EVAL_BSIZE="
                    f"{getattr(benchmark_cls, 'DEFAULT_EVAL_BSIZE', '?')}, "
                    f"ALLOW_CUSTOMIZE_BSIZE="
                    f"{getattr(benchmark_cls, 'ALLOW_CUSTOMIZE_BSIZE', '?')} "
                    f"--- exception={e}"
                ) from e
        if self.verbose:
            print(f"[{self.__class__.__name__}.load_model] clsname={benchmark}")
        model, example_inputs = benchmark.get_module()
        if self.verbose:
            print(f"[{self.__class__.__name__}.load_model] clsname={benchmark}, done loading")
        if model_name in [
            "basic_gnn_edgecnn",
            "basic_gnn_gcn",
            "basic_gnn_sage",
            "basic_gnn_gin",
        ]:
            self._reassign_parameters(model)

        if dtype is None:
            model = model.to(self.device)
        else:
            model = model.to(self.device, dtype=dtype)

        if self.enable_activation_checkpointing:
            model.gradient_checkpointing_enable()

        # Models that must be in train mode while training
        if is_training and (not use_eval_mode or model_name in self._config["only_training"]):
            model.train()
        else:
            model.eval()

        gc.collect()
        batch_size = benchmark.batch_size
        if model_name == "torchrec_dlrm":
            batch_namedtuple = NamedTuple("Batch", "dense_features sparse_features labels")
            example_inputs = tuple(
                batch_namedtuple(
                    dense_features=batch.dense_features,
                    sparse_features=batch.sparse_features,
                    labels=batch.labels,
                )
                for batch in example_inputs
            )
        # Torchbench has quite different setup for yolov3, so directly passing
        # the right example_inputs
        if model_name == "yolov3":
            example_inputs = (torch.rand(batch_size, 3, 384, 512).to(self.device),)
        # See https://github.com/pytorch/benchmark/issues/1561
        if model_name == "maml_omniglot":
            batch_size = 5
            assert example_inputs[0].shape[0] == batch_size
        if model_name == "vision_maskrcnn":
            batch_size = 1

        if hasattr(model, "config"):
            model.config.to_tuple = False
        else:
            model.config = MakeConfig(to_tuple=False)

        return ModelRunner(
            model,
            example_inputs,
            None,  # kwargs
            device=self.device,
            dtype=self.dtype,
            warmup=self.warmup,
            repeat=self.repeat,
            suite=self.SUITE,
            autocast=self.autocast,
            model_name=model_name,
            attn_impl=self.attn_impl,
        )

    def iter_model_names(self):
        from torchbenchmark import _list_canary_model_paths, _list_model_paths

        expected_models = {_.strip() for _ in self.EXPECTED_MODELS.split("\n") if _}

        models = _list_model_paths()
        models += [
            f
            for f in _list_canary_model_paths()
            if os.path.basename(f) in self._config["canary_models"]
        ]
        model_names = [m for m in models if os.path.basename(m) in expected_models]
        assert len(model_names) >= len(expected_models), (
            f"Unexpected names {len(model_names)} < {len(expected_models)} (expected)"
            f"\n--missing=\n{pprint.pformat(sorted(set(expected_models)-{os.path.basename(m) for m in model_names}))}"  # noqa: E501
            f"\n--canary_models=\n{pprint.pformat(self._config['canary_models'])}"
            f"\n--_list_canary_model_paths()=\n{pprint.pformat(_list_canary_model_paths())}"
            f"\n--_list_model_paths()=\n{pprint.pformat(_list_model_paths())}"
        )
        model_names = [os.path.basename(m) for m in model_names]
        model_names.sort()

        start, end = self.get_benchmark_indices(len(model_names))
        yield from self.enumerate_model_names(model_names, start=start, end=end)
