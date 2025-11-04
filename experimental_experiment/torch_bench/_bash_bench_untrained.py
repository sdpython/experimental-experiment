from typing import Any, Callable, Dict, Optional, Set, Tuple
from torch._dynamo.testing import reset_rng_state
from ._bash_bench_benchmark_runner import BenchmarkRunner
from ._bash_bench_model_runner import ModelRunner
from ..helpers import string_type
from ..torch_interpreter import DEFAULT_TARGET_OPSET


def get_untrained_model_inputs(model_id: str, attn_implementation: str = ""):
    from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs

    model_kwargs = (
        None
        if attn_implementation not in ("", None, "eager")
        else dict(attn_implementation=attn_implementation)
    )
    if model_id.endswith("//pretrained"):
        use_pretrained = True
        model_id = model_id.replace("//pretrained", "")
    else:
        use_pretrained = False
        model_id = model_id
    res = get_untrained_model_with_inputs(
        model_id,
        add_second_input=True,
        model_kwargs=model_kwargs,
        use_pretrained=use_pretrained,
        same_as_pretrained=use_pretrained,
    )
    assert "inputs2" in res, "Second set of inputs is missing."
    return res, dict(strict=False)


class UntrainedRunner(BenchmarkRunner):
    SUITE = "Untrained"
    MODELS: Dict[str, Callable] = {}
    MODEL_IDS = [
        "arnir0/Tiny-LLM",
        "codellama/CodeLlama-7b-hf",
        "emilyalsentzer/Bio_ClinicalBERT//pretrained",
        "facebook/bart-base",
        "google-bert/bert-base-multilingual-cased",
        "google/gemma-2b",
        "hf-internal-testing/tiny-random-BeitForImageClassification",
        "hf-tiny-model-private/tiny-random-PLBartForConditionalGeneration",
        "HuggingFaceM4/tiny-random-idefics",
        "HuggingFaceM4/tiny-random-idefics",
        "Intel/bert-base-uncased-mrpc",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-3.2-1B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3.5-MoE-instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "microsoft/Phi-4-mini-reasoning",
        "microsoft/Phi-4-reasoning",
        "microsoft/beit-base-patch16-224-pt22k-ft22k",
        "microsoft/phi-2",
        "nateraw/vit-age-classifier",
        "openai/clip-vit-base-patch16",
        "openai/whisper-tiny",
        "sentence-transformers/all-MiniLM-L6-v1",
        "sshleifer/tiny-marian-en-de",
        "sshleifer/tiny-marian-en-de",
        "tiiuae/falcon-mamba-tiny-dev",
    ]

    def initialize(self):
        """Steps to run before running the benchmark."""
        model_ids = {
            k: (
                lambda _model_id_=k: get_untrained_model_inputs(
                    _model_id_, attn_implementation=self.attn_impl
                )
            )
            for k in self.MODEL_IDS
        }
        self.MODELS.update({**model_ids})

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
            "untrained",
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
            nvtx=nvtx,
            dump_ort=dump_ort,
            attn_impl=attn_impl,
        )
        self.initialize()

    def _get_model_cls_and_config(self, model_name: str) -> Tuple[Callable, Any]:
        assert (
            model_name in self.MODELS
        ), f"Unable to find {model_name!r} in {sorted(self.MODELS)}"
        return self.MODELS[model_name]

    def load_model(
        self,
        model_name: str,
        batch_size: Optional[int] = None,
    ) -> ModelRunner:
        is_training = self.training
        use_eval_mode = self.use_eval_mode
        reset_rng_state()
        tu = self._get_model_cls_and_config(model_name)
        tu = tu()
        task = None

        dynamic_shapes = None
        inputs2 = None
        if isinstance(tu, dict):
            model_cls, example_inputs = tu["model"], tu["inputs"]
            dynamic_shapes = tu.get("dynamic_shapes", None)
            task = tu.get("task", "")
            inputs2 = tu.get("inputs2", None)
            export_options = None
            config = tu.get("configuration", None)
        elif len(tu) == 2:
            if isinstance(tu[0], dict):
                model_cls, example_inputs = tu[0]["model"], tu[0]["inputs"]
                dynamic_shapes = tu[0].get("dynamic_shapes", None)
                inputs2 = tu[0].get("inputs2", None)
                config = tu[0].get("configuration", None)
                task = tu[0].get("task", "")
                export_options = tu[1]
            else:
                model_cls, example_inputs = tu
                export_options = None
                config = None
        elif len(tu) == 3:
            model_cls, example_inputs, export_options = tu
            config = None
        else:
            raise AssertionError(
                f"Unable to handle {len(tu)} elements: {string_type(tu)}, "
                f"it can be (dict, dict), (model, inputs), (model, inputs, dict))"
            )

        model = model_cls() if isinstance(model_cls, type) else model_cls
        if str(type(model)) == "<class 'function'>":
            model = model()

        if is_training and not use_eval_mode:
            model.train()
        else:
            model.eval()

        if export_options and "patch_transformers" in export_options:
            patch_options = dict(patch_transformers=export_options.pop("patch_transformers"))
        else:
            patch_options = None

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
            wrap_kind="nowrap",
            model_name=model_name,
            export_options=export_options,
            patch_options=patch_options,
            dynamic_shapes=dynamic_shapes,
            inputs2=inputs2,
            task=task,
            attn_impl=self.attn_impl,
            config=config,
        )

    def iter_model_names(self):
        model_names = list(self.MODELS)
        assert model_names, "Empty list of models"
        model_names.sort()

        start, end = self.get_benchmark_indices(len(model_names))
        yield from self.enumerate_model_names(model_names, start=start, end=end)
