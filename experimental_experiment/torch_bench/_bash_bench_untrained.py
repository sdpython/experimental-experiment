from typing import Any, Callable, Dict, Optional, Set, Tuple
from torch._dynamo.testing import reset_rng_state
from ._bash_bench_benchmark_runner import BenchmarkRunner
from ._bash_bench_model_runner import ModelRunner
from ._bash_bench_models_helper import get_llama_model_layer
from ..helpers import string_type
from ..torch_models.diffusion_model_helper import (
    get_stable_diffusion_2_unet,
)
from ..torch_models.llm_model_helper import (
    LLMInputKind,
    get_ai21_jamba_15_mini,
    get_all_mini_ml_l6_v1,
    get_falcon_mamba_7b,
    get_llama32_9b_vision,
    get_phi2,
    get_phi35_mini_instruct,
    get_phi35_vision_instruct,
    get_phi4,
    get_smollm_1_7b,
    get_tiny_llm,
)


class UntrainedRunner(BenchmarkRunner):
    SUITE = "Untrained"
    MODELS: Dict[str, Callable] = {}

    @classmethod
    def initialize(cls):
        """Steps to run before running the benchmark."""
        cls.MODELS.update(
            {
                # diffusion
                "StableDiffusion2Unet": get_stable_diffusion_2_unet,
                # LLM simple
                "Llama2Layer": (lambda: get_llama_model_layer(num_hidden_layers=2)),
                # LLM with Cache
                "AI21Jamba15MiniLM_1LayerNoCache": (
                    lambda: (
                        get_ai21_jamba_15_mini(
                            num_hidden_layers=1,
                            input_cache=False,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        dict(strict=True, replace_dynamic_cache=False),
                    )
                ),
                "AI21Jamba15MiniLM_1Layer": (
                    lambda: (
                        get_ai21_jamba_15_mini(
                            num_hidden_layers=1,
                            input_cache=True,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        dict(replace_dynamic_cache=True, strict=False),
                    )
                ),
                "AI21Jamba15MiniLM_2Layer": (
                    lambda: (
                        get_ai21_jamba_15_mini(
                            num_hidden_layers=2,
                            input_cache=True,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        dict(replace_dynamic_cache=True, strict=False),
                    )
                ),
                "AllMiniLML6v1_1Layer": (
                    lambda: (
                        get_all_mini_ml_l6_v1(
                            num_hidden_layers=1,
                            input_cache=True,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        dict(strict=False, replace_dynamic_cache=False),
                    )
                ),
                "AllMiniLML6v1_1LayerNoCache": (
                    lambda: (
                        get_all_mini_ml_l6_v1(
                            num_hidden_layers=1,
                            input_cache=False,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        dict(strict=False, replace_dynamic_cache=False),
                    )
                ),
                "AllMiniLML6v1_2Layer": (
                    lambda: (
                        get_all_mini_ml_l6_v1(
                            num_hidden_layers=2,
                            input_cache=True,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        dict(strict=False, replace_dynamic_cache=False),
                    )
                ),
                "AllMiniLML6v1_2LayerNoCache": (
                    lambda: (
                        get_all_mini_ml_l6_v1(
                            num_hidden_layers=2,
                            input_cache=False,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        dict(strict=False, replace_dynamic_cache=False),
                    )
                ),
                "FalconMamba7bLM_1LayerNoCache": (
                    lambda: (
                        get_falcon_mamba_7b(
                            num_hidden_layers=1,
                            input_cache=False,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        {},
                    )
                ),
                "FalconMamba7bLM_2LayerNoCache": (
                    lambda: (
                        get_falcon_mamba_7b(
                            num_hidden_layers=1,
                            input_cache=False,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        {},
                    )
                ),
                "FalconMamba7bLM_1Layer": (
                    lambda: (
                        get_falcon_mamba_7b(
                            num_hidden_layers=1,
                            input_cache=True,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        {},
                    )
                ),
                "FalconMamba7bLM_2Layer": (
                    lambda: (
                        get_falcon_mamba_7b(
                            num_hidden_layers=1,
                            input_cache=True,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        {},
                    )
                ),
                "Llama_9b_vision_8Layer": (lambda: get_llama32_9b_vision(num_hidden_layers=8)),
                "Phi2LM_1LayerNoCache": (
                    lambda: (
                        get_phi2(
                            num_hidden_layers=1,
                            input_cache=False,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                        ),
                        dict(strict=False, replace_dynamic_cache=True),
                    )
                ),
                "Phi2LM_1Layer": (
                    lambda: (
                        get_phi2(
                            num_hidden_layers=1,
                            input_cache=True,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        dict(strict=False, replace_dynamic_cache=True),
                    )
                ),
                "Phi2LM_2LayerNoCache": (
                    lambda: (
                        get_phi2(
                            num_hidden_layers=2,
                            input_cache=False,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                        ),
                        dict(strict=False, replace_dynamic_cache=True),
                    )
                ),
                "Phi2LM_2Layer": (
                    lambda: (
                        get_phi2(
                            num_hidden_layers=2,
                            input_cache=True,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        dict(strict=False, replace_dynamic_cache=True),
                    )
                ),
                "Phi35MiniInstructLM_2Layer": (
                    lambda: (
                        get_phi35_mini_instruct(
                            num_hidden_layers=2,
                            input_cache=True,
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        dict(strict=False, replace_dynamic_cache=True),
                    )
                ),
                "Phi35MiniInstructLMVision_2Layer": (
                    lambda: (
                        get_phi35_vision_instruct(
                            num_hidden_layers=2,
                            input_cache=True,
                            input_kind=LLMInputKind.input_ids
                            | LLMInputKind.attention_mask
                            | LLMInputKind.past_key_values,
                            common_dynamic_shapes=True,
                        ),
                        dict(strict=False, replace_dynamic_cache=True),
                    )
                ),
                "Phi35MiniInstructLMVision_1Layer_Images": (
                    lambda: (
                        get_phi35_vision_instruct(
                            num_hidden_layers=1,
                            input_cache=True,
                            _attn_implementation="eager",
                            input_kind=LLMInputKind.ALL,
                            common_dynamic_shapes=True,
                        ),
                        dict(strict=False, replace_dynamic_cache=True),
                    )
                ),
                "Phi4LM_2LayerNoCache": (
                    lambda: (
                        get_phi4(
                            num_hidden_layers=2,
                            input_cache=False,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        dict(strict=False, replace_dynamic_cache=True),
                    )
                ),
                "Phi4LM_2Layer": (
                    lambda: (
                        get_phi4(
                            num_hidden_layers=2,
                            input_cache=True,
                            _attn_implementation="eager",
                            common_dynamic_shapes=True,
                            batch_size=2,
                        ),
                        dict(strict=False, replace_dynamic_cache=True),
                    )
                ),
                "SmolLM17b_2LayerNoCache": (
                    lambda: (
                        get_smollm_1_7b(
                            input_cache=False,
                            num_hidden_layers=2,
                            batch_size=2,
                            common_dynamic_shapes=True,
                        ),
                        dict(replace_dynamic_cache=False),
                    )
                ),
                "SmolLM17b_2Layer": (
                    lambda: (
                        get_smollm_1_7b(
                            input_cache=True,
                            num_hidden_layers=2,
                            batch_size=2,
                            common_dynamic_shapes=True,
                        ),
                        dict(replace_dynamic_cache=True, strict=False),
                    )
                ),
                "TinyLLM_NoCache": (
                    lambda: (
                        get_tiny_llm(
                            input_cache=False,
                            batch_size=2,
                            common_dynamic_shapes=True,
                        ),
                        dict(replace_dynamic_cache=False),
                    )
                ),
                "TinyLLM": (
                    lambda: (
                        get_tiny_llm(
                            input_cache=True,
                            batch_size=2,
                            common_dynamic_shapes=True,
                        ),
                        dict(replace_dynamic_cache=False),
                    )
                ),
            }
        )

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

        dynamic_shapes = None
        inputs2 = None
        if isinstance(tu, dict):
            model_cls, example_inputs = tu["model"], tu["inputs"]
            dynamic_shapes = tu.get("dynamic_shapes", None)
            inputs2 = tu.get("inputs2", None)
            export_options = None
        elif len(tu) == 2:
            if isinstance(tu[0], dict):
                model_cls, example_inputs = tu[0]["model"], tu[0]["inputs"]
                dynamic_shapes = tu[0].get("dynamic_shapes", None)
                inputs2 = tu[0].get("inputs2", None)
                export_options = tu[1]
            else:
                model_cls, example_inputs = tu
                export_options = None
        elif len(tu) == 3:
            model_cls, example_inputs, export_options = tu
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

        if export_options and "replace_dynamic_cache" in export_options:
            patch_options = dict(
                replace_dynamic_cache=export_options.pop("replace_dynamic_cache")
            )
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
        )

    def iter_model_names(self):
        model_names = list(self.MODELS)
        assert model_names, "Empty list of models"
        model_names.sort()

        start, end = self.get_benchmark_indices(len(model_names))
        yield from self.enumerate_model_names(model_names, start=start, end=end)
