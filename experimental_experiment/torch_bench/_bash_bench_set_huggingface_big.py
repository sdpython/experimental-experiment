from typing import Any, Callable, Dict, Optional, Set, Tuple
from torch._dynamo.testing import reset_rng_state
from ..torch_interpreter import DEFAULT_TARGET_OPSET
from ._bash_bench_benchmark_runner import BenchmarkRunner
from ._bash_bench_model_runner import ModelRunner
from .big_models import CACHE as CACHE_DEFAULT
from .big_models.try_codellama import get_model_inputs as get_codellama
from .big_models.try_falcon_mamba import get_model_inputs as get_falcon_mamba
from .big_models.try_minilm import get_model_inputs as get_minilm
from .big_models.try_smollm import get_model_inputs as get_smollm
from .big_models.try_stable_diffusion_3 import get_model_inputs as get_stable_diffusion_3


class HuggingfaceBigRunner(BenchmarkRunner):
    SUITE = "HuggingFaceBig"
    MODELS: Dict[str, Callable] = {}
    CACHE = CACHE_DEFAULT

    def initialize(self):
        """Steps to run before running the benchmark."""
        self.MODELS.update(
            {
                "all_MiniLM_L6_v1": get_minilm,
                "code_llama": get_codellama,
                "stable_diffusion_3": get_stable_diffusion_3,
                "falcon_mamba_7b": get_falcon_mamba,
                "SmolLM_1_7b": get_smollm,
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
        target_opset: int = DEFAULT_TARGET_OPSET,
        dtype: Optional[Any] = None,
        nvtx: bool = False,
        dump_ort: bool = False,
        attn_impl: str = "eager",
    ):
        super().__init__(
            "huggingface_big",
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
        model_cls, example_inputs = self._get_model_cls_and_config(model_name)(
            dtype=str(self.dtype).replace("torch.", ""),
            device=self.device,
            verbose=self.verbose,
            cache=self.CACHE,
        )

        model = model_cls()

        if is_training and not use_eval_mode:
            model.train()
        elif hasattr(model, "eval"):
            model.eval()

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
            attn_impl=self.attn_impl,
        )

    def iter_model_names(self):
        model_names = list(self.MODELS)
        assert model_names, "Empty list of models"
        model_names.sort()

        start, end = self.get_benchmark_indices(len(model_names))
        yield from self.enumerate_model_names(model_names, start=start, end=end)
