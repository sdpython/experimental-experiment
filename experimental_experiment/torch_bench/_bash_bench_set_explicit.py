from typing import Any, Callable, Dict, Optional, Set, Tuple
from torch._dynamo.testing import reset_rng_state
from ._bash_bench_benchmark_runner import BenchmarkRunner
from ._bash_bench_model_runner import ModelRunner
from ._bash_bench_models_helper import (
    get_dummy_model,
    get_dummy_model_fail,
    get_dummy_model_fail_convert,
    get_llama_model_layer,
    get_speech2text2_causal_ml_not_trained_model,
)


class ExplicitRunner(BenchmarkRunner):
    SUITE = "Explicit"
    MODELS: Dict[str, Callable] = {}

    @classmethod
    def initialize(cls):
        """Steps to run before running the benchmark."""
        cls.MODELS.update(
            {
                "1001Fail": get_dummy_model_fail,
                "1001Fail2": get_dummy_model_fail_convert,
                "101Dummy": get_dummy_model,
                "Speech2Text2ForCausalLMNotTrained": get_speech2text2_causal_ml_not_trained_model,  # noqa: E501
                "Llama2Layer": lambda: get_llama_model_layer(num_hidden_layers=2),
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
            "explicit",
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
        verbose: int = 0,
    ) -> ModelRunner:
        is_training = self.training
        use_eval_mode = self.use_eval_mode
        reset_rng_state()
        model_cls, example_inputs = self._get_model_cls_and_config(model_name)()

        model = model_cls()

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
            suite=self.SUITE,
            autocast=self.autocast,
            wrap_kind="nowrap",
        )

    def iter_model_names(self):
        model_names = list(self.MODELS)
        assert model_names, "Empty list of models"
        model_names.sort()

        start, end = self.get_benchmark_indices(len(model_names))
        yield from self.enumerate_model_names(model_names, start=start, end=end)
