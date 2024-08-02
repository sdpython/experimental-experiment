import importlib
import textwrap
from typing import Any, Optional, Set, Tuple
import torch
from torch._dynamo.testing import collect_results, reset_rng_state
from torch._dynamo.utils import clone_inputs
from ._bash_bench_model_runner import (
    download_retry_decorator,
    _rand_int_tensor,
    ModelRunner,
)
from ._bash_bench_benchmark_runner import BenchmarkRunner
from ._bash_bench_set_dummies import Neuron, Neuron16, NeuronTuple, Neuron2Outputs


class ExplicitRunner(BenchmarkRunner):

    @classmethod
    def initialize(container):
        """
        Steps to run before running the benchmark.
        """
        container.EXTRA_MODELS.update(
            {
                "Speech2Text2": (
                    lambda: Neuron.config,
                    Neuron,
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

    def load_model(
        self,
        model_name: str,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ) -> ModelRunner:
        is_training = self.training
        use_eval_mode = self.use_eval_mode
        dtype = self.dtype
        reset_rng_state()
        model_cls, config, batch_size = self._get_model_cls_and_config(model_name)
        assert isinstance(
            batch_size, int
        ), f"Unexpected value for batch_size={batch_size}"

        model = model_cls()
        if dtype is None:
            model = model.to(self.device)
        else:
            model = model.to(self.device, dtype=dtype)

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
            suite="HuggingFace",
        )

    def iter_model_names(self):
        model_names = list(self.BATCH_SIZE_KNOWN_MODELS.keys()) + list(
            self.EXTRA_MODELS.keys()
        )
        model_names = set(model_names)
        assert model_names, "Empty list of models"
        model_names = sorted(model_names)

        start, end = self.get_benchmark_indices(len(model_names))
        for _ in self.enumerate_model_names(model_names, start=start, end=end):
            yield _

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
