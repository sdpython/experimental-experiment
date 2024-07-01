import collections
import time
from typing import Any, Callable, Set, Optional, Tuple, List, Dict
import numpy as np
import torch


def _rand_int_tensor(
    device: str, low: int, high: int, shape: Tuple[int, ...]
) -> torch.Tensor:
    """Creates a random input integer tensor."""
    return torch.randint(
        low,
        high,
        shape,
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )


def download_retry_decorator(retry: int = 5) -> Callable:  # type: ignore[arg-type]
    """
    Decorator function for applying retry logic to a download function.

    The wrapped function will be called up to 5 times
    and raises an exception if the function fails each time.
    After each unsuccessful attempt, there is a delay before
    the next attempt, which is increased linearly with the number of tries.

    Usage:

    ::

        @download_retry_decorator(retry=5)
        def download_function(model_name: str):
            # download logic goes here
            # ...
    """

    def decorator(download_fn):
        def wrapper(*args, **kwargs) -> Any:
            tries = 0
            total_allowed_tries = retry
            while tries <= total_allowed_tries:
                try:
                    model = download_fn(*args, **kwargs)
                    return model
                except RuntimeError as e:
                    tries += 1
                    if tries <= total_allowed_tries:
                        wait = tries * 30
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            f"Failed to load model {args!r} "
                            f"with following error(s): {e!r}."
                        )

        return wrapper

    return decorator


def get_dynamo_stats():
    # TODO: consider deepcopy'ing the entire counters struct and
    # adding a helper to do subtraction on it
    return collections.Counter(
        {
            "calls_captured": torch._dynamo.utils.counters["stats"]["calls_captured"],
            "unique_graphs": torch._dynamo.utils.counters["stats"]["unique_graphs"],
            "graph_breaks": sum(torch._dynamo.utils.counters["graph_break"].values()),
            # NB: The plus removes zero counts
            "unique_graph_breaks": len(+torch._dynamo.utils.counters["graph_break"]),
            "autograd_captures": torch._dynamo.utils.counters["compiled_autograd"][
                "captures"
            ],
            "autograd_compiles": torch._dynamo.utils.counters["compiled_autograd"][
                "compiles"
            ],
            "cudagraph_skips": torch._dynamo.utils.counters["inductor"][
                "cudagraph_skips"
            ],
        }
    )


def get_peak_memory():
    return torch.cuda.max_memory_allocated() / 10**9


class ModelRunner:
    def __init__(self, model: Any, inputs: Any, device: str, dtype: torch.dtype):
        if dtype is None:
            cvt = lambda o: o.to(device)  # noqa: E731
        else:
            cvt = lambda o: o.to(dtype).to(device)  # noqa: E731
        self.model = cvt(model)
        self.device = device
        self.dtype = dtype
        if isinstance(inputs, dict):
            self.inputs = {k: cvt(v) for k, v in inputs.items()}
            self.run = self._run_dict
        elif isinstance(inputs, tuple):
            self.inputs = tuple(cvt(v) for v in inputs)
            self.run = self._run_tuple
        else:
            raise AssertionError(
                f"Unexpected type {type(inputs)} for inputs and model {type(model)}."
            )

    def _run_dict(self) -> Any:
        return self.model(**self.inputs)

    def _run_tuple(self) -> Any:
        return self.model(*self.inputs)

    def parameters_size(self) -> int:
        res = 0
        for p in self.model.parameters():
            res += np.prod(list(p.shape))
        return res


class BenchmarkRunner:
    def __init__(
        self,
        suite_name: str,
        device: str,
        partition_id: int = 0,
        total_partitions: int = 1,
        include_model_names: Optional[Set[str]] = None,
        exclude_model_names: Optional[Set[str]] = None,
        training: bool = False,
        use_eval_mode: bool = False,
        enable_activation_checkpointing: bool = False,
        dtype: Optional[torch.dtype] = None,
        verbose: int = 0,
    ):
        self.suite_name = suite_name
        self.device = device
        self.partition_id = partition_id
        self.total_partitions = total_partitions
        self.include_model_names = include_model_names
        self.exclude_model_names = exclude_model_names or set()
        self.verbose = verbose
        self.training = training
        self.use_eval_mode = use_eval_mode
        self.enable_activation_checkpointing = enable_activation_checkpointing
        self.dtype = dtype

    def get_model_name_list(self):
        return list(self.iter_model_names())

    def get_benchmark_indices(self, length):
        start = self.partition_id * (length // self.total_partitions)
        end = (
            (self.partition_id + 1) * (length // self.total_partitions)
            if self.partition_id < self.total_partitions - 1
            else length
        )
        return start, end

    def iter_models(self):
        """
        Loads the models,
        """
        names = self.get_model_name_list()
        for model_name in names:
            if self.verbose:
                begin = time.perf_counter()
                print(f"[BenchmarkRunner.benchmark] load model {model_name!r}")
            res = self.load_model(model_name)
            if self.verbose:
                print(
                    f"[BenchmarkRunner.benchmark] loaded model {model_name!r} in {time.perf_counter() - begin}"
                )
            yield res

    def run_models(self, process: bool = False) -> List[Dict[str, Any]]:
        names = self.get_model_name_list()
        for model_name in names:
            if self.verbose:
                print(f"[BenchmarkRunner.benchmark] run model {model_name!r}")
            model_runner = self.load_model(model_name)
            if self.verbose > 1:
                print(
                    f"[BenchmarkRunner.benchmark] parameters size {model_runner.parameters_size()!r}"
                )
            model_runner.run()
