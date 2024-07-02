import collections
import os
import gc
import time
from typing import Any, Callable, Set, Optional, Tuple, Iterator, Dict, List
import numpy as np
import onnx
import torch
from .export_model_helper import WrapInferenceSessionForTorch


class MakeConfig:
    """Creates a dummy configtation."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


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
    def __init__(
        self,
        model: Any,
        inputs: Any,
        device: str,
        dtype: torch.dtype,
        warmup: int,
        repeat: int,
    ):
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
        self.repeat = repeat
        self.warmup = warmup

    def _run_dict(self) -> Any:
        return self.model(**self.inputs)

    def _run_tuple(self) -> Any:
        return self.model(*self.inputs)

    def parameters_size(self) -> int:
        res = 0
        for p in self.model.parameters():
            res += np.prod(list(p.shape))
        return res

    def to_onnx(
        self,
        exporter: str,
        name: str,
        dynamic: bool,
        fake_tensor: bool,
        no_grad: bool,
        optimization: str,
        verbose: int,
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        if exporter == "custom":
            return self._to_onnx_custom(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
            )
        raise AssertionError(f"Exporter {exporter!r} is not implemented.")

    def _to_onnx_custom(
        self,
        name: str,
        dynamic: bool,
        fake_tensor: bool,
        no_grad: bool,
        optimization: str,
        verbose: int,
    ):
        assert not dynamic, "dynamic true not implemented yet"
        from ..torch_interpreter import to_onnx

        onx = to_onnx(
            self.model,
            self.inputs,
            optimize=bool(optimization),
            large_model=True,
            verbose=verbose,
        )
        onx.save(name)

    def make_feeds(self, filename):
        onx = onnx.load(filename, load_external_data=False)
        names = [_.name for _ in onx.graph.input]
        if isinstance(self.inputs, dict):
            assert set(names) == set(
                self.inputs
            ), f"Input names mismatch, got {set(self.inputs)}, expecting {set(names)}."
            return self.inputs
        assert len(names) == len(
            self.inputs
        ), f"Mismatch number of outputs, {list(self.inputs)} != {names}"
        return dict(zip(names, self.inputs))


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
        warmup: int = 10,
        repeat: int = 30,
        fake_tensor: bool = False,
        no_grad: bool = False,
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
        self.repeat = repeat
        self.warmup = warmup
        self.fake_tensor = fake_tensor
        self.no_grad = no_grad

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

    def enumerate_load_models(self) -> Iterator[Tuple[Any, Any]]:
        """
        Loads the models and returns them.
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
            if self.verbose:
                print(
                    f"[BenchmarkRunner.benchmark] loaded model {model_name!r} in {time.perf_counter() - begin}"
                )
            yield res

    def enumerate_run_models(self) -> Iterator[Any]:
        names = self.get_model_name_list()
        for model_name in names:
            if self.verbose:
                begin = time.perf_counter()
                print(f"[BenchmarkRunner.benchmark] run model {model_name!r}")
            model_runner = self.load_model(model_name)
            if self.verbose:
                print(
                    f"[BenchmarkRunner.benchmark] run model {model_name!r} in {time.perf_counter() - begin}"
                )
            if self.verbose > 1:
                print(
                    f"[BenchmarkRunner.benchmark] parameters size {model_runner.parameters_size()!r}"
                )
            yield model_runner.run()

    def enumerate_test_models(
        self,
        exporter: str,
        process: bool = False,
        folder: str = "dump_test_models",
        dynamic: bool = False,
        optimization: str = "",
    ) -> Iterator[Dict[Any, Any]]:
        assert not process, "process=True not implemented."
        assert not dynamic, "dynamic=True not implemented."
        assert not optimization, "optimization=True not implemented."

        import onnxruntime

        if not os.path.exists(folder):
            os.makedirs(folder)
        names = self.get_model_name_list()
        for model_name in names:
            if self.verbose:
                print(f"[BenchmarkRunner.benchmark] test model {model_name!r}")
            if self.verbose > 1:
                print(f"[BenchmarkRunner.benchmark] load model {model_name!r}")

            stats = {}
            begin = time.perf_counter()
            model_runner = self.load_model(model_name)
            repeat = model_runner.repeat
            warmup = model_runner.warmup
            stats["model_name"] = model_name
            stats["time_load"] = time.perf_counter() - begin
            stats["size_params"] = model_runner.parameters_size()
            stats["warmup"] = warmup
            stats["repeat"] = repeat
            stats["exporter"] = exporter

            # warmup
            if self.verbose > 1:
                print(f"[BenchmarkRunner.benchmark] warmup model {model_name!r}")
            begin = time.perf_counter()
            for w in range(warmup):
                expected = model_runner.run()
            stats["time_eager_warmup"] = (time.perf_counter() - begin) / warmup

            # repeat
            if self.verbose > 1:
                print(f"[BenchmarkRunner.benchmark] repeat model {model_name!r}")
            begin = time.perf_counter()
            for w in range(repeat):
                expected = model_runner.run()
            stats["time_eager_repeat"] = (time.perf_counter() - begin) / repeat

            if self.verbose > 1:
                print(f"[BenchmarkRunner.benchmark] export model {model_name!r}")

            # export
            filename = os.path.join(
                folder, f"{model_name}-{exporter}-{self.device}-{self.dtype}.onnx"
            )
            stats["filename"] = filename
            begin = time.perf_counter()
            model_runner.to_onnx(
                exporter,
                name=filename,
                dynamic=dynamic,
                optimization=optimization,
                verbose=self.verbose + 1,
                fake_tensor=self.fake_tensor,
                no_grad=self.no_grad,
            )
            stats["time_export"] = time.perf_counter() - begin

            feeds = model_runner.make_feeds(filename)
            del model_runner
            gc.collect()

            if self.verbose > 1:
                print(f"[BenchmarkRunner.benchmark] inference model {model_name!r}")

            # session
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cpu":
                providers = providers[1:]
            stats["providers"] = ",".join(providers)
            begin = time.perf_counter()
            sess = WrapInferenceSessionForTorch(
                onnxruntime.InferenceSession(filename, providers=providers)
            )
            stats["time_session"] = time.perf_counter() - begin

            if self.verbose > 1:
                print(f"[BenchmarkRunner.benchmark] warmup ort {model_name!r}")

            # warmup
            begin = time.perf_counter()
            for w in range(warmup):
                got = self.ort_run(sess, feeds)
            stats["time_onnx_warmup"] = (time.perf_counter() - begin) / warmup

            if self.verbose > 1:
                print(f"[BenchmarkRunner.benchmark] repeat ort {model_name!r}")

            # repeat
            begin = time.perf_counter()
            for w in range(repeat):
                got = self.ort_run(sess, feeds)
            stats["time_onnx_repeat"] = (time.perf_counter() - begin) / repeat
            stats["speedup"] = stats["time_eager_repeat"] / stats["time_onnx_repeat"]

            # discrepancies
            a, r = self.max_diff(expected, got)
            stats["discrepancies_abs"] = a
            stats["discrepancies_rel"] = r
            yield stats

    def ort_run(
        self, sess: WrapInferenceSessionForTorch, feeds: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        list_feeds = [feeds[k] for k in sess.input_names]
        return sess.run_dlpack(*list_feeds)

    def max_diff(self, expected: Any, got: Any) -> Tuple[float, float]:
        if isinstance(expected, torch.Tensor):
            if isinstance(got, torch.Tensor):
                diff = (got - expected).abs()
                return float(diff.max()), float(
                    ((diff + 1e-7) / (expected + 1e-7)).max()
                )
            if isinstance(got, (list, tuple)):
                if len(got) != 1:
                    return np.inf, np.inf
                return self.max_diff(expected, got[0])
        if isinstance(expected, (tuple, list)):
            if len(expected) == 1:
                return self.max_diff(expected[0], got)
            if not isinstance(got, (tuple, list)):
                return np.inf, np.inf
            if len(got) != len(expected):
                return np.inf, np.inf
            am, rm = 0, 0
            for e, g in zip(expected, got):
                a, r = self.max_diff(e, g)
                am = max(am, a)
                rm = max(rm, r)
            return am, rm

        raise AssertionError(
            f"Not implemented with type(expected)={type(expected)}, type(results)={type(got)}"
        )
