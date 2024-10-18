import gc
import os
import pickle
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import onnx
from onnx.helper import tensor_dtype_to_np_dtype
import torch
from torch._dynamo.testing import collect_results
from torch._dynamo.utils import clone_inputs
from .export_model_helper import (
    WrapForTorch,
    WrapInferenceSessionForTorch,
    compute_weight_size,
    obj_size,
    size_type,
    str_dtype,
    str_shape,
)
from ..bench_run import max_diff
from ..memory_peak import flatten, start_spying_on

from ..ext_test_case import has_onnxruntime_training
from ..xbuilder._dtype_helper import torch_dtype_to_onnx_dtype


class BenchmarkRunner:
    """
    Class running the benchmark.

    :param suite_name: suite name
    :param device: device
    :param partition_id: partition id
    :param total_partition: number of total partition
    :param include_model_names: models to include
    :param exclude_model_names: models to exclude
    :param training: training mode (CHECK)
    :param use_eval_mode: use eval mode (CHECK)
    :param enable_activation_checkpointing: (CHECK)
    :param dtype: default dtype (None to change nothing)
    :param verbose: verbosity
    :param warmup: number of iteration to warmup the model
    :param repeat: number of iteration to repeat the model
    :param fake_tensor: use fake_tensor
    :param no_grad: use no_grad
    :param target_opset: target opset
    :param nvtx: add events to profile
    :param dump_ort: dumps onnxruntime optimized graph
    """

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
        dtype: Optional[Union[str, torch.dtype]] = None,
        verbose: int = 0,
        warmup: int = 10,
        repeat: int = 30,
        fake_tensor: bool = False,
        no_grad: bool = True,
        target_opset: int = 18,
        nvtx: bool = False,
        dump_ort: bool = False,
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
        if isinstance(dtype, str):
            if dtype in ("default", ""):
                dtype = ""
                self.dtype = None
                self.autocast = False
            elif dtype.startswith("autocast_"):
                dtype = dtype.replace("autocast_", "")
                assert hasattr(torch, dtype), f"Unexpected dtype={dtype!r}"
                self.dtype = getattr(torch, dtype)
                self.autocast = True
            else:
                assert hasattr(torch, dtype), f"Unexpected dtype={dtype!r}"
                self.dtype = getattr(torch, dtype)
                self.autocast = False
        else:
            self.dtype = dtype
            self.autocast = False
        self.repeat = repeat
        self.warmup = warmup
        self.fake_tensor = fake_tensor
        self.no_grad = no_grad
        self.target_opset = target_opset
        self.nvtx = nvtx
        self.dump_ort = dump_ort
        assert no_grad, "no_grad false not implemented yet"
        assert not fake_tensor, "fake_tensor true not implemented yet"
        self.dlpack = self.dtype not in {
            "bfloat16",
            onnx.TensorProto.BFLOAT16,
            torch.bfloat16,
        } and has_onnxruntime_training(push_back_batch=True)

    def forward_pass(self, mod, inputs, collect_outputs=True):
        return mod(**inputs)

    def forward_pass_autocast(self, mod, inputs, collect_outputs=True):
        with torch.autocast(device_type=self.device, dtype=self.dtype):
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

    def get_model_name_list(self) -> List[str]:
        """Returns the model list."""
        return list(self.iter_model_names())

    def get_benchmark_indices(self, length):
        """Returns the model indices in the benchmark to run."""
        start = self.partition_id * (length // self.total_partitions)
        end = (
            (self.partition_id + 1) * (length // self.total_partitions)
            if self.partition_id < self.total_partitions - 1
            else length
        )
        return start, end

    def enumerate_model_names(
        self, model_names: List[str], start: int = 0, end: int = -1
    ) -> Iterator[str]:
        """
        Enumerates model names.

        :param model_names: list of names
        :param start: index of the first model
        :param end: index of the last model (excluded) or -1 for the end
        :return: iterator

        The method uses `self.include_model_names` and `self.exclude_model_names`
        to filter in or out the models to run.
        """
        if end == -1:
            end = len(model_names)
        assert (
            start < end
        ), f"Empty partition (start={start}, end={end}, model_names={model_names!r})"
        has_one_model = False
        done = set()
        for index, model_name in enumerate(model_names):
            if index < start or index >= end:
                continue
            if model_name in self.exclude_model_names:
                continue
            if not self.include_model_names:
                yield model_name
                has_one_model = True
                continue
            if model_name in self.include_model_names:
                yield model_name
                has_one_model = True
                done.add(model_name)
                continue

        if self.include_model_names and len(self.include_model_names) != len(done):
            # Some names were not found.
            not_found = [_ for _ in self.include_model_names if _ and _ not in done]
            for sub in not_found:
                for model_name in model_names[start:end]:
                    if model_name not in done and sub in model_name:
                        yield model_name
                        has_one_model = True
                        done.add(model_name)

        assert has_one_model, (
            f"No model listed, model_names={model_names}, start={start}, "
            f"end={end}, self.include_model_names={self.include_model_names}, "
            f"self.exclude_model_names={self.exclude_model_names}"
        )

    def enumerate_load_models(self) -> Iterator[Tuple[Any, Any]]:
        """Loads the models and returns them."""
        names = self.get_model_name_list()
        for model_name in names:
            if self.verbose:
                begin = time.perf_counter()
                print(f"[BenchmarkRunner.benchmark] load model {model_name!r}")
            res = self.load_model(model_name)
            if self.verbose:
                print(
                    f"[BenchmarkRunner.benchmark] loaded model {model_name!r} "
                    f"in {time.perf_counter() - begin}"
                )
            yield res

    def enumerate_run_models(self) -> Iterator[Any]:
        """Runs the models once."""
        names = self.get_model_name_list()
        for model_name in names:
            if self.verbose:
                begin = time.perf_counter()
                print(f"[BenchmarkRunner.benchmark] run model {model_name!r}")
            model_runner = self.load_model(model_name)
            if self.verbose:
                print(
                    f"[BenchmarkRunner.benchmark] run model {model_name!r} "
                    f"in {time.perf_counter() - begin}"
                )
            if self.verbose > 1:
                print(
                    f"[BenchmarkRunner.benchmark] parameters size "
                    f"{model_runner.compute_weight_size()!r}"
                )
            yield model_runner.run()

    def move_to(self, device: str, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, tuple):
            return tuple(self.move_to(device, o) for o in obj)
        if isinstance(obj, list):
            return tuple(self.move_to(device, o) for o in obj)
        if isinstance(obj, dict):
            return {k: self.move_to(device, o) for k, o in obj.items()}
        if obj is None:
            return None
        if hasattr(obj, "to"):
            return obj.to(device)
        if isinstance(obj, np.ndarray):
            t = torch.Tensor(obj).to(device)
            assert torch_dtype_to_onnx_dtype(t.dtype) == tensor_dtype_to_np_dtype(
                obj.dtype
            ), f"Type mismatch between {obj.dtype} and {t.dtype}"
            return t
        if hasattr(obj, "numpy"):
            # if isinstance(obj, onnxruntime.capi.onnxruntime_pybind11_state.OrtValue):
            # Implicit copy to torch.Tensor
            if hasattr(obj, "to_dlpack"):
                from torch._C import _from_dlpack

                return _from_dlpack(obj.to_dlpack()).to(device)
            return torch.Tensor(obj.numpy()).to(device)
        if "SquashedNormal" in obj.__class__.__name__ and device == "cpu":
            return obj
        if hasattr(obj, "conv_states") and obj.__class__.__name__ == "MambaCache":
            # class MambaCache
            # see https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py
            if ("cuda" in device and obj.conv_states.get_device() < 0) or (
                "cpu" in device and obj.conv_states.get_device() >= 0
            ):
                from transformers.configuration_utils import PretrainedConfig

                config = PretrainedConfig(
                    num_hidden_layers=obj.conv_states.shape[0],
                    intermediate_size=obj.intermediate_size,
                    state_size=obj.ssm_state_size,
                    conv_kernel=obj.conv_kernel_size,
                )
                new_cache = obj.__class__(
                    config,
                    batch_size=obj.batch_size,
                    dtype=obj.conv_states.dtype,
                    device=device,
                )
                new_cache.conv_states[:, :, :, :] = obj.conv_states.to(device)[:, :, :, :]
                new_cache.ssm_states[:, :, :, :] = obj.ssm_states.to(device)[:, :, :, :]
                # AssertionError(
                #    f"Moving class {obj.__class__.__name__!r} from device "
                #    f"{obj.conv_states.get_device()} to {device!r} is not implemented yet,
                #    f"dir(obj)={dir(obj)}"
                # )
                return new_cache
            return obj
        raise AssertionError(f"move_to not implemented for type {type(obj)}, dir={dir(obj)}")

    @classmethod
    def size_type(cls, dtype) -> int:
        return size_type(dtype)

    def obj_size(self, obj: Any) -> int:
        return obj_size(obj)

    def ort_run(
        self,
        sess: WrapInferenceSessionForTorch,
        feeds: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Runs with onnxruntme."""
        list_feeds = [feeds[k] for k in sess.input_names]
        if self.dlpack:
            return sess.run_dlpack(*list_feeds)
        return sess.run_ort_inference(*list_feeds)

    @classmethod
    def _post_process_optimization_statistics(
        cls, opt_stats: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Example::

            [{'pattern': 'check_A', 'time_in': 0.004310695920139551},
             {'pattern': 'remove_identity_nodes', 'removed': 393, 'added': 243,
              'time_in': 0.008972601033747196},
             {'pattern': 'check_B', 'time_in': 0.00272956071421504},
             {'pattern': 'remove_unused', 'removed': 0, 'time_in': 0.007460766937583685},
             {'pattern': 'check_C', 'time_in': 0.002775861881673336},
             {'pattern': 'match_CastPattern', 'iteration': 0, 'instances': 26,
              'time_in': 0.001641636248677969, 'match_index': 26},
             {'pattern': 'match_ExpandPattern', 'iteration': 0, 'instances': 0,
              'time_in': 0.0013782307505607605, 'match_index': 26},
             {'pattern': 'match_IdentityPattern', 'iteration': 0, 'instances': 73,
              'time_in': 0.0037209829315543175, 'match_index': 99},
             {'pattern': 'apply_IdentityPattern', 'added': 1, 'removed': 1, 'iteration': 0,
              'match_index': 88, 'instances': 1, 'time_in': 0.0004087090492248535}
        """
        if opt_stats is None:
            return dict(onnx_optimized=0)
        new_stat = {}
        if "optimization" in opt_stats:
            added, removed, time_in = 0, 0, 0.0
            max_iter = 0
            applied = set()
            matched = set()
            n_applied = 0
            by_pattern = {}
            by_pattern_n = {}
            by_iter = {}
            cst_added, cst_removed, cst_time_in = 0, 0, 0.0

            for obs in opt_stats["optimization"]:
                pattern = obs["pattern"]
                if pattern == "constant_folding":
                    cst_added += obs.get("added", 0)
                    cst_removed += obs.get("removed", 0)
                    cst_time_in += obs.get("time_in", 0)
                if pattern not in by_pattern:
                    by_pattern[pattern] = 0
                    by_pattern_n[pattern] = 0
                    by_iter[pattern] = 0
                time_in += obs.get("time_in", 0)
                added += obs.get("added", 0)
                removed += obs.get("removed", 0)
                max_iter = max(max_iter, obs.get("iteration", 0))
                by_pattern[pattern] += obs.get("time_in", 0)
                by_pattern_n[pattern] += obs.get("added", 0) - obs.get("removed", 0)
                if not pattern.startswith("match"):
                    by_iter[pattern] = max(by_iter[pattern], obs.get("iteration", 0))
                p = obs["pattern"]
                if p.startswith("match_"):
                    matched.add(p)
                elif p.startswith("apply_"):
                    applied.add(p)
                    n_applied += 1
            new_stat.update(
                dict(
                    onnx_opt_optimized=1,
                    onnx_opt_all_time_in=time_in,
                    onnx_opt_all_added=added,
                    onnx_opt_all_removed=removed,
                    onnx_opt_max_iter=max_iter,
                    onnx_opt_unique_matched=len(matched),
                    onnx_opt_unique_applied=len(applied),
                    onnx_opt_n_applied=n_applied,
                    time_export_optimization=time_in,
                    onnx_opt_cst_time_in=cst_time_in,
                    onnx_opt_cst_added=cst_added,
                    onnx_opt_cst_removed=cst_removed,
                )
            )
            sorted_time = sorted([(v, k) for k, v in by_pattern.items()], reverse=True)
            if sorted_time:
                for i in range(min(10, len(sorted_time))):
                    new_stat.update(
                        {
                            f"onnx_opt_toptime{i}": sorted_time[i][0],
                            f"onnx_opt_toptimename{i}": sorted_time[i][1],
                            f"time_opt_toptime{i}": sorted_time[i][0],
                        }
                    )
            sorted_n = sorted((v, k) for k, v in by_pattern_n.items())
            if sorted_n:
                for i in range(min(10, len(sorted_n))):
                    new_stat.update(
                        {
                            f"onnx_opt_topn{i}": sorted_n[i][0],
                            f"onnx_opt_topnname{i}": sorted_n[i][1],
                        }
                    )
            sorted_iter = sorted([(v, k) for k, v in by_iter.items()], reverse=True)
            if sorted_iter:
                for i in range(min(10, len(sorted_iter))):
                    new_stat.update(
                        {
                            f"onnx_opt_topiter{i}": sorted_iter[i][0],
                            f"onnx_opt_topitername{i}": sorted_iter[i][1],
                        }
                    )

        if "builder" in opt_stats:
            builder = opt_stats["builder"]
            if "aten" in builder:
                new_stat.update({f"op_torch_{k}": v for k, v in builder["aten"].items()})

        new_stat.update({k: v for k, v in opt_stats.items() if k.startswith("time_")})
        return new_stat

    @classmethod
    def _post_process_onnx_statistics(cls, model: onnx.ModelProto) -> Dict[str, Any]:
        stats = {}
        nodes = list(model.graph.node)
        for f in model.functions:
            nodes.extend(f.node)
        stats["onnx_n_nodes"] = len(nodes)
        stats["onnx_n_initializer"] = len(model.graph.initializer)
        stats["onnx_n_sparse_initializer"] = len(model.graph.sparse_initializer)
        stats["onnx_n_functions"] = len(model.functions)
        stats["onnx_n_sequence"] = len(
            [n for n in model.graph.node if n.op_type == "Sequence"]
        )
        stats["onnx_n_inputs"] = len(model.graph.input)
        stats["onnx_n_outputs"] = len(model.graph.output)
        stats["onnx_input_names"] = "|".join(i.name for i in model.graph.input)
        stats["onnx_output_names"] = "|".join(i.name for i in model.graph.output)
        stats["op_onnx_initializer"] = len(model.graph.initializer)
        stats["op_onnx_sparse_initializer"] = len(model.graph.sparse_initializer)
        for node in nodes:
            if node.domain == "":
                key = f"op_onnx_{node.op_type}"
            else:
                key = f"op_onnx_{node.domain}_{node.op_type}"
            if key in stats:
                stats[key] += 1
            else:
                stats[key] = 1
        return stats

    @classmethod
    def _flatten(cls, value):
        res = []
        if isinstance(value, dict):
            # We assume the dictionary is ordered.
            return cls._flatten(list(value.values()))
        if isinstance(value, (list, tuple)):
            for v in value:
                res.extend(cls._flatten(v))
        else:
            res.append(value)
        return tuple(res)

    @classmethod
    def max_diff(
        cls,
        expected: Any,
        got: Any,
        verbose: int = 0,
        level: int = 0,
        flatten: bool = False,
        debug_info: Optional[List[str]] = None,
        begin: int = 0,
        end: int = -1,
        _index: int = 0,
    ) -> Dict[str, float]:
        """
        Returns the maximum discrepancy.

        :param expected: expected values
        :param got: values
        :param verbose: verbosity level
        :param level: for embedded outputs, used for debug purpposes
        :param flatten: flatten outputs
        :param debug_info: debug information
        :param begin: first output to considered
        :param end: last output to considered (-1 for the last one)
        :param _index: used with begin and end
        :return: dictionary with many values

        * abs: max abolute error
        * rel: max relative error
        * sum: sum of the errors
        * n: number of outputs values, if there is one
          output, this number will be the number of elements
          of this output
        """
        if flatten:
            return cls.max_diff(
                cls._flatten(expected),
                cls._flatten(got),
                verbose=verbose,
                flatten=False,
                debug_info=(
                    debug_info
                    if verbose < 10
                    else (
                        [f"{' ' * level}flatten"]
                        if not debug_info
                        else ([*debug_info, f"{' ' * level}flatten"])
                    )
                ),
                level=level,
                begin=begin,
                end=end,
                _index=_index,
            )

        return max_diff(
            expected=expected,
            got=got,
            verbose=verbose,
            level=level,
            flatten=flatten,
            debug_info=debug_info,
            begin=begin,
            end=end,
            _index=_index,
        )

    def enumerate_test_models(
        self,
        exporter: str,
        process: bool = False,
        folder: str = "dump_test_models",
        dynamic: bool = False,
        optimization: str = "",
        quiet: bool = True,
        memory_peak: bool = False,
        part: Optional[int] = None,
        pickled_name: Optional[str] = None,
        rtopt: bool = True,
        shape_again: bool = False,
    ) -> Iterator[Dict[Any, Any]]:
        """
        Runs the benchmarks, run, export, run in onnx, measure the speedup.

        :param exporter: exporter to run
        :param process: unused
        :param folder: where to dump the models
        :param dynamic: unused now
        :param optimization: optimization string to run
        :param quiet: True to catch exception
        :param memory_peak: True to measure the memory peak in a secondary process
        :param part: None to run both path, 1 to run the first part
            (load the model + eager mode + export),
            2 to run the run the inference
        :param pickled_name: name used to store everything on disk if *part* is True
        :param rtopt: disable onnxruntime optimization
        :param shape_again: run shape inference after the export,
            erases whatever the model already contains
        """
        assert not process, "process=True not implemented."

        from experimental_experiment.bench_run import get_machine

        if "-" in optimization:
            # Removes value "-" as empty strings are sometimes not allowed.
            spl = optimization.split(",") if "," in optimization else [optimization]
            spl = [("" if _ == "-" else _) for _ in spl]
            optimization = ",".join(spl)

        def _end():
            # restore the initial state
            torch.set_grad_enabled(initial_no_grad)

        machine_specs = get_machine()
        initial_no_grad = torch.is_grad_enabled()

        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        names = self.get_model_name_list()
        assert len(names) > 0, "no model to run"
        assert (
            len(names) == 1 or part is None
        ), f"part={part} only works with 1 model at a time not {names}"
        for model_name in names:
            begin_total = time.perf_counter()

            if part is None or part == 0:
                stats, context = self._test_model_part_1(
                    model_name=model_name,
                    machine_specs=machine_specs,
                    quiet=quiet,
                    exporter=exporter,
                    optimization=optimization,
                    dynamic=dynamic,
                    memory_peak=memory_peak,
                    folder=folder,
                    initial_no_grad=initial_no_grad,
                    rtopt=rtopt,
                    shape_again=shape_again,
                )
                if part == 0:
                    stats["STEP"] = "export"
                    assert pickled_name, f"pickled_name cannot be empty with part={part}"
                    if os.path.exists(pickled_name):
                        os.remove(pickled_name)
                    if self.verbose:
                        print(
                            f"[enumerate_test_models] dumps everything into {pickled_name!r}"
                        )
                    pkl = dict(stats=stats, context=context)
                    with open(pickled_name, "wb") as f:
                        pickle.dump(pkl, f)
                    if self.verbose:
                        print(
                            f"[enumerate_test_models] pickled size "
                            f"{os.stat(pickled_name).st_size}"
                        )

            if part is None or part == 1:
                if part == 1:
                    assert pickled_name, f"pickled_name cannot be empty with part={part}"
                    assert os.path.exists(pickled_name), f"{pickled_name!r} does not exist"
                    with open(pickled_name, "rb") as f:
                        data = pickle.load(f)
                    context = data["context"]
                    stats = data["stats"]
                    if self.verbose:
                        print(f"[enumerate_test_models] restored data from {pickled_name!r}")
                if context["part1"]:
                    self._test_model_part_2(stats, **context)
                    if "STEP" in stats:
                        del stats["STEP"]

            total_time = time.perf_counter() - begin_total
            stats["time_total"] = total_time
            if self.verbose:
                print(
                    f"[BenchmarkRunner.benchmark] done model {model_name!r} "
                    f"with exporter={exporter!r} in {total_time}"
                )
            yield stats

        # final steps
        _end()

    def _test_model_part_1(
        self,
        model_name: str,
        quiet=None,
        exporter=None,
        optimization=None,
        dynamic=None,
        memory_peak=None,
        folder=None,
        machine_specs=None,
        initial_no_grad=None,
        autocast=None,
        rtopt=None,
        shape_again=None,
    ):
        assert quiet is not None
        assert exporter is not None
        assert optimization is not None
        assert dynamic is not None
        assert memory_peak is not None
        assert folder is not None
        assert machine_specs is not None
        assert initial_no_grad is not None
        assert shape_again is not None

        import onnxruntime
        import onnxscript

        try:
            import transformers
        except ImportError:
            transformers = None
        try:
            import monai
        except ImportError:
            monai = None
        try:
            import timm
        except ImportError:
            timm = None

        from experimental_experiment.ext_test_case import BOOLEAN_VALUES
        from experimental_experiment.bench_run import _clean_string

        #######
        # begin
        #######

        context = {"part1": False}

        if self.device.startswith("cuda"):
            device_id = 0 if ":" not in self.device else int(self.device.split(":")[1])
            torch.cuda.reset_peak_memory_stats(device_id)
        torch.set_grad_enabled(initial_no_grad)
        if self.verbose:
            print(
                f"[BenchmarkRunner.benchmark] test model {model_name!r} "
                f"with exporter={exporter!r}"
            )
        if self.verbose > 1:
            print(f"[BenchmarkRunner.benchmark] load model {model_name!r}")

        stats = {
            "version_python": ".".join(str(i) for i in sys.version_info[:3]),
            "version_torch": getattr(torch, "__version__", "dev"),
            "version_transformers": (
                "-" if transformers is None else getattr(transformers, "__version__", "dev")
            ),
            "version_onnxruntime": getattr(onnxruntime, "__version__", "dev"),
            "version_onnxscript": getattr(onnxscript, "__version__", "dev"),
            "version_onnx": getattr(onnx, "__version__", "dev"),
            "version_monai": (
                "-" if monai is None else getattr(monai, "__version__", "dev")
            ),
            "version_timm": ("-" if timm is None else getattr(timm, "__version__", "dev")),
        }
        stats.update(machine_specs)
        if self.device.startswith("cuda"):
            stats["device_id"] = device_id
            stats["mema_gpu_0_before_loading"] = torch.cuda.max_memory_allocated(device_id)
            if self.verbose > 1:
                print(
                    f"[benchmarkrunner.benchmark] gpu_allocation="
                    f"{stats['mema_gpu_0_before_loading']} "
                    f"reserved={torch.cuda.memory_reserved(device_id)} "
                    f"before loading"
                )

        begin = time.perf_counter()
        if quiet:
            try:
                model_runner = self.load_model(model_name)
            except Exception as e:
                if self.verbose:
                    print(
                        f"[benchmarkrunner.benchmark] unable to load model "
                        f"{model_name} due to {e}"
                    )
                stats["ERR_load"] = _clean_string(str(e)).replace("\n", " ")
                return stats, context
        else:
            model_runner = self.load_model(model_name)
        if self.verbose:
            print(
                f"[benchmarkrunner.benchmark] model wrapped with class "
                f"{type(model_runner.model)}"
            )
        if self.device.startswith("cuda") and self.verbose > 1:
            print(
                f"[benchmarkrunner.benchmark] gpu_allocation="
                f"{torch.cuda.max_memory_allocated(device_id)} "
                f"reserved={torch.cuda.memory_reserved(device_id)} "
                f"just after loading"
            )
        repeat = model_runner.repeat
        warmup = model_runner.warmup
        stats["model_name"] = model_name
        stats["suite"] = model_runner.suite
        stats["time_load"] = time.perf_counter() - begin
        stats["params_size"] = model_runner.compute_weight_size()
        stats["params_dtype"] = model_runner.parameters_dtype()
        stats["warmup"] = warmup
        stats["repeat"] = repeat
        stats["dynamic"] = 1 if dynamic else 0
        stats["flag_no_grad"] = self.no_grad
        stats["flag_fake_tensor"] = self.fake_tensor
        stats["flag_training"] = self.training
        stats["exporter"] = exporter
        stats["input_size"] = self.obj_size(model_runner.inputs)
        stats["_index"] = (
            f"{model_name}-{exporter}-{optimization}-"
            f"d{1 if dynamic else 0}-rt{1 if rtopt else 0}"
        )
        stats["date_start"] = f"{datetime.now():%Y-%m-%d}"
        stats["opt_patterns"] = optimization
        stats["rtopt"] = 1 if rtopt else 0

        if self.device.startswith("cuda"):
            is_cuda = True
            stats["mema_gpu_1_after_loading"] = torch.cuda.max_memory_allocated(device_id)
            if self.verbose > 1:
                print(f"[benchmarkrunner.benchmark] input_size={stats['input_size']}")
                print(
                    f"[benchmarkrunner.benchmark] gpu_allocation="
                    f"{stats['mema_gpu_1_after_loading']} "
                    f"reserved={torch.cuda.memory_reserved(device_id)} "
                    f"after loading"
                )
        else:
            is_cuda = False

        if self.verbose > 1:
            print(
                f"[BenchmarkRunner.benchmark] model size and dtype "
                f"{stats['params_size']}, {stats['params_dtype']}"
            )

        ########
        # warmup
        ########

        if self.verbose > 1:
            print(
                f"[BenchmarkRunner.benchmark] warmup model {model_name!r} "
                f"- {warmup} times"
            )
            print(f"[BenchmarkRunner.benchmark] device={model_runner.device!r}")
            devices = [
                (
                    i.get_device()
                    if hasattr(i, "get_device")
                    else (None if i is None else type(i))
                )
                for i in model_runner.inputs
            ]
            print(f"[BenchmarkRunner.benchmark] input device={devices}")

        begin = time.perf_counter()
        if quiet:
            try:
                with torch.no_grad():
                    # training mode consumes too much memory
                    for w in range(warmup):
                        if self.nvtx:
                            torch.cuda.nvtx.range_push("EAGER-WARMUP")
                        if w == warmup - 1:
                            expected = model_runner.run()
                        else:
                            model_runner.run()
                            # we don't plan to keep expected on CUDA
                            # del expected
                        if self.nvtx:
                            torch.cuda.nvtx.range_pop()
                        if self.device.startswith("cuda") and self.verbose > 1:
                            print(
                                f"[benchmarkrunner.benchmark] gpu_allocation="
                                f"{torch.cuda.max_memory_allocated(device_id)} "
                                f"reserved={torch.cuda.memory_reserved(device_id)} "
                                f"after iteration {w}"
                            )
            except Exception as e:
                stats["ERR_warmup_eager"] = _clean_string(str(e)).replace("\n", "_ ")
                stats["time_warmup_eager"] = (time.perf_counter() - begin) / warmup
                if self.verbose:
                    print(f"[benchmarkrunner.benchmark] time_warmup_eager {e}")
                return stats, context
            stats["time_warmup_eager"] = (time.perf_counter() - begin) / warmup
        else:
            with torch.no_grad():
                # training mode consumes too much memory
                for w in range(warmup):
                    if self.nvtx:
                        torch.cuda.nvtx.range_push("EAGER-WARMUP")
                    if w == warmup - 1:
                        expected = model_runner.run()
                    else:
                        model_runner.run()
                        # we don't plan to keep expected on CUDA
                        # del expected
                    if self.nvtx:
                        torch.cuda.nvtx.range_pop()
                    if self.device.startswith("cuda") and self.verbose > 1:
                        print(
                            f"[benchmarkrunner.benchmark] gpu_allocation="
                            f"{torch.cuda.max_memory_allocated(device_id)} "
                            f"reserved={torch.cuda.memory_reserved(device_id)} "
                            f"after iteration {w}"
                        )
            stats["time_warmup_eager"] = (time.perf_counter() - begin) / warmup

        expected = self.move_to("cpu", expected)
        stats["output_size"] = self.obj_size(expected)
        if self.verbose > 1:
            print(f"[benchmarkrunner.benchmark] output_size={stats['output_size']}")

        if self.device.startswith("cuda"):
            stats["mema_gpu_2_after_warmup"] = torch.cuda.max_memory_allocated(device_id)
            if self.verbose > 1:
                print(
                    f"[benchmarkrunner.benchmark] gpu_allocation="
                    f"{stats['mema_gpu_2_after_warmup']} "
                    f"reserved={torch.cuda.memory_reserved(device_id)} "
                    f"after warmup"
                )
            torch.cuda.empty_cache()
            stats["mema_gpu_3_empty_cache"] = torch.cuda.max_memory_allocated(device_id)
            if self.verbose > 1:
                print(
                    f"[benchmarkrunner.benchmark] gpu_allocation="
                    f"{stats['mema_gpu_3_empty_cache']} "
                    f"reserved={torch.cuda.memory_reserved(device_id)} "
                    f"after empty_cache"
                )

        ########
        # repeat
        ########

        if self.verbose > 1:
            print(
                f"[BenchmarkRunner.benchmark] repeat model {model_name!r} "
                f"- {repeat} times"
            )

        with torch.no_grad():
            # training mode consumes too much memory
            lats = []
            for _ in range(repeat):
                if is_cuda:
                    torch.cuda.synchronize()
                if self.nvtx:
                    torch.cuda.nvtx.range_push("EAGER-ITER")
                begin = time.perf_counter()
                model_runner.run()
                if is_cuda:
                    torch.cuda.synchronize()
                lats.append(time.perf_counter() - begin)
                if self.nvtx:
                    torch.cuda.nvtx.range_pop()
        if len(lats) > 0:
            stats["time_latency_eager"] = sum(lats) / len(lats)
            stats["time_latency_eager_t_qu"] = "/".join(
                map(str, np.quantile(lats, np.arange(11) / 10.0))
            )
            stats["time_latency_eager_t_min"] = min(lats)
            stats["time_latency_eager_t_max"] = max(lats)
            stats["time_latency_eager_t_std"] = np.std(lats)
            stats["time_latency_eager_t_med"] = np.median(lats)
            h = max(1, len(lats) // 10)
            stats["time_latency_eager_t_qu_10t"] = "/".join(map(str, lats[::h]))
            stats["time_latency_eager_t_delta"] = (
                stats["time_latency_eager_t_max"] - stats["time_latency_eager_t_min"]
            ) / (stats["time_latency_eager_t_med"])
            if len(lats) > 1:
                stats["time_latency_eager_t_corrt"] = np.corrcoef(
                    lats, list(range(len(lats)))
                )[0, 1]
                if len(lats) > 2:
                    stats["time_latency_eager_t_corrp"] = np.corrcoef(lats[1:], lats[:-1])[
                        0, 1
                    ]

        if self.device.startswith("cuda"):
            stats["mema_gpu_4_after_repeat"] = torch.cuda.max_memory_allocated(device_id)
            if self.verbose > 1:
                print(
                    f"[benchmarkrunner.benchmark] gpu_allocation="
                    f"{stats['mema_gpu_4_after_repeat']} "
                    f"reserved={torch.cuda.memory_reserved(device_id)} "
                    f"after repeat"
                )
        if self.verbose > 1:
            print(f"[BenchmarkRunner.benchmark] export model {model_name!r}")

        ########
        # export
        ########

        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device_id)
            stats["mema_gpu_4_reset"] = torch.cuda.max_memory_allocated(device_id)

        sopt = (
            ("-" + optimization.replace("+", "_").replace("/", "_")) if optimization else ""
        )
        sdtype = str(self.dtype).lower().split(".")[-1]
        pfilename = os.path.join(
            folder,
            (
                f"{model_name}-{exporter}-{self.device.replace(':', '')}"
                f"-{sdtype}{sopt}-"
                f"d{1 if dynamic in BOOLEAN_VALUES else 0}"
                f"rt{1 if rtopt in BOOLEAN_VALUES else 0}"
            ),
        )
        if pfilename and not os.path.exists(pfilename):
            os.makedirs(pfilename)
        cleaned_name = model_name.replace(".", "_").replace("/", "_")
        filename = os.path.join(
            pfilename,
            (
                f"model_{cleaned_name}-{exporter}{sopt}-"
                f"d{1 if dynamic in BOOLEAN_VALUES else 0}"
                f"rt{1 if rtopt in BOOLEAN_VALUES else 0}.onnx"
            ),
        )

        memory_session = (
            start_spying_on(cuda=self.device.startswith("cuda")) if memory_peak else None
        )
        if memory_session is not None and self.verbose:
            print("[BenchmarkRunner.benchmark] start_spying_on")

        if self.verbose:
            if dynamic:
                print(
                    f"[BenchmarkRunner.benchmark] dynamic_shapes="
                    f"{model_runner.get_dynamic_shapes(dynamic)}"
                )
            print(
                f"[BenchmarkRunner.benchmark] input shapes="
                f"{model_runner.get_input_shapes(dynamic=dynamic)}"
            )
            _ishapes = model_runner.get_input_shapes(dynamic=dynamic, export=True)
            print(f"[BenchmarkRunner.benchmark] export input shapes={_ishapes}")

        begin = time.perf_counter()
        if quiet:
            try:
                exported_model, opt_stats = model_runner.export_as(
                    exporter,
                    name=filename,
                    dynamic=dynamic,
                    optimization=optimization,
                    verbose=max(self.verbose - 1, 0),
                    fake_tensor=self.fake_tensor,
                    no_grad=self.no_grad,
                    target_opset=self.target_opset,
                )
            except Exception as e:
                stats["time_export"] = time.perf_counter() - begin
                stats["ERR_export"] = _clean_string(str(e)).replace("\n", "_ ")
                if self.verbose:
                    print(f"[benchmarkrunner.benchmark] err_export {e}")
                if memory_session is not None:
                    memory_results = memory_session.stop()
                    memory_stats = flatten(memory_results, prefix="memory_")
                    stats.update(memory_stats)
                if self.verbose:
                    print("[BenchmarkRunner.benchmark] stop_spying_on")
                model_runner.dump_std(f"{filename}.log.txt")
                return stats, context

            stats["time_export"] = time.perf_counter() - begin
            stats["time_export_success"] = time.perf_counter() - begin
        else:
            exported_model, opt_stats = model_runner.export_as(
                exporter,
                name=filename,
                dynamic=dynamic,
                optimization=optimization,
                verbose=max(self.verbose - 1, 0),
                fake_tensor=self.fake_tensor,
                no_grad=self.no_grad,
                target_opset=self.target_opset,
            )
            stats["time_export"] = time.perf_counter() - begin
            stats["time_export_success"] = time.perf_counter() - begin
        model_runner.dump_std(f"{filename}.log.txt")

        if memory_session is not None:
            memory_results = memory_session.stop()
            print(f"[export_model] ends memory monitoring {memory_results}")
            memory_stats = flatten(memory_results, prefix="memory_")
            stats.update(memory_stats)
            if self.verbose:
                print("[BenchmarkRunner.benchmark] stop_spying_on")

        if self.device.startswith("cuda"):
            stats["mema_gpu_5_after_export"] = torch.cuda.max_memory_allocated(device_id)
            if self.verbose > 1:
                print(
                    f"[benchmarkrunner.benchmark] gpu_allocation="
                    f"{stats['mema_gpu_5_after_export']} "
                    f"reserved={torch.cuda.memory_reserved(device_id)} "
                    f"after export"
                )

        stats.update(self._post_process_optimization_statistics(opt_stats))
        stats["filename"] = filename
        stats["onnx_weight_size_proto"] = compute_weight_size(exported_model)
        stats["onnx_weight_size_torch"] = model_runner.compute_weight_size()

        if shape_again:
            if self.verbose:
                print(f"[benchmarkrunner.benchmark] redo shapes {filename!r}")
                if self.verbose > 1:
                    print(f"[benchmarkrunner.benchmark] load {filename!r}")
            onx_with_shapes = onnx.load(filename, load_external_data=False)
            if self.verbose > 1:
                print("[benchmarkrunner.benchmark] wipe shapes out")
            del onx_with_shapes.graph.value_info[:]
            if self.verbose > 1:
                print("[benchmarkrunner.benchmark] do shape inference again")
            onx_with_shapes = onnx.shape_inference.infer_shapes(onx_with_shapes)
            if self.verbose > 1:
                print(f"[benchmarkrunner.benchmark] saves {filename!r}")
            onnx.save(onx_with_shapes, filename, save_as_external_data=False)
            if self.verbose:
                print(f"[benchmarkrunner.benchmark] done shapes again {filename!r}")

        if quiet:
            try:
                feeds = model_runner.make_feeds(exporter, filename)
                feeds_dynamic = (
                    model_runner.make_feeds(exporter, filename, dynamic=True)
                    if dynamic
                    else None
                )
            except AssertionError as e:
                stats["ERR_feeds"] = _clean_string(str(e)).replace("\n", "_ ")
                if self.verbose:
                    print(f"[benchmarkrunner.benchmark] err_feeds {e}")
                return stats, context

        else:
            feeds = model_runner.make_feeds(exporter, filename)
            feeds_dynamic = (
                model_runner.make_feeds(exporter, filename, dynamic=True)
                if dynamic
                else None
            )
            assert (dynamic and feeds_dynamic is not None) or (
                not dynamic and feeds_dynamic is None
            ), (
                f"dynamic={dynamic}, feeds_dynamic is "
                f"{'' if feeds_dynamic is None else 'not'} None"
            )

        assert isinstance(feeds, tuple) or all(
            isinstance(v, torch.Tensor) for v in feeds.values()
        ), f"One input is not a tensor {dict((k,type(v)) for k,v in feeds.items())}"  # noqa: C402
        if feeds_dynamic:
            assert isinstance(feeds_dynamic, tuple) or all(
                isinstance(v, torch.Tensor) for v in feeds_dynamic.values()
            ), (
                f"One dynamic input is not a tensor "
                f"{dict((k,type(v)) for k,v in feeds_dynamic.items())}"  # noqa: C402
            )
        context["feeds"] = feeds
        context["feeds_dynamic"] = feeds_dynamic

        #########
        # dynamic
        #########

        if dynamic:
            expected_dynamic = model_runner.run_dynamic(wrapped=True)
            expected_dynamic = self.move_to("cpu", expected_dynamic)
        else:
            expected_dynamic = None

        del model_runner
        gc.collect()

        if isinstance(feeds, dict):
            # This is the type for onnx inputs
            feeds_values = list(feeds.values())
            stats["onnx_input_dtypes"] = "/".join(
                str_dtype(getattr(_, "dtype", "?")) for _ in feeds_values
            )
            stats["onnx_input_shapes"] = "/".join(
                str_shape(getattr(_, "shape", "?")) for _ in feeds_values
            )
        if isinstance(feeds_dynamic, dict):
            # This is the type for onnx inputs
            feeds_dynamic_values = list(feeds_dynamic.values())
            stats["onnx_input_dynamic_shapes"] = "/".join(
                str_shape(getattr(_, "shape", "?")) for _ in feeds_dynamic_values
            )

        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device_id)
            stats["mema_gpu_6_after_gcollect"] = torch.cuda.max_memory_allocated(device_id)
            if self.verbose > 1:
                print(
                    f"[benchmarkrunner.benchmark] gpu_allocation="
                    f"{stats['mema_gpu_6_after_gcollect']} "
                    f"reserved={torch.cuda.memory_reserved(device_id)} "
                    f"after gc.collect"
                )
        context["part1"] = True
        context["model_name"] = model_name
        context["filename"] = filename
        context["exported_model"] = exported_model
        context["exporter"] = exporter
        context["quiet"] = quiet
        context["expected"] = expected
        context["expected_dynamic"] = expected_dynamic
        context["warmup"] = warmup
        context["repeat"] = repeat
        context["rtopt"] = rtopt

        assert (feeds_dynamic is not None and expected_dynamic is not None) or (
            feeds_dynamic is None and expected_dynamic is None
        ), (
            f"feeds_dynamic is {'' if feeds_dynamic is None else 'not'} None, "
            f"expected_dynamic is {'' if expected_dynamic is None else 'not'} None, "
            f"dynamic={dynamic}"
        )

        return stats, context

    def _test_model_part_2(
        self,
        stats: Dict[str, Any],
        model_name: str,
        filename=None,
        exported_model=None,
        exporter=None,
        quiet=None,
        expected=None,
        expected_dynamic=None,
        feeds=None,
        feeds_dynamic=None,
        warmup=None,
        repeat=None,
        part1=None,
        rtopt=None,
    ):
        assert expected is not None
        assert feeds is not None
        assert repeat is not None
        assert warmup is not None
        assert quiet is not None
        assert exporter is not None
        assert exported_model is not None
        assert filename is not None
        assert part1 is not None
        assert part1, "Part 1 was not sucessful"
        assert (feeds_dynamic is not None and expected_dynamic is not None) or (
            feeds_dynamic is None and expected_dynamic is None
        ), (
            f"feeds_dynamic is {'' if feeds_dynamic is None else 'not'} None, "
            f"expected_dynamic is {'' if expected_dynamic is None else 'not'} None"
        )

        import onnxruntime

        from experimental_experiment.bench_run import _clean_string

        if self.device.startswith("cuda"):
            is_cuda = True
            device_id = 0 if ":" not in self.device else int(self.device.split(":")[1])
            torch.cuda.reset_peak_memory_stats(device_id)
            stats["device_id2"] = device_id
            stats["mema_gpu_6_before_session"] = torch.cuda.max_memory_allocated(device_id)
            if self.verbose > 1:
                print(
                    f"[benchmarkrunner.benchmark] gpu_allocation="
                    f"{stats['mema_gpu_6_before_session']} "
                    f"reserved={torch.cuda.memory_reserved(device_id)} "
                    f"before creating a session"
                )
        else:
            is_cuda = False

        #########
        # session
        #########

        if self.verbose > 1:
            print(f"[BenchmarkRunner.benchmark] inference model {model_name!r}")

        if self.device == "cpu":
            providers = ["CPUExecutionProvider"]
            stats["providers"] = providers[0]
        else:
            providers = [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
            stats["providers"] = f"CUDAExecutionProvider:{device_id},CPUExecutionProvider"

        begin = time.perf_counter()
        if isinstance(exported_model, onnx.ModelProto):
            domains = {d.domain for d in exported_model.opset_import}
            session_options = onnxruntime.SessionOptions()
            if not rtopt:
                session_options.graph_optimization_level = (
                    onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                )
            if self.dump_ort:
                session_options.optimized_model_filepath = f"{filename}-ortopt.onnx"
                if self.verbose > 1:
                    print(
                        f"[BenchmarkRunner.benchmark] saves optimized "
                        f"model by onnxruntime in "
                        f"{session_options.optimized_model_filepath!r}"
                    )
            if "onnx_extended.ortops.optim.cuda" in domains:
                try:
                    from onnx_extended.ortops.optim.cuda import get_ort_ext_libs
                except ImportError as e:
                    stats["ERR_ort"] = _clean_string(str(e)).replace("\n", " ")
                    if self.verbose:
                        print(f"[benchmarkrunner.benchmark] err_ort {e}")
                    return stats

                if self.verbose > 2:
                    print(f"[BenchmarkRunner.benchmark] register {get_ort_ext_libs()[0]!r}")
                session_options.register_custom_ops_library(get_ort_ext_libs()[0])

            if self.verbose > 2:
                print("[BenchmarkRunner.benchmark] create session")
            is_onnx = True
            stats["onnx_model"] = "1"
            if quiet:
                try:
                    ort_sess = onnxruntime.InferenceSession(
                        filename, session_options, providers=providers
                    )
                except Exception as e:
                    stats["ERR_ort"] = _clean_string(str(e)).replace("\n", " ")
                    if self.verbose:
                        print(f"[benchmarkrunner.benchmark] err_ort {e}")
                    return stats
            else:
                ort_sess = onnxruntime.InferenceSession(
                    filename, session_options, providers=providers
                )
            if self.verbose > 1:
                print("[BenchmarkRunner.benchmark] WrapInferenceSessionForTorch")
            sess = WrapInferenceSessionForTorch(ort_sess, nvtx=self.nvtx)
            stats.update(self._post_process_onnx_statistics(exported_model))

            if self.dump_ort and os.path.exists(session_options.optimized_model_filepath):
                # Let's save the optimized model with external weights.
                fold = os.path.join(
                    os.path.split(session_options.optimized_model_filepath)[0],
                    "ort_optimized",
                )
                cleaned_name = model_name.replace(".", "_").replace("/", "_")
                new_filename = os.path.join(fold, f"model_rtopt_{cleaned_name}.onnx")
                if self.verbose > 1:
                    print(
                        f"[BenchmarkRunner.benchmark] load and saves with "
                        f"external weights the optimized model by onnxruntime in "
                        f"{new_filename!r}"
                    )
                if fold and not os.path.exists(fold):
                    os.makedirs(fold)
                ortops = onnx.load(session_options.optimized_model_filepath)
                onnx.save(ortops, new_filename, save_as_external_data=True)
                # Let's free some space.
                os.remove(session_options.optimized_model_filepath)
        else:
            if self.verbose > 1:
                print("[BenchmarkRunner.benchmark] WrapForTorch")
            is_onnx = False
            sess = WrapForTorch(exported_model)
            stats["onnx_model"] = "0"

        stats["time_session"] = time.perf_counter() - begin

        if self.device.startswith("cuda"):
            stats["mema_gpu_7_after_session"] = torch.cuda.max_memory_allocated(device_id)
            if self.verbose > 1:
                print(
                    f"[benchmarkrunner.benchmark] gpu_allocation="
                    f"{stats['mema_gpu_7_after_session']} "
                    f"reserved={torch.cuda.memory_reserved(device_id)} "
                    f"after session"
                )

        if self.verbose > 1:
            print(f"[BenchmarkRunner.benchmark] warmup {exporter} - {model_name!r}")
        stats["device"] = self.device

        if os.path.exists(filename):
            stats["onnx_filesize"] = os.stat(filename).st_size

        torch.set_grad_enabled(not self.no_grad)

        #########
        # dynamic
        #########

        if feeds_dynamic is None or not isinstance(exported_model, onnx.ModelProto):
            got_dynamic = None
        else:
            if self.verbose:
                print("[benchmarkrunner.benchmark] check dynamic")
            if self.nvtx:
                torch.cuda.nvtx.range_push("ORT-DYNAMIC")
            got_dynamic = self.ort_run(sess, feeds_dynamic)
            if self.nvtx:
                torch.cuda.nvtx.range_pop()

        ################
        # warmup session
        ################

        if self.verbose > 1:
            print(
                f"[benchmarkrunner.benchmark] no_grad={self.no_grad} "
                f"torch.is_grad_enabled()={torch.is_grad_enabled()} before warmup"
            )

        got = None
        if isinstance(exported_model, onnx.ModelProto):
            # warmup session
            begin = time.perf_counter()
            time_first_iter = None
            if quiet:
                try:
                    for _ in range(warmup):
                        if self.nvtx:
                            torch.cuda.nvtx.range_push("ORT-WARMUP")
                        if _ == warmup - 1:
                            got = self.ort_run(sess, feeds)
                        else:
                            self.ort_run(sess, feeds)
                        if time_first_iter is None:
                            time_first_iter = time.perf_counter() - begin
                        if self.nvtx:
                            torch.cuda.nvtx.range_pop()
                except Exception as e:
                    if self.verbose:
                        print(f"[benchmarkrunner.benchmark] err_warmup {e}")
                        traceback.print_tb(e.__traceback__, file=sys.stdout)
                    stats["ERR_warmup"] = _clean_string(str(e)).replace("\n", "_ ")
                    stats["time_warmup_fail"] = time.perf_counter() - begin
                    return stats
            else:
                for _ in range(warmup):
                    if self.nvtx:
                        torch.cuda.nvtx.range_push("ORT-WARMUP")
                    if _ == warmup - 1:
                        got = self.ort_run(sess, feeds)
                    else:
                        self.ort_run(sess, feeds)
                    if time_first_iter is None:
                        time_first_iter = time.perf_counter() - begin
                    if self.nvtx:
                        torch.cuda.nvtx.range_pop()
            stats["time_warmup"] = (time.perf_counter() - begin) / warmup
            if time_first_iter is not None:
                stats["time_warmup_first_iteration"] = time_first_iter
            if self.device.startswith("cuda"):
                stats["mema_gpu_8_after_export_warmup"] = torch.cuda.max_memory_allocated(
                    device_id
                )
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] gpu_allocation="
                        f"{stats['mema_gpu_8_after_export_warmup']} "
                        f"reserved={torch.cuda.memory_reserved(device_id)} "
                        f"after export warmup"
                    )
            if self.verbose > 1:
                print(
                    f"[benchmarkrunner.benchmark] torch.is_grad_enabled()="
                    f"{torch.is_grad_enabled()} after warmup"
                )
            got = self.move_to("cpu", got)
            if got_dynamic is not None:
                got_dynamic = self.move_to("cpu", got_dynamic)

            if self.verbose > 1:
                print(f"[BenchmarkRunner.benchmark] repeat ort {model_name!r}")

            ################
            # repeat session
            ################

            if "ERR_warmup" not in stats:
                lats = []
                for _ in range(repeat):
                    if is_cuda:
                        torch.cuda.synchronize()
                    if self.nvtx:
                        torch.cuda.nvtx.range_push("ORT-ITER")
                    begin = time.perf_counter()
                    self.ort_run(sess, feeds)
                    if is_cuda:
                        torch.cuda.synchronize()
                    lats.append(time.perf_counter() - begin)
                    if self.nvtx:
                        torch.cuda.nvtx.range_pop()
                if len(lats) > 0:
                    stats["time_latency"] = sum(lats) / len(lats)
                    stats["time_latency_t_qu"] = "/".join(
                        map(str, np.quantile(lats, np.arange(11) / 10.0))
                    )
                    stats["time_latency_t_min"] = min(lats)
                    stats["time_latency_t_max"] = max(lats)
                    stats["time_latency_t_std"] = np.std(lats)
                    stats["time_latency_t_med"] = np.median(lats)
                    h = max(1, len(lats) // 10)
                    stats["time_latency_t_qu_10t"] = "/".join(map(str, lats[::h]))
                    stats["time_latency_t_delta"] = (
                        stats["time_latency_t_max"] - stats["time_latency_t_min"]
                    ) / (stats["time_latency_t_med"])
                    if len(lats) > 1:
                        stats["time_latency_t_corrt"] = np.corrcoef(
                            lats, list(range(len(lats)))
                        )[0, 1]
                    if len(lats) > 2:
                        stats["time_latency_t_corrp"] = np.corrcoef(lats[1:], lats[:-1])[
                            0, 1
                        ]

            if self.device.startswith("cuda"):
                stats["mema_gpu_9_after_export_repeat"] = torch.cuda.max_memory_allocated(
                    device_id
                )
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] gpu_allocation="
                        f"{stats['mema_gpu_9_after_export_repeat']} "
                        f"reserved={torch.cuda.memory_reserved(device_id)} "
                        f"after export repeat"
                    )
        else:
            # warmup session
            if exporter == "eager":
                # no try, catch needed for eager mode.
                begin = time.perf_counter()
                time_first_iter = None
                for _ in range(warmup):
                    if self.nvtx:
                        torch.cuda.nvtx.range_push("EAGER-WARMUP")
                    if _ == warmup - 1:
                        got = sess.run(feeds)
                    else:
                        sess.run(feeds)
                    if time_first_iter is None:
                        time_first_iter = time.perf_counter() - begin
                    if self.nvtx:
                        torch.cuda.nvtx.range_pop()
                stats["time_warmup"] = (time.perf_counter() - begin) / warmup
                if time_first_iter is not None:
                    stats["time_warmup_first_iteration"] = time_first_iter
            else:
                begin = time.perf_counter()
                time_first_iter = None
                if quiet:
                    try:
                        for _ in range(warmup):
                            if self.nvtx:
                                torch.cuda.nvtx.range_push("CPL-WARMUP")
                            if _ == warmup - 1:
                                got = sess.run(feeds)
                            else:
                                sess.run(feeds)
                            if time_first_iter is None:
                                time_first_iter = time.perf_counter() - begin
                            if self.nvtx:
                                torch.cuda.nvtx.range_pop()
                    except Exception as e:
                        if self.verbose:
                            print(f"[benchmarkrunner.benchmark] err_warmup {e}")
                            traceback.print_tb(e.__traceback__, file=sys.stdout)
                        stats["ERR_warmup"] = _clean_string(str(e)).replace("\n", "_ ")
                        stats["time_warmup_fail"] = time.perf_counter() - begin
                        return stats
                else:
                    for _ in range(warmup):
                        if self.nvtx:
                            torch.cuda.nvtx.range_push("CPL-WARMUP")
                        if _ == warmup - 1:
                            got = sess.run(feeds)
                        else:
                            sess.run(feeds)
                        if time_first_iter is None:
                            time_first_iter = time.perf_counter() - begin
                        if self.nvtx:
                            torch.cuda.nvtx.range_pop()
                stats["time_warmup"] = (time.perf_counter() - begin) / warmup
                if time_first_iter is not None:
                    stats["time_warmup_first_iteration"] = time_first_iter
            if self.device.startswith("cuda"):
                stats["mema_gpu_8_after_export_warmup"] = torch.cuda.max_memory_allocated(
                    device_id
                )
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] gpu_allocation="
                        f"{stats['mema_gpu_8_after_export_warmup']} "
                        f"reserved={torch.cuda.memory_reserved(device_id)} "
                        f"after export warmup"
                    )
            if self.verbose > 1:
                print(
                    f"[benchmarkrunner.benchmark] torch.is_grad_enabled()="
                    f"{torch.is_grad_enabled()} after warmup"
                )
            got = self.move_to("cpu", got)
            if got_dynamic:
                got_dynamic = self.move_to("cpu", got_dynamic)

            if self.verbose > 1:
                print(f"[BenchmarkRunner.benchmark] repeat torch {model_name!r}")

            ################
            # repeat session
            ################

            if "ERR_warmup" not in stats:
                lats = []
                for _ in range(repeat):
                    if is_cuda:
                        torch.cuda.synchronize()
                    if self.nvtx:
                        torch.cuda.nvtx.range_push("CPL-ITER")
                    begin = time.perf_counter()
                    sess.run(feeds)
                    if is_cuda:
                        torch.cuda.synchronize()
                    lats.append(time.perf_counter() - begin)
                    if self.nvtx:
                        torch.cuda.nvtx.range_pop()
                if len(lats) > 0:
                    stats["time_latency"] = sum(lats) / len(lats)
                    stats["time_latency_t_qu"] = "/".join(
                        map(str, np.quantile(lats, np.arange(11) / 10.0))
                    )
                    stats["time_latency_t_min"] = min(lats)
                    stats["time_latency_t_max"] = max(lats)
                    stats["time_latency_t_std"] = np.std(lats)
                    stats["time_latency_t_med"] = np.median(lats)
                    h = max(1, len(lats) // 10)
                    stats["time_latency_t_qu_10t"] = "/".join(map(str, lats[::h]))
                    stats["time_latency_t_delta"] = (
                        stats["time_latency_t_max"] - stats["time_latency_t_min"]
                    ) / (stats["time_latency_t_med"])
                    if len(lats) > 1:
                        stats["time_latency_t_corrt"] = np.corrcoef(
                            lats, list(range(len(lats)))
                        )[0, 1]
                    if len(lats) > 2:
                        stats["time_latency_t_corrp"] = np.corrcoef(lats[1:], lats[:-1])[
                            0, 1
                        ]

            if self.device.startswith("cuda"):
                stats["mema_gpu_9_after_export_repeat"] = torch.cuda.max_memory_allocated(
                    device_id
                )
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] gpu_allocation="
                        f"{stats['mema_gpu_9_after_export_repeat']} "
                        f"reserved={torch.cuda.memory_reserved(device_id)} "
                        f"after export repeat"
                    )

        if "time_latency" in stats:
            stats["speedup"] = stats["time_latency_eager"] / stats["time_latency"]
            stats["speedup_med"] = (
                stats["time_latency_eager_t_med"] / stats["time_latency_t_med"]
            )
            stats["speedup_increase"] = stats["speedup"] - 1

        ###############
        # discrepancies
        ###############

        if got is not None:
            d = self.max_diff(expected, got, verbose=self.verbose, flatten=is_onnx)
            stats["discrepancies_abs"] = d["abs"]
            stats["discrepancies_rel"] = d["rel"]
            stats["discrepancies_avg"] = d["sum"] / max(d["n"], 1)

        if got_dynamic is not None:
            assert (
                expected_dynamic is not None
            ), "expected_dynamic is None and got_dynamic is not."
            d = self.max_diff(
                expected_dynamic, got_dynamic, verbose=self.verbose, flatten=is_onnx
            )
            stats["discrepancies_dynamic_abs"] = d["abs"]
            stats["discrepancies_dynamic_rel"] = d["rel"]
            stats["discrepancies_dynamic_avg"] = d["sum"] / max(d["n"], 1)

        return stats
