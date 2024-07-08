import os
import gc
import time
from datetime import datetime
from typing import Any, Set, Optional, Tuple, Iterator, Dict, List, Union
import numpy as np
import onnx
import torch
from .export_model_helper import WrapInferenceSessionForTorch, WrapForTorch


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
            self.dtype = getattr(torch, dtype) if dtype else None
        else:
            self.dtype = dtype
        self.repeat = repeat
        self.warmup = warmup
        self.fake_tensor = fake_tensor
        self.no_grad = no_grad
        self.target_opset = target_opset
        assert no_grad, "no_grad true not implemented yet"

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
            yield res

    def enumerate_run_models(self) -> Iterator[Any]:
        """
        Runs the models once.
        """
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

    def move_to(self, device: str, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, tuple):
            return tuple(self.move_to(device, o) for o in obj)
        if obj is None:
            return None
        raise AssertionError(f"move_to not implemented for type {type(obj)}")

    def obj_size(self, obj: Any) -> int:
        if isinstance(obj, torch.Tensor):
            return np.prod(list(obj.shape))
        if isinstance(obj, tuple):
            return sum(self.obj_size(o) for o in obj)
        if obj is None:
            return 0
        raise AssertionError(f"input_size not implemented for type {type(obj)}")

    def enumerate_test_models(
        self,
        exporter: str,
        process: bool = False,
        folder: str = "dump_test_models",
        dynamic: bool = False,
        optimization: str = "",
        quiet: bool = True,
    ) -> Iterator[Dict[Any, Any]]:
        """
        Runs the benchmarks, run, export, run in onnx, measure the speedup.
        """
        assert not process, "process=True not implemented."
        assert not dynamic, "dynamic=True not implemented."
        assert not optimization, "optimization=True not implemented."

        import transformers
        import onnxruntime
        from experimental_experiment.bench_run import get_machine, _clean_string

        machine_specs = get_machine()
        initial_no_grad = torch.is_grad_enabled()

        if not os.path.exists(folder):
            os.makedirs(folder)
        names = self.get_model_name_list()
        for model_name in names:

            #######
            # begin
            #######

            torch.set_grad_enabled(initial_no_grad)
            begin_total = time.perf_counter()
            if self.verbose:
                print(
                    f"[BenchmarkRunner.benchmark] test model {model_name!r} "
                    f"with exporter={exporter!r}"
                )
            if self.verbose > 1:
                print(f"[BenchmarkRunner.benchmark] load model {model_name!r}")

            stats = {
                "version_torch": torch.__version__,
                "version_transformers": transformers.__version__,
                "version_onnxruntime": onnxruntime.__version__,
            }
            stats.update(machine_specs)
            if self.device == "cuda":
                stats["mema_gpu_0_before_loading"] = torch.cuda.memory_allocated(0)
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] gpu_allocation={stats['mema_gpu_0_before_loading']} "
                        f"reserved={torch.cuda.memory_reserved(0)} before loading"
                    )

            begin = time.perf_counter()
            model_runner = self.load_model(model_name)
            if self.verbose:
                print(
                    f"[benchmarkrunner.benchmark] model wrapped with class {type(model_runner.model)}"
                )
            if self.device == "cuda" and self.verbose > 1:
                print(
                    f"[benchmarkrunner.benchmark] gpu_allocation={torch.cuda.memory_allocated(0)} "
                    f"reserved={torch.cuda.memory_reserved(0)} just after loading"
                )
            repeat = model_runner.repeat
            warmup = model_runner.warmup
            stats["model_name"] = model_name
            stats["time_load"] = time.perf_counter() - begin
            stats["params_size"] = model_runner.parameters_size()
            stats["params_dtype"] = model_runner.parameters_dtype()
            stats["warmup"] = warmup
            stats["repeat"] = repeat
            stats["flag_no_grad"] = self.no_grad
            stats["flag_fake_tensor"] = self.fake_tensor
            stats["flag_training"] = self.training
            stats["exporter"] = exporter
            stats["input_size"] = self.obj_size(model_runner.inputs)
            stats["_index"] = f"{model_name}-{exporter}"
            stats["date_start"] = f"{datetime.now():%Y-%m-%d}"

            if self.device == "cuda":
                stats["mema_gpu_1_after_loading"] = torch.cuda.memory_allocated(0)
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] input_size={stats['input_size']}"
                    )
                    print(
                        f"[benchmarkrunner.benchmark] gpu_allocation={stats['mema_gpu_1_after_loading']} "
                        f"reserved={torch.cuda.memory_reserved(0)} after loading"
                    )

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

            begin = time.perf_counter()
            if quiet:
                try:
                    with torch.no_grad():
                        # training mode consumes too much memory
                        for w in range(warmup):
                            if w == warmup - 1:
                                expected = model_runner.run()
                            else:
                                model_runner.run()
                                # we don't plan to keep expected on CUDA
                                # del expected
                            if self.device == "cuda" and self.verbose > 1:
                                print(
                                    f"[benchmarkrunner.benchmark] gpu_allocation={torch.cuda.memory_allocated(0)} "
                                    f"reserved={torch.cuda.memory_reserved(0)} after iteration {w}"
                                )
                except Exception as e:
                    stats["ERR_warmup_eager"] = _clean_string(str(e)).replace(
                        "\n", "_ "
                    )
                    stats["time_warmup_eager"] = (time.perf_counter() - begin) / warmup
                    if self.verbose:
                        print(f"[benchmarkrunner.benchmark] time_warmup_eager {e}")
                    yield stats
                    continue
                stats["time_warmup_eager"] = (time.perf_counter() - begin) / warmup
            else:
                with torch.no_grad():
                    # training mode consumes too much memory
                    for w in range(warmup):
                        if w == warmup - 1:
                            expected = model_runner.run()
                        else:
                            model_runner.run()
                            # we don't plan to keep expected on CUDA
                            # del expected
                        if self.device == "cuda" and self.verbose > 1:
                            print(
                                f"[benchmarkrunner.benchmark] gpu_allocation={torch.cuda.memory_allocated(0)} "
                                f"reserved={torch.cuda.memory_reserved(0)} after iteration {w}"
                            )
                stats["time_warmup_eager"] = (time.perf_counter() - begin) / warmup

            expected = self.move_to("cpu", expected)
            stats["output_size"] = self.obj_size(expected)
            if self.verbose > 1:
                print(f"[benchmarkrunner.benchmark] output_size={stats['output_size']}")

            if self.device == "cuda":
                stats["mema_gpu_2_after_warmup"] = torch.cuda.memory_allocated(0)
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] gpu_allocation={stats['mema_gpu_2_after_warmup']} "
                        f"reserved={torch.cuda.memory_reserved(0)} after warmup"
                    )
                torch.cuda.empty_cache()
                stats["mema_gpu_3_empty_cache"] = torch.cuda.memory_allocated(0)
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] gpu_allocation={stats['mema_gpu_3_empty_cache']} "
                        f"reserved={torch.cuda.memory_reserved(0)} after empty_cache"
                    )

            ########
            # repeat
            ########

            if self.verbose > 1:
                print(
                    f"[BenchmarkRunner.benchmark] repeat model {model_name!r} "
                    f"- {repeat} times"
                )

            begin = time.perf_counter()
            with torch.no_grad():
                # training mode consumes too much memory
                for w in range(repeat):
                    model_runner.run()
            stats["time_repeat_eager"] = (time.perf_counter() - begin) / repeat

            if self.device == "cuda":
                stats["mema_gpu_4_after_repeat"] = torch.cuda.memory_allocated(0)
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] gpu_allocation={stats['mema_gpu_4_after_repeat']} "
                        f"reserved={torch.cuda.memory_reserved(0)} after repeat"
                    )
            if self.verbose > 1:
                print(f"[BenchmarkRunner.benchmark] export model {model_name!r}")

            ########
            # export
            ########

            pfilename = os.path.join(
                folder, f"{model_name}-{exporter}-{self.device}-{self.dtype or ''}"
            )
            if not os.path.exists(pfilename):
                os.mkdir(pfilename)
            filename = os.path.join(pfilename, "model.onnx")

            begin = time.perf_counter()
            if quiet:
                try:
                    exported_model = model_runner.export_as(
                        exporter,
                        name=filename,
                        dynamic=dynamic,
                        optimization=optimization,
                        verbose=self.verbose + 1,
                        fake_tensor=self.fake_tensor,
                        no_grad=self.no_grad,
                        target_opset=self.target_opset,
                    )
                except Exception as e:
                    stats["time_export"] = time.perf_counter() - begin
                    stats["ERR_export"] = _clean_string(str(e)).replace("\n", "_ ")
                    if self.verbose:
                        print(f"[benchmarkrunner.benchmark] err_export {e}")
                    yield stats
                    continue
                stats["time_export"] = time.perf_counter() - begin
            else:
                exported_model = model_runner.export_as(
                    exporter,
                    name=filename,
                    dynamic=dynamic,
                    optimization=optimization,
                    verbose=self.verbose + 1,
                    fake_tensor=self.fake_tensor,
                    no_grad=self.no_grad,
                    target_opset=self.target_opset,
                )
                stats["time_export"] = time.perf_counter() - begin

            if self.device == "cuda":
                stats["mema_gpu_5_after_export"] = torch.cuda.memory_allocated(0)
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] gpu_allocation={stats['mema_gpu_5_after_export']} "
                        f"reserved={torch.cuda.memory_reserved(0)} after export"
                    )

            stats["filename"] = filename
            if quiet:
                try:
                    feeds = model_runner.make_feeds(exporter, filename)
                except AssertionError as e:
                    stats["ERR_feeds"] = _clean_string(str(e)).replace("\n", "_ ")
                    if self.verbose:
                        print(f"[benchmarkrunner.benchmark] err_feeds {e}")
                    yield stats
                    continue
            else:
                feeds = model_runner.make_feeds(exporter, filename)

            del model_runner
            gc.collect()

            if self.device == "cuda":
                stats["mema_gpu_6_after_gcollect"] = torch.cuda.memory_allocated(0)
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] gpu_allocation={stats['mema_gpu_6_after_gcollect']} "
                        f"reserved={torch.cuda.memory_reserved(0)} after gc.collect"
                    )

            if self.verbose > 1:
                print(f"[BenchmarkRunner.benchmark] inference model {model_name!r}")

            #########
            # session
            #########

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cpu":
                providers = providers[1:]
            stats["providers"] = ",".join(providers)
            begin = time.perf_counter()
            if isinstance(exported_model, onnx.ModelProto):
                is_onnx = True
                ort_sess = onnxruntime.InferenceSession(filename, providers=providers)
                sess = WrapInferenceSessionForTorch(ort_sess)
                stats["onnx_model"] = "1"
                onx_inputs = ort_sess.get_inputs()
                onx_outputs = ort_sess.get_outputs()
                stats["onnx_n_inputs"] = len(onx_inputs)
                stats["onnx_n_outputs"] = len(onx_outputs)
                stats["onnx_input_names"] = "|".join(i.name for i in onx_inputs)
                stats["onnx_output_names"] = "|".join(i.name for i in onx_outputs)
            else:
                is_onnx = False
                sess = WrapForTorch(exported_model)
                stats["onnx_model"] = "0"

            stats["time_session"] = time.perf_counter() - begin

            if self.device == "cuda":
                stats["mema_gpu_7_after_session"] = torch.cuda.memory_allocated(0)
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] gpu_allocation={stats['mema_gpu_7_after_session']} "
                        f"reserved={torch.cuda.memory_reserved(0)} after session"
                    )

            if self.verbose > 1:
                print(
                    f"[BenchmarkRunner.benchmark] warmup "
                    f"{exporter} - {model_name!r}"
                )
            stats["device"] = self.device

            if os.path.exists(filename):
                stats["onnx_filesize"] = os.stat(filename).st_size

            torch.set_grad_enabled(not self.no_grad)

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
                if quiet:
                    try:
                        for _ in range(warmup):
                            if _ == warmup - 1:
                                got = self.ort_run(sess, feeds)
                            else:
                                self.ort_run(sess, feeds)
                    except Exception as e:
                        if self.verbose:
                            print(f"[benchmarkrunner.benchmark] err_warmup {e}")
                        stats["ERR_warmup"] = _clean_string(str(e)).replace("\n", "_ ")
                        stats["time_warmup"] = (time.perf_counter() - begin) / warmup
                        yield stats
                        continue
                else:
                    for _ in range(warmup):
                        if _ == warmup - 1:
                            got = self.ort_run(sess, feeds)
                        else:
                            self.ort_run(sess, feeds)
                stats["time_warmup"] = (time.perf_counter() - begin) / warmup
                if self.device == "cuda":
                    stats["mema_gpu_8_after_export_warmup"] = (
                        torch.cuda.memory_allocated(0)
                    )
                    if self.verbose > 1:
                        print(
                            f"[benchmarkrunner.benchmark] gpu_allocation={stats['mema_gpu_8_after_export_warmup']} "
                            f"reserved={torch.cuda.memory_reserved(0)} after export warmup"
                        )
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] torch.is_grad_enabled()="
                        f"{torch.is_grad_enabled()} after warmup"
                    )
                got = self.move_to("cpu", got)

                if self.verbose > 1:
                    print(f"[BenchmarkRunner.benchmark] repeat ort {model_name!r}")

                ################
                # repeat session
                ################

                if "ERR_warmup" not in stats:
                    begin = time.perf_counter()
                    for _ in range(repeat):
                        self.ort_run(sess, feeds)
                    stats["time_repeat"] = (time.perf_counter() - begin) / repeat
                if self.device == "cuda":
                    stats["mema_gpu_9_after_export_repeat"] = (
                        torch.cuda.memory_allocated(0)
                    )
                    if self.verbose > 1:
                        print(
                            f"[benchmarkrunner.benchmark] gpu_allocation={stats['mema_gpu_9_after_export_repeat']} "
                            f"reserved={torch.cuda.memory_reserved(0)} after export repeat"
                        )
            else:
                # warmup session
                if exporter == "eager":
                    # no try, catch needed for eager mode.
                    begin = time.perf_counter()
                    for _ in range(warmup):
                        if _ == warmup - 1:
                            got = sess.run(feeds)
                        else:
                            sess.run(feeds)
                    stats["time_warmup"] = (time.perf_counter() - begin) / warmup
                else:
                    begin = time.perf_counter()
                    if quiet:
                        try:
                            for _ in range(warmup):
                                if _ == warmup - 1:
                                    got = sess.run(feeds)
                                else:
                                    sess.run(feeds)
                        except Exception as e:
                            if self.verbose:
                                print(f"[benchmarkrunner.benchmark] err_warmup {e}")
                            stats["ERR_warmup"] = _clean_string(str(e)).replace(
                                "\n", "_ "
                            )
                            stats["time_warmup"] = (
                                time.perf_counter() - begin
                            ) / warmup
                            yield stats
                            continue
                    else:
                        for _ in range(warmup):
                            if _ == warmup - 1:
                                got = sess.run(feeds)
                            else:
                                sess.run(feeds)
                    stats["time_warmup"] = (time.perf_counter() - begin) / warmup
                if self.device == "cuda":
                    stats["mema_gpu_8_after_export_warmup"] = (
                        torch.cuda.memory_allocated(0)
                    )
                    if self.verbose > 1:
                        print(
                            f"[benchmarkrunner.benchmark] gpu_allocation={stats['mema_gpu_8_after_export_warmup']} "
                            f"reserved={torch.cuda.memory_reserved(0)} after export warmup"
                        )
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] torch.is_grad_enabled()="
                        f"{torch.is_grad_enabled()} after warmup"
                    )
                got = self.move_to("cpu", got)

                if self.verbose > 1:
                    print(f"[BenchmarkRunner.benchmark] repeat torch {model_name!r}")

                ################
                # repeat session
                ################

                if "ERR_warmup" not in stats:
                    begin = time.perf_counter()
                    for _ in range(repeat):
                        sess.run(feeds)
                    stats["time_repeat"] = (time.perf_counter() - begin) / repeat

                if self.device == "cuda":
                    stats["mema_gpu_9_after_export_repeat"] = (
                        torch.cuda.memory_allocated(0)
                    )
                    if self.verbose > 1:
                        print(
                            f"[benchmarkrunner.benchmark] gpu_allocation={stats['mema_gpu_9_after_export_repeat']} "
                            f"reserved={torch.cuda.memory_reserved(0)} after export repeat"
                        )

            if "time_repeat" in stats:
                stats["speedup"] = stats["time_repeat_eager"] / stats["time_repeat"]
                stats["speedup_increase"] = stats["speedup"] - 1

            ###############
            # discrepancies
            ###############

            if got is not None:
                a, r = self.max_diff(
                    expected, got, verbose=self.verbose, flatten=is_onnx
                )
                stats["discrepancies_abs"] = a
                stats["discrepancies_rel"] = r
                if self.verbose:
                    print(
                        f"[BenchmarkRunner.benchmark] done model with {len(stats)} metrics"
                    )

            total_time = time.perf_counter() - begin_total
            stats["time_total"] = total_time
            if self.verbose:
                print(
                    f"[BenchmarkRunner.benchmark] done model {model_name!r} "
                    f"with exporter={exporter!r} in {total_time}"
                )
            yield stats

        # restore the initial state
        torch.set_grad_enabled(initial_no_grad)

    def ort_run(
        self, sess: WrapInferenceSessionForTorch, feeds: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Runs with onnxruntme."""
        list_feeds = [feeds[k] for k in sess.input_names]
        return sess.run_dlpack(*list_feeds)

    @classmethod
    def _flatten(cls, value):
        res = []
        assert not isinstance(value, dict), "Unable to flatten a dictionary."
        if isinstance(value, (list, tuple)):
            for v in value:
                res.extend(cls._flatten(v))
        else:
            res.append(value)
        return tuple(res)

    def max_diff(
        self,
        expected: Any,
        got: Any,
        verbose: int = 0,
        level: int = 0,
        flatten: bool = False,
    ) -> Tuple[float, float]:
        """
        Returns the maximum discrepancy.
        """
        if flatten:
            return self.max_diff(
                self._flatten(expected),
                self._flatten(got),
                verbose=verbose,
                flatten=False,
            )
        if hasattr(expected, "to_tuple"):
            return self.max_diff(
                expected.to_tuple(), got, verbose=verbose, level=level + 1
            )

        if hasattr(got, "to_tuple"):
            return self.max_diff(
                expected, got.to_tuple(), verbose=verbose, level=level + 1
            )

        if isinstance(expected, torch.Tensor):
            if isinstance(got, torch.Tensor):
                diff = (got - expected).abs()
                return float(diff.max()), float(
                    ((diff.abs()) / (expected.abs() + 1e-7)).max()
                )
            if isinstance(got, (list, tuple)):
                if len(got) != 1:
                    if verbose > 2:
                        print(
                            f"[max_diff] (a) inf because len(expected)={len(expected)}!=1, "
                            f"len(got)={len(got)}, level={level}"
                        )
                        for i, (a, b) in enumerate(zip(expected, got)):
                            if isinstance(a, torch.Tensor) and isinstance(
                                b, torch.Tensor
                            ):
                                print(
                                    f"    i={i} expected {a.dtype}:{a.shape}, has {b.dtype}:{b.shape}"
                                )
                            else:
                                print(f"    i={i} a is {type(a)}, b is {type(b)}")
                    return np.inf, np.inf
                return self.max_diff(expected, got[0], verbose=verbose, level=level + 1)
        if isinstance(expected, (tuple, list)):
            if len(expected) == 1:
                return self.max_diff(expected[0], got, verbose=verbose, level=level + 1)
            if not isinstance(got, (tuple, list)):
                if verbose > 2:
                    print(
                        f"[max_diff] inf because type(expected)={type(expected)}, "
                        f"type(got)={type(got)}, level={level}"
                    )
                return np.inf, np.inf
            if len(got) != len(expected):
                if verbose > 2:
                    print(
                        f"[max_diff] (b) inf because len(expected)={len(expected)}, "
                        f"len(got)={len(got)}, level={level}"
                    )
                    for i, (a, b) in enumerate(zip(expected, got)):
                        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                            print(
                                f"    i={i} expected {a.dtype}:{a.shape}, has {b.dtype}:{b.shape}"
                            )
                        else:
                            print(f"    i={i} a is {type(a)}, b is {type(b)}")
                return np.inf, np.inf
            am, rm = 0, 0
            for e, g in zip(expected, got):
                a, r = self.max_diff(e, g, verbose=verbose, level=level + 1)
                am = max(am, a)
                rm = max(rm, r)
            return am, rm

        raise AssertionError(
            f"Not implemented with type(expected)={type(expected)}, type(results)={type(got)}, "
            f"dir(expected)={dir(expected)}, level={level}"
        )
