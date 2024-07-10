import os
import gc
import time
from datetime import datetime
from typing import Any, Set, Optional, Tuple, Iterator, Dict, List, Union
import numpy as np
import onnx
import torch
from .export_model_helper import WrapInferenceSessionForTorch, WrapForTorch
from ..memory_peak import start_spying_on, flatten


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
        memory_peak: bool = False,
    ) -> Iterator[Dict[Any, Any]]:
        """
        Runs the benchmarks, run, export, run in onnx, measure the speedup.
        """
        assert not process, "process=True not implemented."
        assert not dynamic, "dynamic=True not implemented."

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
            stats["suite"] = model_runner.suite
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
            stats["opt_patterns"] = optimization

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
            stats["time_latency_eager"] = (time.perf_counter() - begin) / repeat

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

            sopt = (
                ("-" + optimization.replace("+", "_").replace("/", "_"))
                if optimization
                else ""
            )
            pfilename = os.path.join(
                folder,
                f"{model_name}-{exporter}-{self.device}-{self.dtype or ''}{sopt}",
            )
            if not os.path.exists(pfilename):
                os.mkdir(pfilename)
            filename = os.path.join(pfilename, "model.onnx")

            memory_session = (
                start_spying_on(cuda=self.device == "cuda") if memory_peak else None
            )

            begin = time.perf_counter()
            if quiet:
                try:
                    exported_model, opt_stats = model_runner.export_as(
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
                    if memory_session is not None:
                        memory_results = memory_session.stop()
                        memory_stats = flatten(memory_results, prefix="memory_")
                        stats.update(memory_stats)
                    yield stats
                    continue
                stats["time_export"] = time.perf_counter() - begin
            else:
                exported_model, opt_stats = model_runner.export_as(
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

            if memory_session is not None:
                memory_results = memory_session.stop()
                print(f"[export_model] ends memory monitoring {memory_results}")
                memory_stats = flatten(memory_results, prefix="memory_")
                stats.update(memory_stats)

            if self.device == "cuda":
                stats["mema_gpu_5_after_export"] = torch.cuda.memory_allocated(0)
                if self.verbose > 1:
                    print(
                        f"[benchmarkrunner.benchmark] gpu_allocation={stats['mema_gpu_5_after_export']} "
                        f"reserved={torch.cuda.memory_reserved(0)} after export"
                    )

            stats.update(self._post_process_optimization_statistics(opt_stats))
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
                stats["onnx_model"] = "1"
                if quiet:
                    try:
                        ort_sess = onnxruntime.InferenceSession(
                            filename, providers=providers
                        )
                    except Exception as e:
                        stats["ERR_ort"] = str(e)
                        if self.verbose:
                            print(f"[benchmarkrunner.benchmark] err_ort {e}")
                        yield stats
                        continue
                else:
                    ort_sess = onnxruntime.InferenceSession(
                        filename, providers=providers
                    )
                sess = WrapInferenceSessionForTorch(ort_sess)
                stats.update(self._post_process_onnx_statistics(exported_model))
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
                    stats["time_latency"] = (time.perf_counter() - begin) / repeat
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
                    stats["time_latency"] = (time.perf_counter() - begin) / repeat

                if self.device == "cuda":
                    stats["mema_gpu_9_after_export_repeat"] = (
                        torch.cuda.memory_allocated(0)
                    )
                    if self.verbose > 1:
                        print(
                            f"[benchmarkrunner.benchmark] gpu_allocation={stats['mema_gpu_9_after_export_repeat']} "
                            f"reserved={torch.cuda.memory_reserved(0)} after export repeat"
                        )

            if "time_latency" in stats:
                stats["speedup"] = stats["time_latency_eager"] / stats["time_latency"]
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
    def _post_process_optimization_statistics(
        cls, opt_stats: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Example:

        ::

            [{'pattern': 'check_A', 'time_in': 0.004310695920139551},
             {'pattern': 'remove_identity_nodes', 'removed': 393, 'added': 243, 'time_in': 0.008972601033747196},
             {'pattern': 'check_B', 'time_in': 0.00272956071421504},
             {'pattern': 'remove_unused', 'removed': 0, 'time_in': 0.007460766937583685},
             {'pattern': 'check_C', 'time_in': 0.002775861881673336},
             {'pattern': 'match_CastPattern', 'iteration': 0, 'instances': 26, 'time_in': 0.001641636248677969, 'match_index': 26},
             {'pattern': 'match_ExpandPattern', 'iteration': 0, 'instances': 0, 'time_in': 0.0013782307505607605, 'match_index': 26},
             {'pattern': 'match_IdentityPattern', 'iteration': 0, 'instances': 73, 'time_in': 0.0037209829315543175, 'match_index': 99},
             {'pattern': 'apply_IdentityPattern', 'added': 1, 'removed': 1, 'iteration': 0, 'match_index': 88, 'instances': 1, 'time_in': 0.0004087090492248535}
        """
        if opt_stats is None:
            return dict(onnx_optimized=0)
        new_stat = {}
        if "optimization" in opt_stats:
            time_in = 0.0
            added = 0
            removed = 0
            max_iter = 0
            applied = set()
            matched = set()
            n_applied = 0
            for obs in opt_stats["optimization"]:
                time_in += obs.get("time_in", 0)
                added += obs.get("added", 0)
                removed += obs.get("removed", 0)
                max_iter = max(max_iter, obs.get("iteration", 0))
                p = obs["pattern"]
                if p.startswith("match_"):
                    matched.add(p)
                elif p.startswith("apply_"):
                    applied.add(p)
                    n_applied += 1
            new_stat.update(
                dict(
                    onnx_optimized=1,
                    onnx_opt_time_in=time_in,
                    onnx_opt_added=added,
                    onnx_opt_removed=removed,
                    onnx_opt_max_iter=max_iter,
                    onnx_opt_unique_matched=len(matched),
                    onnx_opt_unique_applied=len(applied),
                    onnx_opt_n_applied=n_applied,
                )
            )
        if "builder" in opt_stats:
            builder = opt_stats["builder"]
            if "aten" in builder:
                new_stat.update(
                    {f"op_torch_{k}": v for k, v in builder["aten"].items()}
                )
        return new_stat

    @classmethod
    def _post_process_onnx_statistics(cls, model: onnx.ModelProto) -> Dict[str, Any]:
        stats = {}
        stats["onnx_n_nodes"] = len(model.graph.node)
        stats["onnx_n_functions"] = len(model.functions)
        stats["onnx_n_sequence"] = len(
            [n for n in model.graph.node if n.op_type == "Sequence"]
        )
        stats["onnx_n_inputs"] = len(model.graph.input)
        stats["onnx_n_outputs"] = len(model.graph.output)
        stats["onnx_input_names"] = "|".join(i.name for i in model.graph.input)
        stats["onnx_output_names"] = "|".join(i.name for i in model.graph.output)
        for node in model.graph.node:
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
                rdiff = diff / (expected.abs() + 1e-3)
                abs_diff, rel_diff = float(diff.max()), float(rdiff.max())
                if self.verbose >= 10 and (abs_diff >= 10 or rel_diff >= 10):
                    # To understand the value if comes from.
                    print(
                        f"[max_diff] abs_diff={abs_diff}, rel_diff={rel_diff}, "
                        f"dtype={expected.dtype}, shape={expected.shape}"
                    )
                    if abs_diff >= 10:
                        idiff = torch.argmax(diff.reshape((-1,)))
                        x = expected.reshape((-1,))[idiff]
                        y = got.reshape((-1,))[idiff]
                        print(f"   [max_diff] abs diff={abs_diff}, x={x}, y={y}")

                    if rel_diff >= 10:
                        idiff = torch.argmax(rdiff.reshape((-1,)))
                        x = expected.reshape((-1,))[idiff]
                        y = got.reshape((-1,))[idiff]
                        print(f"   [max_diff] rel diff={rel_diff}, x={x}, y={y}")

                return abs_diff, rel_diff

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


def merge_benchmark_reports(
    data: List[str],
    model="model_name",
    keys=(
        "suite",
        "exporter",
        "opt_patterns",
        "device",
        "dtype",
        "dynamic",
        "version",
        "version_onnxruntime",
        "version_torch",
        "version_transformers",
        "flag_fake_tensor",
        "flag_no_grad",
        "flag_training",
        "machine",
    ),
    report_on=(
        "speedup",
        "speedup_increase",
        "discrepancies_*",
        "TIME_ITER",
        "time_*",
        "ERR_*",
        "onnx_*",
        "op_*",
        "memory_*",
    ),
    formulas=("memory_peak_load", "buckets", "status"),
    excel_output: Optional[str] = None,
) -> Dict[str, "pandas.DataFrame"]:  # noqa: F821
    """
    Merges multiple files produced by bash_benchmark...

    ::

        _index,DATE,ERR_export,ITER,TIME_ITER,capability,cpu,date_start,device,device_name,...
        101Dummy-custom,2024-07-08,,0,7.119158490095288,7.0,40,2024-07-08,cuda,...
        101Dummy-script,2024-07-08,,1,6.705480073112994,7.0,40,2024-07-08,cuda,...
        101Dummy16-custom,2024-07-08,,2,6.970448340754956,7.0,40,2024-07-08,cuda,...

    Every key with a unique value is removed.
    Every column with a unique value is displayed on main.
    List of knowns columns::

        DATE
        ERR_export
        ERR_warmup
        ITER
        TIME_ITER
        capability
        cpu
        date_start
        device
        device_name
        discrepancies_abs
        discrepancies_rel
        dtype
        dump_folder
        dynamic
        executable
        exporter
        filename
        flag_fake_tensor
        flag_no_grad
        flag_training
        has_cuda
        input_size
        machine
        mema_gpu_0_before_loading
        mema_gpu_1_after_loading
        mema_gpu_2_after_warmup
        mema_gpu_3_empty_cache
        mema_gpu_4_after_repeat
        mema_gpu_5_after_export
        mema_gpu_6_after_gcollect
        mema_gpu_7_after_session
        mema_gpu_8_after_export_warmup
        mema_gpu_9_after_export_repeat
        memory_begin,
        memory_end,
        memory_gpu0_begin,
        memory_gpu0_end,
        memory_gpu0_mean,
        memory_gpu0_n,
        memory_gpu0_peak,
        memory_mean,
        memory_n,
        memory_peak,
        model,
        model_name,
        onnx_filesize
        onnx_input_names,
        onnx_model
        onnx_n_inputs,
        onnx_n_outputs,
        onnx_optimized,
        onnx_output_names,
        opt_patterns,
        output_data,
        output_size,
        params_dtype,
        params_size,
        process,
        processor,
        providers,
        quiet,
        repeat,
        speedup,
        speedup_increase,
        target_opset,
        time_export,
        time_load,
        time_latency,
        time_latency_eager,
        time_session,
        time_total,
        time_warmup,
        time_warmup_eager,
        verbose,
        version,
        version_onnxruntime,
        version_torch,
        version_transformers,
        warmup,
        ERROR,
        OUTPUT,
        CMD
    """
    import pandas

    dfs = []
    for filename in data:
        df = pandas.read_csv(filename)
        dfs.append(df)

    df = pandas.concat(dfs, axis=0)

    # replace nan values
    set_columns = set(df.columns)
    for c in ["opt_patterns", "ERR_export", "ERR_warmup"]:
        if c in set_columns:
            df[c] = df[c].fillna("-")

    res = {"0raw": df}

    # uniques keys
    report_on_star = [r for r in report_on if "*" in r]
    report_on = [r for r in report_on if "*" not in r]
    keys = [k for k in keys if k in set_columns]
    report_on = [k for k in report_on if k in set_columns]
    for col in sorted(set_columns):
        for star in report_on_star:
            assert star[-1] == "*", f"Unsupported position for * in {star!r}"
            s = star[:-1]
            if col.startswith(s):
                report_on.append(col)
                break

    unique = {}
    for c in df.columns:
        u = df[c].unique()
        if len(u) == 1:
            unique[c] = u.tolist()[0]

    main = [dict(column="dates", value=", ".join(sorted(df["DATE"].unique().tolist())))]
    for k, v in unique.items():
        main.append(dict(column=k, value=v))
    res["0main"] = pandas.DataFrame(main)
    new_keys = [k for k in keys if k not in unique]

    # formulas
    bucket_columns = []
    for expr in formulas:
        if expr == "memory_peak_load":
            if (
                "mema_gpu_5_after_export" in set_columns
                and "mema_gpu_1_after_loading" in set_columns
            ):
                df[expr] = (
                    df["mema_gpu_5_after_export"] - df["mema_gpu_1_after_loading"]
                )
                report_on.append(expr)
            continue

        if expr == "status":
            if "discrepancies_abs" in set_columns:
                df["status_convert"] = (~df["discrepancies_abs"].isna()).astype(int)
                df["status_err<1e-2"] = (
                    ~df["discrepancies_abs"].isna() & (df["discrepancies_abs"] < 1e-2)
                ).astype(int)
                df["status_err<1e-3"] = (
                    ~df["discrepancies_abs"].isna() & (df["discrepancies_abs"] < 1e-3)
                ).astype(int)
                df["status_err<1e-4"] = (
                    ~df["discrepancies_abs"].isna() & (df["discrepancies_abs"] < 1e-4)
                ).astype(int)
                df["status_<=eager+2%"] = (
                    ~df["discrepancies_abs"].isna()
                    & (df["time_latency"] <= df["time_latency_eager"] * 1.02)
                ).astype(int)
                report_on.extend(
                    [
                        "status_convert",
                        "status_err<1e-2",
                        "status_err<1e-3",
                        "status_err<1e-4",
                        "status_lat<=eager+2%",
                    ]
                )
            continue

        if expr == "buckets":
            if "exporter" in set_columns and "script" in set(df.exporter):
                # Do the same with the exporter as a baseline.
                keep = [model, *new_keys, "speedup"]
                gr = df[df.exporter == "script"][keep].copy()
                gr["speedup_script"] = gr["speedup"]
                gr = gr.drop("speedup", axis=1)
                on = [k for k in keep[:-1] if k != "exporter"]
                joined = pandas.merge(df, gr, left_on=on, right_on=on, how="left")
                joined["speedup_increase_script"] = (
                    joined["speedup_script"] / joined["speedup"] - 1
                ).fillna(np.inf)
                assert df.shape[0] == joined.shape[0]
                df = joined.drop("exporter_y", axis=1).copy()
                df["exporter"] = df["exporter_x"]
                df = df.drop("exporter_x", axis=1)
                set_columns = set(df.columns)
                df["status_<=script+2%"] = (
                    df["speedup_increase_script"] >= 1 / 1.02
                ).astype(int)
                report_on.append("status_lat<=script+2%")

            for c in ["speedup_increase", "speedup_increase_script"]:
                if c not in set_columns:
                    continue
                scale = [-np.inf, -20, -10, -5, -2, 2, 5, 10, 20, np.inf]
                for i in range(1, len(scale)):
                    val = (df[c] >= scale[i - 1] / 100) & (df[c] < scale[i] / 100)
                    v1 = f"{scale[i-1]}%" if not np.isinf(scale[i - 1]) else ""
                    v2 = f"{scale[i]}%" if not np.isinf(scale[i]) else ""
                    d = f"[{v1},{v2}[" if v1 and v2 else (f"<{v2}" if v2 else f">={v1}")
                    if c == "speedup_increase_script":
                        d = f"script {d}"
                    bucket_columns.append(d)
                    df[d] = val.astype(int)

            continue

    # values
    for c in report_on:
        keep = [model, *new_keys, c]
        dfc = df[keep]
        pivot = dfc.pivot(index=model, columns=new_keys, values=c)
        res[c] = pivot

    # buckets
    if bucket_columns:
        table = df[[*new_keys, model, *bucket_columns, "speedup_increase"]].copy()
        pivot = table.pivot(
            index=[*[c for c in new_keys if c != "exporter"], model],
            columns=["exporter"],
            values=bucket_columns,
        )

        # the following code switches places between exporter and buckets
        tpiv = pivot.T.reset_index(drop=False)

        def _order(index):
            if index.name == "exporter":
                return index
            order = [
                "<-20%",
                "[-20%,-10%[",
                "[-10%,-5%[",
                "[-5%,-2%[",
                "[-2%,2%[",
                "[2%,5%[",
                "[5%,10%[",
                "[10%,20%[",
                ">=20%",
                "script <-20%",
                "script [-20%,-10%[",
                "script [-10%,-5%[",
                "script [-5%,-2%[",
                "script [-2%,2%[",
                "script [2%,5%[",
                "script [5%,10%[",
                "script [10%,20%[",
                "script >=20%",
            ]
            position = {v: i for i, v in enumerate(order)}
            return [position[s] for s in index]

        tpiv1 = tpiv[~tpiv.level_0.str.startswith("script")]
        tpiv2 = tpiv[tpiv.level_0.str.startswith("script")].copy()
        tpiv1 = tpiv1.set_index(["exporter", "level_0"]).sort_index(key=_order).T.copy()
        tpiv1.loc["SUM"] = tpiv1.sum(axis=0).copy()
        res["bucket"] = tpiv1.fillna(0).astype(int)

        if tpiv2.shape[0] > 0:
            tpiv2["level_0"] = tpiv2["level_0"].apply(lambda s: s[len("script ") :])
            tpiv2 = (
                tpiv2.set_index(["exporter", "level_0"]).sort_index(key=_order).T.copy()
            )
            tpiv2.loc["SUM"] = tpiv2.sum(axis=0).copy()
            res["bucket_script"] = tpiv2.fillna(0).astype(int)

    # add summary at the end
    mean_med = [
        c
        for c in res
        if c.startswith("time_")
        or c.startswith("onnx_")
        or c.startswith("op_")
        or c.startswith("discrepancies_")
        or c.startswith("status_")
    ]
    for c in [*mean_med, "speedup_increase"]:
        if c in res and set(res[c].dtypes) in {np.float64, np.float32}:
            summary = res[c].mean(axis=0).copy()
            med = res[c].median(axis=0)
            res[c].loc["MEAN"] = summary
            res[c].loc["MED"] = med
    for c in ["speedup"]:
        if c in res:
            summary = np.exp(np.log(res[c]).mean(axis=0))
            med = res[c].median(axis=0)
            res[c].loc["GMEAN"] = summary
            res[c].loc["MED"] = med

    # final fusion

    def _merge(res, merge, prefix, reverse=True):
        m = None
        for name in merge:
            df = res[name].T
            cols = set(df.columns)
            df = df.reset_index(drop=False).copy()
            index_cols = set(df.columns) - cols
            df["stat"] = name[len(prefix) :]
            df = df.set_index([*list(index_cols), "stat"]).T
            if m is None:
                m = df
                continue
            m = pandas.merge(m, df, how="outer", left_index=True, right_index=True)

        # We need to change the columns index order.
        if reverse:
            df = m.T
            setc = set(df.columns)
            df = df.reset_index(drop=False)
            index = set(df.columns) - setc
            if index == {"stat", "exporter"}:
                m = df.set_index(["stat", "exporter"]).T
        else:
            m = m.T.sort_index().T
        return m

    for prefix in [
        "status_",
        "discrepancies_",
        "memory_",
        "onnx_",
        "time_",
        "ERR_",
        "op_onnx_",
        "op_torch_",
    ]:
        merge = [k for k in res if k.startswith(prefix)]
        if len(merge) == 0:
            continue
        res[prefix[:-1]] = _merge(res, merge, prefix, reverse=prefix != "status_")
        res = {k: v for k, v in res.items() if k not in set(merge)}

    if excel_output:
        with pandas.ExcelWriter(excel_output) as writer:
            for k, v in sorted(res.items()):
                v.to_excel(writer, sheet_name=k, index=k not in {"0raw", "0main"})
    return res
