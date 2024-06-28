import time
import os
from typing import Any, Dict, List
import numpy as np
import onnx
from onnx import ModelProto
from ..convert.convert_helper import optimize_model_proto
from ..bench_run import measure_discrepancies


def common_export(
    model: Any,
    inputs: List[Any],
    exporter: str = "dynamo",
    target_opset: int = 18,
    folder: str = "",
    filename: str = "model.onnx",
    dynamic_shapes: Any | None = None,
    verbose: int = 0,
    optimization: str | None = None,
    stats: Dict[str, Any] | None = None,
):
    """
    Exports a model into a folder.

    :param model: model
    :param exporter: script, dynamo
    :param folder: folder to export into
    :param filename: onnx filename
    :param inputs: inputs
    :param dynamic_shapes: dynamic shapes
    :param target_opset: target opset
    :param optimization: optimization scenario, '/' separated values
    :param verbose: verbosity
    :param stats: if not None, populates this
        dictionary with statistics about time
    :returns: onnx proto
    """
    import torch.onnx

    if folder:
        if not os.path.exists(folder):
            os.mkdir(folder)
        filename = os.path.join(folder, filename)

    if verbose:
        print(f"[common_export] start exporting with {exporter!r} in {filename!r}")
    begin = time.perf_counter()
    if exporter == "script":
        torch.onnx.export(
            model,
            inputs,
            filename,
            do_constant_folding=False,
            input_names=[f"input{i}" for i in range(len(inputs))],
            opset_version=target_opset,
            dynamic_axes=dynamic_shapes,
        )
    elif exporter == "dynamo":
        assert (
            dynamic_shapes is None
        ), f"dynamic_shapes={dynamic_shapes} is not implemented yet"
        torch.onnx.export(
            model,
            inputs,
            filename,
            do_constant_folding=False,
            input_names=[f"input{i}" for i in range(len(inputs))],
            opset_version=target_opset,
            dynamic_axes=dynamic_shapes,
            dynamo=True,
        )
    elif exporter == "dynamo_export":
        with torch.no_grad():
            prog = torch.onnx.dynamo_export(model, *inputs)
        onnx.save(prog.model_proto, filename)
    else:
        raise ValueError(f"Unknown exporter {exporter!r}")

    if stats is not None:
        stats["export_time"] = time.perf_counter() - begin
        stats["filesize"] = os.stat(filename).st_size

    if verbose:
        print(f"[common_export] exporter done in {time.perf_counter() - begin}s")
        print(
            f"[common_export] size of the export: {os.stat(filename).st_size / 2**20} Mb"
        )

    with open(filename, "rb") as f:
        onx = onnx.load(f)

    if optimization:
        if verbose:
            print(f"[common_export] start optimization with {optimization!r}")
        begin = time.perf_counter()
        optimized_model = optimize_model_proto(
            onx, optimization, verbose=verbose, stats=stats
        )
        end = time.perf_counter() - begin
        if stats is not None:
            stats["optimization_time"] = end
        if verbose:
            print(f"[common_export] optimization done in {end}")
            print(f"[common_export] saves the model in {filename!r}")
            begin = time.perf_counter()

        onnx.save(optimized_model, filename)
        if verbose:
            print(f"[common_export] done saving in {time.perf_counter() - begin}")

    return onx


def run_inference(
    model: Any,
    example_inputs: List[Any],
    warmup: int = 5,
    repeat: int = 5,
    verbose: int = 0,
) -> dict[str, Any]:
    """
    Runs multiple times the same inference.

    Args:
        model: torch model to run
        example_inputs: dummy inputs
        warmup: number of iterations to warmup
        repeat: number of iterations to repeat
        verbose: verbosity

    Returns:
        statistcs
    """
    if verbose:
        print(f"[run_inference] start {warmup} warmup iterations")

    stats: dict[str, Any] = {}
    iterations: list[float] = []
    begin = time.perf_counter()
    for i in range(warmup):
        t0 = time.perf_counter()
        model(*example_inputs[i % len(example_inputs)])
        iterations.append(time.perf_counter() - t0)
    end = time.perf_counter() - begin
    stats["warmup"] = warmup
    stats["warmup_time"] = end
    stats["warmup_iter"] = iterations

    if verbose:
        print(f"[run_inference] warmup done in {time.perf_counter() - begin}")
        print(f"[run_inference] start {repeat} iterations")

    iterations = []
    begin = time.perf_counter()
    for i in range(warmup):
        t0 = time.perf_counter()
        model(*example_inputs[i % len(example_inputs)])
        iterations.append(time.perf_counter() - t0)
    end = time.perf_counter() - begin
    stats["repeat"] = repeat
    stats["repeat_time"] = end
    stats["repeat_iter"] = iterations

    if verbose:
        print(f"[run_inference] measure done in {time.perf_counter() - begin}")

    return stats


class WrapInferenceSessionForTorch:
    def __init__(self, sess: Any):
        # onnxruntime is importing when needed as it takes a couple of seconds if it contains CUDA EP.
        import onnxruntime
        import torch
        from onnxruntime.capi import _pybind_state as ORTC  # noqa: N812

        self.sess = sess
        self.input_names = [i.name for i in sess.get_inputs()]
        self.output_names = [i.name for i in sess.get_outputs()]
        self.bind = onnxruntime.SessionIOBinding(sess._sess)
        self.OrtValue = ORTC.OrtValue
        self.ORTC = ORTC
        self.torch = torch
        self.run_options = onnxruntime.RunOptions()

        self.TORCH_DTYPE_TO_NUMPY_DTYPE = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.bool: np.bool_,
        }

        DEVICES = {
            -1: ORTC.OrtDevice(ORTC.OrtDevice.cpu(), ORTC.OrtDevice.default_memory(), 0)
        }

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                DEVICES[i] = ORTC.OrtDevice(
                    ORTC.OrtDevice.cuda(), ORTC.OrtDevice.default_memory(), i
                )

        self.DEVICES = DEVICES

    def _get_ortvalues_from_torch_tensors(
        self,
        tensors: tuple[Any, ...],  # tuple["torch.Tensor", ...],
        n_outputs: int,
    ) -> tuple[Any, Any]:  # tuple[tuple["torch.Tensor", ...], tuple["OrtDevice", ...]]:
        ortvalues = self.ORTC.OrtValueVector()
        ortvalues.reserve(len(tensors))
        dtypes = []
        shapes = []
        data_ptrs = []
        devices = []

        max_device = -1
        assert isinstance(max_device, int), f"unexpected type for device={max_device!r}"
        assert tensors is not None, "tensors cannot be None"
        new_tensors = []
        for tensor in tensors:
            assert isinstance(
                tensor, self.torch.Tensor
            ), f"Unexpected type {type(tensor)}"
            dtypes.append(self.TORCH_DTYPE_TO_NUMPY_DTYPE[tensor.dtype])
            shapes.append(tensor.size())
            data_ptrs.append(tensor.data_ptr())
            d = tensor.get_device()
            devices.append(self.DEVICES[d])
            new_tensors.append(tensor)
            max_device = max(max_device, tensor.get_device())

        ortvalues.push_back_batch(new_tensors, data_ptrs, dtypes, shapes, devices)
        output_devices = []
        for _ in range(n_outputs):
            dev = self.DEVICES[max_device]
            output_devices.append(dev)

        return ortvalues, output_devices

    def _ortvalues_to_torch_tensor(
        self,
        ortvalues: Any,  #  "onnxruntime.OrtValueVector",
    ) -> tuple[Any, ...]:  # tuple["torch.Tensor", ...]:
        if len(ortvalues) == 0:
            return tuple()

        from torch._C import _from_dlpack

        if all(
            map(lambda i: ortvalues[i].has_value(), range(len(ortvalues)))
        ):  # noqa: C417
            res = ortvalues.to_dlpacks(_from_dlpack)
        else:
            res = []
            for i in range(len(ortvalues)):
                res.append(
                    _from_dlpack(ortvalues[i].to_dlpack())
                    if ortvalues[i].has_value()
                    else None
                )
        return tuple(res)

    def run(self, output_names, feeds):
        inputs = [feeds[i] for i in self.input_names]
        return self.run_dlpack(*inputs, output_names=output_names)

    def run_dlpack(self, *inputs, output_names=None):
        if output_names is None:
            output_names = self.output_names
        ortvalues, output_devices = self._get_ortvalues_from_torch_tensors(
            inputs, len(output_names)
        )

        ort_outputs = self.ORTC.OrtValueVector()
        self.sess.run_with_ortvaluevector(
            self.run_options,
            self.input_names,
            ortvalues,
            output_names,
            ort_outputs,
            output_devices,
        )
        pth_outputs = self._ortvalues_to_torch_tensor(ort_outputs)
        return pth_outputs


def run_onnx_inference(
    model: ModelProto,
    example_inputs: List[Any],
    warmup: int = 5,
    repeat: int = 5,
    verbose: int = 0,
    ort_optimize: bool = True,
    torch_model: Any | None = None,
) -> dict[str, Any]:
    """
    Runs multiple times the same inference with onnxruntime.

    Args:
        model: torch model to run
        example_inputs: dummy inputs
        warmup: number of iterations to warmup
        repeat: number of iterations to repeat
        verbose: verbosity
        ort_optimize: enable, disable onnxruntime optimizations
        torch_model: if not empty, measure the discrepancies

    Returns:
        statistcs
    """
    stats: dict[str, Any] = {}
    device = example_inputs[0][0].get_device()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device >= 0
        else ["CPUExecutionProvider"]
    )
    stats["providers"] = ",".join(providers)
    if verbose:
        print(f"[run_inference] create session with providers {providers!r}")

    begin = time.perf_counter()
    # onnxruntime is importing when needed as it takes a couple of seconds if it contains CUDA EP.
    import onnxruntime

    so = onnxruntime.SessionOptions()
    if ort_optimize:
        so.add_session_config_entry("session.disable_aot_function_inlining", "0")
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        so.add_session_config_entry("session.disable_aot_function_inlining", "1")
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    sess = onnxruntime.InferenceSession(model.SerializeToString(), so, providers)
    wrapped_session = WrapInferenceSessionForTorch(sess)

    end = time.perf_counter() - begin
    stats["ort_session_create_time"] = end
    if verbose:
        print(f"[run_inference] created session in {end}")
        print(f"[run_inference] start {warmup} warmup iterations")

    if torch_model:
        expected = [
            torch_model(*example_inputs[i % len(example_inputs)]) for i in range(warmup)
        ]

    got = []
    iterations = []
    begin = time.perf_counter()
    for i in range(warmup):
        t0 = time.perf_counter()
        got.append(wrapped_session.run_dlpack(*example_inputs[i % len(example_inputs)]))
        iterations.append(time.perf_counter() - t0)
    end = time.perf_counter() - begin
    stats["warmup"] = warmup
    stats["warmup_time"] = end / warmup
    stats["warmup_iter"] = iterations
    if torch_model:
        abs_err, rel_err = measure_discrepancies(expected, got)
        stats["discrepancies_abs"] = abs_err
        stats["discrepancies_rel"] = rel_err

    if verbose:
        print(f"[run_inference] warmup done in {time.perf_counter() - begin}")
        print(f"[run_inference] start {repeat} iterations")

    iterations = []
    begin = time.perf_counter()
    for i in range(repeat):
        t0 = time.perf_counter()
        wrapped_session.run_dlpack(*example_inputs[i % len(example_inputs)])
        iterations.append(time.perf_counter() - t0)
    end = time.perf_counter() - begin
    stats["repeat"] = repeat
    stats["repeat_time"] = end / repeat
    stats["repeat_iter"] = iterations

    if verbose:
        print(f"[run_inference] measure done in {time.perf_counter() - begin}")

    return stats
