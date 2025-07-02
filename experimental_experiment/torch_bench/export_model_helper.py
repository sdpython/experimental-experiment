import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
from onnx_diagnostic.helpers import max_diff
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
)
from ..convert.convert_helper import optimize_model_proto_oxs
from ..torch_interpreter import DEFAULT_TARGET_OPSET


def size_type(dtype: Any) -> int:
    if isinstance(dtype, int):
        # It is a TensorProto.DATATYPE
        if dtype in {TensorProto.DOUBLE, TensorProto.INT64, TensorProto.UINT64}:
            return 8
        if dtype in {TensorProto.FLOAT, TensorProto.INT32, TensorProto.UINT32}:
            return 4
        if dtype in {
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
            TensorProto.INT16,
            TensorProto.UINT16,
        }:
            return 2
        if dtype in {TensorProto.INT8, TensorProto.UINT8, TensorProto.BOOL}:
            return 1
        raise AssertionError(f"Unable to return the element size for type {dtype}")

    import torch

    if dtype in {torch.float64, torch.int64}:
        return 8
    if dtype in {torch.float32, torch.int32}:
        return 4
    if dtype in {torch.float16, torch.int16, torch.bfloat16}:
        return 2
    if dtype in {torch.int8, torch.uint8, torch.bool}:
        return 1
    if hasattr(torch, "uint64"):
        # it fails on mac
        if dtype in {torch.uint64}:
            return 8
    if hasattr(torch, "uint32"):
        # it fails on mac
        if dtype in {torch.uint32}:
            return 4
    if hasattr(torch, "uint16"):
        # it fails on mac
        if dtype in {torch.uint16}:
            return 2
    raise AssertionError(f"Unexpected dtype={dtype}")


def obj_size(obj: Any) -> int:
    import torch

    if isinstance(obj, torch.Tensor):
        assert not obj.is_sparse, "Sparse tensors not supported yet."
        return int(np.prod(list(obj.shape)) * size_type(obj.dtype))
    if isinstance(obj, (tuple, list)):
        return sum(obj_size(o) for o in obj)
    if isinstance(obj, dict):
        return sum(obj_size(o) for o in obj.values())
    if obj is None:
        return 0
    if obj.__class__.__name__.endswith("KeyedJaggedTensor"):
        # Not implemented yet.
        return 0
    if isinstance(obj, (int, float, str, bytes)):
        return sys.getsizeof(obj)
    if hasattr(obj, "_fields") and isinstance(obj._fields, dict):
        # detectron2.structures.instances.Instances
        return obj_size(obj._fields)
    if hasattr(obj, "tensor") and isinstance(obj.tensor, torch.Tensor):
        # detectron2.structures.instances.Bowes
        return obj_size(obj.tensor)
    if "SquashedNormal" in obj.__class__.__name__:
        return sys.getsizeof(obj)
    if obj.__class__.__name__ == "MambaCache":
        return obj_size(obj.conv_states) + obj_size(obj.ssm_states)
    if obj.__class__.__name__ == "DynamicCache":
        return obj_size(obj.key_cache) + obj_size(obj.value_cache)
    if obj.__class__.__name__ == "EncoderDecoderCache":
        return obj_size(obj.self_attention_cache) + obj_size(obj.cross_attention_cache)
    raise AssertionError(f"input_size not implemented for type {type(obj)}")


def compute_weight_size(model: Any) -> int:
    """
    Returns the model size for a torch model or an onnx model.
    That includes the weights.
    """
    if isinstance(model, ModelProto):
        size = compute_weight_size(model.graph)
        for f in model.functions:
            size += compute_weight_size(f)
        assert isinstance(size, int), f"Unexpected type {type(size)}: {size}"
        return size

    if isinstance(model, GraphProto):
        size = 0
        for init in model.initializer:
            size += compute_weight_size(init)
        for init in model.sparse_initializer:
            size += compute_weight_size(init)
        for node in model.node:
            size += compute_weight_size(node)
        assert isinstance(size, int), f"Unexpected type {type(size)}: {size}"
        return size

    if isinstance(model, FunctionProto):
        size = 0
        for node in model.node:
            size += compute_weight_size(node)
        assert isinstance(size, int), f"Unexpected type {type(size)}: {size}"
        return size

    if isinstance(model, TensorProto):
        p = int(np.prod(model.dims))
        size = p * size_type(model.data_type)
        assert isinstance(size, int), f"Unexpected type {type(size)}: {size}"
        return size

    if isinstance(model, NodeProto):
        if model.op_type == "Constant":
            return len(model.SerializeToString())
        size = 0
        for att in model.attribute:
            if att.type == AttributeProto.GRAPH:
                size += compute_weight_size(att.g)
        assert isinstance(size, int), f"Unexpected type {type(size)}: {size}"
        return size

    if hasattr(model, "parameters"):
        import torch

        size = 0
        for p in model.parameters():
            assert isinstance(p, torch.Tensor), f"Unexpected type {type(p)}"
            size += obj_size(p)
        assert isinstance(size, int), f"Unexpected type {type(size)}: {size}"
        return size

    if hasattr(model, "buffer"):
        # executorch
        return len(model.buffer)

    return np.nan


def common_export(
    model: Any,
    inputs: List[Any],
    exporter: str = "custom",
    target_opset: int = DEFAULT_TARGET_OPSET,
    folder: str = "",
    filename: str = "model.onnx",
    dynamic_shapes: Optional[Any] = None,
    verbose: int = 0,
    optimize_oxs: str = "",
    ort_optimize: bool = False,
    large_model: bool = False,
    order: bool = False,
    enable_pattern: Optional[Union[str, List[str]]] = None,
    disable_pattern: Optional[Union[str, List[str]]] = None,
    stats: Optional[Dict[str, Any]] = None,
):
    """
    Exports a model into a folder.

    :param model: model
    :param exporter: torchscript, onnx_dynamo, custom, ...
    :param folder: folder to export into
    :param filename: onnx filename
    :param inputs: inputs
    :param dynamic_shapes: dynamic shapes
    :param target_opset: target opset
    :param optimize_oxs: run optimization with onnxscript
    :param enable_pattern: patterns to apply
    :param disable_pattern: patterns not to apply
    :param verbose: verbosity
    :param stats: if not None, populates this
        dictionary with statistics about time
    :param optimize_oxs: optimize
    :param ort_optimize: produces a file showing onnxruntime optimizations
    :param large_model: save weights as external
    :param order: optimize order
    :returns: onnx proto
    """
    import torch.onnx

    if folder:
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, filename)

    if verbose:
        print(
            f"[common_export] start exporting with {exporter!r}, "
            f"{len(inputs)} inputs in {filename!r}"
        )
    begin = time.perf_counter()
    if exporter == "torch_script":
        assert isinstance(inputs, tuple), f"{type(inputs)}"
        assert len(inputs) == 2
        torch.onnx.export(
            model,
            inputs,
            filename,
            do_constant_folding=False,
            input_names=[f"input{i}" for i in range(len(inputs))],
            opset_version=target_opset,
            dynamic_axes=dynamic_shapes,
        )
    elif exporter == "onnx_dynamo":
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
    elif exporter in ("custom", "custom-fallback"):
        from ..torch_interpreter import to_onnx
        from ..xbuilder import OptimizationOptions
        from ..xoptim import get_pattern_list

        patterns = get_pattern_list(enable_pattern, disable_pattern, verbose=verbose)
        onx = to_onnx(
            model,
            inputs,
            input_names=[f"input{i}" for i in range(len(inputs))],
            options=OptimizationOptions(patterns=patterns, order=order),
            verbose=verbose,
            target_opset=target_opset,
            optimize=bool(enable_pattern),
            large_model=large_model,
        )
        with open(filename, "wb") as f:
            f.write(onx.SerializeToString())
    else:
        raise ValueError(f"Unknown exporter {exporter!r}")

    if stats is not None:
        stats["export_time"] = time.perf_counter() - begin
        stats["filesize"] = os.stat(filename).st_size

    if verbose:
        print(f"[common_export] exporter done in {time.perf_counter() - begin}s")
        print(f"[common_export] size of the export: {os.stat(filename).st_size / 2**20} Mb")

    with open(filename, "rb") as f:
        onx = onnx.load(f)

    if optimize_oxs:
        if verbose:
            print("[common_export] start optimization with onnxscript")
        begin = time.perf_counter()
        optimized_model = optimize_model_proto_oxs(onx, verbose=verbose, stats=stats)
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

    if ort_optimize and filename:
        output = f"{filename}.opt.onnx"
        if verbose:
            print(f"[common_export] onnxruntime optimize in {output!r}")
        from ..convert.convert_helper import ort_optimize as fopt

        is_cuda = next(model.parameters()).is_cuda
        if is_cuda:
            device_id = next(model.parameters()).get_device()
            providers = [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                ("CPUExecutionProvider", {}),
            ]
        else:
            providers = ["CPUExecutionProvider"]

        fopt(onx, output, providers=providers, disable_aot=False)
        if verbose:
            print("[common_export] done")

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


class WrapForTorch:
    """Wraps a torch model."""

    def __init__(self, torch_model: Any):
        if hasattr(torch_model, "graph_module"):
            self.model = torch_model.graph_module
        else:
            self.model = torch_model

    def run(self, inputs):
        if isinstance(inputs, dict):
            return self.model(**inputs)
        return self.model(*inputs)

    @property
    def input_names(self):
        res = []
        for node in self.model.graph.nodes:
            if node.op == "placeholder":
                res.append(node.target)
        return res


class WrapExecutorchForTorch(WrapForTorch):
    """Wraps a executorch model."""

    def __init__(self, model: Any, forward_method: Callable):
        self.model = model
        self.forward_method = forward_method

    def run(self, inputs):
        return self.forward_method.execute(inputs)

    @property
    def input_names(self):
        raise NotImplementedError(f"Not implemented yet {dir(self.model)}.")
        res = []
        for node in self.model.graph.nodes:
            if node.op == "placeholder":
                res.append(node.target)
        return res


class WrapInferenceSessionForTorch:
    """
    Wraps an `onnxruntime.InferenceSession` to overload method `run`
    to support :class:`torch.Tensor`.
    """

    def __init__(self, sess: Any, nvtx: bool = False):
        # onnxruntime is importing when needed as it takes a
        # couple of seconds if it contains CUDA EP.
        import onnxruntime
        import torch
        from onnxruntime.capi import _pybind_state as ORTC  # noqa: N812

        self.sess = sess
        self.input_names = [i.name for i in sess.get_inputs()]
        self.output_names = [i.name for i in sess.get_outputs()]
        self.OrtValue = ORTC.OrtValue
        self.ORTC = ORTC
        self.torch = torch
        self.nvtx = nvtx
        self.run_options = onnxruntime.RunOptions()
        self.dlpack = False

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

        self.TORCH_DTYPE_TO_ONNX_DTYPE = {
            torch.float16: TensorProto.FLOAT16,
            torch.bfloat16: TensorProto.BFLOAT16,
            torch.float32: TensorProto.FLOAT,
            torch.float64: TensorProto.DOUBLE,
            torch.uint8: TensorProto.UINT8,
            torch.int8: TensorProto.INT8,
            torch.int16: TensorProto.INT16,
            torch.int32: TensorProto.INT32,
            torch.int64: TensorProto.INT64,
            torch.bool: TensorProto.BOOL,
        }

        DEVICES = {
            -1: ORTC.OrtDevice(ORTC.OrtDevice.cpu(), ORTC.OrtDevice.default_memory(), 0)
        }

        if torch.cuda.device_count() > 0:
            for i in range(torch.cuda.device_count()):
                DEVICES[i] = ORTC.OrtDevice(
                    ORTC.OrtDevice.cuda(), ORTC.OrtDevice.default_memory(), i
                )

        self.DEVICES = DEVICES
        self.use_run = not isinstance(sess, onnxruntime.InferenceSession)

    def _get_ortvalues_from_torch_tensors(
        self,
        tensors: tuple[Any, ...],  # tuple["torch.Tensor", ...],
        n_outputs: int,
    ) -> tuple[Any, Any]:  # tuple[tuple["torch.Tensor", ...], tuple["OrtDevice", ...]]:
        assert tensors is not None, "tensors cannot be None"
        ortvalues = self.ORTC.OrtValueVector()
        ortvalues.reserve(len(tensors))
        dtypes = []
        shapes = []
        data_ptrs = []
        devices = []

        if self.nvtx:
            self.torch.cuda.nvtx.range_push("_get_ortvalues_from_torch_tensors.1")
        max_device = -1
        new_tensors = []
        for tensor in tensors:
            assert isinstance(tensor, self.torch.Tensor), f"Unexpected type {type(tensor)}"
            dtypes.append(self.TORCH_DTYPE_TO_NUMPY_DTYPE[tensor.dtype])
            shapes.append(tensor.size())
            data_ptrs.append(tensor.data_ptr())
            d = tensor.get_device()
            devices.append(self.DEVICES[d])
            new_tensors.append(tensor)
            max_device = max(max_device, d)

        if self.nvtx:
            self.torch.cuda.nvtx.range_pop()
            self.torch.cuda.nvtx.range_push("_get_ortvalues_from_torch_tensors.2")

        assert isinstance(max_device, int), f"unexpected type for device={max_device!r}"
        ortvalues.push_back_batch(new_tensors, data_ptrs, dtypes, shapes, devices)
        output_devices = []
        for _ in range(n_outputs):
            dev = self.DEVICES[max_device]
            output_devices.append(dev)

        if self.nvtx:
            self.torch.cuda.nvtx.range_pop()
        return ortvalues, output_devices

    def _ortvalues_to_torch_tensor(
        self,
        ortvalues: Any,  #  "onnxruntime.OrtValueVector",
    ) -> tuple[Any, ...]:  # tuple["torch.Tensor", ...]:
        if len(ortvalues) == 0:
            return tuple()

        from torch._C import _from_dlpack

        if all(ortvalues[i].has_value() for i in range(len(ortvalues))):
            if self.nvtx:
                self.torch.cuda.nvtx.range_push("_ortvalues_to_torch_tensor.1")
            res = ortvalues.to_dlpacks(_from_dlpack)
            if self.nvtx:
                self.torch.cuda.nvtx.range_pop()
        else:
            if self.nvtx:
                self.torch.cuda.nvtx.range_push("_ortvalues_to_torch_tensor.2")
            res = []
            for i in range(len(ortvalues)):
                res.append(
                    _from_dlpack(ortvalues[i].to_dlpack())
                    if ortvalues[i].has_value()
                    else None
                )
            if self.nvtx:
                self.torch.cuda.nvtx.range_pop()
        return tuple(res)

    def run(self, output_names, feeds):
        if self.use_run:
            return self.sess.run(output_names, feeds)
        inputs = [feeds[i] for i in self.input_names]
        if self.dlpack:
            return self.run_dlpack(*inputs, output_names=output_names)
        return self.run_ort_inference(*inputs)

    def _bind_torch_tensors(
        self,
        tensors: tuple[Any, ...],  # tuple["torch.Tensor", ...],
        output_names: List[str],
    ) -> "SessionIBinding":  # noqa: F821
        assert tensors is not None, "tensors cannot be None"
        assert len(tensors) == len(self.input_names), (
            f"Mismatch got {len(tensors)}, {len(self.input_names)} are expected, "
            f"names={self.input_names}"
        )
        bind = self.ORTC.SessionIOBinding(self.sess._sess)
        max_device = -1
        for name, tensor in zip(self.input_names, tensors):
            assert isinstance(tensor, self.torch.Tensor), f"Unexpected type {type(tensor)}"
            d = tensor.get_device()
            max_device = max(d, max_device)
            bind.bind_input(
                name,
                self.DEVICES[d],
                self.TORCH_DTYPE_TO_NUMPY_DTYPE.get(
                    # it works on CI
                    tensor.dtype,
                    # it does not seem to write for all releases
                    self.TORCH_DTYPE_TO_ONNX_DTYPE[tensor.dtype],
                ),
                tensor.shape or (1,),  # onnxruntime does not like empty shape here
                tensor.data_ptr(),
            )

        device = self.DEVICES[max_device]
        for o in output_names:
            bind.bind_output(o, device)
        return bind

    def run_ort_inference(self, *inputs, output_names=None):
        if output_names is None:
            output_names = self.output_names

        if self.nvtx:
            self.torch.cuda.nvtx.range_push("_bind_torch_tensors")

        bind = self._bind_torch_tensors(inputs, output_names=output_names)

        if self.nvtx:
            self.torch.cuda.nvtx.range_pop()
            self.torch.cuda.nvtx.range_push("run_with_iobinding")

        self.sess._sess.run_with_iobinding(bind, self.run_options)
        ort_vector_outputs = bind.get_outputs()
        # The function returns OrtValue, the code computing the discrepancies will
        # have to convert (so the necessary copy is not included here).
        # DlPack mechanism should be implemented in onnxruntime
        # (not only in onnxruntime-training).
        ort_outputs = [ort_vector_outputs[i] for i in range(len(ort_vector_outputs))]

        if self.nvtx:
            self.torch.cuda.nvtx.range_pop()
        return ort_outputs

    def run_dlpack(self, *inputs, output_names=None):
        if output_names is None:
            output_names = self.output_names
        ortvalues, output_devices = self._get_ortvalues_from_torch_tensors(
            inputs, len(output_names)
        )

        if self.nvtx:
            self.torch.cuda.nvtx.range_push("run_with_ortvaluevector")

        ort_outputs = self.ORTC.OrtValueVector()
        self.sess.run_with_ortvaluevector(
            self.run_options,
            self.input_names,
            ortvalues,
            output_names,
            ort_outputs,
            output_devices,
        )

        if self.nvtx:
            self.torch.cuda.nvtx.range_pop()

        pth_outputs = self._ortvalues_to_torch_tensor(ort_outputs)
        return pth_outputs


def run_onnx_inference(
    model: ModelProto,
    example_inputs: List[Any],
    warmup: int = 5,
    repeat: int = 5,
    verbose: int = 0,
    ort_optimize: bool = True,
    torch_model: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
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
        [("CUDAExecutionProvider", {"device_id": device}), "CPUExecutionProvider"]
        if device >= 0
        else ["CPUExecutionProvider"]
    )
    stats["providers"] = ",".join(providers)
    if verbose:
        print(f"[run_inference] create session with providers {providers!r}")

    begin = time.perf_counter()
    # onnxruntime is importing when needed as it
    # takes a couple of seconds if it contains CUDA EP.
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
        d = max_diff(expected, got)
        stats["discrepancies_abs"] = d["abs"]
        stats["discrepancies_dnan"] = d["dnan"]
        stats["discrepancies_rel"] = d["rel"]
        stats["discrepancies_avg"] = d["avg"]

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


def str_shape(shape: Tuple[Any, ...]) -> str:
    s = "x".join(str(i) for i in shape)
    return s


def str_dtype(dtype: Any) -> str:
    s = str(dtype)
    return s.replace("torch.", "").replace("'", "").replace("<", "").replace(">", "")
