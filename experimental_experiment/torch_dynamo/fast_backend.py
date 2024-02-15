import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import ModelProto
import torch
from torch._C import _from_dlpack
from ..torch_exp.onnx_export import to_onnx
from onnxruntime.capi import _pybind_state as ORTC


def _get_session(
    onx: ModelProto,
    impl: str = "ort",
    providers: Optional[List[str]] = None,
    exc: bool = True,
) -> Tuple[Union["ReferenceEvaluator", "InferenceSession"], "RunOptions"]:  # noqa: F821
    assert impl == "ort", f"Unexpected impl={impl!r}"
    assert exc, f"Silent mode is not allowed but exc={exc!r}"
    import onnxruntime

    run_options = onnxruntime.RunOptions()
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
    opts = onnxruntime.SessionOptions()
    opts.add_session_config_entry("session.disable_aot_function_inlining", "1")

    return (
        onnxruntime.InferenceSession(
            onx.SerializeToString(), opts, providers=providers
        ),
        run_options,
    )


def _get_ortvalues_from_torch_tensors(
    torch_type_to_np_type: Dict[Any, Any],
    devices_list: Dict[int, Any],
    OrtValueVector: type,
    tensors: Tuple["torch.Tensor", ...],  # noqa: F821
    n_outputs: int,
) -> Tuple[Tuple["torch.Tensor", ...], Tuple["OrtDevice", ...]]:  # noqa: F821
    ortvalues = OrtValueVector()
    ortvalues.reserve(len(tensors))
    dtypes = []
    shapes = []
    data_ptrs = []
    devices = []
    max_device = -1

    for tensor in tensors:
        dtypes.append(torch_type_to_np_type[tensor.dtype])
        shapes.append(tensor.size())
        data_ptrs.append(tensor.data_ptr())
        d = tensor.get_device()
        devices.append(devices_list[d])
        max_device = max(max_device, d)

    ortvalues.push_back_batch(tensors, data_ptrs, dtypes, shapes, devices)
    return ortvalues, [devices_list[max_device] for i in range(n_outputs)]


def _ortvalues_to_torch_tensor(
    from_dlpack: Callable, ortvalues: "onnxruntime.OrtValueVector"  # noqa: F821
) -> Tuple["torch.Tensor", ...]:  # noqa: F821
    if len(ortvalues) == 0:
        return tuple()

    res = ortvalues.to_dlpacks(from_dlpack)
    return tuple(res)


def _run_onnx_session_with_ortvaluevector(
    OrtValueVector: type,
    from_dlpack: Callable,
    torch_type_to_np_type: Dict[Any, Any],
    devices: Dict[int, Any],
    run_options: "onnxruntime.RunOptions",  # noqa: F821
    sess: "onnxruntime.InferenceSession",  # noqa: F821
    input_names: Tuple[str, ...],
    inputs: Tuple["torch.Tensor", ...],  # noqa: F821
    output_names: List[str],
    input_value_infos: Optional[Tuple["onnx.ValueInfoProto", ...]] = None,  # noqa: F821
) -> Tuple["torch.Tensor"]:  # noqa: F821
    # _nvtx_range_push("contiguous")
    contiguous_inputs = tuple(a.contiguous() for a in inputs)
    # _nvtx_range_pop()

    # _nvtx_range_push("push_back_batch")
    ort_inputs, output_devices = _get_ortvalues_from_torch_tensors(
        torch_type_to_np_type,
        devices,
        OrtValueVector,
        contiguous_inputs,
        len(output_names),
    )
    # _nvtx_range_pop()

    # _nvtx_range_push("run_with_ortvaluevector")
    ort_outputs = OrtValueVector()
    sess.run_with_ortvaluevector(
        run_options, input_names, ort_inputs, output_names, ort_outputs, output_devices
    )
    # _nvtx_range_pop()

    # _nvtx_range_push("after run_with_ortvaluevector")
    # Map ORTValue to torch.Tensor.
    pth_outputs = _ortvalues_to_torch_tensor(from_dlpack, ort_outputs)
    # _nvtx_range_pop()

    return pth_outputs


def onnx_custom_backend(
    graph_module: "torch.fx.GraphModule",  # noqa: F821
    args: List["torch.Tensor"],  # noqa: F821
    target_opset: Optional[int] = None,
    backend: str = "ort",
    verbose: Union[int, Tuple[int, int]] = 0,
    dump_prefix: Optional[None] = None,
    providers: Optional[Tuple[str]] = None,
    raise_exc: bool = True,
    storage: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Custom backend to export torch models into onnx
    (see :epkg:`torch.compiler`).
    This backend relies on :epkg:`onnxruntime` and tries to be
    as efficient as possible.

    :param graph_module: graph to export
    :param args: arguments
    :param target_opset: opset to use for the conversion
    :param backend: only `'ort'` is allowed
    :param verbose: adjust verbosity, if tuple, if gives different verbosity level
        to the exporter and the runtime
    :param dump_prefix
    :param providers: where to run the model, by default
    :param raise_exc: raise an exception whenever something goes wrong
    :param storage: to store any interesting objects during the process
    :return: Callable

    See :ref:`l-plot-onnxrt-diff` for an example.
    If not empty, `storage` keeps the memory of the data generated,
    onnx models, graph module as well the inputs and outputs when
    the model is run.
    """
    TORCH_DTYPE_TO_NUMPY_DTYPE = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.longlong,
        torch.bool: np.bool_,
    }

    DEVICES = {
        -1: ORTC.OrtDevice(ORTC.OrtDevice.cpu(), ORTC.OrtDevice.default_memory(), 0)
    }
    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            DEVICES[i] = ORTC.OrtDevice(
                ORTC.OrtDevice.cuda(), ORTC.OrtDevice.default_memory(), i
            )
        max_device = max(i.get_device() for i in args)
        if max_device >= 0:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    input_names = (
        ["input"] if len(args) == 1 else [f"input{i}" for i in range(len(args))]
    )

    verbose_onnx, verbose_backend = (
        verbose if isinstance(verbose, tuple) else (verbose, verbose)
    )

    onx, builder = to_onnx(
        graph_module,
        tuple(args),
        input_names=input_names,
        remove_unused=True,
        constant_folding=False,
        verbose=verbose_onnx,
        target_opset=target_opset,
        return_builder=True,
    )

    value = os.environ.get("ONNXRT_DUMP_PATH", None)
    if value:
        dump_prefix = value

    if dump_prefix:
        counter = 0
        name = f"{dump_prefix}_{counter}.onnx"
        while os.path.exists(name):
            counter += 1
            name = f"{dump_prefix}_{counter}.onnx"

        with open(name, "wb") as f:
            f.write(onx.SerializeToString())
        name = f"{dump_prefix}_{counter}.txt"
        with open(name, "w") as f:
            f.write(str(graph_module.graph))
            f.write("\n")

    sess, run_options = _get_session(onx, backend, providers, exc=raise_exc)

    input_names = [i.name for i in onx.graph.input]
    output_names = [i.name for i in onx.graph.output]

    if storage is not None:
        stor = {}
        if "instance" in storage:
            storage["instance"].append(stor)
        else:
            storage["instance"] = [stor]
        stor["graph_module"] = graph_module
        stor["onnx"] = onx
        stor["builder"] = builder
        stor["sess"] = sess
        stor["inputs"] = []
        stor["outputs"] = []
        stor["providers"] = providers
    else:
        stor = None

    def run(
        *inputs,
        sess=sess,
        stor=stor,
        input_names=input_names,
        output_names=output_names,
    ):
        res = _run_onnx_session_with_ortvaluevector(
            ORTC.OrtValueVector,
            _from_dlpack,
            TORCH_DTYPE_TO_NUMPY_DTYPE,
            DEVICES,
            run_options,
            sess,
            input_names,
            inputs,
            output_names,
        )
        if stor:
            stor["inputs"].append(args)
            stor["outputs"].append(res)
        return res

    return run
