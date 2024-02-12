import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import ModelProto


def _get_session(
    onx: ModelProto, impl: str = "ort", exc: bool = True
) -> Tuple[Union["ReferenceEvaluator", "InferenceSession"], "RunOptions"]:  # noqa: F821
    assert impl == "ort", f"Unexpected impl={impl!r}"
    assert exc, f"Silent mode is not allowed but exc={exc!r}"
    import onnxruntime

    run_options = onnxruntime.RunOptions()
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")

    return (
        onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        ),
        run_options,
    )


def _adjust_scalar_from_fx_to_onnx(
    dynamo_value: "torch.Tensor",  # noqa: F821
    value_info: "onnx.ValueInfoProto",  # noqa: F821
) -> "torch.Tensor":  # noqa: F821
    """Helper function to wrap PyTorch variables as torch.Tensor"""
    assert hasattr(dynamo_value, "contiguous"), f"Unexpected type={type(dynamo_value)}"
    return dynamo_value.contiguous()


def _get_ortvalues_from_torch_tensors(
    torch_type_to_np_type: Dict[Any, Any],
    OrtValueVector: type,
    tensors: Tuple["torch.Tensor", ...],  # noqa: F821
    devices: Tuple["ORTC.OrtDevice", ...],  # noqa: F821
) -> Tuple["torch.Tensor", ...]:  # noqa: F821
    ortvalues = OrtValueVector()
    ortvalues.reserve(len(tensors))
    dtypes = []
    shapes = []
    data_ptrs = []

    for tensor in tensors:
        dtypes.append(torch_type_to_np_type[tensor.dtype])
        shapes.append(tensor.size())
        data_ptrs.append(tensor.data_ptr())
    ortvalues.push_back_batch(tensors, data_ptrs, dtypes, shapes, devices)
    return ortvalues


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
    run_options: "onnxruntime.RunOptions",  # noqa: F821
    sess: "onnxruntime.InferenceSession",  # noqa: F821
    input_names: Tuple[str, ...],
    inputs: Tuple["torch.Tensor", ...],  # noqa: F821
    input_devices: Tuple["ORTC.OrtDevice", ...],  # noqa: F821
    output_names: Optional[Tuple[str, ...]] = None,
    outputs: Optional[Tuple["torch.Tensor", ...]] = None,  # noqa: F821
    output_devices: Optional[Tuple["ORTC.OrtDevice", ...]] = None,  # noqa: F821
    input_value_infos: Optional[Tuple["onnx.ValueInfoProto", ...]] = None,  # noqa: F821
) -> Tuple["torch.Tensor"]:  # noqa: F821
    # _nvtx_range_push("contiguous")
    inputs = tuple(
        _adjust_scalar_from_fx_to_onnx(arg, value_info)
        for arg, value_info in zip(inputs, input_value_infos)
    )
    # _nvtx_range_pop()

    # _nvtx_range_push("push_back_batch")
    ort_inputs = _get_ortvalues_from_torch_tensors(
        torch_type_to_np_type, inputs, input_devices
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
    Custom backend to export torch models into onnx.
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
    import torch
    from torch._C import _from_dlpack
    from ..torch_exp.onnx_export import to_onnx
    from onnxruntime.capi import _pybind_state as ORTC

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

    sess, run_options = _get_session(onx, backend, exc=raise_exc)

    names = [i.name for i in onx.graph.input]

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
    else:
        stor = None

    def run(*inputs, sess=sess, names=names, stor=stor):
        res = _run_onnx_session_with_ortvaluevector(
            ORTC.OrtValueVector,
            _from_dlpack,
            TORCH_DTYPE_TO_NUMPY_DTYPE,
            run_options,
            sess,
            names,
            inputs,
            input_devices=None,
            output_names=None,
            outputs=None,
            output_devices=None,
            input_value_infos=None,
        )
        if stor:
            stor["inputs"].append(args)
            stor["outputs"].append(res)
        return res

    return run
