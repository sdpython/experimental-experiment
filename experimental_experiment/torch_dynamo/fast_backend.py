import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import ModelProto
import torch
from torch._C import _from_dlpack
from ..torch_exp._torch_helper import create_input_names, create_symint
from ..torch_exp.onnx_export import to_onnx, OptimizationOptions
from ..torch_exp.optimization_patterns import get_pattern_list
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
    # opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
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
    is_dimension_in: List[Tuple[bool, int, str]],
) -> Tuple[Tuple["torch.Tensor", ...], Tuple["OrtDevice", ...], Any]:  # noqa: F821
    ortvalues = OrtValueVector()
    ortvalues.reserve(len(tensors))
    dtypes = []
    shapes = []
    data_ptrs = []
    devices = []
    dimensions = []

    max_device = max(t.get_device() for t in tensors if isinstance(t, torch.Tensor))
    sdev = "cpu" if max_device < 0 else f"cuda:{max_device}"
    new_tensors = []
    for tensor, (dim, rk, name) in zip(tensors, is_dimension_in):
        if dim:
            assert isinstance(
                tensor, (int, torch.SymInt)
            ), f"Unexpected type {type(tensor)} for name={name!r}."
            dtypes.append(np.int64)
            if rk == 1:
                t = torch.tensor([int(tensor)], dtype=torch.int64)
            else:
                t = torch.tensor(int(tensor), dtype=torch.int64)
            t = t.to(sdev)
            devices.append(max_device)
            new_tensors.append(t)
            dimensions.append(t)
            shapes.append(t.size())
            data_ptrs.append(t.data_ptr())
        else:
            assert isinstance(tensor, torch.Tensor), (
                f"Unexpected type {type(tensor)}, dim={dim}, rk={rk}, name={name!r}, "
                f"len(tensors)={len(tensors)}, len(is_dimension_in)={len(is_dimension_in)}"
            )
            dtypes.append(torch_type_to_np_type[tensor.dtype])
            shapes.append(tensor.size())
            data_ptrs.append(tensor.data_ptr())
            d = tensor.get_device()
            devices.append(devices_list[d])
            max_device = max(max_device, d)
            new_tensors.append(tensor)

    ortvalues.push_back_batch(new_tensors, data_ptrs, dtypes, shapes, devices)
    return ortvalues, [devices_list[max_device] for i in range(n_outputs)], dimensions


def _post_process(output: Any, name: Optional[str], dim: bool) -> Any:
    if name is None:
        # None value required by torch
        return None
    if dim:
        # a dimension to replace
        if output.shape == (1,):
            yi = int(output[0])
        else:
            yi = int(output)
        return create_symint(yi)
    return output


def _ortvalues_to_torch_tensor(
    from_dlpack: Callable,
    ortvalues: "onnxruntime.OrtValueVector",  # noqa: F821
    is_dimension_out: List[Tuple[bool, int, Optional[str]]],
) -> Tuple["torch.Tensor", ...]:  # noqa: F821
    if len(ortvalues) == 0:
        return tuple()

    res = ortvalues.to_dlpacks(from_dlpack)
    return tuple(_post_process(r, d[2], d[0]) for r, d in zip(res, is_dimension_out))


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
    is_dimension_in: Optional[List[Tuple[bool, int, str]]] = None,
    is_dimension_out: Optional[List[Tuple[bool, int, Optional[str]]]] = None,
    input_value_infos: Optional[Tuple["onnx.ValueInfoProto", ...]] = None,  # noqa: F821
) -> Tuple["torch.Tensor"]:  # noqa: F821
    # _nvtx_range_push("contiguous")
    contiguous_inputs = tuple(
        (a.contiguous() if isinstance(a, torch.Tensor) else a) for a in inputs
    )
    # _nvtx_range_pop()

    # _nvtx_range_push("push_back_batch")
    ort_inputs, output_devices, dimensions = _get_ortvalues_from_torch_tensors(
        torch_type_to_np_type,
        devices,
        OrtValueVector,
        contiguous_inputs,
        len(output_names),
        is_dimension_in,
    )
    # _nvtx_range_pop()

    # _nvtx_range_push("run_with_ortvaluevector")
    ort_outputs = OrtValueVector()
    sess.run_with_ortvaluevector(
        run_options,
        input_names,
        ort_inputs,
        output_names,
        ort_outputs,
        output_devices,
    )

    # _nvtx_range_pop()

    # _nvtx_range_push("after run_with_ortvaluevector")
    # Map ORTValue to torch.Tensor.
    pth_outputs = _ortvalues_to_torch_tensor(from_dlpack, ort_outputs, is_dimension_out)
    # _nvtx_range_pop()

    # dimensions is only kept to avoid the garbage collector to delete temporary tensors
    assert dimensions is not None
    return pth_outputs


def _serialize(args: Any) -> Any:
    if isinstance(args, torch.Tensor):
        return args
    if isinstance(args, tuple):
        return tuple(_serialize(a) for a in args)
    if isinstance(args, list):
        return list(_serialize(a) for a in args)
    if isinstance(args, (int, torch.SymInt)):
        return args
    raise RuntimeError(f"Unable to serialize type {type(args)}.")


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
    disable_pattern: Optional[List[Union[str, type]]] = None,
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
    :param disable_pattern: optimization pattern to disable
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
        max_device = max(i.get_device() for i in args if hasattr(i, "get_device"))
        if max_device >= 0:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    input_names = input_names = create_input_names(graph_module, args)

    verbose_onnx, verbose_backend = (
        verbose if isinstance(verbose, tuple) else (verbose, verbose)
    )

    patterns = get_pattern_list("default", disable_pattern)

    options = OptimizationOptions(
        remove_unused=True,
        constant_folding=False,
        patterns=patterns,
        verbose=verbose_onnx,
    )

    onx, builder = to_onnx(
        graph_module,
        tuple(args),
        input_names=input_names,
        options=options,
        verbose=verbose_onnx,
        target_opset=target_opset,
        return_builder=True,
    )

    value = os.environ.get("ONNXRT_DUMP_PATH", None)
    if value:
        dump_prefix = value

    dump_first_inputs = [False]
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
        dump_first_inputs = [True]

    sess, run_options = _get_session(onx, backend, providers, exc=raise_exc)

    input_names = [i.name for i in onx.graph.input]
    output_names = [i.name for i in onx.graph.output]

    is_dimension_in = []
    for o in onx.graph.input:
        b = "_dim_" in o.name
        rk = len(o.type.tensor_type.shape.dim)
        is_dimension_in.append((b, rk, o.name))

    is_dimension_out = []
    for o in onx.graph.output:
        b = "_dim_" in o.name
        rk = len(o.type.tensor_type.shape.dim)
        is_dimension_out.append((b, rk, None if "_NONE_" in o.name else o.name))

    if storage is not None:
        stor = {}
        if "instance" in storage:
            storage["instance"].append(stor)
        else:
            storage["instance"] = [stor]
        stor["graph_module"] = graph_module
        stor["onnx"] = onx
        stor["is_dimension_in"] = is_dimension_in
        stor["is_dimension_out"] = is_dimension_out
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
        dump_first_inputs=dump_first_inputs,
        is_dimension_in=is_dimension_in,
        is_dimension_out=is_dimension_out,
    ):
        if dump_first_inputs[0]:
            dump_first_inputs[0] = False
            with open(name + ".pkl", "wb") as f:
                pickle.dump([input_names, _serialize(inputs), output_names], f)

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
            is_dimension_in,
            is_dimension_out,
        )
        if stor:
            stor["inputs"].append(args)
            stor["outputs"].append(res)
        return res

    return run
