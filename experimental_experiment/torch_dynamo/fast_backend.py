import os
import pickle
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import ModelProto, TensorProto, load
from onnx.numpy_helper import from_array, to_array
import torch
from torch._C import _from_dlpack
from onnxruntime.capi import _pybind_state as ORTC
from ..xbuilder import OptimizationOptions
from ..torch_interpreter import to_onnx
from ..torch_interpreter._torch_helper import create_input_names
from ..xoptim.patterns import get_pattern_list
from .backend_helper import get_dimensions


def _get_session(
    onx: ModelProto,
    impl: str = "ort",
    providers: Optional[List[str]] = None,
    ort_optimization_level: Optional[str] = None,
    exc: bool = True,
) -> Tuple[Union["ReferenceEvaluator", "InferenceSession"], "RunOptions"]:  # noqa: F821
    assert impl == "ort", f"Unexpected impl={impl!r}"
    assert exc, f"Silent mode is not allowed but exc={exc!r}"
    import onnxruntime

    run_options = onnxruntime.RunOptions()
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
    opts = onnxruntime.SessionOptions()
    if ort_optimization_level is not None:
        assert hasattr(onnxruntime.GraphOptimizationLevel, ort_optimization_level), (
            f"Unexpected value {ort_optimization_level!r} for GraphOptimizationLevel, "
            f"expecting one of the values in {dir(onnxruntime.GraphOptimizationLevel)}"
        )
        opts.graph_optimization_level = getattr(
            onnxruntime.GraphOptimizationLevel, ort_optimization_level
        )
    opts.add_session_config_entry("session.disable_aot_function_inlining", "1")

    return (
        onnxruntime.InferenceSession(
            onx.SerializeToString(), opts, providers=providers
        ),
        run_options,
    )


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
        return yi
    return output


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


class OrtBackend:
    """
    Wraps method ``run_with_ortvaluevector`` from ``InferenceSession``
    to implement a backend for ``torch.dynamo``.
    """

    TORCH_DTYPE_TO_NUMPY_DTYPE = {
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

    NUMPY_DTYPE_TO_TORCH_DTYPE = {
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.int64: torch.int64,
        np.int32: torch.int32,
        np.dtype("float16"): torch.float16,
        np.dtype("float32"): torch.float32,
        np.dtype("int64"): torch.int64,
        np.dtype("int32"): torch.int32,
    }

    def __init__(
        self,
        sess: "onnxruntime.InferenceSession",  # noqa: F821
        run_options: Optional["onnxruntime.RunOptions"] = None,  # noqa: F821
        devices: Optional[Dict[int, Any]] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        is_dimension_in: Optional[List[Tuple[bool, int, str]]] = None,
        is_dimension_out: Optional[List[Tuple[bool, int, Optional[str]]]] = None,
        dump_first_inputs: Optional[str] = None,
        stor: Optional[Dict[str, Any]] = None,
        onnx_model: Optional[ModelProto] = None,
    ):
        self.sess = sess
        self.input_names = input_names
        self.output_names = output_names
        self.is_dimension_in = is_dimension_in
        self.is_dimension_out = is_dimension_out
        self.dump_first_inputs = dump_first_inputs
        self.stor = stor
        self.run_options = run_options
        self.devices = devices
        self.OrtValueVector = ORTC.OrtValueVector
        self.from_dlpack = _from_dlpack
        self.onnx_model = onnx_model

        if self.devices is None:
            DEVICES = {
                -1: ORTC.OrtDevice(
                    ORTC.OrtDevice.cpu(), ORTC.OrtDevice.default_memory(), 0
                )
            }

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    DEVICES[i] = ORTC.OrtDevice(
                        ORTC.OrtDevice.cuda(), ORTC.OrtDevice.default_memory(), i
                    )
            self.devices = DEVICES

        if self.run_options is None:
            import onnxruntime

            self.run_options = onnxruntime.RunOptions()
            self.run_options.add_run_config_entry(
                "disable_synchronize_execution_providers", "1"
            )
        if self.input_names is None:
            self.input_names = [i.name for i in sess.get_inputs()]
        if self.output_names is None:
            self.output_names = [i.name for i in sess.get_outputs()]
        if self.is_dimension_in is None:
            self.is_dimension_in = []
            for o in sess.get_inputs():
                b = "_dim_" in o.name
                rk = len(o.shape)
                self.is_dimension_in.append((b, rk, o.name))
        if self.is_dimension_out is None:
            self.is_dimension_out = []
            for o in sess.get_outputs():
                b = "_dim_" in o.name
                rk = len(o.shape)
                self.is_dimension_out.append(
                    (b, rk, None if "_NONE_" in o.name else o.name)
                )

    def __call__(self, *inputs):
        if self.dump_first_inputs:
            name = self.dump_first_inputs
            self.dump_first_inputs = None
            with open(name + ".pkl", "wb") as f:
                pickle.dump(
                    [self.input_names, _serialize(inputs), self.output_names], f
                )

        res, dimensions = self._run_onnx_session_with_ortvaluevector(inputs)
        for x, name in zip(res, self.output_names):
            if isinstance(x, (torch.SymInt, int)):
                if x == 0:
                    self.dump_for_debug("debug_data", *inputs)
                assert x != 0, (
                    f"Dimension is null for name={name!r}, "
                    f"input dimensions={dimensions}"
                )

        if self.stor:
            self.stor["inputs"].append(inputs)
            self.stor["outputs"].append(res)
        return res

    def _get_ortvalues_from_torch_tensors(
        self, tensors: Tuple["torch.Tensor", ...]  # noqa: F821
    ) -> Tuple[Tuple["torch.Tensor", ...], Tuple["OrtDevice", ...], Any]:  # noqa: F821
        ortvalues = self.OrtValueVector()
        ortvalues.reserve(len(tensors))
        dtypes = []
        shapes = []
        data_ptrs = []
        devices = []
        dimensions = []

        max_device = -1
        assert isinstance(max_device, int), f"unexpected type for device={max_device!r}"
        assert tensors is not None, "tensors cannot be None"
        new_tensors = []
        for tensor, (dim, rk, name) in zip(tensors, self.is_dimension_in):
            if dim:
                assert isinstance(
                    tensor, (int, torch.SymInt)
                ), f"Unexpected type {type(tensor)} for name={name!r}."
                dtypes.append(np.int64)
                ti = int(tensor)
                assert ti != 0, (
                    f"Null value for a dimension ti={ti}, "
                    f"tensor={tensor}, rk={rk}, name={name!r}, "
                    f"type(tensor)={type(tensor)}, "
                    f"dimension={[t for t in tensors if isinstance(t, (int,torch.SymInt))]}"
                )
                t = torch.tensor([ti] if rk == 1 else ti, dtype=torch.int64)
                devices.append(self.devices[-1])
                new_tensors.append(t)
                dimensions.append(t)
                shapes.append(t.size())
                data_ptrs.append(t.data_ptr())
            else:
                assert isinstance(tensor, torch.Tensor), (
                    f"Unexpected type {type(tensor)}, dim={dim}, rk={rk}, name={name!r}, "
                    f"len(tensors)={len(tensors)}, len(is_dimension_in)={len(self.is_dimension_in)}"
                )
                dtypes.append(self.TORCH_DTYPE_TO_NUMPY_DTYPE[tensor.dtype])
                shapes.append(tensor.size())
                data_ptrs.append(tensor.data_ptr())
                d = tensor.get_device()
                devices.append(self.devices[d])
                new_tensors.append(tensor)
                max_device = max(max_device, tensor.get_device())

        ortvalues.push_back_batch(new_tensors, data_ptrs, dtypes, shapes, devices)
        output_devices = []
        for dim, rk, name in self.is_dimension_out:
            if dim:
                dev = self.devices[-1]
            else:
                dev = self.devices[max_device]
            output_devices.append(dev)

        return (ortvalues, output_devices, dimensions)

    def _ortvalues_to_torch_tensor(
        self, ortvalues: "onnxruntime.OrtValueVector"  # noqa: F821
    ) -> Tuple["torch.Tensor", ...]:  # noqa: F821
        if len(ortvalues) == 0:
            return tuple()

        res = ortvalues.to_dlpacks(self.from_dlpack)
        return tuple(
            _post_process(r, d[2], d[0]) for r, d in zip(res, self.is_dimension_out)
        )

    def _run_onnx_session_with_ortvaluevector(
        self, inputs: Tuple["torch.Tensor", ...]  # noqa: F821
    ) -> Tuple["torch.Tensor"]:  # noqa: F821
        # _nvtx_range_push("contiguous")
        contiguous_inputs = tuple(
            (a.contiguous() if isinstance(a, torch.Tensor) else a) for a in inputs
        )
        # _nvtx_range_pop()

        # _nvtx_range_push("push_back_batch")
        ort_inputs, output_devices, dimensions = self._get_ortvalues_from_torch_tensors(
            contiguous_inputs
        )
        # _nvtx_range_pop()

        # _nvtx_range_push("run_with_ortvaluevector")
        ort_outputs = self.OrtValueVector()
        self.sess.run_with_ortvaluevector(
            self.run_options,
            self.input_names,
            ort_inputs,
            self.output_names,
            ort_outputs,
            output_devices,
        )

        # _nvtx_range_pop()

        # _nvtx_range_push("after run_with_ortvaluevector")
        # Map ORTValue to torch.Tensor.
        pth_outputs = self._ortvalues_to_torch_tensor(ort_outputs)
        # _nvtx_range_pop()

        return pth_outputs, dimensions

    def to_tensor_proto(self, value: Any) -> TensorProto:
        if isinstance(value, np.ndarray):
            proto = from_array(value)
        elif isinstance(value, int):
            proto = from_array(np.array([value], dtype=np.int64))
        elif isinstance(value, torch.Tensor):
            return self.to_tensor_proto(value.detach().cpu().numpy())
        else:
            raise RuntimeError(
                f"Unexpected type {type(value)}, unable to convert to TensorProto"
            )
        return proto

    def dump_for_debug(self, folder: str, *inputs, test_case: int = 0):
        """
        Dumps everything in a folder.
        """
        assert self.onnx_model is not None, "Cannot dump if the onnx model is not here"
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, "model.onnx"), "wb") as f:
            f.write(self.onnx_model.SerializeToString())
        case = os.path.join(folder, f"test_case_{test_case}")
        if not os.path.exists(case):
            os.mkdir(case)
        assert (
            len(inputs) > 0
        ), f"Empty sequence of inputs, cannot save into {folder!r}."
        for i, inp in enumerate(inputs):
            name = os.path.join(case, f"input_{i}.pb")
            pb = self.to_tensor_proto(inp)
            with open(name, "wb") as f:
                f.write(pb.SerializeToString())

    @classmethod
    def replay_dumped_data(
        cls,
        folder: str,
        test_case: int = 0,
        providers: Optional[List[str]] = None,
        impl: str = "ort",
        ort_optimization_level: Optional[str] = None,
    ) -> Tuple["OrtBackend", List[Any]]:
        """
        Loads the data save by :meth:`dump_for_debug`.
        """
        onx = load(os.path.join(folder, "model.onnx"))
        test = os.path.join(folder, f"test_case_{test_case}")
        inputs = []
        i = 0
        name = os.path.join(test, f"input_{i}.pb")
        while os.path.exists(name):
            with open(name, "rb") as f:
                b = f.read()
            t = TensorProto()
            t.ParseFromString(b)
            a = to_array(t)
            inputs.append(a)
            i += 1
            name = os.path.join(test, f"input_{i}.pb")

        if providers is None:
            providers = (
                [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
            device = 0 if torch.cuda.is_available() else -1
        else:
            device = 0 if "CUDAExecutionProvider" in providers else -1

        sess, run_options = _get_session(
            onx,
            impl,
            providers,
            exc=True,
            ort_optimization_level=ort_optimization_level,
        )
        bck = OrtBackend(sess, run_options=run_options, onnx_model=onx)

        new_inputs = []
        for value, dim in zip(inputs, bck.is_dimension_in):
            if dim[0]:
                v = int(value[0]) if value.shape == (1,) else int(value)
            else:
                v = torch.Tensor(value.copy()).to(
                    cls.NUMPY_DTYPE_TO_TORCH_DTYPE[value.dtype]
                )
                if device >= 0:
                    v = v.to(device)
            new_inputs.append(v)
        return bck, new_inputs


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
    enable_pattern: Optional[Union[str, List[Union[str, type]]]] = "default",
    disable_pattern: Optional[Union[str, List[Union[str, type]]]] = None,
    pre_ort_model_transforms: Optional[
        Union[Callable[ModelProto, ModelProto], List[Callable[ModelProto, ModelProto]]]
    ] = None,
    ort_optimization_level: Optional[str] = None,
    dispatcher: Optional["Dispatcher"] = None,  # noqa: F821
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
    :param dump_prefix: to dump the models and the inputs
    :param providers: where to run the model, by default
    :param raise_exc: raise an exception whenever something goes wrong
    :param storage: to store any interesting objects during the process
    :param enable_pattern: optimization patterns to enable
    :param disable_pattern: optimization patterns to disable
    :param pre_ort_model_transforms: list of transformations applied on the final ModelProto
    :param ort_optimization_level: graph optimization level for onnxruntime,
        the default value is the same as what :epkg:`onnxruntime` defines
    :param dispatcher: see :class:`experimental_experiment.torch_interpreter.Dispatcher`
    :return: Callable

    See :ref:`l-plot-onnxrt-diff` or :ref:`l-plot-custom-backend` for examples.
    If not empty, `storage` keeps the memory of the data generated,
    onnx models, graph module as well the inputs and outputs when
    the model is run.
    """

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
            providers = [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
    else:
        max_device = -1

    input_names = input_names = create_input_names(graph_module, args)

    verbose_onnx, verbose_backend = (
        verbose if isinstance(verbose, tuple) else (verbose, verbose)
    )

    patterns = get_pattern_list(enable_pattern, disable_pattern, verbose=verbose_onnx)

    options = OptimizationOptions(
        remove_unused=True,
        constant_folding=False,
        patterns=patterns,
        verbose=verbose_onnx,
    )

    begin = time.perf_counter()
    if verbose:
        if max_device >= 0:
            print(
                f"[onnx_custom_backend] CUDA memory "
                f"allocated={torch.cuda.memory_allocated(max_device)}, "
                f"reserved={torch.cuda.memory_reserved(max_device)}, "
                f"max_device={max_device}"
            )
        print("[onnx_custom_backend] starts conversion to onnx.")

    onx, builder = to_onnx(
        graph_module,
        tuple(args),
        input_names=input_names,
        options=options,
        verbose=verbose_onnx,
        target_opset=target_opset,
        return_builder=True,
        dispatcher=dispatcher,
    )

    if verbose:
        print(
            f"[onnx_custom_backend] to_onnx done in {time.perf_counter() - begin} with "
            f"{len(onx.graph.node)} nodes and {onx.functions} local functions."
        )
        if max_device >= 0:
            print(
                f"[onnx_custom_backend] CUDA memory "
                f"allocated={torch.cuda.memory_allocated(max_device)}, "
                f"reserved={torch.cuda.memory_reserved(max_device)}, "
                f"max_device={max_device}"
            )

    if pre_ort_model_transforms is not None:

        if not isinstance(pre_ort_model_transforms, list):
            pre_ort_model_transforms = [pre_ort_model_transforms]
        for tr in pre_ort_model_transforms:
            begin = time.perf_counter()
            if verbose:
                if max_device >= 0:
                    print(
                        f"[onnx_custom_backend] CUDA memory "
                        f"allocated={torch.cuda.memory_allocated(max_device)}, "
                        f"reserved={torch.cuda.memory_reserved(max_device)}, "
                        f"max_device={max_device}"
                    )
                print(f"[onnx_custom_backend] starts pre_ort_model_transforms {tr}")

            onx = tr(onx)

            if verbose:
                print(
                    f"[onnx_custom_backend] pre_ort_model_transforms "
                    f"done in {time.perf_counter() - begin} with "
                    f"{len(onx.graph.node)} nodes and {len(onx.functions)} local functions."
                )
                if max_device >= 0:
                    print(
                        f"[onnx_custom_backend] CUDA memory "
                        f"allocated={torch.cuda.memory_allocated(max_device)}, "
                        f"reserved={torch.cuda.memory_reserved(max_device)}, "
                        f"max_device={max_device}"
                    )

    value = os.environ.get("ONNXRT_DUMP_PATH", None)
    if value:
        dump_prefix = value

    dump_first_inputs = None
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
        dump_first_inputs = name

    begin = time.perf_counter()
    if verbose:
        if max_device >= 0:
            print(
                f"[onnx_custom_backend] CUDA memory "
                f"allocated={torch.cuda.memory_allocated(max_device)}, "
                f"reserved={torch.cuda.memory_reserved(max_device)}, "
                f"max_device={max_device}"
            )
        print("[onnx_custom_backend] starts creating InferenceSession")

    sess, run_options = _get_session(
        onx,
        backend,
        providers,
        exc=raise_exc,
        ort_optimization_level=ort_optimization_level,
    )

    if verbose:
        print(
            f"[onnx_custom_backend] InferenceSession done in {time.perf_counter() - begin}"
        )
        if max_device >= 0:
            print(
                f"[onnx_custom_backend] CUDA memory "
                f"allocated={torch.cuda.memory_allocated(max_device)}, "
                f"reserved={torch.cuda.memory_reserved(max_device)}, "
                f"max_device={max_device}"
            )

    input_names = [i.name for i in onx.graph.input]
    output_names = [i.name for i in onx.graph.output]
    is_dimension_in, is_dimension_out = get_dimensions(onx)

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

    run = OrtBackend(
        sess=sess,
        run_options=run_options,
        stor=stor,
        input_names=input_names,
        output_names=output_names,
        dump_first_inputs=dump_first_inputs,
        is_dimension_in=is_dimension_in,
        is_dimension_out=is_dimension_out,
        devices=DEVICES,
        onnx_model=onx,
    )
    return run
