import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import onnx
import numpy as np
import torch
from torch._C import _from_dlpack


class _InferenceSession:

    @classmethod
    def has_onnxruntime_training(cls):
        """Tells if onnxruntime_training is installed."""
        try:
            from onnxruntime import training
        except ImportError:
            # onnxruntime not training
            training = None
        if training is None:
            return False

        try:
            from onnxruntime.capi.onnxruntime_pybind11_state import OrtValueVector
        except ImportError:
            return False

        if not hasattr(OrtValueVector, "push_back_batch"):
            return False
        return True

    def __init__(
        self,
        sess: Union[onnx.ModelProto, str, "onnxruntime.InferenceSession"],  # noqa: F821
        session_options: Optional["onnxruntime.SessionOptions"] = None,  # noqa: F821
        providers: Optional[Union[str, List[str]]] = None,
        nvtx: bool = False,
        enable_profiling: bool = False,
        graph_optimization_level: Union[
            "onnxruntime.GraphOptimizationLevel", bool  # noqa: F821
        ] = None,
        log_severity_level: Optional[int] = None,
        log_verbosity_level: Optional[int] = None,
        optimized_model_filepath: Optional[str] = None,
        disable_aot_function_inlining: Optional[bool] = None,
        use_training_api: Optional[bool] = None,
    ):
        # onnxruntime is importing when needed as it takes a
        # couple of seconds if it contains CUDA EP.
        import onnxruntime
        from onnxruntime.capi import _pybind_state as ORTC  # noqa: N812

        if isinstance(sess, (onnx.ModelProto, str)):
            assert session_options is None or (
                providers is None
                and graph_optimization_level is None
                and log_severity_level is None
                and log_verbosity_level is None
            ), "session_options is defined, it is impossible to overwrite any option."
            if session_options is None:
                session_options = onnxruntime.SessionOptions()
                if enable_profiling:
                    session_options.enable_profiling = enable_profiling
                if optimized_model_filepath:
                    session_options.optimized_model_filepath = optimized_model_filepath
                    session_options.optimized_model_external_initializers_file_name = (
                        f"{os.path.splitext(optimized_model_filepath)[0]}.data"
                    )
                if log_severity_level is not None:
                    session_options.log_severity_level = log_severity_level
                if log_verbosity_level is not None:
                    session_options.log_verbosity_level = log_verbosity_level
                if graph_optimization_level is not None:
                    if isinstance(graph_optimization_level, bool):
                        session_options.graph_optimization_level = (
                            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                            if graph_optimization_level
                            else onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                        )
                    else:
                        session_options.graph_optimization_level = graph_optimization_level
                if disable_aot_function_inlining:
                    session_options.add_session_config_entry(
                        "session.disable_aot_function_inlining", "1"
                    )
            if providers is None:
                providers = ["CPUExecutionProvider"]
            if isinstance(providers, str):
                if providers.lower() == "cpu":
                    providers = ["CPUExecutionProvider"]
                elif providers.lower() == "cuda":
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    raise ValueError(f"Unexpected value for providers={providers!r}")
            sess = onnxruntime.InferenceSession(
                sess.SerializeToString(), session_options, providers=providers
            )
        else:
            assert (
                session_options is None
                and providers is None
                and graph_optimization_level is None
                and log_severity_level is None
                and log_verbosity_level is None
            ), f"First input is {type(sess)}, it is impossible to overwrite any option."

        self.sess = sess
        self.input_names = [i.name for i in sess.get_inputs()]
        self.output_names = [i.name for i in sess.get_outputs()]
        self.OrtValue = ORTC.OrtValue
        self.ORTC = ORTC
        self.torch = torch
        self.nvtx = nvtx
        self.run_options = onnxruntime.RunOptions()

        if log_severity_level is not None:
            self.run_options.log_severity_level = log_severity_level
        if log_verbosity_level is not None:
            self.run_options.log_verbosity_level = log_verbosity_level

        self.use_training_api = (
            self.has_onnxruntime_training() if use_training_api is None else use_training_api
        )

        DEVICES = {-1: ORTC.OrtDevice(ORTC.OrtDevice.cpu(), ORTC.OrtDevice.default_memory(), 0)}

        if torch.cuda.device_count() > 0:
            for i in range(torch.cuda.device_count()):
                DEVICES[i] = ORTC.OrtDevice(
                    ORTC.OrtDevice.cuda(), ORTC.OrtDevice.default_memory(), i
                )

        self.DEVICES = DEVICES
        self._torch_from_dlpack = _from_dlpack


class InferenceSessionForNumpy(_InferenceSession):
    """
    Wraps an `onnxruntime.InferenceSession` to overload method `run`
    to support :class:`numpy.ndarray`.

    :param sess: model or inference session
    :param session_options: options
    :param providers: providers
    :param nvtx: enable nvidia events
    :param providers: `None`, `"CPU"`, `"CUDA"` or a list of providers
    :param graph_optimization_level: see :class:`onnxruntime.SessionOptions`
    :param log_severity_level: see :class:`onnxruntime.SessionOptions`
    :param log_verbosity_level: see :class:`onnxruntime.SessionOptions`
    :param optimized_model_filepath:  see :class:`onnxruntime.SessionOptions`
    :param disable_aot_function_inlining:  see :class:`onnxruntime.SessionOptions`
    :param use_training_api: use onnxruntime-traning API
    """

    def __init__(
        self,
        sess: Union[onnx.ModelProto, str, "onnxruntime.InferenceSession"],  # noqa: F821
        session_options: Optional["onnxruntime.SessionOptions"] = None,  # noqa: F821
        providers: Optional[Union[str, List[str]]] = None,
        nvtx: bool = False,
        enable_profiling: bool = False,
        graph_optimization_level: Union[
            "onnxruntime.GraphOptimizationLevel", bool  # noqa: F821
        ] = None,
        log_severity_level: Optional[int] = None,
        log_verbosity_level: Optional[int] = None,
        optimized_model_filepath: Optional[str] = None,
        disable_aot_function_inlining: Optional[bool] = None,
        use_training_api: Optional[bool] = None,
    ):
        super().__init__(
            sess,
            session_options=session_options,
            providers=providers,
            nvtx=nvtx,
            enable_profiling=enable_profiling,
            graph_optimization_level=graph_optimization_level,
            log_severity_level=log_severity_level,
            log_verbosity_level=log_verbosity_level,
            optimized_model_filepath=optimized_model_filepath,
            disable_aot_function_inlining=disable_aot_function_inlining,
            use_training_api=use_training_api,
        )

    def run(
        self, output_names: Optional[List[str]], feeds: Dict[str, np.ndarray]
    ) -> List[np.ndarray]:
        """
        Calls :meth:`onnxruntime.InferenceSession.run`.
        """
        return self.sess.run(output_names, feeds)


class InferenceSessionForTorch(_InferenceSession):
    """
    Wraps an `onnxruntime.InferenceSession` to overload method `run`
    to support :class:`torch.Tensor`.

    :param sess: model or inference session
    :param session_options: options
    :param providers: providers
    :param nvtx: enable nvidia events
    :param providers: `None`, `"CPU"`, `"CUDA"` or a list of providers
    :param graph_optimization_level: see :class:`onnxruntime.SessionOptions`
    :param log_severity_level: see :class:`onnxruntime.SessionOptions`
    :param log_verbosity_level: see :class:`onnxruntime.SessionOptions`
    :param optimized_model_filepath:  see :class:`onnxruntime.SessionOptions`
    :param disable_aot_function_inlining:  see :class:`onnxruntime.SessionOptions`
    :param use_training_api: use onnxruntime-traning API
    """

    def __init__(
        self,
        sess: Union[onnx.ModelProto, str, "onnxruntime.InferenceSession"],  # noqa: F821
        session_options: Optional["onnxruntime.SessionOptions"] = None,  # noqa: F821
        providers: Optional[Union[str, List[str]]] = None,
        nvtx: bool = False,
        enable_profiling: bool = False,
        graph_optimization_level: Union[
            "onnxruntime.GraphOptimizationLevel", bool  # noqa: F821
        ] = None,
        log_severity_level: Optional[int] = None,
        log_verbosity_level: Optional[int] = None,
        optimized_model_filepath: Optional[str] = None,
        disable_aot_function_inlining: Optional[bool] = None,
        use_training_api: Optional[bool] = None,
    ):
        super().__init__(
            sess,
            session_options=session_options,
            providers=providers,
            nvtx=nvtx,
            enable_profiling=enable_profiling,
            graph_optimization_level=graph_optimization_level,
            log_severity_level=log_severity_level,
            log_verbosity_level=log_verbosity_level,
            optimized_model_filepath=optimized_model_filepath,
            disable_aot_function_inlining=disable_aot_function_inlining,
            use_training_api=use_training_api,
        )

        self.TORCH_DTYPE_TO_ONNX_DTYPE = {
            torch.float16: onnx.TensorProto.FLOAT16,
            torch.bfloat16: onnx.TensorProto.BFLOAT16,
            torch.float32: onnx.TensorProto.FLOAT,
            torch.float64: onnx.TensorProto.DOUBLE,
            torch.uint32: onnx.TensorProto.UINT32,
            torch.uint16: onnx.TensorProto.UINT16,
            torch.uint8: onnx.TensorProto.UINT8,
            torch.int8: onnx.TensorProto.INT8,
            torch.int16: onnx.TensorProto.INT16,
            torch.int32: onnx.TensorProto.INT32,
            torch.int64: onnx.TensorProto.INT64,
            torch.bool: onnx.TensorProto.BOOL,
        }

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

    def _get_ortvalues_from_torch_tensors(
        self, tensors: Tuple[torch.Tensor, ...], n_outputs: int
    ) -> Tuple[Tuple[torch.Tensor, "onnxruntime.OrtDevice"], ...]:  # noqa: F821
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
        ortvalues: Union[
            List["onnxruntime.OrtValue"], "onnxruntime.OrtValueVector"  # noqa: F821
        ],
    ) -> Tuple[torch.Tensor, ...]:
        if len(ortvalues) == 0:
            return tuple()

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
                    self._torch_from_dlpack(ortvalues[i].to_dlpack())
                    if ortvalues[i].has_value()
                    else None
                )
            if self.nvtx:
                self.torch.cuda.nvtx.range_pop()
        return tuple(res)

    def run(
        self, output_names: Optional[List[str]], feeds: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Same as :meth:`onnxruntime.InferenceSession.run` except that
        feeds is a dictionary of :class:`torch.Tensor`.
        """
        if self.use_training_api:
            inputs = [feeds[i] for i in self.input_names]
            return self.run_training_api(*inputs, output_names=output_names)
        return self.run_dlpack(output_names, feeds)

    def run_training_api(
        self, *inputs, output_names: Optional[List[str]] = None
    ) -> List[torch.Tensor]:
        """
        Calls the former training API now implemented in onnxruntime as well.

        :param inputs: list of :class:`torch.Tensor`
        :param output_names: requested outputs or None for all
        :return: list of :class:`torch.Tensor`
        """
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

    def run_dlpack(
        self, output_names: Optional[List[str]], feeds: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Same as :meth:`onnxruntime.InferenceSession.run` except that
        feeds is a dictionary of :class:`torch.Tensor`.
        The output device is CPU even if the outputs are on CUDA.
        """
        new_feeds = {}
        for k, v in feeds.items():
            new_feeds[k] = self.OrtValue.from_dlpack(v.__dlpack__(), v.dtype == torch.bool)
        if self.nvtx:
            self.torch.cuda.nvtx.range_push("run_with_ort_values")
        ort_outputs = self.sess._sess.run_with_ort_values(
            new_feeds, output_names or self.output_names, self.run_options
        )
        if self.nvtx:
            self.torch.cuda.nvtx.range_pop()
        pth_outputs = self._ortvalues_to_torch_tensor(ort_outputs)
        return pth_outputs


def investigate_onnxruntime_issue(
    proto: Union[onnx.ModelProto, str],  # noqa: F821
    session_options: Optional["onnxruntime.SessionOptions"] = None,  # noqa: F821
    providers: Optional[Union[str, List[str]]] = None,
    nvtx: bool = False,
    enable_profiling: bool = False,
    graph_optimization_level: Union[
        "onnxruntime.GraphOptimizationLevel", bool  # noqa: F821
    ] = None,
    log_severity_level: Optional[int] = None,
    log_verbosity_level: Optional[int] = None,
    optimized_model_filepath: Optional[str] = None,
    disable_aot_function_inlining: Optional[bool] = None,
    use_training_api: Optional[bool] = None,
    onnx_to_session: Optional[
        Union[str, Callable[onnx.ModelProto, "onnxruntime.InferenceSession"]]  # noqa: F821
    ] = None,
    # if model needs to be run.
    feeds: Optional[Union[Dict[str, torch.Tensor], Dict[str, np.ndarray]]] = None,
    verbose: int = 0,
    dump_filename: Optional[str] = None,
    infer_shapes: bool = True,
):
    """
    Invgestigates a crashing model. It tries every node until
    it crashes by adding the ones one by one in the model.

    :param proto: model or inference session
    :param session_options: options
    :param providers: providers
    :param nvtx: enable nvidia events
    :param providers: `None`, `"CPU"`, `"CUDA"` or a list of providers
    :param graph_optimization_level: see :class:`onnxruntime.SessionOptions`
    :param log_severity_level: see :class:`onnxruntime.SessionOptions`
    :param log_verbosity_level: see :class:`onnxruntime.SessionOptions`
    :param optimized_model_filepath:  see :class:`onnxruntime.SessionOptions`
    :param disable_aot_function_inlining:  see :class:`onnxruntime.SessionOptions`
    :param use_training_api: use onnxruntime-traning API
    :param onnx_to_session: function to load a model into an inference session if
        automated way implemented in this function is not enough,
        if it is equal ``cpu_session``, the callable becomes:
        ``lambda model: onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"])``
    :param feeds: run onnxruntime as well
    :param verbosity: verbosity level
    :param dump_filename: if not None, the function dumps the last model run
    :param infer_shapes: run shape inference

    The most simple use:

    .. code-block:: python

        investigate_onnxruntime_issue(
            model,
            feeds=feeds,
            verbose=10,
            dump_filename="test_investigate_onnxruntime_issue_callable.onnx",
            onnx_to_session="cpu_session",
        )

    Full example:

    .. runpython::
        :showcode:

        import numpy as np
        import onnx
        import onnx.helper as oh
        from experimental_experiment.ort_session import investigate_onnxruntime_issue

        TFLOAT = onnx.TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["x", "y"], ["gggg"]),
                    oh.make_node("Add", ["gggg", "z"], ["final"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("x", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("y", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("z", TFLOAT, [None, None]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        onnx.checker.check_model(model)
        feeds = {
            "x": np.random.rand(5, 6).astype(np.float32),
            "y": np.random.rand(5, 6).astype(np.float32),
            "z": np.random.rand(5, 6).astype(np.float32),
        }
        investigate_onnxruntime_issue(
            model,
            feeds=feeds,
            verbose=1,
            graph_optimization_level=False,
            dump_filename="last_issue.onnx",
        )
    """
    onx = (
        proto
        if isinstance(proto, onnx.ModelProto)
        else onnx.load(proto, load_external_data=False)
    )
    input_names = [i.name for i in onx.graph.input]
    if verbose:
        print(
            f"[investigate_onnxruntime_issue] found "
            f"{len(onx.graph.node)} nodes and {len(input_names)} inputs"
        )
    if infer_shapes:
        if verbose:
            print("[investigate_onnxruntime_issue] run shape inference")
        onx = onnx.shape_inference.infer_shapes(onx)

    if isinstance(onnx_to_session, str):
        if onnx_to_session == "cpu_session":
            import onnxruntime

            onnx_to_session = lambda model: onnxruntime.InferenceSession(  # noqa: E731
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        else:
            raise ValueError(f"Unexpected value onnx_to_session={onnx_to_session!r}")
    else:
        cls = (
            InferenceSessionForNumpy
            if feeds is None or any(isinstance(v, np.ndarray) for v in feeds.values())
            else InferenceSessionForTorch
        )
    if verbose and not onnx_to_session:
        print(f"[investigate_onnxruntime_issue] cls={cls}")

    for i in range(len(onx.graph.node)):
        node = onx.graph.node[i]
        if verbose:
            print(
                f"[investigate_onnxruntime_issue] + node {i}: "
                f"{node.op_type}({', '.join(node.input)}) -> "
                f"{', '.join(node.output)}"
            )
        e = onnx.utils.Extractor(onx)
        extracted = e.extract_model(input_names, node.output)
        if dump_filename:
            if verbose > 1:
                print(f"[investigate_onnxruntime_issue]   save into {dump_filename}")
            onnx.save(extracted, dump_filename)

        if verbose > 1:
            print("[investigate_onnxruntime_issue]   create the session")

        if onnx_to_session:
            sess = onnx_to_session(onx)
        else:
            sess = cls(
                extracted,
                session_options=session_options,
                providers=providers,
                nvtx=nvtx,
                enable_profiling=enable_profiling,
                graph_optimization_level=graph_optimization_level,
                log_severity_level=log_severity_level,
                log_verbosity_level=log_verbosity_level,
                optimized_model_filepath=optimized_model_filepath,
                disable_aot_function_inlining=disable_aot_function_inlining,
                use_training_api=use_training_api,
            )

        if not feeds:
            if verbose > 1:
                print("[investigate_onnxruntime_issue]   session created")
            continue

        if verbose > 1:
            print("[investigate_onnxruntime_issue]   running session")

        sess.run(None, feeds)

    if verbose > 0:
        print("[investigate_onnxruntime_issue] done.")
