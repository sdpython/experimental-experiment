from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import ModelProto, NodeProto, TensorProto, TypeProto, helper as oh, load
from onnx.numpy_helper import to_array


class OrtEval:
    """
    This class loads an onnx model and the executes one by one the nodes
    with onnxruntime. This class is mostly meant for debugging.

    :param proto: ModelProto or filaname
    :param providers: providers
    :param options: session options
    :param verbose: verbosity
    :param whole: run the whole model instead instead of node
        by node
    :param incremental: run the model node by node, but for every node,
        executes the graph up to that node
    :param optimized_model_filepath: export the optimized graph
    """

    def __init__(
        self,
        proto: Union[str, ModelProto],
        providers: Optional[Union[str, List[str]]] = None,
        options: Optional["onnxruntime.SessionOptions"] = None,  # noqa: F821
        verbose: int = 0,
        whole: bool = False,
        incremental: bool = False,
        optimized_model_filepath: Optional[str] = None,
    ):
        self.session_options = options
        if self.session_options is None:
            import onnxruntime

            self.session_options = onnxruntime.SessionOptions()
        if optimized_model_filepath:
            self.session_options.optimized_model_filepath = optimized_model_filepath
        if verbose >= 30:
            import onnxruntime

            self.session_options.log_severity_level = 0
            self.session_options.log_verbosity_level = 0
            self.run_options = onnxruntime.RunOptions()
            self.run_options.log_severity_level = 0
            self.run_options.log_verbosity_level = 0
        else:
            import onnxruntime

            self.run_options = onnxruntime.RunOptions()

        if providers is None or providers in ("cpu", "CPU"):
            providers = ["CPUExecutionProvider"]
        elif providers in ("cuda", "CUDA"):
            providers = ["CUDAExecutionProvider"]
        self.providers = providers
        self._cache = {}
        if isinstance(proto, str):
            proto = load(proto)
        assert isinstance(proto, ModelProto), f"Unexpected type {type(proto)}"
        self.proto = proto
        self.nodes = list(proto.graph.node)
        self.rt_inits_ = {init.name: to_array(init) for init in proto.graph.initializer}
        self.rt_nodes_ = list(self.proto.graph.node)
        self.verbose = verbose
        self.whole = whole
        self.incremental = incremental
        assert not whole or not incremental, (
            f"whole={whole} and incremental={incremental} "
            f"cannot be both True at the same time."
        )

        try:
            import torch
        except ImportError:
            return

        self.torch = torch
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

    def _get_torch_dtype(self, dt: Any) -> "torch.dtype":  # noqa: F821
        if dt == np.bool_:
            return self.torch.bool
        if dt == np.float32:
            return self.torch.float32
        if dt == np.float16:
            return self.torch.float16
        if dt == np.int64:
            return self.torch.int64
        assert False, f"Unexpected type {type(dt)}: {dt}"

    def _get_itype(self, dt: Any) -> int:
        if isinstance(dt, int):
            return dt
        if dt in self.TORCH_DTYPE_TO_NUMPY_DTYPE:
            dt = self.TORCH_DTYPE_TO_NUMPY_DTYPE[dt]
        if dt == np.bool_:
            return TensorProto.BOOL
        if dt == np.float32:
            return TensorProto.FLOAT
        if dt == np.float16:
            return TensorProto.FLOAT16
        if dt == np.int64:
            return TensorProto.INT64
        return oh.np_dtype_to_tensor_dtype(dt)

    def _log(self, level: int, pattern: str, *args: List[Any]) -> None:
        if level < self.verbose:
            new_args = [self._log_arg(a) for a in args]
            print(pattern % tuple(new_args))

    def _log_arg(self, a: Any) -> Any:
        if isinstance(a, (str, int, float)):
            return a
        if hasattr(a, "detach"):
            device = f"D{a.get_device()}:"
            a = a.detach().cpu().numpy()
        else:
            device = -1
        if isinstance(a, np.ndarray):
            if self.verbose < 4:  # noqa: PLR2004
                return f"{device}{a.dtype}:{a.shape} in [{a.min()}, {a.max()}]"
            elements = a.ravel().tolist()
            if len(elements) > 5:  # noqa: PLR2004
                elements = elements[:5]
                return f"{device}{a.dtype}:{a.shape}:{','.join(map(str, elements))}..."
            return f"{device}{a.dtype}:{a.shape}:{elements}"
        if hasattr(a, "append"):
            return ", ".join(map(self._log_arg, a))
        return a

    def _log(self, level: int, pattern: str, *args: List[Any]) -> None:
        if level < self.verbose:
            new_args = [self._log_arg(a) for a in args]
            print(pattern % tuple(new_args))

    def run(
        self, outputs: Optional[List[str]], feed_inputs: Dict[str, Any]
    ) -> List[Any]:
        """
        Runs the model.
        It only works with numpy arrays.

        :param outputs: required outputs or None for all
        :param feed_inputs: inputs
        :return: outputs
        """
        if self.whole:
            if "" in self._cache:
                sess = self._cache[""]
            else:
                import onnxruntime

                sess = onnxruntime.InferenceSession(
                    self.proto.SerializeToString(),
                    self.session_options,
                    providers=self.providers,
                )
                self._cache[""] = sess
            return sess.run(outputs, feed_inputs, run_options=self.run_options)

        if outputs is None:
            outputs = [o.name for o in self.proto.graph.output]
        results = self.rt_inits_.copy()

        for k, v in self.rt_inits_.items():
            self._log(2, " +C %s: %s", k, v)
        for k, v in feed_inputs.items():
            self._log(2, " +I %s: %s", k, v)
            results[k] = v

        for node in self.rt_nodes_:
            self._log(1, "%s(%s) -> %s", node.op_type, node.input, node.output)
            for i in node.input:
                if i != "" and i not in results:
                    raise RuntimeError(
                        f"Unable to find input {i!r} in known results {sorted(results)}, "
                        f"self.rt_inits_ has {sorted(self.rt_inits_)}, "
                        f"feed_inputs has {sorted(feed_inputs)}."
                    )
            inputs = [(results[i] if i != "" else None) for i in node.input]
            outputs = self._run(node, inputs, results)
            for name, value in zip(node.output, outputs):
                if name == "":
                    continue
                self._log(2, " + %s: %s", name, value)  # type: ignore[arg-type]
                results[name] = value

        output_names = [o.name for o in self.proto.graph.output]
        for name in output_names:
            if name not in results:
                raise RuntimeError(
                    f"Unable to find output name {name!r} in {sorted(results)}, proto is\n{self.proto_}"
                )
        return [results[name] for name in output_names]

    def _get_sess_incremental(
        self, node: NodeProto
    ) -> Tuple[ModelProto, "onnxruntime.InferenceSession"]:  # noqa: F821
        import onnxruntime
        from ..xbuilder import GraphBuilder, OptimizationOptions

        builder = GraphBuilder(
            self.proto, optimization_options=OptimizationOptions(patterns=None)
        )
        builder.select_outputs(node.output)
        onx = builder.to_onnx()
        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), self.session_options, self.providers
        )
        return onx, sess

    def _get_sess(
        self, node: NodeProto, inputs: List[Any]
    ) -> Tuple[ModelProto, "onnxruntime.InferenceSession"]:  # noqa: F821
        if self.incremental:
            return self._get_sess_incremental(node)

        import onnxruntime

        vinputs = [
            (
                oh.make_tensor_value_info(i, TensorProto.FLOAT, None)
                if i == ""
                else oh.make_tensor_value_info(
                    i,
                    self._get_itype(it.dtype),
                    it.shape,
                )
            )
            for i, it in zip(node.input, inputs)
        ]

        voutputs = [oh.make_value_info(o, TypeProto()) for o in node.output]

        onx = oh.make_model(
            oh.make_graph([node], "node", vinputs, voutputs),
            ir_version=self.proto.ir_version,
        )
        del onx.opset_import[:]
        onx.opset_import.extend(self.proto.opset_import)

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), self.session_options, self.providers
        )
        return onx, sess

    def _run(
        self, node: NodeProto, inputs: List[Any], results: Dict[str, Any]
    ) -> List[Any]:
        """
        Runs a node.
        """
        types = [(None if a is None else (a.dtype, a.shape)) for a in inputs]
        key = tuple([id(node), *types])
        if key in self._cache:
            sess = self._cache[key][1]
        else:
            self._cache[key] = onx, sess = self._get_sess(node, inputs)

        if self.incremental:
            # Inputs are the inputs of the model not the node.
            feeds = {}
            for i in self.proto.graph.input:
                feeds[i.name] = results[i.name]
        else:
            feeds = dict(zip(node.input, inputs))
            if "" in feeds:
                feeds[""] = np.array([0], dtype=np.float32)

        outputs = sess.run(None, feeds, run_options=self.run_options)
        return outputs

    def run_dlpack(
        self, outputs: Optional[List[str]], feed_inputs: Dict[str, Any]
    ) -> List[Any]:
        """
        Runs the model using :epkg:`run_with_ortvaluevector`.
        It only works with :class:`torch.Tensor`.

        :param outputs: required outputs or None for all
        :param feed_inputs: inputs
        :return: outputs
        """
        if self.whole:
            from onnxruntime.capi import _pybind_state as ORTC

            if "" in self._cache:
                sess = self._cache[""]
            else:
                import onnxruntime

                sess = onnxruntime.InferenceSession(
                    self.proto.SerializeToString(),
                    self.session_options,
                    providers=self.providers,
                )
                self._cache[""] = sess

            input_names = [i.name for i in self.proto.graph.input]
            output_names = [i.name for i in self.proto.graph.output]
            inputs = [feed_inputs[i] for i in input_names]
            ortvalues, output_devices = self._get_ortvalues_from_torch_tensors(
                inputs, len(self.proto.graph.output)
            )

            ort_outputs = ORTC.OrtValueVector()
            sess.run_with_ortvaluevector(
                self.run_options,
                input_names,
                ortvalues,
                output_names,
                ort_outputs,
                output_devices,
            )
            pth_outputs = self._ortvalues_to_torch_tensor(ort_outputs)
            return pth_outputs

        if outputs is None:
            outputs = [o.name for o in self.proto.graph.output]
        if not hasattr(self, "rt_inits_torch_"):
            import torch

            self.rt_inits_torch_ = {
                k: torch.Tensor(v).to(self._get_torch_dtype(v.dtype))
                for k, v in self.rt_inits_.items()
            }
            if "CUDAExecutionProvider" in self.providers:
                ts = self.rt_inits_torch_
                self.rt_inits_torch_ = {}
                for k, v in ts.items():
                    if v.dtype in (torch.float32,) and len(v.shape) == 0:
                        pass
                    elif v.dtype not in (torch.int64, torch.bool):
                        v = v.cuda()
                    self.rt_inits_torch_[k] = v
        results = self.rt_inits_torch_.copy()

        for k, v in self.rt_inits_.items():
            self._log(2, " +C %s: %s", k, v)
        for k, v in feed_inputs.items():
            self._log(2, " +I %s: %s", k, v)
            results[k] = v

        for node in self.rt_nodes_:
            self._log(1, "%s(%s) -> %s", node.op_type, node.input, node.output)
            for i in node.input:
                if i != "" and i not in results:
                    raise RuntimeError(
                        f"Unable to find input {i!r} in known results {sorted(results)}, "
                        f"self.rt_inits_ has {sorted(self.rt_inits_)}, "
                        f"feed_inputs has {sorted(feed_inputs)}."
                    )
            inputs = [(results[i] if i != "" else None) for i in node.input]
            outputs = self._run_dlpack(node, inputs, results)
            for name, value in zip(node.output, outputs):
                if name == "":
                    continue
                self._log(2, " + %s: %s", name, value)  # type: ignore[arg-type]
                results[name] = value

        output_names = [o.name for o in self.proto.graph.output]
        for name in output_names:
            if name not in results:
                raise RuntimeError(
                    f"Unable to find output name {name!r} in {sorted(results)}, proto is\n{self.proto_}"
                )
        return [results[name] for name in output_names]

    def _get_ortvalues_from_torch_tensors(
        self,
        tensors: Tuple["torch.Tensor", ...],  # noqa: F821
        n_outputs: int,
        log_set: Optional[List[Any]] = None,
    ) -> Tuple[Tuple["torch.Tensor", ...], Tuple["OrtDevice", ...], Any]:  # noqa: F821
        import torch
        from onnxruntime.capi import _pybind_state as ORTC

        DEVICES = {
            -1: ORTC.OrtDevice(ORTC.OrtDevice.cpu(), ORTC.OrtDevice.default_memory(), 0)
        }

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                DEVICES[i] = ORTC.OrtDevice(
                    ORTC.OrtDevice.cuda(), ORTC.OrtDevice.default_memory(), i
                )

        ortvalues = ORTC.OrtValueVector()
        ortvalues.reserve(len(tensors))
        dtypes = []
        shapes = []
        data_ptrs = []
        devices = []

        max_device = -1
        assert isinstance(max_device, int), f"unexpected type for device={max_device!r}"
        assert tensors is not None, "tensors cannot be None"
        new_tensors = []
        for pos, tensor in enumerate(tensors):
            if tensor is None:
                tensor = torch.Tensor(np.array([0], dtype=np.float32)).to(
                    "cuda" if max_device >= 0 else "cpu"
                )
            assert isinstance(tensor, torch.Tensor), f"Unexpected type {type(tensor)}"
            dtypes.append(self.TORCH_DTYPE_TO_NUMPY_DTYPE[tensor.dtype])
            shapes.append(tensor.size())
            data_ptrs.append(tensor.data_ptr())
            d = tensor.get_device()
            if self.verbose > 10:
                if log_set is None:
                    print(
                        f"     < p={pos} d={d} dtype={dtypes[-1]} shape={tensor.shape}"
                    )
            devices.append(DEVICES[d])
            new_tensors.append(tensor)
            max_device = max(max_device, tensor.get_device())

        if self.verbose > 10 and log_set:
            for pos, tensor in enumerate(log_set):
                if tensor is None:
                    continue
                d = tensor.get_device()
                dt = self.TORCH_DTYPE_TO_NUMPY_DTYPE[tensor.dtype]
                print(f"     < p={pos} d={d} dtype={dt} shape={tensor.shape}")

        ortvalues.push_back_batch(new_tensors, data_ptrs, dtypes, shapes, devices)
        output_devices = []
        for i in range(n_outputs):
            dev = DEVICES[max_device]
            output_devices.append(dev)
            if self.verbose > 10:
                print(f"     > p={i} d={max_device}")

        return ortvalues, output_devices

    def _ortvalues_to_torch_tensor(
        self, ortvalues: "onnxruntime.OrtValueVector"  # noqa: F821
    ) -> Tuple["torch.Tensor", ...]:  # noqa: F821
        if len(ortvalues) == 0:
            return tuple()

        from torch._C import _from_dlpack

        if all(map(lambda i: ortvalues[i].has_value(), range(len(ortvalues)))):
            res = ortvalues.to_dlpacks(_from_dlpack)
        else:
            res = []
            for i in range(len(ortvalues)):
                if ortvalues[i].has_value():
                    res.append(_from_dlpack(ortvalues[i].to_dlpack()))
                else:
                    res.append(None)
        return tuple(res)

    def _run_dlpack(
        self, node: NodeProto, inputs: List[Any], results: Dict[str, Any]
    ) -> List[Any]:
        """
        Runs a node.
        """
        from onnxruntime.capi import _pybind_state as ORTC

        types = [(None if a is None else (a.dtype, a.shape)) for a in inputs]
        key = tuple([id(node), *types])
        if key in self._cache:
            sess = self._cache[key][1]
        else:
            self._cache[key] = onx, sess = self._get_sess(node, inputs)

        if self.incremental:
            # Inputs are the inputs of the model not the node.
            former_inputs = inputs
            inputs = []
            input_names = []
            for i in self.proto.graph.input:
                inputs.append(results[i.name])
                input_names.append(i.name)

            ortvalues, output_devices = self._get_ortvalues_from_torch_tensors(
                inputs, len(node.output), log_set=former_inputs
            )

            ort_outputs = ORTC.OrtValueVector()
            sess.run_with_ortvaluevector(
                self.run_options,
                input_names,
                ortvalues,
                node.output,
                ort_outputs,
                output_devices,
            )
            pth_outputs = self._ortvalues_to_torch_tensor(ort_outputs)
            return pth_outputs

        ortvalues, output_devices = self._get_ortvalues_from_torch_tensors(
            inputs, len(node.output)
        )

        ort_outputs = ORTC.OrtValueVector()
        sess.run_with_ortvaluevector(
            self.run_options,
            node.input,
            ortvalues,
            node.output,
            ort_outputs,
            output_devices,
        )
        pth_outputs = self._ortvalues_to_torch_tensor(ort_outputs)
        return pth_outputs
