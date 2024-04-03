from typing import Any, Dict, List, Optional, Union
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
    """

    def __init__(
        self,
        proto: Union[str, ModelProto],
        providers: Optional[Union[str, List[str]]] = None,
        options: Optional["SessionOptions"] = None,  # noqa: F821
        verbose: int = 0,
    ):
        self.session_options = options
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

    def _log(self, level: int, pattern: str, *args: List[Any]) -> None:
        if level < self.verbose:
            new_args = [self._log_arg(a) for a in args]
            print(pattern % tuple(new_args))

    def _log_arg(self, a: Any) -> Any:
        if isinstance(a, (str, int, float)):
            return a
        if isinstance(a, np.ndarray):
            if self.verbose < 4:  # noqa: PLR2004
                return f"{a.dtype}:{a.shape} in [{a.min()}, {a.max()}]"
            elements = a.ravel().tolist()
            if len(elements) > 5:  # noqa: PLR2004
                elements = elements[:5]
                return f"{a.dtype}:{a.shape}:{','.join(map(str, elements))}..."
            return f"{a.dtype}:{a.shape}:{elements}"
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
        Runs the model. For

        :param outputs: required outputs or None for all
        :param feed_inputs: inputs
        :return: outputs
        """
        if outputs is None:
            outputs = [o.name for o in self.proto.graph.output]
        results = self.rt_inits_.copy()

        for k, v in self.rt_inits_.items():
            self._log(2, " +C %s: %s", k, v)  # type: ignore[arg-type]
        for k, v in feed_inputs.items():
            self._log(2, " +I %s: %s", k, v)  # type: ignore[arg-type]
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
            outputs = self._run(node, inputs)
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

    def _run(self, node: NodeProto, inputs: List[Any]) -> List[Any]:
        """
        Runs a node.
        """
        types = [(None if a is None else (a.dtype, a.shape)) for a in inputs]
        key = tuple([id(node), *types])
        if key in self._cache:
            sess = self._cache[key][1]
        else:
            import onnxruntime

            vinputs = [
                (
                    oh.make_tensor_value_info(i, TensorProto.FLOAT, None)
                    if i == ""
                    else oh.make_tensor_value_info(
                        i, oh.np_dtype_to_tensor_dtype(it.dtype), it.shape
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
            self._cache[key] = onx, sess

        feeds = dict(zip(node.input, inputs))
        if "" in feeds:
            feeds[""] = np.array([0], dtype=np.float32)

        outputs = sess.run(None, feeds)
        return outputs
