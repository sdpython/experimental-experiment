import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import ModelProto


def _get_session(
    onx: ModelProto, impl: str = "ref", exc: bool = True
) -> Union["ReferenceEvaluator", "InferenceSession"]:  # noqa: F821
    if exc:
        try:
            return _get_session(onx, impl, exc=False)
        except Exception as e:
            from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

            raise AssertionError(
                f"Unable to build session ({str(e)})\n{onnx_simple_text_plot(onx)}"
            ) from e

    if impl == "ref":
        from onnx.reference import ReferenceEvaluator

        return ReferenceEvaluator(onx, verbose=10)
    else:
        import onnxruntime

        return onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )


def onnx_debug_backend(
    graph_module: "torch.fx.GraphModule",  # noqa: F821
    args: List["torch.Tensor"],  # noqa: F821
    target_opset: Optional[int] = None,
    backend: str = "ort",
    verbose: Union[int, Tuple[int, int]] = 0,
    dump_prefix: Optional[None] = str,
    providers: Optional[Tuple[str]] = None,
    raise_exc: bool = True,
    storage: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Custom backend to export torch models into onnx.
    This backend is not meant to be efficient, it is more to check
    the conversion is ok. It relies either on :epkg:`onnxruntime`
    or the python reference implementation.

    :param graph_module: graph to export
    :param args: arguments
    :param target_opset: opset to use for the conversion
    :param backend: after the conversion, the model is executed with a runtime,
        :epkg:`onnxruntime` or the reference implementation,
        it must be a value among `'ort'`, `'ref'` or a class
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
    from ..torch_exp.onnx_export import to_onnx

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

    sess = _get_session(onx, backend, exc=raise_exc)

    names = [i.name for i in onx.graph.input]

    _dtype = {
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("bool"): torch.bool,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.bool_: torch.bool,
    }

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

    def run(*inputs, sess=sess, names=names, stor=stor):
        xnp = [x.detach().numpy() for x in inputs]
        feeds = dict(zip(names, xnp))
        results = sess.run(None, feeds)
        res = tuple(torch.Tensor(y).to(_dtype[y.dtype]) for y in results)
        if storage:
            stor["inputs"].append(feeds)
            stor["outputs"].append(res)
        return res

    return run
