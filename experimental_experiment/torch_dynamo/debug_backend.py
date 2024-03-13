import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from onnx import ModelProto
import torch
from ..xbuilder import OptimizationOptions
from ..torch_interpreter._torch_helper import create_input_names
from ..torch_interpreter import to_onnx
from ..xoptim.patterns import get_pattern_list
from .backend_helper import get_dimensions


def _get_session(
    onx: ModelProto,
    impl: str = "ref",
    exc: bool = True,
    verbose: int = 0,
    providers: Optional[List[str]] = None,
    ort_optimization_level: Optional[str] = None,
) -> Union["ReferenceEvaluator", "InferenceSession"]:  # noqa: F821
    if exc:
        try:
            return _get_session(
                onx,
                impl,
                exc=False,
                verbose=verbose,
                ort_optimization_level=ort_optimization_level,
            )
        except Exception as e:
            from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

            raise AssertionError(
                f"Unable to build session ({str(e)})\n{onnx_simple_text_plot(onx)}"
            ) from e

    if impl == "ref":

        from ..reference import ExtendedReferenceEvaluator

        return ExtendedReferenceEvaluator(onx, verbose=verbose)
    else:
        import onnxruntime

        providers = providers or ["CPUExecutionProvider"]
        opts = onnxruntime.SessionOptions()
        if ort_optimization_level is not None:
            if ort_optimization_level is not None:
                assert hasattr(
                    onnxruntime.GraphOptimizationLevel, ort_optimization_level
                ), (
                    f"Unexpected value {ort_optimization_level!r} for GraphOptimizationLevel, "
                    f"expecting one of the values in {dir(onnxruntime.GraphOptimizationLevel)}"
                )
                opts.graph_optimization_level = getattr(
                    onnxruntime.GraphOptimizationLevel, ort_optimization_level
                )

        return onnxruntime.InferenceSession(
            onx.SerializeToString(), opts, providers=providers
        )


def onnx_debug_backend(
    graph_module: "torch.fx.GraphModule",  # noqa: F821
    args: List[Union["torch.Tensor", "torch.SymInt"]],  # noqa: F821
    target_opset: Optional[int] = None,
    backend: str = "ort",
    verbose: Union[int, Tuple[int, int]] = 0,
    dump_prefix: Optional[None] = None,
    providers: Optional[Tuple[str]] = None,
    raise_exc: bool = True,
    storage: Optional[Dict[str, Any]] = None,
    raise_list: Optional[Set[str]] = None,
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
    :param dump_prefix: prefix used to dump the model generated by the backend
    :param providers: where to run the model, by default
    :param raise_exc: raise an exception whenever something goes wrong
    :param storage: to store any interesting objects during the process
    :param raise_list: the builder stops any time a name falls into that list,
        this is a debbuging tool
    :param enable_pattern: optimization patterns to enable
    :param disable_pattern: optimization patterns to disable
    :param pre_ort_model_transforms: list of transformations applied on the final ModelProto
    :param ort_optimization_level: graph optimization level for onnxruntime,
        the default value is the same as what :epkg:`onnxruntime` defines
    :param dispatcher: see :class:`experimental_experiment.torch_interpreter.Dispatcher`
    :return: Callable

    See :ref:`l-plot-onnxrt-diff` for an example.
    If not empty, `storage` keeps the memory of the data generated,
    onnx models, graph module as well the inputs and outputs when
    the model is run.
    """
    input_names = create_input_names(graph_module, args)

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

    onx, builder = to_onnx(
        graph_module,
        tuple(args),
        input_names=input_names,
        options=options,
        verbose=verbose_onnx,
        target_opset=target_opset,
        return_builder=True,
        raise_list=raise_list,
        dispatcher=dispatcher,
    )

    if pre_ort_model_transforms is not None:
        if not isinstance(pre_ort_model_transforms, list):
            pre_ort_model_transforms = [pre_ort_model_transforms]
        for tr in pre_ort_model_transforms:
            onx = tr(onx)

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
            f.write(builder.get_debug_msg())
            f.write("\n")

    sess = _get_session(
        onx,
        backend,
        exc=raise_exc,
        verbose=verbose_backend,
        providers=providers,
        ort_optimization_level=ort_optimization_level,
    )

    names = [i.name for i in onx.graph.input]

    _dtype = {
        np.dtype("float16"): torch.float16,
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("bool"): torch.bool,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.bool_: torch.bool,
    }

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
    else:
        stor = None

    def run(
        *inputs,
        sess=sess,
        names=names,
        stor=stor,
        is_dimension_in=is_dimension_in,
        is_dimension_out=is_dimension_out,
    ):
        max_device = max(x.get_device() for x in inputs if isinstance(x, torch.Tensor))

        xnp = []
        for x, (dim, rk, name) in zip(inputs, is_dimension_in):
            if isinstance(x, torch.Tensor):
                assert not dim, (
                    f"Input {name!r} is declared as a dimension but is not, "
                    f"dim={dim}, rk={rk}, dtype={x.dtype}, shape={x.shape}"
                )
                nx = x.detach().cpu().numpy()
            elif isinstance(x, (torch.SymInt, int)):
                assert dim and rk <= 1, (
                    f"Input {name!r} is not declared as a dimension but is, "
                    f"dim={dim}, rk={rk}, x={x}, type={type(x)}, names={names}"
                )
                if isinstance(x, int):
                    vi = x
                else:
                    vi = int(x)
                nx = np.array(vi, dtype=np.int64)
                if rk == 1:
                    nx = nx.reshape((-1,))
            else:
                raise AssertionError(f"Unexpected input type {type(x)}")
            assert nx.dtype not in (
                object,
                np.object_,
            ), f"unexpected dtype {nx.dtype} for an input"
            xnp.append(nx)

        feeds = dict(zip(names, xnp))
        results = sess.run(None, feeds)
        res = []
        for y, (dim, rk, name) in zip(results, is_dimension_out):
            if name is None:
                res.append(None)
                continue
            if dim:
                assert len(y.shape) <= 1, (
                    f"Unexpected shape {y.shape} ({y}) for a dimension {name!r} "
                    f"(rk={rk})"
                )
                if y.shape == (1,):
                    yi = int(y[0])
                else:
                    yi = int(y)
                res.append(yi)
                continue
            if max_device >= 0:
                res.append(torch.Tensor(y).to(_dtype[y.dtype]).to("cuda"))
            else:
                res.append(torch.Tensor(y).to(_dtype[y.dtype]))
        if stor:
            stor["inputs"].append(feeds)
            stor["outputs"].append(res)
        return res

    return run
