import inspect
import time
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Union
from onnx import ModelProto
from onnx.defs import onnx_opset_version
from onnx.model_container import ModelContainer
from ..xbuilder.graph_builder import GraphBuilder, OptimizationOptions


def _retrieve(
    name: str,
    value: Any,
    weights: Dict[str, "torch.Tensor"],  # noqa: F821
    buffers: Dict[str, "torch.Tensor"],  # noqa: F821
    constants: Dict[str, "torch.Tensor"],  # noqa: F821
    mapping: Dict[str, Tuple[str, bool]],
    graph_builder: "GraphBuilder",  # noqa: F821
) -> "torch.Tensor":  # noqa: F821
    """
    Sent to the :class:`DynamoInterpreter
    <experimental_experiment.torch_interpreter.interpreter.DynamoInterpreter>`.
    It retrieves the weights.

    :param name: name to retrieve
    :param value: value
    :param weights: mapping name, weights
    :param buffers: mapping name, buffer
    :param constants: mapping name, constants
    :param mapping: mapping name, (new_name, is_weight)
    :param graph_builder: graph builder
    """
    if name not in mapping:
        import torch

        # This is not a weight but a constant.
        if isinstance(value, torch.Tensor) and "FakeTensor" not in str(type(value)):
            return value
        if len(weights) == 0 and len(buffers) == 0 and len(constants) == 0:
            # It has to be an input.
            return None
        raise RuntimeError(
            f"Unable to find {name!r}."
            f"\nAvailable weights: {list(sorted(weights))}. "
            f"\nAvailable buffers: {list(sorted(buffers))}. "
            f"\nAvailable constants: {list(sorted(constants))}. "
            f"\nmapping={mapping}"
            f"{graph_builder.get_debug_msg() if graph_builder else ''}"
            f"\nvalue={value.dtype}:{value.shape}\n{value}"
        )

    new_name, is_weight = mapping[name]

    if is_weight:
        # weights
        if new_name not in weights:
            if (
                new_name.startswith("L__self___")
                and new_name[len("L__self___") :] in weights
            ):
                new_name = new_name[len("L__self___") :]
        assert new_name in weights, (
            f"Unexpected name {name!r} for input "
            f"{name!r} mapped to weight {new_name!r}, "
            f"cannot be found in {', '.join(sorted(weights))}."
        )
        import torch

        value = weights[new_name]
        assert isinstance(value, torch.Tensor), (
            f"Unexpected type {type(value)} for input "
            f"{name!r} mapped to weight {new_name!r}."
        )
        return value

    # buffers and constants or lieft tensors
    if new_name in buffers:
        value = buffers[new_name]
        import torch

        assert isinstance(value, torch.Tensor), (
            f"Unexpected type {type(value)} for buffer "
            f"{name!r} mapped to buffer {new_name!r}."
        )
        return value
    if new_name in constants:
        value = constants[new_name]
        import torch

        assert isinstance(value, torch.Tensor), (
            f"Unexpected type {type(value)} for constant "
            f"{name!r} mapped to constant {new_name!r}."
        )
        return value

    if new_name.startswith("L__self___") and new_name[len("L__self___") :] in buffers:
        new_name = new_name[len("L__self___") :]
        value = buffers[new_name]
        import torch

        assert isinstance(value, torch.Tensor), (
            f"Unexpected type {type(value)} for buffer "
            f"{name!r} mapped to buffer {new_name!r}."
        )
        return value

    if new_name.startswith("c_") and new_name[len("c_") :] in constants:
        new_name = new_name[len("c_") :]
        value = constants[new_name]
        import torch

        assert isinstance(value, torch.Tensor), (
            f"Unexpected type {type(value)} for constant "
            f"{name!r} mapped to constant {new_name!r}."
        )
        return value

    raise ValueError(
        f"Unexpected name {name!r} for input "
        f"{name!r} mapped to buffer or constant {new_name!r}, "
        f"cannot be found in {', '.join(sorted(buffers))} or "
        f"{', '.join(sorted(constants))}"
    )


def _export(
    mod,
    args,
    tracing_mode,
    dynamic_shapes,
    same_signature,
    decomposition_table,
    use_dynamo,
    strict,
):
    import torch

    if not use_dynamo:
        try:
            exported_mod = torch.export.export(
                mod, args, dynamic_shapes=dynamic_shapes, strict=strict
            )
        except torch._export.verifier.SpecViolationError:
            # see https://github.com/pytorch/pytorch/issues/128394
            exported_mod = torch.export._trace._export(
                mod,
                args,
                dynamic_shapes=dynamic_shapes,
                pre_dispatch=False,
                strict=strict,
            )
        return exported_mod

    # other issues
    # https://github.com/pytorch/pytorch/issues/127571

    # import torch.utils._pytree as pytree
    # flat_args, orig_in_spec = pytree.tree_flatten((args, ))
    # print("+++++", orig_in_spec, type(flat_args), len(flat_args))
    res = torch._dynamo.export(
        mod,
        aten_graph=True,
        tracing_mode=tracing_mode,
        dynamic_shapes=dynamic_shapes,
        same_signature=same_signature,
        decomposition_table=decomposition_table,
        assume_static_by_default=dynamic_shapes is None,
    )(*args)

    return res


def _make_builder_interpreter(
    mod: Union["torch.nn.Module", "torch.fx.GraphModule"],  # noqa: F821
    args: Optional[Sequence["torch.Tensor"]] = None,  # noqa: F821
    input_names: Optional[Sequence[str]] = None,
    target_opset: Union[int, Dict[str, int]] = 18,
    as_function: bool = False,
    optimization_options: Optional[OptimizationOptions] = None,
    verbose: int = 0,
    raise_list: Optional[Set[str]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    tracing_mode: str = "symbolic",
    same_signature: bool = True,
    decomposition_table: Optional[
        Dict["torch._ops.OpOverload", Callable[..., Any]]  # noqa: F821
    ] = None,
    dispatcher: Optional["Dispatcher"] = None,  # noqa: F821
    use_dynamo: bool = True,
    strict: bool = True,
) -> Tuple["torch.fx.GraphModule", GraphBuilder, "DynamoInterpreter"]:  # noqa: F821
    """
    Exports a torch model into ONNX using
    `dynamo export
    <https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html>`_.

    :param mod: torch module
    :param args: input arguments
    :param input_names: input names
    :param target_opset: targeted opset or targeted opsets as a dictionary
    :param as_function: export as a ModelProto or a FunctionProto
    :param optimization_options: optimization options
    :param verbose: verbosity level
    :param raise_list: the builder stops any time a name falls into that list,
        this is a debbuging tool
    :param dynamic_shapes: see :epkg:`torch.export.export` or ``torch._dynamo.export``
    :param same_signature: same signature
    :param tracing_mode: tracing model
    :param decomposition_table: decomposition table
    :param dispatcher: see :class:`experimental_experiment.torch_interpreter.Dispatcher`
    :param use_dynamo: use ``torch.export.export`` or ``torch._dynamo.export``
    :param strict: given to ``torch.export.export``
    :return: onnx model
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
        import torch.export

    if isinstance(mod, torch.fx.GraphModule):
        if verbose > 0:
            print(f"[_make_builder_interpreter] use existing {type(mod)}")
        graph_module = mod
        weights = dict(graph_module.named_parameters())
        buffers = dict(graph_module.named_buffers())
        constants = mod.state_dict()
        mapping = {}
    else:
        exported_mod = _export(
            mod,
            args,
            tracing_mode=tracing_mode,
            dynamic_shapes=dynamic_shapes,
            same_signature=same_signature,
            decomposition_table=decomposition_table,
            use_dynamo=use_dynamo,
            strict=strict,
        )

        if verbose > 0:
            msg = ", ".join(
                (
                    "None"
                    if a is None
                    else (
                        f"{a.dtype}:{tuple(a.shape)})"
                        if hasattr(a, "dtype")
                        else str(type(a))
                    )
                )
                for a in args
            )
            print(f"[_make_builder_interpreter] args={msg}")
            print(f"[_make_builder_interpreter] dynamic_shapes={dynamic_shapes}")
            if verbose > 2:
                print(f"[_make_builder_interpreter] exported_mod {exported_mod}")
        graph_module = exported_mod.graph_module
        try:
            weights = dict(exported_mod.named_parameters())
        except AttributeError:
            weights = dict(mod.named_parameters())
        try:
            buffers = dict(exported_mod.named_buffers())
        except AttributeError:
            buffers = dict(mod.named_buffers())
        if hasattr(exported_mod, "tensor_constants"):
            constants = exported_mod.tensor_constants or {}
        else:
            # A bug may appear later.
            constants = {}
        if hasattr(exported_mod, "graph_signature"):
            sig_mismatch_constants = set(k.replace(".", "_") for k in constants)
            signature = exported_mod.graph_signature
            mapping = {}
            for k, v in signature.inputs_to_parameters.items():
                mapping[k] = v, True
            for k, v in signature.inputs_to_buffers.items():
                mapping[k] = v, False
            for k, v in signature.inputs_to_lifted_tensor_constants.items():
                mapping[k] = v, False
                assert (
                    k in constants
                    or k[2:] in constants
                    or k[2:] in sig_mismatch_constants
                ), (
                    f"A constant {k!r}, v={v!r} was detected "
                    f"in the signature was not retrieved from the model. "
                    f"\nlist(constants)={list(sorted(constants))}"
                    f"\nexported_mod.tensor_constants="
                    f"{getattr(exported_mod, 'tensor_constants', '?')}"
                    f"\nexported_mod._constants={getattr(exported_mod, '_constants', '?')}"
                    f"\ndir(export_mod)={dir(exported_mod)}"
                    f"\ndir(mod)={dir(mod)}"
                )
        else:
            mapping = {}
            for k in weights:
                mapping[k] = k, True
            for k in buffers:
                mapping[k] = k, False
            for k in constants:
                mapping[k] = k, False

    builder = GraphBuilder(
        target_opset,
        input_names=input_names,
        as_function=as_function,
        optimization_options=optimization_options,
        args=args,
        verbose=verbose,
        raise_list=raise_list,
        dynamic_shapes=dynamic_shapes,
    )

    def retrieve(
        name,
        value,
        weights=weights,
        buffers=buffers,
        mapping=mapping,
        constants=constants,
        builder=builder,
    ):
        return _retrieve(name, value, weights, buffers, constants, mapping, builder)

    from .interpreter import DynamoInterpreter

    interpreter = DynamoInterpreter(
        builder, retrieve, dispatcher=dispatcher, use_dynamo=use_dynamo
    )
    return graph_module, builder, interpreter


def _model_signature(
    model: Union["torch.nn.Module", Callable]  # noqa: F821
) -> inspect.Signature:
    import torch

    return inspect.signature(
        model.forward if isinstance(model, torch.nn.Module) else model
    )


def to_onnx(
    mod: Union["torch.nn.Module", "torch.fx.GraphModule"],  # noqa: F821
    args: Sequence["torch.Tensor"],  # noqa: F821
    input_names: Optional[Sequence[str]] = None,
    target_opset: Optional[Union[int, Dict[str, int]]] = None,
    as_function: bool = False,
    options: Optional[OptimizationOptions] = None,
    verbose: int = 0,
    return_builder: bool = False,
    raise_list: Optional[Set[str]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    optimize: bool = True,
    dispatcher: Optional["Dispatcher"] = None,  # noqa: F821
    large_model: bool = False,
    external_threshold: int = 1024,
    api_two: bool = False,
    return_optimize_report: bool = False,
    strict: bool = True,
) -> Union[
    Union[ModelProto, ModelContainer],
    Tuple[Union[ModelProto, ModelContainer], GraphBuilder],
]:
    """
    Exports a torch model into ONNX using
    `dynamo export
    <https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html>`_.

    :param mod: torch module
    :param args: input arguments
    :param input_names: input names
    :param target_opset: targeted opset or targeted opsets as a dictionary
    :param as_function: export as a ModelProto or a FunctionProto
    :param options: optimization options
    :param verbose: verbosity level
    :param return_builder: returns the builder as well
    :param raise_list: the builder stops any time a name falls into that list,
        this is a debbuging tool
    :param dynamic_shapes: see :epkg:`torch.export.export`
    :param optimize: optimize the model before exporting into onnx
    :param dispatcher: see :class:`experimental_experiment.torch_interpreter.Dispatcher`
    :param large_model: if True returns a :class:`onnx.model_container.ModelContainer`,
        it lets the user to decide later if the weights should be part of the model
        or saved as external weights
    :param external_threshold: if large_model is True, every tensor above this limit
        is stored as external
    :param api_two: use ``torch._dynamo.export`` instead of ``torch.export.export``
    :param return_optimize_report: returns statistics on the optimization as well
    :param strict: given to ``torch.export.export``
    :return: onnx model
    """
    if target_opset is None:
        target_opset = min(18, onnx_opset_version() - 1)
    if options is None:
        options = OptimizationOptions()
    begin = time.perf_counter()

    if verbose:
        print(f"[to_onnx] build the graph module with input_names={input_names}")

    if dynamic_shapes is not None and input_names:
        # Let's rewrite the dynamic shapes with the true name.
        new_dynamic_shapes = {}
        sig = _model_signature(mod)
        true_input_names = [
            name
            for name, p in sig.parameters.items()
            if p.default in (None, inspect.Parameter.empty)
        ]
        replacements = dict(zip(input_names, true_input_names))

        for k, v in dynamic_shapes.items():
            new_dynamic_shapes[replacements.get(k, k)] = v
        dynamic_shapes = new_dynamic_shapes

    if verbose and dynamic_shapes:
        print(f"[to_onnx] dynamic_shapes={dynamic_shapes}")

    graph_module, builder, interpreter = _make_builder_interpreter(
        mod=mod,
        args=args,
        input_names=input_names,
        target_opset=target_opset,
        as_function=as_function,
        optimization_options=options,
        verbose=verbose,
        raise_list=raise_list,
        dynamic_shapes=dynamic_shapes,
        dispatcher=dispatcher,
        use_dynamo=api_two,
        strict=strict,
    )

    if verbose:
        t = time.perf_counter()
        print(f"[to_onnx] graph module done in {t - begin} s")
        print("[to_onnx] start creating the onnx nodes")
        begin = t

    builder.process(graph_module, interpreter)

    if verbose:
        t = time.perf_counter()
        print(f"[to_onnx] {len(builder.nodes)} onnx nodes done in {t - begin} s")
        print("[to_onnx] start conversion to onnx (before optimization)")
        begin = t

    onx, stats = builder.to_onnx(
        optimize=optimize,
        large_model=large_model,
        external_threshold=external_threshold,
        return_optimize_report=True,
    )
    all_stats = dict(builder=builder.statistics_)
    if stats is not None:
        all_stats["optimization"] = stats
    if verbose:
        t = time.perf_counter()
        proto = onx if isinstance(onx, ModelProto) else onx.model_proto
        print(
            f"[to_onnx] to_onnx done in {t - begin}s "
            f"and {len(proto.graph.node)} nodes, "
            f"{len(proto.graph.initializer)} initializers, "
            f"{len(proto.graph.input)} inputs, "
            f"{len(proto.graph.output)} outputs"
        )
        if verbose >= 10:
            print(builder.get_debug_msg())

    if return_builder:
        return (onx, builder, all_stats) if return_optimize_report else (onx, builder)
    return (onx, all_stats) if return_optimize_report else onx
