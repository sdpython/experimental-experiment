import contextlib
import inspect
import os
import pprint
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from onnx import ModelProto
from onnx.defs import onnx_opset_version
from onnx.model_container import ModelContainer
from ..xbuilder.graph_builder import GraphBuilder, OptimizationOptions


def _insert_flatten_between_transpose_and_view(
    exported_program: "torch.export.ExportedProgram",  # noqa: F821
) -> "torch.export.ExportedProgram":  # noqa: F821
    """
    Modifies the module inplace to insert a node 'flatten' between a node
    'transpose' followed by a node 'view'.
    The modification takes place inplace.
    See issue https://github.com/pytorch/pytorch/issues/136543.
    """
    modified = False
    graph = exported_program.graph_module.graph
    for node in graph.nodes:
        if (node.op != "call_method" or node.target != "transpose") and (
            node.op != "call_function"
            or not hasattr(node.target, "name")
            or node.target.name() != "aten::transpose.int"
        ):
            continue
        insert = False
        for user in node.users:
            if (user.op == "call_method" and user.target == "view") or (
                user.op == "call_function"
                and hasattr(node.target, "name")
                and user.target.name() == "aten::view"
            ):
                insert = True
                break
        if not insert:
            continue

        modified = True
        with graph.inserting_after(node):
            new_node = graph.call_method("flatten", args=(node,))
            node.replace_all_uses_with(new_node)
            # new_node is replaced as well so we manually revert the replacement
            new_node.update_arg(0, node)
            node.users = {new_node: None}

    if not modified:
        # no rewrite was done.
        return exported_program

    graph.lint()
    return exported_program


def _retrieve(
    name: str,
    value: Any,
    weights: Dict[str, "torch.Tensor"],  # noqa: F821
    buffers: Dict[str, "torch.Tensor"],  # noqa: F821
    constants: Dict[str, "torch.Tensor"],  # noqa: F821
    mapping: Dict[str, Tuple[str, bool]],
    graph_builder: "GraphBuilder",  # noqa: F821
    debug: Optional[Any] = None,
    exc: bool = True,
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
    :param debug: any debug information when an issue is raised
    :param exc: raises an exception if not found
    """
    if name not in mapping:
        import torch

        # This is not a weight but a constant.
        if isinstance(value, torch.Tensor) and "FakeTensor" not in str(type(value)):
            return value
        if len(weights) == 0 and len(buffers) == 0 and len(constants) == 0:
            # It has to be an input.
            return None
        if exc:
            raise RuntimeError(
                f"Unable to find {name!r}."
                f"\nAvailable weights: {list(sorted(weights))}. "
                f"\nAvailable buffers: {list(sorted(buffers))}. "
                f"\nAvailable constants: {list(sorted(constants))}. "
                f"\nmapping={mapping}"
                f"{graph_builder.get_debug_msg() if graph_builder else ''}"
                f"\nvalue={value.dtype}:{value.shape}\n{value}"
            )
        return None

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

    if exc:
        raise ValueError(
            f"Unexpected name {name!r} for input "
            f"{name!r} mapped to buffer or constant {new_name!r}, "
            f"cannot be found in {', '.join(sorted(buffers))} or "
            f"{', '.join(sorted(constants))}"
        )
    return None


def _export(
    mod,
    args,
    tracing_mode,
    dynamic_shapes,
    same_signature,
    decomposition_table,
    use_dynamo,
    strict,
    input_names=None,
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
        except torch._dynamo.exc.UserError as e:
            eee = None
            try:
                exported_mod = torch.export.export(mod, args, strict=strict).graph
            except torch._export.verifier.SpecViolationError as ee:
                exported_mod = None
                eee = ee
            raise RuntimeError(
                f"Unable to convert model {type(mod)}, "
                f"type(args)={type(args)}, type(args[0])="
                f"{type(args[0]) if isinstance(args, tuple) and args else '?'}, "
                f"strict={strict}, input_names={input_names}\n--\n"
                f"dynamic_shapes={dynamic_shapes}\n--\ne={e}\n--\neee={eee}"
                f"\n---exported-program---\n{exported_mod}"
            ) from e
        if isinstance(decomposition_table, str):
            from ..torch_dynamo import get_decomposition_table_by_name

            decomposition_table = get_decomposition_table_by_name(decomposition_table)
        if decomposition_table is not None:
            exported_mod = _insert_flatten_between_transpose_and_view(exported_mod)
            exported_mod = exported_mod.run_decompositions(decomposition_table)
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


@contextlib.contextmanager
def bypass_export_some_errors():
    """
    Tries to bypass some functions torch.export.export does not
    support such as ``torch.jit.isinstance``.
    """
    import torch.jit

    f = torch.jit.isinstance
    torch.jit.isinstance = isinstance

    try:
        yield
    finally:
        torch.jit.isinstance = f


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
        Union[str, Dict["torch._ops.OpOverload", Callable[..., Any]]]  # noqa: F821
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
    :param decomposition_table: decomposition table, it can a string as well,
        'default' means :func:`get_decomposition_table
        <experimental_experiment.torch_dynamo.get_decomposition_table>`
        is used
    :param dispatcher: see :class:`experimental_experiment.torch_interpreter.Dispatcher`
    :param use_dynamo: use ``torch.export.export`` or ``torch._dynamo.export``
    :param strict: given to ``torch.export.export``
    :return: onnx model
    """

    def _get(x, att=None):
        if att is None:
            if isinstance(x, dict):
                return list(sorted(x))
            if isinstance(x, list):
                return x
            return [x]
        if hasattr(x, att):
            return _get(getattr(x, att))
        return ["?"]

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
        if os.environ.get("PRINT_GRAPH_MODULE", "0") in (1, "1"):
            print("-- GIVEN GRAPH MODULE")
            print(graph_module.graph)
    else:
        if verbose > 0:
            print(
                f"[_make_builder_interpreter] use decomposition_table="
                f"{decomposition_table!r}"
            )
        with bypass_export_some_errors():
            exported_mod = _export(
                mod,
                args,
                tracing_mode=tracing_mode,
                dynamic_shapes=dynamic_shapes,
                same_signature=same_signature,
                decomposition_table=decomposition_table,
                use_dynamo=use_dynamo,
                strict=strict,
                input_names=input_names,
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
        if os.environ.get("PRINT_GRAPH_MODULE", "0") in (1, "1"):
            print("-- EXPORTED GRAPH MODULE")
            print(graph_module.graph)
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
                    or k[2:].replace("getattr_l__self", "getattr_L__self") in constants
                ), (
                    f"A constant {k!r}, k[2:]={k[2:]!r}, v={v!r} was detected "
                    f"in the signature was not retrieved from the model. "
                    f"k in constants={k in constants}, "
                    f"k[2:] in constants={k[2:] in constants}, "
                    f"type(constants)={type(constants)}, "
                    f"\nlist(constants)={pprint.pformat(list(sorted(constants)))}"
                    f"\nexported_mod.tensor_constants="
                    f"{pprint.pformat(_get(exported_mod, 'tensor_constants'))}"
                    f"\nexported_mod._constants="
                    f"{pprint.pformat(_get(exported_mod, '_constants'))}"
                    f"\nsig_mismatch_constants="
                    f"{pprint.pformat(_get(sig_mismatch_constants))}"
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
        debug=None,
        weights=weights,
        buffers=buffers,
        mapping=mapping,
        constants=constants,
        builder=builder,
        exc=True,
    ):
        return _retrieve(
            name, value, weights, buffers, constants, mapping, builder, debug, exc=exc
        )

    from .interpreter import DynamoInterpreter

    interpreter = DynamoInterpreter(
        builder,
        retrieve,
        dispatcher=dispatcher,
        use_dynamo=use_dynamo,
        example_inputs=args,
        decomposition_table=decomposition_table,
    )
    return graph_module, builder, interpreter


def _model_signature(
    model: Union["torch.nn.Module", Callable],  # noqa: F821
) -> inspect.Signature:
    import torch

    return inspect.signature(model.forward if isinstance(model, torch.nn.Module) else model)


def _replacements_dynamic_shapes(
    mod: Any,
    args: Tuple[Any, ...],
    dict_dynamic_shapes: Dict[str, Any],
    input_names: Optional[List[str]] = None,
):
    new_dynamic_shapes = {}
    sig = _model_signature(mod)
    true_input_names = []
    has_args = None
    n_args = None if input_names is None else len(input_names)
    for name, p in sig.parameters.items():
        if n_args is not None and n_args <= 0:
            break
        if p.kind == p.VAR_POSITIONAL or p.kind == p.VAR_KEYWORD:
            assert not has_args, (
                f"has_args={has_args} is already specified, "
                f"input_names={input_names}, dynamic_shapes="
                f"{dict_dynamic_shapes}"
            )
            assert input_names or None or len(input_names) == len(args), (
                f"Mimsatch number between len(args)={len(args)}, "
                f"input_names={input_names}"
            )
            true_input_names.append(p.name)
            has_args = (p.name, len(args), len(true_input_names))
            n_args -= len(args)
        elif p.default in (None, inspect.Parameter.empty):
            true_input_names.append(name)

    if has_args is None:
        replacements = (
            {} if input_names is None else dict(zip(input_names, true_input_names))
        )
        for k, v in dict_dynamic_shapes.items():
            r = replacements.get(k, k)
            new_dynamic_shapes[r] = v if not has_args or has_args[0] != r else (v,)
        return new_dynamic_shapes

    if has_args:
        assert input_names is not None, (
            f"Not implemented for has_args={has_args}, dynamic_shapes={dict_dynamic_shapes}"
            f", input_names={input_names}"
        )
        assert len(dict_dynamic_shapes) == len(input_names) == has_args[1], (
            f"Mismatch for has_args={has_args}, dynamic_shapes={dict_dynamic_shapes}"
            f", input_names={input_names}"
        )
        new_dynamic_shapes = {
            has_args[0]: tuple(dict_dynamic_shapes[n] for n in input_names)
        }
        return new_dynamic_shapes
    return new_dynamic_shapes


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
    decomposition_table: Optional[
        Union[str, Dict["torch._ops.OpOverload", Callable[..., Any]]]  # noqa: F821
    ] = None,
    inline: bool = False,
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
    :param decomposition_table: decomposition_table, a string as well such as default
        to use the default decomposition table returned by
        :func:`get_decomposition_table
        <experimental_experiment.torch_dynamo.get_decomposition_table>`
    :param inline: inline the model before converting to onnx, this is done before
            any optimization takes place
    :return: onnx model

    If environment variable ``PRINT_GRAPH_MODULE`` is set to one,
    information about the graph module is printed out.
    """
    if target_opset is None:
        target_opset = min(18, onnx_opset_version() - 1)
    if options is None:
        options = OptimizationOptions()
    begin = time.perf_counter()

    if verbose:
        print(f"[to_onnx] build the graph module with input_names={input_names}")

    if isinstance(dynamic_shapes, tuple):
        if len(dynamic_shapes) == 1 and isinstance(dynamic_shapes[0], tuple):
            # Model is wrapped
            dyn_shapes = dynamic_shapes[0]
        else:
            dyn_shapes = dynamic_shapes
        if not input_names:
            input_names = [f"input{i}" for i in range(len(dyn_shapes))]
        assert len(input_names) == len(dyn_shapes), (
            f"Mismatch number of inputs, input_names={input_names!r}, "
            f"dyn_shapes={dyn_shapes!r}"
        )
        dict_dynamic_shapes = dict(zip(input_names, dyn_shapes))
    elif dynamic_shapes is not None:
        assert isinstance(
            dynamic_shapes, dict
        ), f"Unexpected type {type(dynamic_shapes)} for dynamic_shapes={dynamic_shapes}"
        dict_dynamic_shapes = dynamic_shapes
    else:
        dict_dynamic_shapes = None

    if dynamic_shapes is None:
        use_dynamic_shapes = None
    elif input_names:
        # Let's rewrite the dynamic shapes with the true name
        use_dynamic_shapes = _replacements_dynamic_shapes(
            mod,
            args,
            dynamic_shapes if isinstance(dynamic_shapes, dict) else dict_dynamic_shapes,
            input_names,
        )
    else:
        use_dynamic_shapes = (
            dynamic_shapes
            if isinstance(dynamic_shapes, tuple)
            else _replacements_dynamic_shapes(
                mod,
                args,
                dynamic_shapes if isinstance(dynamic_shapes, tuple) else dict_dynamic_shapes,
            )
        )

    if verbose and dynamic_shapes:
        print(
            f"[to_onnx] dynamic_shapes={dynamic_shapes}, "
            f"use_dynamic_shapes={use_dynamic_shapes}"
        )

    assert use_dynamic_shapes is None or isinstance(
        use_dynamic_shapes, (dict, type(args))
    ), (
        f"type(use_dynamic_shapes)={type(use_dynamic_shapes)} should have the "
        f"same type as args or a dictionary: {type(args)}, "
        f"dynamic_shapes={dynamic_shapes}, dict_dynamic_shapes={dict_dynamic_shapes}, "
        f"use_dynamic_shapes={use_dynamic_shapes}"
    )
    graph_module, builder, interpreter = _make_builder_interpreter(
        mod=mod,
        args=args,
        input_names=input_names,
        target_opset=target_opset,
        as_function=as_function,
        optimization_options=options,
        verbose=verbose,
        raise_list=raise_list,
        dynamic_shapes=use_dynamic_shapes,
        dispatcher=dispatcher,
        use_dynamo=api_two,
        strict=strict,
        decomposition_table=decomposition_table,
    )

    t = time.perf_counter()
    add_stats = {"time_export_graph_module": t - begin}
    if verbose:
        print(f"[to_onnx] graph module done in {t - begin} s")
        print("[to_onnx] start creating the onnx nodes")
    begin = t

    builder.process(graph_module, interpreter)

    t = time.perf_counter()
    add_stats["time_export_builder_process"] = t - begin
    if verbose:
        print(f"[to_onnx] {len(builder.nodes)} onnx nodes done in {t - begin} s")
        print("[to_onnx] start conversion to onnx (before optimization)")
    begin = t

    onx, stats = builder.to_onnx(
        optimize=optimize,
        large_model=large_model,
        external_threshold=external_threshold,
        return_optimize_report=True,
        inline=inline,
    )
    all_stats = dict(builder=builder.statistics_)
    if stats:
        all_stats["optimization"] = stats
    t = time.perf_counter()
    add_stats["time_export_to_onnx"] = t - begin
    if verbose:
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

    all_stats.update(add_stats)
    if return_builder:
        return (onx, builder, all_stats) if return_optimize_report else (onx, builder)
    return (onx, all_stats) if return_optimize_report else onx
