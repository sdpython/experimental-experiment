import inspect
import os
import pprint
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from onnx import ModelProto, save_model
from onnx.defs import onnx_opset_version
from onnx.model_container import ModelContainer
from ..helpers import string_type
from ..xbuilder.graph_builder import GraphBuilder, OptimizationOptions
from .export_options import ExportOptions


def match_input_parameters(
    model: Any, names: List[str], args: Optional[Tuple[Any, ...]] = None
) -> Dict[str, Any]:
    """
    Maps the given names with the parameter names in the model.

    :param model: model
    :param names: names to retrieve
    :param args: available inputs
    :return: dictionary with values

    Example:

    .. runpython::
        :showcode:

        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from experimental_experiment.reference import ExtendedReferenceEvaluator
        from experimental_experiment.torch_interpreter import to_onnx, match_input_parameters

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return torch.relu(self.linear(x))

        fake_mode = FakeTensorMode()
        converter = fake_mode.fake_tensor_converter

        fake_x = converter.from_real_tensor(fake_mode, torch.rand(2, 5))
        with fake_mode:
            model = Neuron(5, 3)
            onx = to_onnx(model, (fake_x,))

        # expected values with a different model
        not_fake_model = Neuron(5, 3)
        x = torch.rand(2, 5)
        expected = not_fake_model(x)
        print(expected)

        # converts the model, fill inputs with the weights
        names = [i.name for i in onx.graph.input]
        pfeeds = match_input_parameters(not_fake_model, names, (x,))
        nfeeds = {k:v.detach().numpy() for k,v in pfeeds.items()}
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, nfeeds)
        print(got)
    """

    def cl(s):
        s = s.replace(".", "_")
        return s

    weights = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    constants = model.state_dict()
    mapping = {}
    for k in weights:
        mapping[f"p_{cl(k)}"] = (k, weights[k], 0)
    for k in buffers:
        mapping[f"L__self__{cl(k)}"] = (k, buffers[k], 1)
    for k in constants:
        mapping[k] = (k, constants[k], 2)
    feeds = {}
    pos = 0
    for name in names:
        if name in mapping:
            t = mapping[name]
            feeds[name] = t[1]
        elif args is not None:
            # We assume it is an input.
            assert pos < len(
                args
            ), f"Unable to find argument at position {pos} in args (len(args)={len(args)}"
            feeds[name] = args[pos]
            pos += 1
    assert len(names) == 0 or len(feeds) > 0, (
        f"Unable to retrieve any name from {names!r}, "
        f"len(args)={len(args) if args else 0}, "
        f"mapping={sorted(mapping)}"
    )
    return feeds


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
            if new_name.startswith("L__self___") and new_name[len("L__self___") :] in weights:
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


def _make_builder_interpreter(
    mod: Union["torch.nn.Module", "torch.fx.GraphModule"],  # noqa: F821
    args: Optional[Sequence["torch.Tensor"]] = None,  # noqa: F821
    kwargs: Optional[Dict[str, "torch.Tensor"]] = None,  # noqa: F821
    input_names: Optional[Sequence[str]] = None,
    target_opset: Union[int, Dict[str, int]] = 18,
    as_function: bool = False,
    optimization_options: Optional[OptimizationOptions] = None,
    verbose: int = 0,
    raise_list: Optional[Set[str]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    tracing_mode: str = "symbolic",
    same_signature: bool = True,
    dispatcher: Optional["Dispatcher"] = None,  # noqa: F821
    export_options: Optional[Union[str, ExportOptions]] = None,
) -> Tuple[
    Union["torch.export.ExportedProgram", "torch.fx.GraphModule"],  # noqa: F821
    GraphBuilder,
    "DynamoInterpreter",  # noqa: F821
]:
    """
    Exports a torch model into ONNX using
    `dynamo export
    <https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html>`_.

    :param mod: torch module
    :param args: input arguments
    :param kwargs: keyword attributes
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
    :param dispatcher: see :class:`experimental_experiment.torch_interpreter.Dispatcher`
    :param export_options: Optional[Union[str, ExportOptions]] = None,
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

    if export_options is None:
        export_options = ExportOptions()

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
        exported_program = None
    elif isinstance(mod, torch.nn.Module) and mod.__class__.__name__ == "InterpreterModule":
        # comes from unflatten function
        if verbose > 0:
            print(f"[_make_builder_interpreter] use existing submodule {type(mod)}")
        graph_module = mod
        weights = dict(graph_module.named_parameters())
        buffers = dict(graph_module.named_buffers())
        constants = mod.state_dict()
        mapping = {}
        if os.environ.get("PRINT_GRAPH_MODULE", "0") in (1, "1"):
            print("-- GIVEN GRAPH MODULE")
            print(graph_module.graph)
        exported_program = None
    else:
        if verbose > 0:
            print(f"[_make_builder_interpreter] export_options={export_options!r}")
            print(f"[_make_builder_interpreter] input args={string_type(args)}")
            print(f"[_make_builder_interpreter] input kwargs={string_type(kwargs)}")
            print(f"[_make_builder_interpreter] dynamic_shapes={dynamic_shapes}")
            print(
                f"[_make_builder_interpreter] same_signature={same_signature}, "
                f"tracing_mode={tracing_mode}"
            )
        # If this step fails, try bypass_export_some_errors.
        exported_program = export_options.export(
            mod,
            args if isinstance(args, tuple) else (tuple() if args is None else args),
            kwargs,
            tracing_mode=tracing_mode,
            dynamic_shapes=dynamic_shapes,
            same_signature=same_signature,
            input_names=input_names,
            verbose=verbose,
        )

        graph_module = exported_program.graph_module
        if os.environ.get("PRINT_GRAPH_MODULE", "0") in (1, "1"):
            print("-- EXPORTED GRAPH MODULE")
            print(graph_module.graph)
        try:
            weights = dict(exported_program.named_parameters())
        except AttributeError:
            weights = dict(mod.named_parameters())
        try:
            buffers = dict(exported_program.named_buffers())
        except AttributeError:
            buffers = dict(mod.named_buffers())
        if hasattr(exported_program, "tensor_constants"):
            constants = exported_program.tensor_constants or {}
        else:
            # A bug may appear later.
            constants = {}
        if hasattr(exported_program, "graph_signature"):
            sig_mismatch_constants = set(k.replace(".", "_") for k in constants)
            signature = exported_program.graph_signature
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
                    f"export_options={export_options!r}"
                    f"A constant {k!r}, k[2:]={k[2:]!r}, v={v!r} was detected "
                    f"in the signature was not retrieved from the model. "
                    f"k in constants={k in constants}, "
                    f"k[2:] in constants={k[2:] in constants}, "
                    f"type(constants)={type(constants)}, "
                    f"\nlist(constants)={pprint.pformat(list(sorted(constants)))}"
                    f"\nexported_mod.tensor_constants="
                    f"{pprint.pformat(_get(exported_program, 'tensor_constants'))}"
                    f"\nexported_mod._constants="
                    f"{pprint.pformat(_get(exported_program, '_constants'))}"
                    f"\nsig_mismatch_constants="
                    f"{pprint.pformat(_get(sig_mismatch_constants))}"
                    f"\ndir(export_mod)={dir(exported_program)}"
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
        example_inputs=args,
        export_options=export_options,
    )
    return (exported_program or graph_module), builder, interpreter


def _model_signature(
    model: Union["torch.nn.Module", Callable],  # noqa: F821
) -> inspect.Signature:
    import torch

    return inspect.signature(model.forward if isinstance(model, torch.nn.Module) else model)


def _replacements_dynamic_shapes(
    mod: Any,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dict_dynamic_shapes: Optional[Dict[str, Any]] = None,
    input_names: Optional[List[str]] = None,
    verbose: int = 0,
):
    assert dict_dynamic_shapes is not None, "dict_dynamic_shapes is missing"
    if verbose > 2:
        print(f"[_replacements_dynamic_shapes] type(mod)={type(mod)}")
        print(f"[_replacements_dynamic_shapes] args={string_type(args)}")
        print(f"[_replacements_dynamic_shapes] kwargs={string_type(kwargs)}")
        print(f"[_replacements_dynamic_shapes] dict_dynamic_shapes={dict_dynamic_shapes}")
        print(f"[_replacements_dynamic_shapes] input_names={input_names}")
    new_dynamic_shapes = {}
    sig = _model_signature(mod)
    true_input_names = []
    has_args = None
    n_args = None if input_names is None else len(input_names)
    for name, p in sig.parameters.items():
        if verbose > 3:
            print(
                f"[_replacements_dynamic_shapes] -- {name}: {p.kind} - "
                f"has_args={has_args} - n_args={n_args}"
            )
        if n_args is not None and n_args <= 0:
            break
        if p.kind in (
            p.VAR_POSITIONAL,
            p.VAR_KEYWORD,
            p.POSITIONAL_OR_KEYWORD,
            p.POSITIONAL_ONLY,
        ):
            assert not has_args, (
                f"has_args={has_args} is already specified, "
                f"input_names={input_names}, dynamic_shapes="
                f"{dict_dynamic_shapes}"
            )
            assert input_names is None or len(input_names) == len(
                args
            ), f"Mismatch number between args={string_type(args)}, input_names={input_names}"
            true_input_names.append(p.name)
            if p.kind == p.VAR_POSITIONAL:
                if verbose > 3:
                    print(f"[_replacements_dynamic_shapes]    + {p.name}, has_args={has_args}")
                has_args = (p.name, len(args), len(true_input_names))
            if n_args is not None:
                n_args -= len(args)
            if verbose > 3:
                print(f"[_replacements_dynamic_shapes]    + {p.name}, n_args={n_args}")
        elif p.default in (None, inspect.Parameter.empty):
            true_input_names.append(name)
            if verbose > 3:
                print(f"[_replacements_dynamic_shapes]     + {p.name}")

    if has_args is None:
        replacements = {} if input_names is None else dict(zip(input_names, true_input_names))
        for k, v in dict_dynamic_shapes.items():
            r = replacements.get(k, k)
            new_dynamic_shapes[r] = v if not has_args or has_args[0] != r else (v,)
        if verbose > 2:
            print(f"[_replacements_dynamic_shapes] 1> input_names={input_names}")
            print(f"[_replacements_dynamic_shapes] 1> input_names={true_input_names}")
            print(f"[_replacements_dynamic_shapes] 1> new_dynamic_shapes={new_dynamic_shapes}")
        return new_dynamic_shapes

    if has_args:
        # has_args is supposed to be used when *args is used.
        assert input_names is not None, (
            f"Not implemented for has_args={has_args}, dynamic_shapes={dict_dynamic_shapes}"
            f", input_names={input_names}"
        )
        assert len(dict_dynamic_shapes) == len(input_names) == has_args[1], (
            f"Mismatch for has_args={has_args}, dynamic_shapes={dict_dynamic_shapes}"
            f", input_names={input_names}"
        )
        new_dynamic_shapes = {has_args[0]: tuple(dict_dynamic_shapes[n] for n in input_names)}
        if verbose > 2:
            print(f"[_replacements_dynamic_shapes] 2> has_args={has_args}")
            print(f"[_replacements_dynamic_shapes] 2> input_names={input_names}")
            print(
                f"[_replacements_dynamic_shapes] 2> dict_dynamic_shapes={dict_dynamic_shapes}"
            )
            print(f"[_replacements_dynamic_shapes] 2> new_dynamic_shapes={new_dynamic_shapes}")
        return new_dynamic_shapes
    if verbose > 2:
        print(f"[_replacements_dynamic_shapes] 3> new_dynamic_shapes={new_dynamic_shapes}")
    return new_dynamic_shapes


def is_wrapped(model: Any, dynamic_shapes: Optional[Any] = None) -> bool:
    """
    Tells if a model is wrapped.
    """
    if len(dynamic_shapes) != 1 or not isinstance(dynamic_shapes[0], tuple):
        return False
    raise AssertionError(f"Unable to tell for type {type(model)}")


def to_onnx(
    mod: Union["torch.nn.Module", "torch.fx.GraphModule"],  # noqa: F821
    args: Optional[Sequence["torch.Tensor"]] = None,  # noqa: F821
    kwargs: Optional[Dict[str, "torch.Tensor"]] = None,  # noqa: F821
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
    export_options: Optional[Union[str, ExportOptions]] = None,
    return_optimize_report: bool = False,
    filename: Optional[str] = None,
    inline: bool = False,
    export_modules_as_functions: Union[
        bool, Set[type["torch.nn.Module"]]  # noqa: F821
    ] = False,
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
    :param kwargs: keyword attributes
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
    :param return_optimize_report: returns statistics on the optimization as well
    :param filename: if specified, stores the model into that file
    :param inline: inline the model before converting to onnx, this is done before
            any optimization takes place
    :param export_options: to apply differents options before to get the exported program
    :param export_modules_as_functions: export submodules as local functions,
        this parameter can be filled with a set of class to preserve,
        all this other will be exported as usual
    :return: onnx model

    If environment variable ``PRINT_GRAPH_MODULE`` is set to one,
    information about the graph module is printed out.

    Environment variable ``TO_ONNX_VERBOSE=1`` can be used to
    increase verbosity in this function.
    Environment variable ``ONNX_BUILDER_PROGRESS=1`` can be used to show
    a progress bar on big models.
    """
    if target_opset is None:
        target_opset = min(18, onnx_opset_version() - 1)
    if options is None:
        options = OptimizationOptions()
    begin = time.perf_counter()

    verbose = max(verbose, int(os.environ.get("TO_ONNX_VERBOSE", verbose)))
    if verbose:
        print(f"[to_onnx] build the graph module with input_names={input_names}")

    if isinstance(dynamic_shapes, tuple):
        dyn_shapes = dynamic_shapes[0] if is_wrapped(mod, dynamic_shapes) else dynamic_shapes
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
            kwargs,
            dynamic_shapes if isinstance(dynamic_shapes, dict) else dict_dynamic_shapes,
            input_names,
            verbose=verbose,
        )
    else:
        use_dynamic_shapes = (
            dynamic_shapes
            if isinstance(dynamic_shapes, tuple)
            else _replacements_dynamic_shapes(
                mod,
                args,
                kwargs,
                dynamic_shapes if isinstance(dynamic_shapes, tuple) else dict_dynamic_shapes,
                verbose=verbose,
            )
        )

    if verbose and dynamic_shapes:
        print(f"[to_onnx] dynamic_shapes={dynamic_shapes}")
        print(f"[to_onnx] use_dynamic_shapes={use_dynamic_shapes}")

    assert use_dynamic_shapes is None or isinstance(use_dynamic_shapes, (dict, type(args))), (
        f"type(use_dynamic_shapes)={type(use_dynamic_shapes)} should have the "
        f"same type as args or a dictionary: {type(args)}, "
        f"dynamic_shapes={dynamic_shapes}, dict_dynamic_shapes={dict_dynamic_shapes}, "
        f"use_dynamic_shapes={use_dynamic_shapes}"
    )
    graph_module, builder, interpreter = _make_builder_interpreter(
        mod=mod,
        args=args,
        kwargs=kwargs,
        input_names=input_names,
        target_opset=target_opset,
        as_function=as_function,
        optimization_options=options,
        verbose=verbose,
        raise_list=raise_list,
        dynamic_shapes=use_dynamic_shapes,
        dispatcher=dispatcher,
        export_options=export_options,
    )

    add_stats = {}
    t = time.perf_counter()
    add_stats["time_export_graph_module"] = t - begin
    if verbose:
        print(f"[to_onnx] graph module done in {t - begin} s")

    if export_modules_as_functions:
        import torch.export

        assert isinstance(
            graph_module, torch.export.ExportedProgram
        ), f"Unexpected type {type(graph_module)} for graph_module"

        if export_modules_as_functions is True:
            export_modules_as_functions = set(type(m) for m in mod.modules())
        interpreter.register(export_modules_as_functions, dict(mod.named_modules()))
        if verbose > 1:
            print(
                f"[to_onnx] unflatten the graph_module, "
                f"preserve {sorted(c.__name__ for c in export_modules_as_functions)}"
            )

        a = time.perf_counter()
        new_graph_module = torch.export.unflatten(graph_module)
        add_stats["time_export_unflatten"] = t - a
        graph_module = new_graph_module

    if verbose > 4:
        print(f"[to_onnx] -- fx graph --\n{graph_module.graph}")

    if verbose:
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
        add_stats["optimization"] = stats
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

    if filename:
        if isinstance(onx, ModelProto):
            save_model(onx, filename)
        else:
            onx.save(filename, all_tensors_to_one_file=True)
    all_stats.update(add_stats)
    if return_builder:
        return (onx, builder, all_stats) if return_optimize_report else (onx, builder)
    return (onx, all_stats) if return_optimize_report else onx
