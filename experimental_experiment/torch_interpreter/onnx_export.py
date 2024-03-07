import time
import warnings
from typing import Any, Dict, Optional, Sequence, Set, Tuple, Union
from onnx import ModelProto
from onnx.defs import onnx_opset_version
from ..xbuilder.graph_builder import GraphBuilder, OptimizationOptions


def _retrieve(
    name: str,
    value: Any,
    weights: Dict[str, "torch.Tensor"],  # noqa: F821
    buffers: Dict[str, "torch.Tensor"],  # noqa: F821
    mapping: Dict[str, Tuple[str, bool]],
    graph_builder: "GraphBuilder",  # noqa: F821
) -> "torch.Tensor":  # noqa: F821
    if name not in mapping:
        import torch

        # This is not a weight but a constant.
        if isinstance(value, torch.Tensor) and "FakeTensor" not in str(type(value)):
            return value
        if len(weights) == 0 and len(buffers) == 0:
            # It has to be an input.
            return None
        raise RuntimeError(
            f"Unable to find {name!r}."
            f"\nAvailable weights: {list(sorted(weights))}. "
            f"\nAvailable buffers: {list(sorted(buffers))}. "
            f"\nmapping={mapping}"
            f"{graph_builder.get_debug_msg() if graph_builder else ''}"
            f"\nvalue={value.dtype}:{value.shape}\n{value}"
        )

    new_name, is_weight = mapping[name]

    if is_weight:
        if new_name not in weights:
            if (
                new_name.startswith("L__self___")
                and new_name[len("L__self___") :] in weights
            ):
                new_name = new_name[len("L__self___") :]
        if new_name not in weights:
            raise ValueError(
                f"Unexpected name {name!r} for input "
                f"{name!r} mapped to weight {new_name!r}, "
                f"cannot be found in {', '.join(sorted(weights))}."
            )
        import torch

        value = weights[new_name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Unexpected type {type(value)} for input "
                f"{name!r} mapped to weight {new_name!r}."
            )
        return value
    else:
        if new_name not in buffers:
            if (
                new_name.startswith("L__self___")
                and new_name[len("L__self___") :] in buffers
            ):
                new_name = new_name[len("L__self___") :]
        if new_name not in buffers:
            raise ValueError(
                f"Unexpected name {name!r} for input "
                f"{name!r} mapped to buffer {new_name!r}, "
                f"cannot be found in {', '.join(sorted(buffers))}."
            )
        import torch

        value = buffers[new_name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Unexpected type {type(value)} for constant "
                f"{name!r} mapped to buffer {new_name!r}."
            )
        return value


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
    :param dynamic_shapes: see :epkg:`torch.export.export`
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
        mapping = {}
    else:
        exported_mod = torch.export.export(mod, args, dynamic_shapes=dynamic_shapes)
        if verbose > 0:
            msg = ", ".join(f"{a.dtype}:{tuple(a.shape)})" for a in args)
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
        signature = exported_mod.graph_signature
        mapping = {}
        for k, v in signature.inputs_to_parameters.items():
            mapping[k] = v, True
        for k, v in signature.inputs_to_buffers.items():
            mapping[k] = v, False

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
        name, value, weights=weights, buffers=buffers, mapping=mapping, builder=builder
    ):
        return _retrieve(name, value, weights, buffers, mapping, builder)

    from .interpreter import DynamoInterpreter

    interpreter = DynamoInterpreter(builder, retrieve)
    return graph_module, builder, interpreter


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
) -> Union[ModelProto, Tuple[ModelProto, GraphBuilder]]:
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
    :return: onnx model
    """
    if target_opset is None:
        target_opset = min(18, onnx_opset_version() - 1)
    if options is None:
        options = OptimizationOptions()
    begin = time.perf_counter()

    if verbose:
        print("[to_onnx] build the graph module")

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
    )

    if verbose:
        t = time.perf_counter()
        print(f"[to_onnx] done in {t - begin} s")
        print("[to_onnx] start creating the node after")
        begin = t

    builder.process(graph_module, interpreter)

    if verbose:
        t = time.perf_counter()
        print(f"[to_onnx] done in {t - begin} s")
        print("[to_onnx] start conversion to onnx (before optimization)")
        begin = t

    onx = builder.to_onnx(optimize=optimize)

    if verbose:
        t = time.perf_counter()
        print(f"[to_onnx] done in {t - begin} s")

    if return_builder:
        return onx, builder
    return onx
