import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from onnx import ModelProto
from onnx.defs import onnx_opset_version
from .interpreter import DynamoInterpreter
from .graph_builder import GraphBuilder, OptimizationOptions


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
) -> Tuple["torch.fx.GraphModule", GraphBuilder, DynamoInterpreter]:  # noqa: F821
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
    :return: onnx model
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
        import torch.export

    if isinstance(mod, torch.fx.GraphModule):
        graph_module = mod
        weights = dict(graph_module.named_parameters())
        buffers = dict(graph_module.named_buffers())
        mapping = {}
    else:
        exported_mod = torch.export.export(mod, args)
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
    )

    def retrieve(
        name, value, weights=weights, buffers=buffers, mapping=mapping, builder=builder
    ):
        return _retrieve(name, value, weights, buffers, mapping, builder)

    interpreter = DynamoInterpreter(builder, retrieve)
    return graph_module, builder, interpreter


def to_onnx(
    mod: Union["torch.nn.Module", "torch.fx.GraphModule"],  # noqa: F821
    args: Sequence["torch.Tensor"],  # noqa: F821
    input_names: Optional[Sequence[str]] = None,
    target_opset: Optional[Union[int, Dict[str, int]]] = None,
    as_function: bool = False,
    remove_unused: bool = False,
    constant_folding: bool = False,
    verbose: int = 0,
    return_builder: bool = False,
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
    :param remove_unused: if True, remove unused nodes
    :param constant_folding: if True, fold constants,
        constants are detected while converting the model
    :param verbose: verbosity level
    :return: onnx model
    """
    if target_opset is None:
        target_opset = min(18, onnx_opset_version() - 1)
    graph_module, builder, interpreter = _make_builder_interpreter(
        mod=mod,
        args=args,
        input_names=input_names,
        target_opset=target_opset,
        as_function=as_function,
        optimization_options=OptimizationOptions(
            remove_unused=remove_unused,
            constant_folding=constant_folding,
        ),
        verbose=verbose,
    )

    builder.process(graph_module, interpreter)
    onx = builder.to_onnx()
    if return_builder:
        return onx, builder
    return onx
