import warnings
from typing import Dict, Optional, Sequence, Tuple, Union
from onnx import ModelProto
from .interpreter import DynamoInterpreter
from .graph_builder import GraphBuilder, OptimizationOptions


def _retrieve(
    name: str, weights: Dict[str, "torch.Tensor"], mapping: Dict[str, str]  # noqa: F821
) -> "torch.Tensor":  # noqa: F821
    if name not in mapping:
        raise RuntimeError(
            f"Unable to find {name!r}. " f"Available weights: {list(sorted(weights))}."
        )
    weight = mapping[name]
    if weight not in weights:
        if weight.startswith("L__self___") and weight[len("L__self___") :] in weights:
            weight = weight[len("L__self___") :]
    if weight not in weights:
        raise ValueError(
            f"Unexpected name {name!r} for input "
            f"{name!r} mapped to weight {weight!r}, "
            f"cannot be found in {', '.join(sorted(weights))}."
        )
    import torch

    value = weights[weight]
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"Unexpected type {type(value)} for input "
            f"{name!r} mapped to weight {weight!r}."
        )
    return value


def _make_builder_interpreter(
    mod: Union["torch.nn.Module", "torch.fx.GraphModule"],  # noqa: F821
    args: Optional[Sequence["torch.Tensor"]] = None,  # noqa: F821
    input_names: Optional[Sequence[str]] = None,
    target_opset: Union[int, Dict[str, int]] = 18,
    as_function: bool = False,
    optimization_options: Optional[OptimizationOptions] = None,
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
    :return: onnx model
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
        import torch.export

    if isinstance(mod, torch.fx.GraphModule):
        graph_module = mod
        weights = dict(graph_module.named_parameters())
        mapping = {}
    else:
        exported_mod = torch.export.export(mod, args)
        graph_module = exported_mod.graph_module
        try:
            weights = dict(exported_mod.named_parameters())
        except AttributeError:
            weights = dict(mod.named_parameters())
        signature = exported_mod.graph_signature
        mapping = signature.inputs_to_parameters

    def retrieve(name, weights=weights, mapping=mapping):
        return _retrieve(name, weights, mapping)

    builder = GraphBuilder(
        target_opset,
        input_names=input_names,
        as_function=as_function,
        optimization_options=optimization_options,
    )
    interpreter = DynamoInterpreter(builder, retrieve)
    return graph_module, builder, interpreter


def to_onnx(
    mod: Union["torch.nn.Module", "torch.fx.GraphModule"],  # noqa: F821
    args: Sequence["torch.Tensor"],  # noqa: F821
    input_names: Optional[Sequence[str]] = None,
    target_opset: Union[int, Dict[str, int]] = 18,
    as_function: bool = False,
    remove_unused: bool = False,
    constant_folding: bool = False,
) -> ModelProto:
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
    :return: onnx model
    """
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
    )

    builder.process(graph_module, interpreter)
    return builder.to_onnx()
