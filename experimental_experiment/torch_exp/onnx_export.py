import warnings
from typing import Dict, Optional, Sequence, Union
from onnx import ModelProto
from .walker import DynamoWalker
from .graph_builder import GraphBuilder


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
    :param removed_unused: if True, remove unused nodes
    :param constant_folding: if True, fold constants,
        constants are detected while converting the model
    :return: onnx model
    """
    if as_function:
        raise NotImplementedError("Export to FunctionProto is not implemented yet.")
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

    def retrieve(name: str, weights=weights, mapping=mapping) -> torch.Tensor:
        if name not in mapping:
            raise RuntimeError(
                f"Unable to find {name!r}. "
                f"Available weights: {list(sorted(weights))}."
            )
        weight = mapping[name]
        if weight not in weights:
            if (
                weight.startswith("L__self___")
                and weight[len("L__self___") :] in weights
            ):
                weight = weight[len("L__self___") :]
        if weight not in weights:
            raise ValueError(
                f"Unexpected name {name!r} for input "
                f"{name!r} mapped to weight {weight!r}, "
                f"cannot be found in {', '.join(sorted(weights))}."
            )
        value = weights[weight]
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Unexpected type {type(value)} for input "
                f"{name!r} mapped to weight {weight!r}."
            )
        return value

    builder = GraphBuilder(target_opset, input_names=input_names)
    walker = DynamoWalker(builder, retrieve)

    print("-----------")
    print(mapping)
    print({k: type(v) for k, v in weights.items()})
    for node in graph_module.graph.nodes:
        print("+", node.op, node.name)

    for node in graph_module.graph.nodes:
        walker(node)

    builder.remove_identity_nodes()
    if remove_unused:
        builder.remove_unused()
    if constant_folding:
        builder.constant_folding()
        if remove_unused:
            builder.remove_unused()
    return builder.to_onnx()
