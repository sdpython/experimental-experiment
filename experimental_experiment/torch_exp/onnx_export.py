import warnings
from typing import Dict, Optional, Sequence, Union
from onnx import ModelProto
from .walker import DynamoWalker
from .graph_builder import GraphBuilder


def to_onnx(
    mod: "torch.nn.Module",  # noqa: F821
    args: Sequence["torch.Tensor"],  # noqa: F821
    input_names: Optional[Sequence[str]] = None,
    target_opset: Union[int, Dict[str, int]] = 18,
    as_function: bool = False,
    remove_unused: bool = False,
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
    :return: onnx model
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch.export

    exported_mod = torch.export.export(mod, args)
    signature = exported_mod.graph_signature
    mapping = signature.inputs_to_parameters
    try:
        weights = dict(exported_mod.named_parameters())
    except AttributeError:
        weights = dict(mod.named_parameters())

    def retrieve(name):
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

    graph_module = exported_mod.graph_module
    for node in graph_module.graph.nodes:
        walker(node)

    if remove_unused:
        builder.remove_unused()
    return builder.to_onnx()
