import warnings
from typing import Dict, Tuple, Union
from onnx import ModelProto
from .walker import DynamoWalker
from .graph_builder import GraphBuilder


def to_onnx(
    mod: "torch.nn.Module",  # noqa: F821
    args: Tuple["torch.Tensor"],  # noqa: F821
    target_opset: Union[int, Dict[str, int]] = 18,
    as_function: bool = False,
) -> ModelProto:
    """
    Exports a torch model into ONNX using
    `dynamo export
    <https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html>`_.

    :param mod: torch module
    :param target_opset: targeted opset or targeted opsets as a dictionary
    :param as_function: export as a ModelProto or a FunctionProto
    :return: onnx model
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch.export

    exported_mod = torch.export.export(mod, args)
    signature = exported_mod.graph_signature
    mapping = signature.inputs_to_parameters
    weights = dict(exported_mod.named_parameters())

    def retrieve(name):
        weight = mapping[name]
        value = weights[weight]
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Unexpected type {type(value)} for input "
                f"{name!r} mapped to weight {weight!r}."
            )
        return value.detach().numpy()

    builder = GraphBuilder(target_opset)
    walker = DynamoWalker(builder, retrieve)

    graph_module = exported_mod.graph_module
    for node in graph_module.graph.nodes:
        walker(node)

    return builder.to_onnx()
