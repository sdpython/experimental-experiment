from typing import List, Union
import onnx
from ..xbuilder import GraphBuilder


def unfused_nodes(
    onx: Union[GraphBuilder, onnx.ModelProto], input_names: List[str], output_names: List[str]
) -> List[onnx.NodeProto]:
    """
    Extracts from a model all the nodes starting from ``input_names`` and
    producing ``output_names``.

    This function does not handle subgraphs.

    :param onx: model
    :param input_names: list of inputs to consider
    :param output_names: list of outputs to consider
    :return: list of involved nodes

    One code example:

    .. code-block:: python

        import onnx
        from experimental_experiment.xoptim.unfused import unfused_nodes
        from experimental_experiment.helpers import pretty_onnx


        print("-- load optmized model")
        optimized = onnx.load("<optimized_model.onnx>")

        print("-- loading not optimized model")
        not_optimized = onnx.load("<not_optimized_model.onnx>")
        print("-- done")


        def look_for_pattern(
            not_optimized: onnx.ModelProto, optimized: onnx.ModelProto, node_type: str
        ):
            print()
            print(f"-- looking for fused nodes, type={node_type!r}")

            fused_node = None
            for node in optimized.graph.node:
                if node.op_type == node_type:
                    fused_node = node
                    input_names = [i for i in node.input if i]
                    output_names = [o for o in node.output if o]
                    break

            assert input_names, "No fused node was found."

            print("-- fused_node")
            print(
                pretty_onnx(
                    fused_node, with_attributes=True,
                    highlight=set(input_names) | set(output_names),
                )
            )
            print(f"-- input_names={input_names}")
            print(f"-- output_names={output_names}")
            print("--")

            print("-- looking for fused nodes")
            fused = unfused_nodes(not_optimized, input_names, output_names)

            print("--")
            print(f"-- found {len(fused)} nodes")
            for node in fused:
                print(
                    pretty_onnx(
                        node, with_attributes=True,
                        highlight=set(input_names) | set(output_names),
                    )
                )
            print("--")


        look_for_pattern(not_optimized, optimized, "Attention")
        look_for_pattern(not_optimized, optimized, "SkipLayerNormalization")
    """
    if isinstance(onx, onnx.ModelProto):
        return unfused_nodes(GraphBuilder(onx), input_names, output_names)
    assert isinstance(onx, GraphBuilder), f"Unexpected type {type(onx)} for onx."

    # First step, we go backward.
    nodes = []
    needed = set(output_names)
    not_needed = set(input_names)
    for node in onx.nodes[::-1]:
        so = set(node.output)
        if so & needed:
            nodes.append(node)
            needed |= set(i for i in node.input if i not in not_needed)
            needed -= needed & set(node.output)
        elif so & not_needed == so:
            not_needed |= set(node.input)
        if not needed:
            break

    # Then we go forward.
    keep = []
    input_involded = set(input_names)
    for node in reversed(nodes):
        if set(node.input) & input_involded:
            keep.append(node)
            input_involded |= set(node.output)
    return keep
