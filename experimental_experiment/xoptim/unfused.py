import sys
from typing import List, Union
import onnx
import onnx.helper as oh
from ..xbuilder import GraphBuilder


def unfused_nodes(
    onx: Union[GraphBuilder, onnx.ModelProto],
    input_names: List[str],
    output_names: List[str],
    as_proto: bool = False,
) -> Union[List[onnx.NodeProto], onnx.ModelProto]:
    """
    Extracts from a model all the nodes starting from ``input_names`` and
    producing ``output_names``.

    This function does not handle subgraphs.

    :param onx: model
    :param input_names: list of inputs to consider
    :param output_names: list of outputs to consider
    :param as_proto: produces a ModelProto instead of a list of nodes
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
            fused = unfused_nodes(not_optimized, input_names, output_names, as_proto=True)

            print("-- save onnx")
            onnx.save(fused, f"unfused_{node_type}.onnx")

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
        return unfused_nodes(GraphBuilder(onx), input_names, output_names, as_proto=as_proto)
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

    if not as_proto:
        return keep

    # For a whole model, let's consider the missing inputs.
    input_names_all = input_names.copy()
    known = set(input_names)
    for node in keep:
        for i in node.input:
            if i not in known:
                known.add(i)
                input_names_all.append(i)
            known |= set(node.output)

    init = {}
    shapes = {}
    for name in [*input_names_all, *output_names]:
        if onx.is_constant(name):
            init[name] = onx.get_constant(name)
            continue
        shapes[name] = onx.make_tensor_value_info_from_name(name)

    inits, large_inits = onx._build_initializers(
        switch_low_high=sys.byteorder != "big",
        large_model=False,
        subset=set(init),
        external_threshold=1024,
    )
    assert (
        not large_inits
    ), f"Not yet implemeted with large initializers large_inits={set(large_inits)}"
    model = oh.make_model(
        oh.make_graph(
            keep,
            "unfused_nodes",
            [shapes[n] for n in input_names_all if n not in init],
            [shapes[n] for n in output_names],
        ),
        ir_version=onx.ir_version,
        opset_imports=[oh.make_opsetid(*o) for o in onx.opsets.items()],
    )
    model.graph.initializer.extend(inits)
    return model
