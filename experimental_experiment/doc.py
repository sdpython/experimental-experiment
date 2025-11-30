from typing import Set
import onnx
from onnx_diagnostic.helpers.onnx_helper import onnx_dtype_name


def _get_hidden_inputs(graph: onnx.GraphProto) -> Set[str]:
    hidden = set()
    memo = (
        {i.name for i in graph.initializer}
        | {i.values.name for i in graph.sparse_initializer}
        | {i.name for i in graph.input}
    )
    for node in graph.node:
        for i in node.input:
            if i not in memo:
                hidden.add(i)
        for att in node.attribute:
            if att.type == onnx.AttributeProto.GRAPH and att.g:
                hid = _get_hidden_inputs(att.g)
                less = set(h for h in hid if h not in memo)
                hidden |= less
        memo |= set(node.output)
    return hidden


def _make_node_label(node: onnx.NodeProto) -> str:
    els = [f"{node.domain}.{node.op_type}" if node.domain else node.op_type, "("]
    ee = ["." if i else "" for i in node.input]
    for att in node.attribute:
        if att.name == "to":
            ee.append(f"{att.name}={onnx_dtype_name(att.i)}")
        elif att.name in {"to", "axis", "value_int", "stash_type"}:
            ee.append(f"{att.name}={att.i}")
        elif att.name in {"value_float"}:
            ee.append(f"{att.name}={att.f}")
        elif att.name in {"value_floats"}:
            ee.append(f"{att.name}={att.floats}")
        elif att.name in {"value_ints", "perm"}:
            ee.append(f"{att.name}={att.ints}")
    els.append(", ".join(ee))
    els.append(")")
    return "".join(els)


def to_dot(model: onnx.ModelProto) -> str:
    """Converts a model into a dot graph."""
    model = onnx.shape_inference.infer_shapes(model)

    edge_label = {}
    for val in model.graph.value_info:
        itype = val.type.tensor_type.elem_type
        if itype == onnx.TensorProto.UNDEFINED:
            continue
        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value for d in val.type.tensor_type.shape.dim
        )
        sshape = ",".join(
            map(str, [("?" if isinstance(s, str) and s.startswith("unk") else s) for s in shape])
        )
        edge_label[val.name] = f"{onnx_dtype_name(itype)}({sshape})"

    rows = [
        "digraph {",
        (
            "  graph [rankdir=TB, splines=true, overlap=false, nodesep=0.2, "
            "ranksep=0.2, fontsize=8];"
        ),
        '  node [style="rounded,filled", color="#888888", fontcolor="#222222", shape=box];',
        "  edge [arrowhead=vee, fontsize=6];",
    ]
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    nodes = list(model.graph.node)
    inits = list(model.graph.initializer)
    name_to_ids = {}
    for inp in inputs:
        rows.append(f'  I_{id(inp)} [label="{inp.name}", fillcolor="#eeeeaa"];')
        name_to_ids[inp.name] = f"I_{id(inp)}"
    for init in inits:
        rows.append(f'  i_{id(init)} [label="{init.name}", fillcolor="#cccc00"];')
        name_to_ids[init.name] = f"i_{id(init)}"
    for node in nodes:
        label = _make_node_label(node)
        rows.append(f'  {node.op_type}_{id(node)} [label="{label}", fillcolor="#cccccc"];')
        name_to_ids.update({o: f"{node.op_type}_{id(node)}" for o in node.output if o})

    # nodes
    done = set()
    for node in nodes:
        names = list(node.input)
        for i in names:
            edge = name_to_ids[i], f"{node.op_type}_{id(node)}"
            if edge in done:
                continue
            done.add(edge)
            lab = edge_label.get(i, "")
            if lab:
                lab = f' [label="{lab}"]'
            rows.append(f"  {edge[0]} -> {edge[1]}{lab};")
        if node.op_type in {"Scan", "Loop", "If"}:
            unique = set()
            for att in node.attribute:
                if att.type == onnx.AttributeProto.GRAPH:
                    unique |= _get_hidden_inputs(att.g)
            for i in unique:
                edge = name_to_ids[i], id(node)
                if edge in done:
                    continue
                done.add(edge)
                rows.append(f"  {edge[0]} -> {edge[1]} [style=dotted];")

    # outputs
    for out in outputs:
        rows.append(f'  O_{id(out)} [label="{out.name}", fillcolor="#aaaaee"];')
        edge = name_to_ids[out.name], f"O_{id(out)}"
        rows.append(f"  {edge[0]} -> {edge[1]};")

    rows.append("}")
    return "\n".join(rows)
