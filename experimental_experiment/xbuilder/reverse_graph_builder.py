from typing import Any, Dict, List, Union
import onnx
from onnx_array_api.translate_api.translate import Translater
from onnx_array_api.translate_api.builder_emitter import BuilderEmitter


class CustomBuilderEmitter(BuilderEmitter):
    """
    Custom :class:`onnx_array_api.translate_api.builder_emitter.BuilderEmitter`.
    """

    def __init__(self, make_model_function: str = "make_my_model"):
        super().__init__(make_model_function="make_my_model")

    def _emit_node_type(self, op_type, op_domain):
        if op_type in {"Squeeze", "Unsqueeze"} or op_type.startswith("Reduce"):
            return f"{op_type}AnyOpset"
        return op_type

    def _clean_result_name(self, name):
        return name.replace("#", "__").replace("-", "_")

    def _emit_end_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        rows = super()._emit_end_function(**kwargs)
        return [
            *rows[:-1],
            "    opts = FunctionOptions(",
            f"        name={self.f_name!r},",
            f"        domain={self.f_domain!r},",
            "        move_initializer_to_constant=True,",
            "    )",
            "    g.make_local_function(gr, opts, optimize=False)",
        ]


def to_graph_builder_code(proto: onnx.ModelProto, function_name: str = "build_model") -> str:
    """
    Produces a code building a model with
    :class:`experimental_experiment.xbuilder.GraphBuilder`.

    :param proto: model to convert into a code
    :param function_name: function name
    :return: str

    Example (see also :ref:`l-plot-model-to-code`):

    .. runpython::
        :showcode:

        import numpy as np
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from experimental_experiment.xbuilder.reverse_graph_builder import (
            to_graph_builder_code,
        )

        TFLOAT = onnx.TensorProto.FLOAT
        TINT64 = onnx.TensorProto.INT64

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ConstantOfShape",
                        ["shape"],
                        ["cst"],
                        value=onh.from_array(np.array([0], dtype=np.float32)),
                    ),
                    oh.make_node(
                        "ScatterND",
                        ["cst", "indices", "updates"],
                        ["Z"],
                        reduction="add",
                    ),
                ],
                "create_graph",
                [
                    oh.make_tensor_value_info("shape", TINT64, [None]),
                    oh.make_tensor_value_info("indices", TINT64, [None, None]),
                    oh.make_tensor_value_info("updates", TFLOAT, [None, None, None]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        print(to_graph_builder_code(model))
    """
    tr = Translater(proto, emitter=CustomBuilderEmitter())
    code = tr.export(as_str=True)
    return "\n".join(
        [
            "import numpy as np",
            "from onnx import TensorProto",
            "from onnx.numpy_helper import from_array",
            "from experimental_experiment.xbuilder import GraphBuilder, FunctionOptions",
            "",
            "",
            code.replace("array(nan", "array(np.nan"),
        ]
    )


def to_graph_pattern_matching(
    proto: Union[onnx.FunctionProto, onnx.GraphProto, onnx.ModelProto],
) -> str:
    """
    Produces a code matching a pattern.

    :param proto: model to convert into a code
    :return: str

    Example (see also :ref:`l-plot-model-to-code`):

    .. runpython::
        :showcode:

        import numpy as np
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from experimental_experiment.xbuilder.reverse_graph_builder import (
            to_graph_pattern_matching,
        )

        TFLOAT = onnx.TensorProto.FLOAT
        TINT64 = onnx.TensorProto.INT64

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ConstantOfShape",
                        ["shape"],
                        ["cst"],
                        value=onh.from_array(np.array([0], dtype=np.float32)),
                    ),
                    oh.make_node(
                        "ScatterND",
                        ["cst", "indices", "updates"],
                        ["Z"],
                        reduction="add",
                    ),
                ],
                "create_graph",
                [
                    oh.make_tensor_value_info("shape", TINT64, [None]),
                    oh.make_tensor_value_info("indices", TINT64, [None, None]),
                    oh.make_tensor_value_info("updates", TFLOAT, [None, None, None]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        print(to_graph_pattern_matching(model))
    """
    if isinstance(proto, onnx.FunctionProto):
        nodes = proto.node
        input_names = proto.input
        output_names = proto.output
        inits = set()
        raise NotImplementedError("Not yet implemented for FunctionProto.")
    elif isinstance(proto, onnx.GraphProto):
        nodes = proto.node
        input_names = [i.name for i in proto.input]
        output_names = [i.name for i in proto.output]
        inits = proto.initializer
    elif isinstance(proto, onnx.ModelProto):
        nodes = proto.graph.node
        input_names = [i.name for i in proto.graph.input]
        output_names = [i.name for i in proto.graph.output]
        inits = proto.graph.initializer
    else:
        raise TypeError(f"Unable to process type {type(proto)}.")
    assert nodes, "No node to process."
    assert len(output_names) == 1, (
        f"Function is not implemented yet for "
        f"input_names={input_names!r} and output_names={output_names!r}"
    )

    def _clean(s: str) -> str:
        return s.replace(".", "_").replace("-", "_")

    matches = {
        (node.op_type, node.domain, tuple(node.input), tuple(node.output)): False
        for node in nodes
    }
    position = {
        (node.op_type, node.domain, tuple(node.input), tuple(node.output)): i
        for i, node in enumerate(nodes)
    }
    outside = set(input_names) | set(i.name for i in inits)
    successors = {}
    predecessors = {}
    for node in nodes:
        for i in node.input:
            if i not in successors:
                successors[i] = []
            successors[i].append(node)
        for i in node.output:
            predecessors[i] = node

    first_node = True
    rows = []
    stack_names = [*output_names]
    nodes_names = []
    while stack_names:
        rows.append("")
        name = stack_names.pop()
        if name not in predecessors:
            # stop here
            rows.append(f"# {_clean(name)} has no predecessor.")
            continue

        if name not in outside and name in successors and len(successors[name]) == 1:
            rows.extend(
                [
                    f"if g.is_used_more_than_once({_clean(name)}):",
                    "    return self.none(node, inspect.currentframe().f_lineno)",
                ]
            )

        node = predecessors[name]
        if not node.input:
            # A constant. We skip.
            continue
        key = node.op_type, node.domain, tuple(node.input), tuple(node.output)
        matched = matches[key]
        if matched:
            # We skip for the time being but we should do extract verification.
            rows.append(f"# {_clean(name)} is already processed.")
            continue
        node_name = f"node_{position[key]}_{node.op_type}"
        nodes_names.append(node_name)
        if first_node:
            first_node = False
            assert not matched, f"Algorithm issues, matches={matches}, key={key}"
            rows.extend(
                [
                    f"{node_name} = node",
                    (
                        f"if {_clean(node_name)}.op_type != {node.op_type!r} or "
                        f"{_clean(node_name)}.domain != {node.domain!r}:"
                    ),
                    "    return self.none()",
                ]
            )
            matches[key] = True
            stack_names.extend(node.input)
            for i_, n_ in enumerate(node.input):
                rows.append(f"{_clean(n_)} = {_clean(node_name)}.input[{i_}]")
            continue

        # Another node

        rows.extend(
            [
                f"{_clean(node_name)} = g.node_before({_clean(name)})",
                (
                    f"if {_clean(node_name)} is None or {_clean(node_name)}.op_type != "
                    f"{node.op_type!r} or {_clean(node_name)}.domain != {node.domain!r}:"
                ),
                ("    return self.none(node, inspect.currentframe().f_lineno)"),
            ]
        )
        matches[key] = True
        stack_names.extend(node.input)
        for i_, n_ in enumerate(node.input):
            rows.append(f"{_clean(n_)} = {_clean(node_name)}.input[{i_}]")
        continue

    rows.extend(
        [
            "",
            "# list of nodes",
            f"nodes = [{', '.join(map(_clean,nodes_names[::-1]))}]",
        ]
    )
    return "\n".join(rows)
