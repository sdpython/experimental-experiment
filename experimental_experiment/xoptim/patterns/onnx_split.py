import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ...helpers import make_idn
from ..patterns_api import MatchResult, PatternOptimization


class SlicesSplitPattern(PatternOptimization):
    """
    Merges multiple parallel slices into a split.

    Model with nodes to be fused:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from onnx_array_api.translate_api.make_helper import make_node_extended

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "transpose_1", onnx.TensorProto.FLOAT16, shape=(2, 2, 1024, 512)
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init7_s1_0"],
                value=onh.from_array(np.array([0], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init7_s1_256"],
                value=onh.from_array(np.array([256], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init7_s1_3"],
                value=onh.from_array(np.array([3], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init7_s1_9223372036854775807"],
                value=onh.from_array(
                    np.array([9223372036854775807], dtype=np.int64), name="value"
                ),
            )
        )
        nodes.append(
            make_node_extended(
                "Slice",
                ["transpose_1", "init7_s1_0", "init7_s1_256", "init7_s1_3"],
                ["slice_11"],
            )
        )
        nodes.append(
            make_node_extended(
                "Slice",
                ["transpose_1", "init7_s1_256", "init7_s1_9223372036854775807", "init7_s1_3"],
                ["slice_12"],
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "slice_11", onnx.TensorProto.FLOAT16, shape=(2, 2, 1024, 256)
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "slice_12", onnx.TensorProto.FLOAT16, shape=(2, 2, 1024, 256)
            )
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print(to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from onnx_array_api.translate_api.make_helper import make_node_extended

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "transpose_1", onnx.TensorProto.FLOAT16, shape=(2, 2, 1024, 512)
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init7_s2_256_256"],
                value=onh.from_array(np.array([256, 256], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Split", ["transpose_1", "init7_s2_256_256"], ["slice_11", "slice_12"], axis=3
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "slice_11", onnx.TensorProto.FLOAT16, shape=(2, 2, 1024, 256)
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "slice_12", onnx.TensorProto.FLOAT16, shape=(2, 2, 1024, 256)
            )
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print(to_dot(model))
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Slice" or node.domain != "":
            return self.none()

        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        users = [
            op for op in g.next_nodes(node.input[0]) if op.op_type == "Slice" and op.domain == ""
        ]
        if len(users) <= 1:
            return self.none(node, inspect.currentframe().f_lineno)

        for user in users:
            if len(user.input) == 4:
                continue
            if len(user.input) == 5:
                if not g.is_constant_scalar(user.input[-1]):
                    return self.none(node, inspect.currentframe().f_lineno)
                scalar = g.get_constant_scalar(user.input[-1])
                if scalar != 1:
                    return self.none(node, inspect.currentframe().f_lineno)
                continue
            return self.none(node, inspect.currentframe().f_lineno)

        # axis
        if all(len(op.input) == 2 for op in users):
            axis = 0
        else:
            axes = [op.input[3] for op in users]
            if any(not g.is_constant_scalar(a) for a in axes):
                return self.none(node, inspect.currentframe().f_lineno)

            csts = [g.get_constant_scalar(a) for a in axes]
            if len(set(csts)) != 1:
                return self.none(node, inspect.currentframe().f_lineno)

            axis = csts[0]

        shape = g.get_shape(node.input[0])
        dim = shape[axis]
        if not isinstance(dim, int):
            return self.none(node, inspect.currentframe().f_lineno)

        # starts, ends
        starts = [op.input[1] for op in users]
        ends = [op.input[2] for op in users]

        if not g.is_constant_scalar(starts[0], 0):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(ends[-1]):
            return self.none(node, inspect.currentframe().f_lineno)
        last = g.get_constant_scalar(ends[-1])
        if last not in (dim, 9223372036854775807):
            # 9223372036854775807 is what torch uses to specify the end
            return self.none(node, inspect.currentframe().f_lineno)

        if any(not g.is_constant(i) for i in starts) or any(not g.is_constant(i) for i in ends):
            # no constants
            return self.none(node, inspect.currentframe().f_lineno)

        cst_starts = [None for a in starts]
        cst_ends = [None for a in ends]
        for i in range(len(starts) - 1):
            if ends[i] == starts[i + 1]:
                continue
            end = cst_ends[i] or g.get_computed_constant(ends[i])
            start = cst_starts[i + 1] or g.get_computed_constant(starts[i + 1])
            if all(end == start):
                cst_ends[i] = end
                cst_starts[i + 1] = start
                continue
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, users, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        *nodes: NodeProto,
    ) -> List[NodeProto]:
        # nodes are all slices

        starts = [op.input[1] for op in nodes]
        ends = [op.input[2] for op in nodes]
        cst_starts = [g.get_constant_scalar(a) for a in starts]
        cst_ends = [g.get_constant_scalar(a) for a in ends]
        axis = g.get_constant_scalar(nodes[0].input[3])
        if cst_ends[-1] == 9223372036854775807:
            # 9223372036854775807 is what torch uses to specify the end
            shape = g.get_shape(nodes[0].input[0])
            cst_ends[-1] = shape[axis]
        n_els = [cst_ends[i] - cst_starts[i] for i in range(len(starts))]

        splits = g.make_initializer(
            "", np.array(n_els, dtype=np.int64), source="SlicesSplitPattern.apply.splits"
        )
        outputs = [op.output[0] for op in nodes]
        node = g.make_node(
            "Split",
            [nodes[0].input[0], splits],
            outputs,
            axis=axis,
            name=f"{self.__class__.__name__}--{nodes[0].name}",
        )
        return [node]


class SplitConcatPattern(PatternOptimization):
    """
    Replaces Split + Concat into identity if this is equivalent.

    Model with nodes to be fused:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from onnx_array_api.translate_api.make_helper import make_node_extended

        opset_imports = [
            oh.make_opsetid("", 18),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b")))
        nodes.append(make_node_extended("Split", ["X"], ["s1", "s2"], axis=-1, num_outputs=2))
        nodes.append(make_node_extended("Concat", ["s1", "s2"], ["Y"], axis=-1))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "b")))
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print(to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from onnx_array_api.translate_api.make_helper import make_node_extended

        opset_imports = [
            oh.make_opsetid("", 18),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b")))
        nodes.append(make_node_extended("Identity", ["X"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "b")))
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print(to_dot(model))
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Split" or node.domain != "":
            return self.none()

        only_id = None
        only_node = None
        for o in node.output:
            n = g.next_nodes(o)
            if len(n) != 1:
                return self.none(node, inspect.currentframe().f_lineno)
            i = make_idn(n[0])
            if only_id is None:
                only_id = i
                only_node = n[0]
            elif i != only_id:
                return self.none(node, inspect.currentframe().f_lineno)

        if only_node.op_type != "Concat" or only_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        axis_split = g.get_attribute(node, "axis").i
        axis_concat = g.get_attribute(only_node, "axis").i
        if axis_split < 0 and axis_concat >= 0:
            axis_split += g.get_rank(node.input[0])
        if axis_concat < 0 and axis_split >= 0:
            axis_concat += g.get_rank(node.input[0])
        if axis_split != axis_concat:
            return self.none(node, inspect.currentframe().f_lineno)
        if node.output != only_node.input:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, only_node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        split_node: NodeProto,
        concat_node: NodeProto,
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "Identity",
                split_node.input,
                concat_node.output,
                name=f"{self.__class__.__name__}--{split_node.name}",
            )
        ]
