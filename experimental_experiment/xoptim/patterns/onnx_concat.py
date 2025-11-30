import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ...xshape._onnx_helper import unary_like_op_types
from ..patterns_api import MatchResult, PatternOptimization


class ConcatGatherPattern(PatternOptimization):
    """
    Checks if Gather(Concat) can be replaced by Identity.

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
        inputs.append(oh.make_tensor_value_info("D2", onnx.TensorProto.INT64, shape=(1,)))
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["un"],
                value=onh.from_array(np.array([1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(make_node_extended("Concat", ["D1", "D2"], ["d"], axis=0))
        nodes.append(make_node_extended("Gather", ["d", "un"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.INT64, shape=(1,)))
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
        inputs.append(oh.make_tensor_value_info("D2", onnx.TensorProto.INT64, shape=(1,)))
        nodes.append(make_node_extended("Identity", ["D2"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.INT64, shape=(1,)))
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

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gather" or node.domain != "":
            return self.none()
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(node.input[1])
        if cst is None or cst.dtype != np.int64 or cst.shape != (1,):
            return self.none(node, inspect.currentframe().f_lineno)
        before = g.node_before(node.input[0])
        if before.op_type != "Concat":
            return self.none(node, inspect.currentframe().f_lineno)
        if any(not g.has_shape(i) for i in before.input):
            return self.none(node, inspect.currentframe().f_lineno)
        if any(g.get_shape(i) != (1,) for i in before.input):
            return self.none(node, inspect.currentframe().f_lineno)
        assert cst[0] < len(before.input), (
            f"Concat concatenates many dimensions into one but "
            f"cst={cst} and before.input={before.input}"
        )
        return MatchResult(self, [before, node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        concat_node: NodeProto,
        gather_node: NodeProto,
    ) -> List[NodeProto]:
        index = g.get_constant_scalar(gather_node.input[1])
        new_node = g.make_node(
            "Identity",
            [concat_node.input[index]],
            gather_node.output,
            name=f"{self.__class__.__name__}--{gather_node.name}",
            doc_string=gather_node.doc_string,
        )
        return (
            [concat_node, new_node]
            if g.is_used_more_than_once(concat_node.output[0])
            else [new_node]
        )


class ConcatEmptyPattern(PatternOptimization):
    """
    Checks if one of the concatenated values is empty.

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
        inputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.INT64, shape=("b",)))
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.INT64, shape=("a",)))
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["I"],
                value=onh.from_array(np.array([], dtype=np.int64), name="value"),
            )
        )
        nodes.append(make_node_extended("Concat", ["X", "Y", "I"], ["Z"], axis=0))
        outputs.append(oh.make_tensor_value_info("Z", onnx.TensorProto.INT64, shape=("c",)))
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
        inputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.INT64, shape=("b",)))
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.INT64, shape=("a",)))
        nodes.append(make_node_extended("Concat", ["X", "Y"], ["Z"], axis=0))
        outputs.append(oh.make_tensor_value_info("Z", onnx.TensorProto.INT64, shape=("c",)))
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
        if node.op_type != "Concat" or node.domain != "":
            return self.none()
        rem = self.remove_set(g, node)
        if not rem:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def remove_set(self, g, node):
        att = g.get_attribute(node, "axis")
        axis = att.i
        rem = set()
        for idi, i in enumerate(node.input):
            if not g.has_shape(i):
                continue
            shape = g.get_shape(i)
            if axis < len(shape) and shape[axis] == 0:
                rem.add(idi)
        return rem

    def apply(self, g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
        rem = self.remove_set(g, node)
        assert rem, f"rem is empty for node={node}"
        new_inputs = [n for i, n in enumerate(node.input) if i not in rem]
        if len(rem) == len(node.input) - 1:
            # Identity
            return [
                g.make_node(
                    "Identity",
                    new_inputs,
                    node.output,
                    name=f"{self.__class__.__name__}--{node.name}",
                    doc_string=node.doc_string,
                )
            ]
        new_node = g.make_node(
            "Concat",
            new_inputs,
            node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        new_node.attribute.extend(node.attribute)
        return [new_node]


class ConcatTwiceUnaryPattern(PatternOptimization):
    """
    Sin(Concat(x,x)) -> Concat(Sin(x), Sin(x)).

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
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("b", "c")))
        nodes.append(make_node_extended("Concat", ["X", "X"], ["xx"], axis=0))
        nodes.append(make_node_extended("Sin", ["xx"], ["xsin"]))
        outputs.append(
            oh.make_tensor_value_info("xsin", onnx.TensorProto.FLOAT, shape=("2*b", "c"))
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
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("b", "c")))
        nodes.append(make_node_extended("Sin", ["X"], ["uxsin"]))
        nodes.append(make_node_extended("Concat", ["uxsin", "uxsin"], ["xsin"], axis=0))
        outputs.append(
            oh.make_tensor_value_info("xsin", onnx.TensorProto.FLOAT, shape=("2*b", "c"))
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

    _unary_types = unary_like_op_types()
    _binary_types_scalar_cst = {"Mul", "Add", "Div", "Sub"}

    @classmethod
    def _valid_node(
        cls,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        concat: NodeProto,
        unary: NodeProto,
    ):
        if unary.op_type in cls._unary_types:
            return True
        if unary.op_type == "Unsqueeze" and unary.domain == "":
            if g.is_constant_scalar(unary.input[1]):
                cst = g.get_constant_scalar(unary.input[1])
                axis = g.get_attribute(concat, "axis").i
                if axis == -1 and cst != -1 and cst < g.get_rank(unary.input[0]):
                    return True
        if unary.op_type in cls._binary_types_scalar_cst and g.is_constant_scalar(unary.input[1]):
            return True
        return False

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            g.main_opset < 18
            or node.op_type != "Concat"
            or node.domain != ""
            or len(node.input) != 2
            or node.input[0] != node.input[1]
        ):
            return self.none()

        # Let's check what follows.
        nodes = [n for n in g.next_nodes(node.output[0]) if self._valid_node(g, node, n)]
        if nodes:
            return MatchResult(self, [node, nodes[0]], self.apply)
        return self.none(node, inspect.currentframe().f_lineno)

    def remove_set(self, g, node):
        att = g.get_attribute(node, "axis")
        axis = att.i
        rem = set()
        for idi, i in enumerate(node.input):
            if not g.has_shape(i):
                continue
            shape = g.get_shape(i)
            if axis < len(shape) and shape[axis] == 0:
                rem.add(idi)
        return rem

    def apply(
        self, g: "GraphBuilder", concat: NodeProto, unary: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        new_name = g.unique_name(f"u{unary.output[0]}")
        nodes = [
            g.make_node(
                unary.op_type,
                [concat.input[0], *unary.input[1:]],
                [new_name],
                name=f"{self.__class__.__name__}--{unary.name}",
                doc_string=unary.doc_string,
            ),
            g.make_node(
                concat.op_type,
                [new_name, new_name],
                [unary.output[0]],
                name=f"{self.__class__.__name__}--{concat.name}",
                doc_string=concat.doc_string,
            ),
        ]
        if unary.attribute:
            nodes[0].attribute.extend(unary.attribute)
        if concat.attribute:
            nodes[1].attribute.extend(concat.attribute)

        if g.is_used_more_than_once(concat.output[0]):
            return [concat, *nodes]
        return nodes
