import inspect
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import NodeProto
from ...xshape._onnx_helper import (
    element_wise_binary_op_types,
    element_wise_op_cmp_types,
    unary_like_op_types,
)
from ...xshape._shape_helper import all_int, DYNAMIC_SHAPE
from ..patterns_api import MatchResult, PatternOptimization


class ExpandPattern(PatternOptimization):
    """
    Checks that a Expand is really needed.

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
        inputs.append(
            oh.make_tensor_value_info("init7_s4_32_2_10_8", onnx.TensorProto.INT64, shape=(4,))
        )
        inputs.append(
            oh.make_tensor_value_info("mul", onnx.TensorProto.FLOAT, shape=(32, 2, 10, 8))
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init7_s4_32_2_10_8"],
                value=onh.from_array(np.array([32, 2, 10, 8], dtype=np.int64), name="value"),
            )
        )
        nodes.append(make_node_extended("Expand", ["mul", "init7_s4_32_2_10_8"], ["expand"]))
        outputs.append(
            oh.make_tensor_value_info("expand", onnx.TensorProto.FLOAT, shape=(32, 2, 10, 8))
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
        inputs.append(
            oh.make_tensor_value_info("init7_s4_32_2_10_8", onnx.TensorProto.INT64, shape=(4,))
        )
        inputs.append(
            oh.make_tensor_value_info("mul", onnx.TensorProto.FLOAT, shape=(32, 2, 10, 8))
        )
        nodes.append(make_node_extended("Identity", ["mul", "init7_s4_32_2_10_8"], ["expand"]))
        outputs.append(
            oh.make_tensor_value_info("expand", onnx.TensorProto.FLOAT, shape=(32, 2, 10, 8))
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

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape = g.get_shape(node.input[0])
        if not all_int(shape):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[1]):
            # It may be a symbolic shape.
            return self.none(node, inspect.currentframe().f_lineno)
        value = g.get_computed_constant(node.input[1])
        if value is None:
            return self.none(node, inspect.currentframe().f_lineno)
        with g.builder.maybe_disable_fake_tensor_mode():
            new_shape = tuple(int(i) for i in value)
        if shape != new_shape:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        new_node = g.make_node(
            "Identity",
            node.input,
            node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node]


class ExpandBroadcastPattern(PatternOptimization):
    """
    Checks that a Expand is really needed before an element wise operator.
    The objective is to save one allocation and let the next operator
    do the expansion by broadcasting one input.

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
        inputs.append(
            oh.make_tensor_value_info("mul_25", onnx.TensorProto.FLOAT, shape=(2, 1024, 1))
        )
        inputs.append(
            oh.make_tensor_value_info("input66", onnx.TensorProto.FLOAT, shape=(2, 1024, 1024))
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init7_s3_2_1024_1024"],
                value=onh.from_array(np.array([2, 1024, 1024], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            make_node_extended("Expand", ["mul_25", "init7_s3_2_1024_1024"], ["expand_11"])
        )
        nodes.append(
            make_node_extended("Mul", ["expand_11", "input66"], ["MulMulMulPattern--mul_27"])
        )
        outputs.append(
            oh.make_tensor_value_info(
                "MulMulMulPattern--mul_27", onnx.TensorProto.FLOAT, shape=(2, 1024, 1024)
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
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("mul_25", onnx.TensorProto.FLOAT, shape=(2, 1024, 1))
        )
        inputs.append(
            oh.make_tensor_value_info("input66", onnx.TensorProto.FLOAT, shape=(2, 1024, 1024))
        )
        nodes.append(
            make_node_extended("Mul", ["mul_25", "input66"], ["MulMulMulPattern--mul_27"])
        )
        outputs.append(
            oh.make_tensor_value_info(
                "MulMulMulPattern--mul_27", onnx.TensorProto.FLOAT, shape=(2, 1024, 1024)
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

    _op_types = element_wise_binary_op_types() | element_wise_op_cmp_types()

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape = g.get_shape(node.input[0])
        if not all_int(shape):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[1]):
            # It may be a symbolic shape.
            return self.none(node, inspect.currentframe().f_lineno)
        value = g.get_computed_constant(node.input[1])
        if value is None:
            return self.none(node, inspect.currentframe().f_lineno)
        with g.builder.maybe_disable_fake_tensor_mode():
            new_shape = tuple(int(i) for i in value)

        if g.is_used_more_than_once(node.output[0]):
            # More than one output, not handled right now.
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        assert len(next_nodes) == 1, "The previous test should have cleared out this case."
        next_node = next_nodes[0]

        if next_node.op_type not in self._op_types or next_node.domain != "":
            # Not an element wise operator.
            return self.none(node, inspect.currentframe().f_lineno)

        other = next_node.input[1 if next_node.input[0] == node.output[0] else 0]

        if not g.has_shape(other):
            return self.none(node, inspect.currentframe().f_lineno)

        other_shape = g.get_shape(other)
        if new_shape != other_shape:
            # Expand does not expand to the shape of the other element.
            return self.none(node, inspect.currentframe().f_lineno)
        if len(shape) != len(other_shape):
            # Different ranks.
            return self.none(node, inspect.currentframe().f_lineno)
        for a, b in zip(shape, other_shape):
            if not (a == b or a == 1 or b == 1):
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply, insert_at=next_node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        next_node: NodeProto,
    ) -> List[NodeProto]:
        if next_node.input[0] == node.output[0]:
            inputs = [node.input[0], next_node.input[1]]
        else:
            inputs = [next_node.input[0], node.input[0]]
        return [
            g.make_node(
                next_node.op_type,
                inputs,
                next_node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=next_node.doc_string,
            )
        ]


class ShapeBasedExpandBroadcastPattern(PatternOptimization):
    """
    Similar to
    :class:`experimental_experiment.xoptim.patterns.onnx_expand.ExpandBroadcastPattern`,
    but it allows dynamic shapes as well. It does not look into the second
    argument of Expand, it just infers than an expand is not needed for
    a binary operator following just after.
    """

    _op_types = element_wise_binary_op_types() | element_wise_op_cmp_types()

    @classmethod
    def _is_compatible_shapes_for_expand(
        cls,
        shape_left: DYNAMIC_SHAPE,
        shape_right: DYNAMIC_SHAPE,
        output_shape: Optional[DYNAMIC_SHAPE],
    ) -> bool:
        """
        Checks that the binary operations of the two input shapes returns the output_shape.
        Then no Expand node is needed.
        """
        if output_shape is None:
            return False
        if max(len(shape_left), len(shape_right) if shape_right else 0) < len(output_shape):
            return False
        # Align shapes
        if len(shape_left) < len(shape_right):
            shape_left = (1,) * (len(shape_right) - len(shape_left)) + shape_left
        elif len(shape_left) > len(shape_right):
            shape_right = (1,) * (len(shape_left) - len(shape_right)) + shape_right

        for left, right, out in zip(shape_left, shape_right, output_shape):
            if isinstance(left, int):
                if isinstance(right, int):
                    # static right
                    if left == 1:
                        if right != out:
                            return False
                    elif right == 1:
                        if left != out:
                            return False
                    else:
                        if left != right or left != out or right != out:
                            return False
                else:
                    # dynamic right
                    if left == 1:
                        if right != out:
                            return False
                    else:
                        if left != right or left != out or right != out:
                            return False
            else:
                # dynamic left
                if isinstance(right, int):
                    # static right
                    if right == 1:
                        if left != out:
                            return False
                    else:
                        if left != right or left != out or right != out:
                            return False
                else:
                    # dynamic right
                    if left != right or left != out or right != out:
                        return False
        return True

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in self._op_types or node.domain != "":
            return self.none()
        if (
            not g.has_shape(node.output[0])
            or not g.has_shape(node.input[0])
            or not g.has_shape(node.input[1])
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])
        before = [
            None if n is None or n.op_type != "Expand" else n for n in [node_left, node_right]
        ]
        if before == [None, None]:
            return self.none(node, inspect.currentframe().f_lineno)

        # At least one expand.
        node_left, node_right = before
        shape_left = g.get_shape_renamed(
            node.input[0] if node_left is None else node_left.input[0]
        )
        shape_right = g.get_shape_renamed(
            node.input[1] if node_right is None else node_right.input[0]
        )
        if self._is_compatible_shapes_for_expand(
            shape_left, shape_right, g.get_shape_renamed(node.output[0])
        ):
            if self.verbose:
                print(
                    f"[{self.__class__.__name__}.match] {shape_left} "
                    f"{node.op_type} {shape_right} -> {g.get_shape_renamed(node.output[0])}"
                )
            return MatchResult(self, [node_left, node_right, node], self.apply)
        # We could end up with the following case.
        # shape_left   = (1, 1, 'seq_length', 'cache_length + seq_length')
        # shape_right  = (1, 1, 'seq_length', 'cache_length + seq_length')
        # output_shape = ('batch', 1, 'seq_length', 'cache_length + seq_length')
        # When this happes, it could also be caught by another pattern.
        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        expand_left: NodeProto,
        expand_right: NodeProto,
        binary_node: NodeProto,
    ) -> List[NodeProto]:
        nodes = []
        if expand_left is not None and g.is_used_more_than_once(expand_left.output[0]):
            nodes.append(expand_left)
        if expand_right is not None and g.is_used_more_than_once(expand_right.output[0]):
            nodes.append(expand_right)
        assert (
            not binary_node.attribute
        ), f"Binary operator should not have any attribute, binary_node={binary_node}"
        return [
            *nodes,
            g.make_node(
                binary_node.op_type,
                [
                    binary_node.input[0] if expand_left is None else expand_left.input[0],
                    binary_node.input[1] if expand_right is None else expand_right.input[0],
                ],
                binary_node.output,
                name=f"{self.__class__.__name__}--{binary_node.name}",
                doc_string=binary_node.doc_string,
            ),
        ]


class ExpandSwapPattern(PatternOptimization):
    """
    Tries to move a node Expand forward in the graph.
    Expand + Exp can be changed into Exp + Expand.
    Then Exp applies on a tensor of a smaller or equal size.

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
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("p", onnx.TensorProto.INT64, shape=(1,)))
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(1, 5, 7)))
        inputs.append(oh.make_tensor_value_info("shape", onnx.TensorProto.INT64, shape=(3,)))
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["shape"],
                value=onh.from_array(np.array([3, 1, 1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["p"],
                value=onh.from_array(np.array([2], dtype=np.int64), name="value"),
            )
        )
        nodes.append(make_node_extended("Expand", ["X", "shape"], ["xs"]))
        nodes.append(make_node_extended("Pow", ["xs", "p"], ["Z"]))
        outputs.append(oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, shape=(3, 5, 7)))
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
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("p", onnx.TensorProto.INT64, shape=(1,)))
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(1, 5, 7)))
        inputs.append(oh.make_tensor_value_info("shape", onnx.TensorProto.INT64, shape=(3,)))
        nodes.append(make_node_extended("Pow", ["X", "p"], ["ExpandSwapPattern_X"]))
        nodes.append(make_node_extended("Expand", ["ExpandSwapPattern_X", "shape"], ["Z"]))
        outputs.append(oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, shape=(3, 5, 7)))
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

    _op_types = unary_like_op_types()
    _other_types = {"NegXplus1", "ReplaceZero", "Pow"}

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        assert g.is_used(node.output[0]), (
            f"The match should not even begin, {node.output[0]!r} "
            f"is not used among {node.output} and type={node.op_type!r}"
        )
        if g.is_used_more_than_once(node.output[0]):
            # More than one output so it probably must be done.
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        assert len(next_nodes) == 1, "The previous test should have cleared out this case."
        next_node = next_nodes[0]

        if next_node.op_type not in self._other_types and (
            next_node.op_type not in self._op_types or next_node.domain != ""
        ):
            # Not an unary wise operator.
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        next_node: NodeProto,
    ) -> List[NodeProto]:
        # We need to create a new name for the intermediate results.
        # The optimizer cannot reuse an existing name if the new result
        # has a different shape.
        new_name = g.unique_name(f"{self.__class__.__name__}_{node.input[0]}")
        unary = g.make_node(
            next_node.op_type,
            [node.input[0], *next_node.input[1:]],
            [new_name],
            name=f"{self.__class__.__name__}--{node.name}",
            domain=next_node.domain,
            doc_string=next_node.doc_string,
        )
        unary.attribute.extend(next_node.attribute)
        expand = g.make_node(
            node.op_type,  # Expand
            [new_name, node.input[1]],
            [next_node.output[0]],
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [unary, expand]


class ShapeBasedStaticExpandPattern(PatternOptimization):
    """
    Compares input and output shapes to tell if the expand
    can uses a constant as a second input.
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    @classmethod
    def _find_expand_shape(
        cls, sh1: Tuple[Union[str, int], ...], sh2: Tuple[Union[str, int], ...]
    ) -> Tuple[int, ...]:
        expand_shape = []
        for s1, s2 in zip(sh1, sh2):
            if s1 == s2:
                expand_shape.append(1)
                continue
            if not isinstance(s1, int) or not isinstance(s2, int):
                return None
            expand_shape.append(s2)
        return tuple(expand_shape)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()
        if g.is_constant(node.input[1]):
            # already done
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        sh1 = g.get_shape_renamed(node.input[0])
        sh2 = g.get_shape_renamed(node.output[0])
        if len(sh1) != len(sh2):
            # We ignore that case for the time being.
            return self.none(node, inspect.currentframe().f_lineno)
        expand_shape = self._find_expand_shape(sh1, sh2)
        if expand_shape is None:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        reshape: NodeProto,
    ) -> List[NodeProto]:
        expand_shape = self._find_expand_shape(
            g.get_shape_renamed(reshape.input[0]), g.get_shape_renamed(reshape.output[0])
        )
        new_shape = g.make_initializer(
            "",
            np.array(expand_shape, dtype=np.int64),
            source=f"{self.__class__.__name__}.m1",
        )
        return [
            g.make_node(
                "Expand",
                [reshape.input[0], new_shape],
                reshape.output,
                name=f"{self.__class__.__name__}--{reshape.name}",
                doc_string=reshape.doc_string,
            )
        ]


class ShapeBasedExpandSwapPattern(PatternOptimization):
    """
    Tries to move a node Expand forward in the graph
    for a binary operator. The code is similar to
    :class:`experimental_experiment.xoptim.patterns.onnx_expand.ShapeBasedExpandBroadcastPattern`

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
        inputs.append(
            oh.make_tensor_value_info("full_shape", onnx.TensorProto.INT64, shape=(2,))
        )
        inputs.append(oh.make_tensor_value_info("Xc", onnx.TensorProto.FLOAT, shape=("d", 1)))
        inputs.append(oh.make_tensor_value_info("one", onnx.TensorProto.FLOAT, shape=(1,)))
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["one"],
                value=onh.from_array(np.array([4.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(make_node_extended("Expand", ["Xc", "full_shape"], ["Xce"]))
        nodes.append(make_node_extended("Add", ["Xce", "one"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("d", "d")))
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
        inputs.append(
            oh.make_tensor_value_info("full_shape", onnx.TensorProto.INT64, shape=(2,))
        )
        inputs.append(oh.make_tensor_value_info("Xc", onnx.TensorProto.FLOAT, shape=("d", 1)))
        inputs.append(oh.make_tensor_value_info("one", onnx.TensorProto.FLOAT, shape=(1,)))
        nodes.append(
            make_node_extended("Add", ["Xc", "one"], ["ShapeBasedExpandSwapPattern_Y"])
        )
        nodes.append(
            make_node_extended("Expand", ["ShapeBasedExpandSwapPattern_Y", "full_shape"], ["Y"])
        )
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("d", "d")))
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

    _op_types = element_wise_binary_op_types() | element_wise_op_cmp_types()

    @classmethod
    def _broadcast_shape(
        cls,
        before_expand_shape: DYNAMIC_SHAPE,
        other_term_shape: DYNAMIC_SHAPE,
        exc: bool = False,
    ) -> Optional[DYNAMIC_SHAPE]:
        if len(before_expand_shape) != len(other_term_shape):
            d = abs(len(before_expand_shape) - len(other_term_shape))
            if len(before_expand_shape) < len(other_term_shape):
                before_expand_shape = (1,) * d + before_expand_shape
            else:
                other_term_shape = (1,) * d + other_term_shape
        if len(before_expand_shape) != len(other_term_shape):
            assert not exc, (
                f"Unable to produce a broadcasted shape from "
                f"{before_expand_shape} and {other_term_shape}"
            )
            return None
        res = []
        for a, b in zip(before_expand_shape, other_term_shape):
            if a == b:
                res.append(a)
            elif a == 1:
                res.append(b)
            elif b == 1:
                res.append(a)
            else:
                assert not exc, (
                    f"Unable to produce a broadcasted shape from "
                    f"{before_expand_shape} and {other_term_shape}"
                )
                return None
        return tuple(res)

    @classmethod
    def _get_compatible_expand_shape_for_expand_swap(
        cls,
        before_expand_shape: DYNAMIC_SHAPE,
        expanded_shape: DYNAMIC_SHAPE,
        other_term_shape: DYNAMIC_SHAPE,
        other_expanded_shape: Optional[DYNAMIC_SHAPE],
        output_shape: DYNAMIC_SHAPE,
    ) -> Optional[DYNAMIC_SHAPE]:
        """
        Something like that should work.
        The function returns a shape or None is not possible.

        .. code-block:: python

            _get_compatible_expand_shape_for_expand_swap(
                ("batch", 1, 1, 1),
                ("batch", 1, "seq_length", "cache_length+seq_length"),
                (1,),
                None,
                ("batch", 1, "seq_length", "cache_length+seq_length"),
            )

            >>> ("batch", 1, "seq_length", "cache_length+seq_length")
        )

        """
        if other_expanded_shape is not None and (
            other_expanded_shape != expanded_shape
            or expanded_shape != output_shape
            or len(before_expand_shape) != len(other_term_shape)
        ):
            return None
        if before_expand_shape == expanded_shape or expanded_shape == other_term_shape:
            # This pattern is not meant for that.
            return None
        if output_shape != expanded_shape:
            return None
        if (
            other_expanded_shape is None
            and not ShapeBasedExpandBroadcastPattern._is_compatible_shapes_for_expand(
                before_expand_shape,
                other_term_shape,
                cls._broadcast_shape(before_expand_shape, other_term_shape, exc=False),
            )
        ):
            return None
        if (
            other_expanded_shape is not None
            and not ShapeBasedExpandBroadcastPattern._is_compatible_shapes_for_expand(
                before_expand_shape,
                other_term_shape,
                cls._broadcast_shape(before_expand_shape, other_term_shape, exc=False),
            )
        ):
            return None
        if other_expanded_shape is None:
            return "expand_arg"
        max_dim = cls._broadcast_shape(before_expand_shape, other_term_shape)
        if max_dim == output_shape:
            # Expand is not necessary at all.
            return None
        return tuple(1 if a == b else 0 for a, b in zip(max_dim, output_shape))

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in self._op_types or node.domain != "":
            return self.none()
        if (
            not g.has_shape(node.output[0])
            or not g.has_shape(node.input[0])
            or not g.has_shape(node.input[1])
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])
        before = [
            None if n is None or n.op_type != "Expand" else n for n in [node_left, node_right]
        ]
        if before == [None, None]:
            return self.none(node, inspect.currentframe().f_lineno)

        if None in before:
            # Only one expand
            node_left, node_right = before
            shape_left = g.get_shape_renamed(
                node.input[0] if node_left is None else node_left.input[0]
            )
            shape_right = g.get_shape_renamed(
                node.input[1] if node_right is None else node_right.input[0]
            )
            before_expand_shape = shape_right if node_left is None else shape_left
            expanded_shape = (
                g.get_shape_renamed(node_right.output[0])
                if node_left is None
                else g.get_shape_renamed(node_left.output[0])
            )
            other_term_shape = shape_left if node_left is None else shape_right
            output_shape = g.get_shape_renamed(node.output[0])
            if self._get_compatible_expand_shape_for_expand_swap(
                before_expand_shape, expanded_shape, other_term_shape, None, output_shape
            ):
                if self.verbose:
                    print(
                        f"[{self.__class__.__name__}.match.1] {shape_left} "
                        f"{node.op_type} {shape_right} -> {output_shape}"
                    )
                return MatchResult(self, [node_left, node_right, node], self.apply)
            return self.none(node, inspect.currentframe().f_lineno)

        # Both expand.
        node_left, node_right = before
        if node_left.input[1] != node_right.input[1]:
            # It could work in that case if both expand have different
            # shape argument but the code to make sure it is is not implemented.
            return self.none(node, inspect.currentframe().f_lineno)

        shape_left = g.get_shape_renamed(node_left.input[0])
        shape_right = g.get_shape_renamed(node_right.input[0])
        output_shape = g.get_shape_renamed(node.output[0])
        expand_arg = self._get_compatible_expand_shape_for_expand_swap(
            shape_left,
            g.get_shape_renamed(node.input[0]),
            shape_right,
            g.get_shape_renamed(node.input[1]),
            output_shape,
        )
        if expand_arg:
            if self.verbose:
                print(
                    f"[{self.__class__.__name__}.match.2] {shape_left} "
                    f"{node.op_type} {shape_right} -> {output_shape} with "
                    f"expand_arg={expand_arg}"
                )
            return MatchResult(self, [node_left, node_right, node], self.apply)

        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        expand_left: NodeProto,
        expand_right: NodeProto,
        binary_node: NodeProto,
    ) -> List[NodeProto]:
        nodes = []
        if expand_left is not None and g.is_used_more_than_once(expand_left.output[0]):
            nodes.append(expand_left)
        if expand_right is not None and g.is_used_more_than_once(expand_right.output[0]):
            nodes.append(expand_right)
        assert (
            not binary_node.attribute
        ), f"Binary operator should not have any attribute, binary_node={binary_node}"
        new_name = g.unique_name(f"{self.__class__.__name__}_{binary_node.output[0]}")
        nodes.append(
            g.make_node(
                binary_node.op_type,
                [
                    binary_node.input[0] if expand_left is None else expand_left.input[0],
                    binary_node.input[1] if expand_right is None else expand_right.input[0],
                ],
                [new_name],
                name=f"{self.__class__.__name__}--{binary_node.name}",
                doc_string=binary_node.doc_string,
            ),
        )

        # One or two expand, same rewriting as the expand argument is the same.
        return [
            *nodes,
            g.make_node(
                "Expand",
                [
                    new_name,
                    expand_left.input[1] if expand_right is None else expand_right.input[1],
                ],
                binary_node.output,
                name=f"{self.__class__.__name__}--{binary_node.name}",
                doc_string=binary_node.doc_string,
            ),
        ]


class ShapeBasedExpandBroadcastMatMulPattern(PatternOptimization):
    """
    Similar to
    :class:`experimental_experiment.xoptim.patterns.onnx_expand.ShapeBasedExpandBroadcastPattern`,
    but works only with MatMul.

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
        initializers = [onh.from_array(np.array([1, 1], dtype=np.int64), name="o11")]
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(1, "c", "d"))
        )
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b", "c"))
        )
        nodes.append(oh.make_node("Shape", ["Y"], ["batch"], start=0, end=1))
        nodes.append(oh.make_node("Concat", ["batch", "o11"], ["exp"], axis=0))
        nodes.append(make_node_extended("Expand", ["Y", "exp"], ["Ye"]))
        nodes.append(make_node_extended("MatMul", ["X", "Ye"], ["Z"]))
        outputs.append(
            oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, shape=("a", "b", "d"))
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
        inputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(1, "c", "d"))
        )
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b", "c"))
        )
        nodes.append(make_node_extended("MatMul", ["X", "Y"], ["Z"]))
        outputs.append(
            oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, shape=("a", "b", "d"))
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

    @classmethod
    def _is_compatible_shapes_for_expand(
        cls,
        shape_left: DYNAMIC_SHAPE,
        shape_right: DYNAMIC_SHAPE,
        output_shape: Optional[DYNAMIC_SHAPE],
    ) -> bool:
        """
        Checks that the binary operations of the two input shapes returns the output_shape.
        Then no Expand node is needed.
        """
        if output_shape is None:
            return False
        if len(shape_left) < 2 or len(shape_right) < 2 or len(output_shape) < 2:
            return False
        return ShapeBasedExpandBroadcastPattern._is_compatible_shapes_for_expand(
            shape_left[:-2], shape_right[:-2], output_shape[:-2]
        )

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "MatMul" or node.domain != "":
            return self.none()
        if not g.has_shape(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])
        before = [
            None if n is None or n.op_type != "Expand" else n for n in [node_left, node_right]
        ]
        if before == [None, None]:
            return self.none(node, inspect.currentframe().f_lineno)

        # At least one expand.
        node_left, node_right = before
        shape_left = g.get_shape_renamed(
            node.input[0] if node_left is None else node_left.input[0]
        )
        shape_right = g.get_shape_renamed(
            node.input[1] if node_right is None else node_right.input[0]
        )
        if self._is_compatible_shapes_for_expand(
            shape_left, shape_right, g.get_shape_renamed(node.output[0])
        ):
            if self.verbose:
                print(
                    f"[{self.__class__.__name__}.match] {shape_left} "
                    f"{node.op_type} {shape_right} -> {g.get_shape_renamed(node.output[0])}"
                )
            return MatchResult(self, [node_left, node_right, node], self.apply)
        # We could end up with the following case.
        # shape_left   = (1, 1, 'seq_length', 'cache_length + seq_length')
        # shape_right  = (1, 1, 'seq_length', 'cache_length + seq_length')
        # output_shape = ('batch', 1, 'seq_length', 'cache_length + seq_length')
        # When this happes, it could also be caught by another pattern.
        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        expand_left: NodeProto,
        expand_right: NodeProto,
        binary_node: NodeProto,
    ) -> List[NodeProto]:
        nodes = []
        if expand_left is not None and g.is_used_more_than_once(expand_left.output[0]):
            nodes.append(expand_left)
        if expand_right is not None and g.is_used_more_than_once(expand_right.output[0]):
            nodes.append(expand_right)
        assert (
            not binary_node.attribute
        ), f"Binary operator should not have any attribute, binary_node={binary_node}"
        return [
            *nodes,
            g.make_node(
                binary_node.op_type,
                [
                    binary_node.input[0] if expand_left is None else expand_left.input[0],
                    binary_node.input[1] if expand_right is None else expand_right.input[0],
                ],
                binary_node.output,
                name=f"{self.__class__.__name__}--{binary_node.name}",
                doc_string=binary_node.doc_string,
            ),
        ]


class ShapeBasedExpandCastWhereSwapPattern(PatternOptimization):
    """
    Rewrites Where(Cast(X), X, cond).

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
        inputs.append(oh.make_tensor_value_info("exp", onnx.TensorProto.INT64, shape=(3,)))
        inputs.append(oh.make_tensor_value_info("cst", onnx.TensorProto.FLOAT, shape=(1,)))
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["cst"],
                value=onh.from_array(np.array([-np.inf], dtype=np.float32), name="value"),
            )
        )
        nodes.append(make_node_extended("Expand", ["X", "exp"], ["Xe"]))
        nodes.append(make_node_extended("Cast", ["Xe"], ["Xeb"], to=9))
        nodes.append(make_node_extended("Where", ["Xeb", "Xe", "cst"], ["Y"]))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("b", "b", "c"))
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
        inputs.append(oh.make_tensor_value_info("exp", onnx.TensorProto.INT64, shape=(3,)))
        inputs.append(oh.make_tensor_value_info("cst", onnx.TensorProto.FLOAT, shape=(1,)))
        nodes.append(
            make_node_extended(
                "Cast", ["X"], ["ShapeBasedExpandCastWhereSwapPattern_Xeb"], to=9
            )
        )
        nodes.append(
            make_node_extended(
                "Where",
                ["ShapeBasedExpandCastWhereSwapPattern_Xeb", "X", "cst"],
                ["ShapeBasedExpandCastWhereSwapPattern_Y"],
            )
        )
        nodes.append(
            make_node_extended(
                "Expand", ["ShapeBasedExpandCastWhereSwapPattern_Y", "exp"], ["Y"]
            )
        )
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("b", "b", "c"))
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

    @classmethod
    def _compatible_shapes(
        cls,
        cond: DYNAMIC_SHAPE,
        cst: DYNAMIC_SHAPE,
        output: DYNAMIC_SHAPE,
        before: DYNAMIC_SHAPE,
    ):
        if cond != output:
            return False
        if len(before) < len(output):
            before = (1,) * (len(output) - len(before)) + before
        if len(cst) < len(output):
            cst = (1,) * (len(output) - len(cst)) + cst
        out = ShapeBasedExpandSwapPattern._broadcast_shape(before, cst)
        if len(out) != len(output) or len(out) != len(before):
            return False
        return all(not (o != e and o != b) for b, o, e in zip(before, out, output))

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Where" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]):
            return self.none()
        cast_node = g.node_before(node.input[0])
        if cast_node is None or cast_node.op_type != "Cast" or cast_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if cast_node.input[0] not in node.input[1:]:
            return self.none(node, inspect.currentframe().f_lineno)
        expand_node = g.node_before(cast_node.input[0])
        if expand_node is None or expand_node.op_type != "Expand" or expand_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        nodes = g.next_nodes(cast_node.input[0])
        if len(nodes) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_shape(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(expand_node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        same_index = list(node.input).index(cast_node.input[0])
        if self._compatible_shapes(
            g.get_shape_renamed(node.input[0]),
            g.get_shape_renamed(node.input[3 - same_index]),
            g.get_shape_renamed(node.output[0]),
            g.get_shape_renamed(expand_node.input[0]),
        ):
            return MatchResult(self, [expand_node, cast_node, node], self.apply)
        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        expand_node: NodeProto,
        cast_node: NodeProto,
        where_node: NodeProto,
    ) -> List[NodeProto]:
        to = g.get_attribute(cast_node, "to").i
        pos_index = list(where_node.input).index(expand_node.output[0])
        cast_output = g.unique_name(f"{self.__class__.__name__}_{cast_node.output[0]}")
        where_output = g.unique_name(f"{self.__class__.__name__}_{where_node.output[0]}")
        return [
            g.make_node(
                cast_node.op_type,
                [expand_node.input[0]],
                [cast_output],
                to=to,
                name=f"{self.__class__.__name__}--{cast_node.name}",
                doc_string=cast_node.doc_string,
            ),
            g.make_node(
                where_node.op_type,
                (
                    [cast_output, expand_node.input[0], where_node.input[2]]
                    if pos_index == 1
                    else [cast_output, where_node.input[1], expand_node.input[0]]
                ),
                [where_output],
                name=f"{self.__class__.__name__}--{where_node.name}",
                doc_string=where_node.doc_string,
            ),
            g.make_node(
                expand_node.op_type,
                [where_output, expand_node.input[1]],
                [where_node.output[0]],
                name=f"{self.__class__.__name__}--{expand_node.name}",
                doc_string=expand_node.doc_string,
            ),
        ]


class ShapeBasedConcatExpandPattern(PatternOptimization):
    """
    Rewrites Expand(X, concat(...)) if possible.

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
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", 1)))
        inputs.append(oh.make_tensor_value_info("two", onnx.TensorProto.INT64, shape=(1,)))
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["two"],
                value=onh.from_array(np.array([2], dtype=np.int64), name="value"),
            )
        )
        nodes.append(oh.make_node("Shape", ["X"], ["shx"], start=0, end=1))
        nodes.append(make_node_extended("Concat", ["shx", "two"], ["sh2"], axis=0))
        nodes.append(make_node_extended("Expand", ["X", "sh2"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", 2)))
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
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", 1)))
        inputs.append(oh.make_tensor_value_info("two", onnx.TensorProto.INT64, shape=(1,)))
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init7_12"],
                value=onh.from_array(np.array([1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(make_node_extended("Concat", ["init7_12", "two"], ["sh22"], axis=0))
        nodes.append(make_node_extended("Expand", ["X", "sh22"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", 2)))
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

    @classmethod
    def _compatible_shapes(
        cls,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        shape: DYNAMIC_SHAPE,
        expanded_shape: DYNAMIC_SHAPE,
        concat_input: Sequence[str],
    ) -> Optional[int]:
        if len(shape) != len(expanded_shape) or len(expanded_shape) != len(concat_input):
            return None
        position = []
        for i, (a, b) in enumerate(zip(shape, expanded_shape)):
            if a == b:
                continue
            position.append(i)
        if len(position) != 1:
            # It might be Identity but this should be caught by another pattern.
            return None
        return position[0]

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_constant(node.input[1]):
            # no need
            return self.none(node, inspect.currentframe().f_lineno)
        concat_node = g.node_before(node.input[1])
        if concat_node is None or concat_node.op_type != "Concat" or concat_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_shape(node.input[0]) or not g.has_shape(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape1 = g.get_shape_renamed(node.input[0])
        shape2 = g.get_shape_renamed(node.output[0])
        index = self._compatible_shapes(g, shape1, shape2, concat_node.input)
        if index is None:
            return self.none(node, inspect.currentframe().f_lineno)
        # checking the other values are not 1
        if all(
            (i == index or (g.is_constant(name) and g.get_constant_scalar(name) == 1))
            for i, name in enumerate(concat_node.input)
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [concat_node, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        concat_node: NodeProto,
        expand_node: NodeProto,
    ) -> List[NodeProto]:
        shape1 = g.get_shape_renamed(expand_node.input[0])
        shape2 = g.get_shape_renamed(expand_node.output[0])
        index = self._compatible_shapes(g, shape1, shape2, concat_node.input)
        init1 = g.make_initializer(
            g.unique_name("init7_1"), g.ONE, source="ShapeBasedConcatExpandPattern.1"
        )
        new_input = [
            (iname if i == index else init1) for i, iname in enumerate(concat_node.input)
        ]
        new_name = g.unique_name(concat_node.output[0])
        return [
            g.make_node(
                "Concat",
                new_input,
                [new_name],
                axis=0,
                name=f"{self.__class__.__name__}--{concat_node.name}",
                doc_string=concat_node.doc_string,
            ),
            g.make_node(
                "Expand",
                [expand_node.input[0], new_name],
                expand_node.output,
                name=f"{self.__class__.__name__}--{expand_node.name}",
                doc_string=expand_node.doc_string,
            ),
        ]


class SwapExpandReshapePattern(PatternOptimization):
    """
    Checks if Expand + Reshape can be swapped.

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
        inputs.append(
            oh.make_tensor_value_info("weight", onnx.TensorProto.FLOAT, shape=(1, 4, 1))
        )
        inputs.append(oh.make_tensor_value_info("stat", onnx.TensorProto.INT64, shape=(3,)))
        inputs.append(oh.make_tensor_value_info("shape", onnx.TensorProto.INT64, shape=(3,)))
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["weight"],
                value=onh.from_array(
                    np.array([[[2.0], [3.0], [4.0], [5.0]]], dtype=np.float32), name="value"
                ),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["stat"],
                value=onh.from_array(np.array([0, 1, -1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(make_node_extended("Expand", ["weight", "shape"], ["resh"]))
        nodes.append(make_node_extended("Reshape", ["resh", "stat"], ["Y"]))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", 1, 4))
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
        inputs.append(
            oh.make_tensor_value_info("weight", onnx.TensorProto.FLOAT, shape=(1, 4, 1))
        )
        inputs.append(oh.make_tensor_value_info("stat", onnx.TensorProto.INT64, shape=(3,)))
        inputs.append(oh.make_tensor_value_info("shape", onnx.TensorProto.INT64, shape=(3,)))
        nodes.append(make_node_extended("Reshape", ["weight", "stat"], ["Y2"]))
        nodes.append(make_node_extended("Expand", ["Y2", "shape"], ["Y"]))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", 1, 4))
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

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Reshape" or node.domain != "":
            return self.none()
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        expand_node = g.node_before(node.input[0])
        if expand_node is None or expand_node.op_type != "Expand" or expand_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_rank(expand_node.input[0]) or g.get_rank(expand_node.input[0]) != 3:
            return self.none(node, inspect.currentframe().f_lineno)

        cst = g.get_computed_constant(node.input[1])
        if cst is None:
            return self.none(node, inspect.currentframe().f_lineno)
        shape = g.builder.value_as_shape(expand_node.input[1])
        if shape is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if tuple(cst) != (0, 1, -1) or shape[1:] != (1, 1):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [expand_node, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        expand_node: NodeProto,
        reshape_node: NodeProto,
    ) -> List[NodeProto]:
        new_name = g.unique_name(reshape_node.output[0])
        return [
            g.make_node(
                "Reshape",
                [expand_node.input[0], reshape_node.input[1]],
                [new_name],
                name=f"{self.__class__.__name__}--{reshape_node.name}",
                doc_string=reshape_node.doc_string,
            ),
            g.make_node(
                "Expand",
                [new_name, expand_node.input[1]],
                reshape_node.output,
                name=f"{self.__class__.__name__}--{expand_node.name}",
                doc_string=expand_node.doc_string,
            ),
        ]
