import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization
from ..patterns.onnx_rotary import FunctionHalfRotaryEmbeddingPattern


class ContribRotaryEmbeddingPattern(PatternOptimization):
    """
    Very similar to
    :class:`experimental_experimental.xoptim.patterns.onnx_rotary.RotaryEmbeddingPattern`.
    """

    _operator_name = FunctionHalfRotaryEmbeddingPattern._operator_name
    _domain_name = FunctionHalfRotaryEmbeddingPattern._domain_name

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != self._operator_name or node.domain != self._domain_name:
            # Not ready in opset 23.
            return self.none()
        if not g.has_rank(node.input[0]) or g.get_rank(node.input[0]) != 4:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[1]) or not g.has_shape(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape_cos = g.get_shape(node.input[1])
        shape_sin = g.get_shape(node.input[2])
        if shape_cos != shape_sin:
            return self.none(node, inspect.currentframe().f_lineno)
        if len(shape_cos) != 4:
            return self.none(node, inspect.currentframe().f_lineno)
        if shape_cos[1] != 1 or shape_sin[1] != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if shape_cos[0] != 1 or shape_sin[0] != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        concat_cos = g.node_before(node.input[1])
        if concat_cos is None or concat_cos.op_type != "Concat" or concat_cos.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if concat_cos.input[0] != concat_cos.input[1]:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_attribute(concat_cos, "axis").i != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        concat_sin = g.node_before(node.input[2])
        if concat_sin is None or concat_sin.op_type != "Concat" or concat_sin.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if concat_sin.input[0] != concat_sin.input[1]:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_attribute(concat_sin, "axis").i != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        split_node = g.node_before(node.input[0])
        if split_node is None or split_node.op_type != "Split" or split_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(split_node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape_input = g.get_shape(split_node.input[0])
        if not isinstance(shape_input[1], int):
            # Not a fixed number of heads.
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(split_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(split_node.input[1])
        if cst.shape != (2,):
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        concat_node = next_nodes[0]
        if concat_node.op_type != "Concat" or concat_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if split_node.output[1] != concat_node.input[1]:
            return self.none(node, inspect.currentframe().f_lineno)
        axis = g.get_attribute(concat_node, "axis").i
        if axis != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self,
            [concat_cos, concat_sin, split_node, node, concat_node],
            self.apply,
            insert_at=None if g.is_used_more_than_once(concat_cos.output[0]) else concat_node,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        concat_cos: NodeProto,
        concat_sin: NodeProto,
        split_node: NodeProto,
        half_node: NodeProto,
        concat_node: NodeProto,
    ) -> List[NodeProto]:
        cst = g.get_computed_constant(split_node.input[1])
        rotary_dim = int(cst[0])
        shape = g.get_shape(split_node.input[0])
        assert isinstance(shape[1], int), f"Number of heads is not fixed, shape(X)={shape}"
        num_heads = shape[1]

        rotary_nodes = []
        if g.is_used_more_than_once(concat_cos.output[0]):
            rotary_nodes.append(concat_cos)
        if g.is_used_more_than_once(concat_sin.output[0]):
            rotary_nodes.append(concat_sin)

        cos_name = g.unique_name(f"{self.__class__.__name__}--{half_node.input[1]}")
        sin_name = g.unique_name(f"{self.__class__.__name__}--{half_node.input[2]}")
        batch_name = g.unique_name(f"{self.__class__.__name__}--{half_node.input[0]}--batch")
        zeroone = g.make_initializer(
            "", np.array([0, 1], dtype=np.int64), source=f"{self.__class__.__name__}.01"
        )
        one = g.make_initializer("", g.ONE, source=f"{self.__class__.__name__}.1")
        one_no_dim = g.make_initializer(
            "", g.ONE_NO_DIM, source=f"{self.__class__.__name__}.1d"
        )

        # position_ids
        zero_no_dim = g.make_initializer(
            "", g.ZERO_NO_DIM, source=f"{self.__class__.__name__}.0d"
        )
        seq_length = g.unique_name(
            f"{self.__class__.__name__}--{half_node.input[0]}--seq_length"
        )
        seq_length_squeezed = g.unique_name(
            f"{self.__class__.__name__}--{half_node.input[0]}--seqsq"
        )
        exp_shape = g.unique_name(f"{self.__class__.__name__}--{half_node.input[0]}_pshape")
        flat_pids = g.unique_name(f"{self.__class__.__name__}--{half_node.input[0]}_flat_pids")
        position_ids = g.unique_name(
            f"{self.__class__.__name__}--{half_node.input[0]}_position_ids"
        )

        rotary_nodes.extend(
            [
                g.make_node(
                    "Squeeze",
                    [concat_cos.input[0], zeroone],
                    [cos_name],
                    name=f"{self.__class__.__name__}--{half_node.name}",
                ),
                g.make_node(
                    "Squeeze",
                    [concat_sin.input[0], zeroone],
                    [sin_name],
                    name=f"{self.__class__.__name__}--{half_node.name}",
                ),
                g.make_node(
                    "Shape",
                    [split_node.input[0]],
                    [batch_name],
                    start=0,
                    end=1,
                    name=f"{self.__class__.__name__}--{half_node.name}",
                ),
                g.make_node(
                    "Shape",
                    [split_node.input[0]],
                    [seq_length],
                    start=2,
                    end=3,
                    name=f"{self.__class__.__name__}--{half_node.name}",
                ),
                g.make_node(
                    "Squeeze",
                    [seq_length],
                    [seq_length_squeezed],
                    name=f"{self.__class__.__name__}--{half_node.name}",
                ),
                g.make_node(
                    "Range",
                    [zero_no_dim, seq_length_squeezed, one_no_dim],
                    [flat_pids],
                    name=f"{self.__class__.__name__}--{half_node.name}",
                ),
                g.make_node(
                    "Concat",
                    [batch_name, one],
                    [exp_shape],
                    axis=0,
                    name=f"{self.__class__.__name__}--{half_node.name}",
                ),
                g.make_node(
                    "Expand",
                    [flat_pids, exp_shape],
                    [position_ids],
                    name=f"{self.__class__.__name__}--{half_node.name}",
                ),
                g.make_node(
                    "RotaryEmbedding",
                    [split_node.input[0], position_ids, cos_name, sin_name],
                    [concat_node.output[0]],
                    name=f"{self.__class__.__name__}--{half_node.name}",
                    rotary_embedding_dim=rotary_dim,
                    num_heads=num_heads,
                    domain="com.microsoft",
                ),
            ]
        )
        return rotary_nodes
