import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ...helpers import make_idn
from ..patterns_api import MatchResult, PatternOptimization


class RotaryEmbeddingPattern(PatternOptimization):
    """
    Fuses the y * cos + (rotary(y) * sin) into RotaryEmbedding(y)
    where y = transpose(x, [0, 2, 1, 3]).
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Split" or node.domain != "" or len(node.output) != 2:
            return self.none()

        axis = g.get_attribute(node, "axis")
        if axis is None or axis.i not in (-1, 3):
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.input[0])
        # It should be Split and Mul
        if len(next_nodes) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        split_node = node
        tr_node = g.node_before(node.input[0])
        if tr_node is None or g.is_used_more_than_once(tr_node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        perm = tuple(g.get_attribute(tr_node, "perm").ints)
        if perm != (0, 2, 1, 3):
            return self.none(node, inspect.currentframe().f_lineno)

        # cos part
        mul_node_cos = (
            next_nodes[0] if make_idn(next_nodes[1]) == make_idn(node) else next_nodes[1]
        )
        if mul_node_cos.op_type != "Mul" or mul_node_cos.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        add_nodes = g.next_nodes(mul_node_cos.output[0])
        if len(add_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        add_node = add_nodes[0]
        if add_node.op_type != "Add" or add_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # sin part
        spl1 = g.next_nodes(node.output[0])
        spl2 = g.next_nodes(node.output[1])
        if len(spl1) != 1 or len(spl2) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if spl1[0].op_type == "Concat":
            concat_node = spl1[0]
            neg_node = spl2[0]
        else:
            concat_node = spl2[0]
            neg_node = spl1[0]
        if (
            concat_node.op_type != "Concat"
            or concat_node.domain != ""
            or neg_node.op_type != "Neg"
            or neg_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        check_node = g.next_nodes(neg_node.output[0])
        if len(check_node) != 1 or make_idn(check_node[0]) != make_idn(concat_node):
            return self.none(node, inspect.currentframe().f_lineno)
        axis = g.get_attribute(concat_node, "axis")
        if axis is None or axis.i not in (-1, 3):
            return self.none(node, inspect.currentframe().f_lineno)

        mul_node_sin = g.next_nodes(concat_node.output[0])
        if (
            len(mul_node_sin) != 1
            or mul_node_sin[0].op_type != "Mul"
            or mul_node_sin[0].domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        mul_node_sin = mul_node_sin[0]
        if g.is_used_more_than_once(mul_node_sin.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        # final addition
        if set(add_node.input) != {mul_node_cos.output[0], mul_node_sin.output[0]}:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self,
            [tr_node, split_node, neg_node, concat_node, mul_node_cos, mul_node_sin, add_node],
            self.apply,
            insert_at=split_node,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        tr_node: NodeProto,
        split_node: NodeProto,
        neg_node: NodeProto,
        concat_node: NodeProto,
        mul_node_cos: NodeProto,
        mul_node_sin: NodeProto,
        add_node: NodeProto,
    ) -> List[NodeProto]:
        zero = g.make_initializer(
            "", np.array(0, dtype=np.int64), source="RotaryEmbeddingPattern.zero"
        )
        zero1 = g.make_initializer("", g.ZERO, source="RotaryEmbeddingPattern.zero1")
        mone = g.make_initializer("", g.MINUS_ONE, source="RotaryEmbeddingPattern.mone")
        one = g.make_initializer(
            "", np.array(1, dtype=np.int64), source="RotaryEmbeddingPattern.one"
        )

        shape_name = g.unique_name(f"{self.__class__.__name__}--{tr_node.input[0]}")
        squeezed_name = g.unique_name(f"{self.__class__.__name__}--{tr_node.input[0]}")
        batch_name = g.unique_name(f"{self.__class__.__name__}--{tr_node.input[0]}")
        pids_name = g.unique_name(f"{self.__class__.__name__}--{tr_node.input[0]}")
        known = {tr_node.output[0], concat_node.output[0]}
        cos = mul_node_cos.input[0 if mul_node_cos.input[1] in known else 1]
        sin = mul_node_sin.input[0 if mul_node_sin.input[1] in known else 1]
        expand_shape = g.unique_name(f"{self.__class__.__name__}--{tr_node.input[0]}")
        expand_name = g.unique_name(f"{self.__class__.__name__}--{tr_node.input[0]}")
        cos_shape = g.unique_name(f"{self.__class__.__name__}--{tr_node.input[0]}")
        new_cos_shape = g.unique_name(f"{self.__class__.__name__}--{tr_node.input[0]}")
        cos_reshaped = g.unique_name(f"{self.__class__.__name__}--{tr_node.input[0]}")
        sin_reshaped = g.unique_name(f"{self.__class__.__name__}--{tr_node.input[0]}")

        return [
            g.make_node(
                "Shape",
                [tr_node.input[0]],
                [shape_name],
                start=2,
                end=3,
                name=f"{self.__class__.__name__}--Sh--{split_node.name}",
            ),
            g.make_node(
                "Shape",
                [tr_node.input[0]],
                [batch_name],
                start=0,
                end=1,
                name=f"{self.__class__.__name__}--Sh--{split_node.name}",
            ),
            g.make_node(
                "Squeeze",
                [shape_name, zero1],
                [squeezed_name],
                name=f"{self.__class__.__name__}--Ra--{split_node.name}",
            ),
            g.make_node(
                "Range",
                [zero, squeezed_name, one],
                [pids_name],
                name=f"{self.__class__.__name__}--Ra--{split_node.name}",
            ),
            g.make_node(
                "Concat",
                [batch_name, shape_name],
                [expand_shape],
                axis=0,
                name=f"{self.__class__.__name__}--Co--{split_node.name}",
            ),
            g.make_node(
                "Expand",
                [pids_name, expand_shape],
                [expand_name],
                name=f"{self.__class__.__name__}--Ex--{split_node.name}",
            ),
            g.make_node(
                "Shape",
                [cos],
                [cos_shape],
                start=-1,
                name=f"{self.__class__.__name__}--ShCos--{split_node.name}",
            ),
            g.make_node(
                "Concat",
                [mone, cos_shape],
                [new_cos_shape],
                axis=0,
                name=f"{self.__class__.__name__}--CoCos--{split_node.name}",
            ),
            g.make_node(
                "Reshape",
                [cos, new_cos_shape],
                [cos_reshaped],
                name=f"{self.__class__.__name__}--ReshCos--{split_node.name}",
            ),
            g.make_node(
                "Reshape",
                [sin, new_cos_shape],
                [sin_reshaped],
                name=f"{self.__class__.__name__}--ReshSin--{split_node.name}",
            ),
            g.make_node(
                "RotaryEmbedding",
                [*tr_node.input, expand_name, cos_reshaped, sin_reshaped],
                add_node.output,
                domain="com.microsoft",
                interleaved=0,
                name=f"{self.__class__.__name__}--Rot--{split_node.name}",
            ),
        ]
