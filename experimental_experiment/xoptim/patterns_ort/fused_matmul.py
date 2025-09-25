import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ...xbuilder._shape_helper import all_int
from ..patterns_api import MatchResult, PatternOptimization


class FusedMatMulDivPattern(PatternOptimization):
    """
    Replaces the Matmul, Div into FusedMatMul.
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type != "MatMul" or node.domain != "") and (
            node.op_type != "FusedMatMul" or node.domain != "com.microsoft"
        ):
            return self.none()

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        op_type = next_nodes[0].op_type
        if op_type not in ("Mul", "Div"):
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant_scalar(next_nodes[0].input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_nodes[0]], self.apply, insert_at=next_nodes[0])

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        node_div: NodeProto,
    ) -> List[NodeProto]:
        alpha = 1.0
        atts = []
        if node.op_type == "FusedMatMul":
            for att in node.attribute:
                if att.name == "alpha":
                    alpha *= att.f
                else:
                    atts.append(att)

        cst = g.get_computed_constant(node_div.input[1])
        scale = float(cst if len(cst.shape) == 0 else cst[0])
        if node_div.op_type == "Div":
            alpha /= scale
        else:
            alpha *= scale

        mm = g.make_node(
            "FusedMatMul",
            node.input,
            node_div.output,
            domain="com.microsoft",
            alpha=alpha,
            name=f"{self.__class__.__name__}--{node.name}",
        )
        if atts:
            mm.attribute.extend(atts)
        return [mm]


class FusedMatMulPattern(PatternOptimization):
    """
    Replaces the sequence Transpose, Matmul into FusedMatMul.
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type != "MatMul" or node.domain != "") and (
            node.op_type != "FusedMatMul" or node.domain != "com.microsoft"
        ):
            return self.none()

        if node.op_type == "FusedMatMul":
            transA = g.get_attribute(node, "transA", exc=False) or 0
            transB = g.get_attribute(node, "transB", exc=False) or 0
            if transA != transB:
                # one side is already transposed.
                return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_rank(node.input[0]) or not g.has_rank(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_rank(node.input[0]) < 2 or g.get_rank(node.input[1]) < 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_rank(node.input[0]) <= 2 and g.get_rank(node.input[1]) <= 2:
            # Regular Gemm.
            return self.none(node, inspect.currentframe().f_lineno)

        nodes_before = [g.node_before(node.input[0]), g.node_before(node.input[1])]
        ns = [
            (n if n is not None and n.op_type == "Transpose" and n.domain == "" else None)
            for n in nodes_before
        ]
        if len([_ for _ in ns if _ is not None]) == 0:
            return self.none(node, inspect.currentframe().f_lineno)

        if g.has_processor("CUDA"):
            nns = []
            for n in ns:
                if n is None:
                    nns.append(n)
                    continue
                if g.is_used_more_than_once(n.output[0]):
                    nns.append(None)
                    continue
                nns.append(n)
            if len([_ for _ in ns if _ is not None]) == 0:
                return self.none(node, inspect.currentframe().f_lineno)
            ns = nns

        hints = []
        found = False
        nns = []
        for n in ns:
            if n is None:
                nns.append(None)
                continue
            perm = list(g.get_attribute(n, "perm").ints)
            expecting = list(range(len(perm)))
            expecting[-2], expecting[-1] = expecting[-1], expecting[-2]
            if perm != expecting:
                hints.append(dict(expecting=expecting, perm=perm))
                nns.append(None)
                continue
            found = True
            nns.append(n)

        ns = nns
        if not found:
            # unexpected transpose
            return self.none(node, inspect.currentframe().f_lineno, lambda: f"hints={hints}")

        # At this stage, one or two inputs are transposed before being used.
        # MatMul or Gemm are operating on 2D tensors.
        nodes = [*ns, node]
        if nodes[0] is not None and nodes[1] is not None:
            # Both are available, we only transpose one.
            nodes[0] = None
        if not g.is_used_more_than_once(node.output[0]):
            next_node = g.next_node(node.output[0])
            if (
                next_node.op_type in {"Div", "Mul"}
                and next_node.domain == ""
                and g.is_constant_scalar(next_node.input[1])
            ):
                # The node can be fused with matmul
                nodes.append(next_node)

        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_before_left: Optional[NodeProto],
        node_before_right: Optional[NodeProto],
        node: NodeProto,
        scale: Optional[NodeProto] = None,
    ) -> List[NodeProto]:
        inputs = [
            (node.input[0] if node_before_left is None else node_before_left.input[0]),
            (node.input[1] if node_before_right is None else node_before_right.input[0]),
            *node.input[2:],
        ]

        transA = 0 if node_before_left is None else 1
        transB = 0 if node_before_right is None else 1
        transBatchA = 0
        transBatchB = 0
        keep = []
        for att in node.attribute:
            if att.name in {"alpha", "beta"}:
                keep.append(att)
            elif att.name == "transA":
                transA = (att.i + transA) % 2
            elif att.name == "transB":
                transB = (att.i + transB) % 2
            elif att.name == "transBatchA":
                transBatchA = att.i
            elif att.name == "transBatchB":
                transBatchB = att.i
            else:
                raise NotImplementedError(
                    f"Unexpected attribute {att.name!r}={att} for node={node}"
                )

        kwargs = dict(
            transA=transA,
            transB=transB,
            transBatchA=transBatchA,
            transBatchB=transBatchB,
        )

        if scale is not None:
            # Let's include the scale as well
            cst = g.get_computed_constant(scale.input[1])
            value = float(cst[0] if cst.shape == (1,) else cst)
            assert scale.op_type in {
                "Div",
                "Mul",
            }, f"Match did not check next_node type {scale.op_type!r}"
            alpha = value if scale.op_type == "Mul" else (1.0 / value)
            kwargs["alpha"] = alpha
            output = scale.output[0]
        else:
            output = node.output[0]

        new_node = g.make_node(
            "FusedMatMul",
            inputs,
            [output],
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
            domain="com.microsoft",
            **kwargs,
        )
        new_node.attribute.extend(keep)
        res = [new_node]
        if node_before_left is not None and g.is_used_more_than_once(node_before_left.output[0]):
            # This is not efficient on CUDA.
            res.append(node_before_left)
        if node_before_right is not None and g.is_used_more_than_once(
            node_before_right.output[0]
        ):
            # This is not efficient on CUDA.
            res.append(node_before_right)
        return res


class FusedMatMulx2Pattern(PatternOptimization):
    """
    Replaces the sequence Div by a scalar consumed by two FusedMatMul.
    """

    def __init__(self, verbose: int = 0, priority: int = 3):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type not in "MatMul" or node.domain != "") and (
            node.op_type != "FusedMatMul" or node.domain != "com.microsoft"
        ):
            return self.none()

        div_node = None
        for name in node.input:
            n = g.node_before(name)
            if n is None:
                continue
            if n.op_type not in {"Mul", "Div"} or n.domain != "":
                continue
            if not g.is_constant_scalar(n.input[1]):
                continue
            div_node = n
            break

        if div_node is None:
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(div_node.output[0])
        op_types = [n.op_type for n in next_nodes]
        if any(t not in {"FusedMatMul", "MatMul"} for t in op_types):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [div_node, *next_nodes], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        div_node: Optional[NodeProto],
        *mnodes: Optional[NodeProto],
    ) -> List[NodeProto]:
        cst = g.get_constant_scalar(div_node.input[1])
        if div_node.op_type == "Div":
            cst = 1.0 / cst

        new_nodes = []
        for node in mnodes:
            alpha = 1.0
            atts = []
            for att in node.attribute:
                if att.name == "alpha":
                    alpha = float(att.f)
                else:
                    atts.append(att)
            new_inputs = [
                (div_node.input[0] if i == div_node.output[0] else i) for i in node.input
            ]
            alpha *= cst
            new_node = g.make_node(
                "FusedMatMul",
                new_inputs,
                node.output,
                domain="com.microsoft",
                alpha=alpha,
                name=f"{self.__class__.__name__}--{node.name}",
            )
            if atts:
                new_node.attribute.extend(atts)
            new_nodes.append(new_node)
        return new_nodes


class FusedMatMulTransposePattern(PatternOptimization):
    """
    Replaces the sequence (Fused)Matmul(A,B) + Transpose
    into FusedMatMul(B.T, A.T).
    """

    def __init__(self, verbose: int = 0, priority: int = 3):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type not in "MatMul" or node.domain != "") and (
            node.op_type != "FusedMatMul" or node.domain != "com.microsoft"
        ):
            return self.none()

        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        next_nodes = g.next_nodes(node.output[0])
        if (
            len(next_nodes) != 1
            or next_nodes[0].op_type != "Transpose"
            or next_nodes[0].domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        transpose_node = next_nodes[0]
        perm = list(g.get_attribute(transpose_node, "perm").ints)
        if len(perm) > 2:
            if perm[:-2] != list(range(len(perm) - 2)):
                return self.none(node, inspect.currentframe().f_lineno)
        if perm[-2:] != [len(perm) - 1, len(perm) - 2]:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, transpose_node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        transpose_node: NodeProto,
    ) -> List[NodeProto]:
        default_values = dict(transA=0, transB=0, transBatchA=0, transBatchB=0, alpha=1.0)
        kwargs = g.get_attributes_with_default(node, **default_values)
        kwargs["transA"], kwargs["transB"] = 1 - kwargs["transB"], 1 - kwargs["transA"]
        remove = []
        for k in kwargs:
            if kwargs[k] == default_values[k]:
                remove.append(k)
        for r in remove:
            del kwargs[r]
        new_node = g.make_node(
            "FusedMatMul",
            [node.input[1], node.input[0]],
            transpose_node.output,
            domain="com.microsoft",
            name=f"{self.__class__.__name__}--{node.name}",
            **kwargs,
        )
        return [new_node]


class ReshapeGemmPattern(PatternOptimization):
    """
    Replaces the sequence Reshape(-1, ...) + Gemm
    into FusedMatMul().
    """

    def __init__(self, verbose: int = 0, priority: int = 3):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in "Gemm" or node.domain != "" or len(node.input) == 3:
            return self.none()

        transA = g.get_attributes_with_default(node, transA=0)["transA"]
        if transA != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        node_before = g.node_before(node.input[0])
        if node_before is None or node_before.op_type != "Reshape" or node_before.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape = g.get_shape(node.input[1])
        if not all_int(shape):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node_before.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        shape = g.get_computed_constant(node_before.input[1])
        if shape.shape != (2,) or shape[0] != -1:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node_before, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        reshape_node: NodeProto,
        gemm_node: NodeProto,
    ) -> List[NodeProto]:
        kwargs = {}
        transB = g.get_attributes_with_default(gemm_node, transB=0)["transB"]
        if transB:
            kwargs["transB"] = transB
        gemm_output = g.unique_name(f"{self.__class__.__name__}--{gemm_node.output[0]}")
        new_node = g.make_node(
            "FusedMatMul",
            [reshape_node.input[0], *gemm_node.input[1:]],
            [gemm_output],
            domain="com.microsoft",
            name=f"{self.__class__.__name__}--{gemm_node.name}",
            **kwargs,
        )
        shape = g.get_shape(gemm_node.input[1])
        new_shape = g.make_initializer(
            "",
            np.array([-1, shape[1 - transB]], dtype=np.int64),
            source=f"ReshapeGemm.shape({gemm_node.name})",
        )
        reshape_node = g.make_node(
            "Reshape",
            [gemm_output, new_shape],
            gemm_node.output,
            name=f"{self.__class__.__name__}--{gemm_node.name}",
        )
        return [new_node, reshape_node]


class TransposeFusedMatMulBPattern(PatternOptimization):
    """
    Replaces the sequence Transpose(B, [0, 2, 3, 1] + (Fused)Matmul(A,B)
    into Transpose(A, [0, 2, 1, 3]) + FusedMatMul(A, B, transB=1).
    """

    def __init__(self, verbose: int = 0, priority: int = 3):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type not in "MatMul" or node.domain != "") and (
            node.op_type != "FusedMatMul" or node.domain != "com.microsoft"
        ):
            return self.none()
        if g.is_used_more_than_once(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        transB = g.get_attributes_with_default(node, transB=0)["transB"]
        if transB != 0:
            return self.none(node, inspect.currentframe().f_lineno)

        transpose_node = g.node_before(node.input[1])
        if (
            transpose_node is None
            or transpose_node.op_type != "Transpose"
            or transpose_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        perm = list(g.get_attribute(transpose_node, "perm").ints)
        if perm != [0, 2, 3, 1]:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [transpose_node, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        transpose_node: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        tout = g.unique_name(f"{self.__class__.__name__}--{node.input[1]}")
        nodes = [
            g.make_node(
                "Transpose",
                transpose_node.input,
                [tout],
                name=f"{self.__class__.__name__}--{node.name}",
                perm=[0, 2, 1, 3],
            ),
            g.make_node(
                "FusedMatMul",
                [node.input[0], tout],
                node.output,
                domain="com.microsoft",
                name=f"{self.__class__.__name__}--{node.name}",
                transB=1,
            ),
        ]
        for att in node.attribute:
            if att.name != "transB":
                nodes[-1].attribute.append(att)
        return nodes
