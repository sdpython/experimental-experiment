import inspect
from collections import Counter
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class FunctionPackedMatMulPattern(PatternOptimization):
    """
    Replaces multiple MatMul (X,A), (X,B) by (X, concat(A,B))...
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "MatMul" or node.domain != "":
            return self.none()
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        side_nodes = g.next_nodes(node.input[0])
        matmul_nodes = [
            n
            for n in side_nodes
            if n.op_type == "MatMul"
            and n.input[0] == node.input[0]
            and g.is_constant(n.input[1])
        ]
        if len(matmul_nodes) < 2 or not g.has_rank(matmul_nodes[0].output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        mm = []
        for n in matmul_nodes:
            if g.has_shape(n.input[1]):
                mm.append((n, g.get_shape(n.input[1])))
        shapes = [(v, k) for k, v in Counter(_[1] for _ in mm).items()]
        shapes.sort()
        best_shape = shapes[-1][1]
        matmul_nodes = [n for n, s in mm if s == best_shape]
        if len(matmul_nodes) < 2:
            # Nothing to do.
            return self.none(node, inspect.currentframe().f_lineno)

        # Reshape
        next_nodes = [g.next_nodes(n.output[0]) for n in matmul_nodes]
        if any(len(nn) != 1 or nn[0].op_type != "Reshape" for nn in next_nodes):
            # More than one users after.
            return self.none(node, inspect.currentframe().f_lineno)
        next_node = [_[0] for _ in next_nodes]
        first_shape = next_node[0].input[1]
        if any(_.input[1] != first_shape for _ in next_node):
            # Not the same shape for all.
            return self.none(node, inspect.currentframe().f_lineno)
        reshape_nodes = next_node
        if len(reshape_nodes) != len(matmul_nodes):
            return self.none(node, inspect.currentframe().f_lineno)

        # Transpose
        next_nodes = [g.next_nodes(n.output[0]) for n in reshape_nodes]
        if any(len(nn) != 1 or nn[0].op_type != "Transpose" for nn in next_nodes):
            # More than one users after.
            return self.none(node, inspect.currentframe().f_lineno)
        next_node = [_[0] for _ in next_nodes]
        first_perm = list(g.get_attribute(next_node[0], "perm").ints)
        if any(list(g.get_attribute(_, "perm").ints) != first_perm for _ in next_node):
            # Not the same shape for all.
            return self.none(node, inspect.currentframe().f_lineno)
        tr_nodes = next_node
        if len(tr_nodes) != len(matmul_nodes):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(
            self, [*matmul_nodes, *reshape_nodes, *tr_nodes], self.apply, insert_at=node
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        *nodes: NodeProto,
    ) -> List[NodeProto]:
        matmul_nodes = [n for n in nodes if n.op_type == "MatMul"]
        reshape_nodes = [n for n in nodes if n.op_type == "Reshape"]
        tr_nodes = [n for n in nodes if n.op_type == "Transpose"]

        perm = list(g.get_attribute(tr_nodes[0], "perm").ints)
        str_perm = "_".join(map(str, perm))
        f_name = f"PackedMatMulReshapeTranspose{len(matmul_nodes)}_{str_perm}"
        return [
            g.make_node(
                f_name,
                [
                    matmul_nodes[0].input[0],
                    *[_.input[1] for _ in matmul_nodes],
                    reshape_nodes[0].input[1],
                ],
                [_.output[0] for _ in tr_nodes],
                domain="SimplifyingFunction",
                name=f"{self.__class__.__name__}--{matmul_nodes[0].name}",
            )
        ]

    """
        mm_name = g.unique_name(
            f"{self.__class__.__name__}--{matmul_nodes[0].input[1]}-matmul"
        )
        c_name = g.unique_name(f"{self.__class__.__name__}--{matmul_nodes[0].input[1]}-concat")
        concat_node = g.make_node(
            "Concat",
            [n.input[1] for n in matmul_nodes],
            [c_name],
            axis=-1,
            name=f"{self.__class__.__name__}--{matmul_nodes[0].name}",
            doc_string=matmul_nodes[0].doc_string,
        )
        matmul_node = g.make_node(
            "MatMul",
            [matmul_nodes[0].input[0], c_name],
            [mm_name],
            name=f"{self.__class__.__name__}--{matmul_nodes[0].name}",
            doc_string=matmul_nodes[0].doc_string,
        )
        r_nodes = [concat_node, matmul_node]

        # Reshape
        reshape_nodes = [n for n in nodes if n.op_type == "Reshape"]
        sh_names = [
            g.unique_name(f"{self.__class__.__name__}--{reshape_nodes[0].input[1]}-last1"),
            g.unique_name(f"{self.__class__.__name__}--{reshape_nodes[0].input[1]}-last2"),
        ]
        rk = g.get_rank(matmul_nodes[0].output[0])
        splits = g.make_initializer("", np.array([rk, 1], dtype=np.int64))
        split_node = g.make_node(
            "Split",
            [reshape_nodes[0].input[1], splits],
            sh_names,
            name=f"{self.__class__.__name__}--{reshape_nodes[0].name}",
        )
        r_nodes.append(split_node)
        multiply = g.make_initializer("", np.array([rk], dtype=np.int64))
        mul_name = g.unique_name(f"{self.__class__.__name__}--{reshape_nodes[0].input[1]}-mul")
        mul_node = g.make_node(
            "Mul",
            [sh_names[1], multiply],
            [mul_name],
            name=f"{self.__class__.__name__}--{reshape_nodes[0].name}",
        )
        r_nodes.append(mul_node)
        new_shape = g.unique_name(
            f"{self.__class__.__name__}--{reshape_nodes[0].input[1]}-shape"
        )
        concat_node = g.make_node(
            "Concat",
            [sh_names[0], mul_name],
            [new_shape],
            axis=0,
            name=f"{self.__class__.__name__}--{reshape_nodes[0].name}",
        )
        r_nodes.append(concat_node)
        reshaped = g.unique_name(
            f"{self.__class__.__name__}--{reshape_nodes[0].input[1]}-reshaped"
        )
        reshape_node = g.make_node(
            "Reshape",
            [mm_name, new_shape],
            [reshaped],
            name=f"{self.__class__.__name__}--{reshape_nodes[0].name}",
        )
        r_nodes.append(reshape_node)

        # Transpose
        transpose_nodes = [n for n in nodes if n.op_type == "Transpose"]
        perm = list(g.get_attribute(transpose_nodes[0], "perm").ints)
        transposed = g.unique_name(
            f"{self.__class__.__name__}--{transpose_nodes[0].input[0]}-tr"
        )
        tr_node = g.make_node(
            "Transpose",
            [reshaped],
            [transposed],
            perm=perm,
            name=f"{self.__class__.__name__}--{transpose_nodes[0].name}",
        )
        r_nodes.append(tr_node)

        # Split
        split_node = g.make_node(
            "Split",
            [transposed],
            [_.output[0] for _ in transpose_nodes],
            num_outputs=len(transpose_nodes),
            axis=perm[-1],
            name=f"{self.__class__.__name__}--{transpose_nodes[0].name}",
        )
        r_nodes.append(split_node)
        return r_nodes
    """
