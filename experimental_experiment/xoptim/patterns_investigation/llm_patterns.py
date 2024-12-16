import inspect
from collections import Counter
from typing import List, Optional
from onnx import NodeProto
from ...xbuilder import GraphBuilder, FunctionOptions
from ..patterns_api import MatchResult, PatternOptimization
from . import SimplifyingEasyPatternFunction


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
            self,
            [*matmul_nodes, *reshape_nodes, *tr_nodes],
            self.apply,
            insert_at=reshape_nodes[0],
        )

    def apply(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        *nodes: NodeProto,
    ) -> List[NodeProto]:
        matmul_nodes = [n for n in nodes if n.op_type == "MatMul"]
        reshape_nodes = [n for n in nodes if n.op_type == "Reshape"]
        tr_nodes = [n for n in nodes if n.op_type == "Transpose"]

        perm = list(g.get_attribute(tr_nodes[0], "perm").ints)
        str_perm = "_".join(map(str, perm))
        f_name = f"PackedMatMulReshapeTranspose{len(matmul_nodes)}_{str_perm}"
        domain = "SimplifyingFunction"
        nodes_to_return = [
            g.make_node(
                f_name,
                [
                    matmul_nodes[0].input[0],
                    *[_.input[1] for _ in matmul_nodes],
                    reshape_nodes[0].input[1],
                ],
                [_.output[0] for _ in tr_nodes],
                domain=domain,
                name=f"{self.__class__.__name__}--{matmul_nodes[0].name}",
            )
        ]

        # Creates the local function
        if g.builder.has_local_function(f_name, domain=domain):
            return nodes_to_return

        self._add_local_function(
            g.builder,
            domain,
            f_name,
            len(matmul_nodes),
            g.get_rank(matmul_nodes[0].output[0]),
            perm,
        )
        assert g.builder.has_local_function(
            f_name, domain=domain
        ), f"The function {domain}.{f_name} was not added to the builder."
        return nodes_to_return

    @classmethod
    def _add_local_function(
        cls, g: GraphBuilder, domain: str, f_name: str, n_nodes: int, rk: int, perm: List[int]
    ):
        local_g = GraphBuilder(g.main_opset, as_function=True)
        local_g.make_tensor_input("X")
        for i in range(n_nodes):
            local_g.make_tensor_input(f"W{i}")
        local_g.make_tensor_input("shape")

        all_weights = local_g.op.Concat(
            *[f"W{i}" for i in range(n_nodes)], axis=-1, name="merge_weights"
        )
        y = local_g.op.MatMul("X", all_weights, name="packed_matmul")
        names = local_g.op.Split(
            y,
            num_outputs=n_nodes,
            name="split",
            outputs=[f"p{i}" for i in range(n_nodes)],
            axis=-1,
        )
        reshaped = [local_g.op.Reshape(n, "shape", name="reshape") for n in names]
        _transposed = [
            local_g.op.Transpose(n, perm=perm, outputs=[f"Z{i}"], name="transpose")
            for i, n in enumerate(reshaped)
        ]
        for i in range(n_nodes):
            local_g.make_tensor_output(f"Z{i}")

        function_options = FunctionOptions(export_as_function=True, name=f_name, domain=domain)
        g.make_local_function(local_g, function_options=function_options)


class FunctionSplitRotaryMulPattern(SimplifyingEasyPatternFunction):
    """
    Moves the nodes in match_pattern into a local function.

    .. runpython::
        :showcode:

        from experimental_experiment.xbuilder import GraphBuilder
        from experimental_experiment.xoptim import GraphBuilderPatternOptimization
        from experimental_experiment.xoptim.patterns_investigation.llm_patterns import (
            FunctionSplitRotaryMulPattern,
        )

        pat = FunctionSplitRotaryMulPattern()
        g = GraphBuilderPatternOptimization(GraphBuilder(18))
        print(pat._pattern_to_string(g))
    """

    def match_pattern(self, g: GraphBuilder, X, split1, split2, C1, C2):
        rot, part = g.op.Split(X, split1, outputs=2, axis=-1)
        s1, s2 = g.op.Split(rot, split2, outputs=2, axis=-1)
        neg = g.op.Neg(s2)
        fullrot = g.op.Concat(neg, s1, axis=-1)
        add = g.op.Add(
            g.op.Mul(rot, C2),
            g.op.Mul(fullrot, C1),
        )
        return g.op.Concat(add, part, axis=-1)

    def apply_pattern(self, g, X, split1, split2, C1, C2):
        assert self.f_name() == "SplitRotaryMul", f"Name mismatch {self.f_name()!r}"
        return g.anyop.SplitRotaryMul(X, split1, split2, C1, C2, domain="SimplifyingFunction")


class FunctionPowTanhPattern(SimplifyingEasyPatternFunction):
    """
    Moves the nodes in match_pattern into a local function.

    .. runpython::
        :showcode:

        from experimental_experiment.xbuilder import GraphBuilder
        from experimental_experiment.xoptim import GraphBuilderPatternOptimization
        from experimental_experiment.xoptim.patterns_investigation.llm_patterns import (
            FunctionPowTanhPattern,
        )

        pat = FunctionPowTanhPattern()
        g = GraphBuilderPatternOptimization(GraphBuilder(18))
        print(pat._pattern_to_string(g))
    """

    def match_pattern(self, g: GraphBuilder, X, three, o_o_four, half, o_height, one):
        add = g.op.Add(X, g.op.Mul(g.op.Pow(X, three), o_o_four))
        return g.op.Mul(g.op.Mul(X, half), g.op.Add(g.op.Tanh(g.op.Mul(add, o_height)), one))

    def apply_pattern(self, g, X, three, o_o_four, half, o_height, one):
        assert self.f_name() == "PowTanh", f"Name mismatch {self.f_name()!r}"
        return g.anyop.PowTanh(
            X, three, o_o_four, half, o_height, one, domain="SimplifyingFunction"
        )
