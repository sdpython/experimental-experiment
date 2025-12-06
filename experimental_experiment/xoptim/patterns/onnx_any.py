import inspect
from typing import List, Optional
from onnx import NodeProto
from ...helpers import make_idn
from ...xshape._onnx_helper import unary_like_op_types
from ..patterns_api import MatchResult, PatternOptimization


class SameChildrenPattern(PatternOptimization):
    """
    Checks there is no duplicated node doing the same than another one beside.

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

        opset_imports = [
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", 1, 3, 4))
        )
        inputs.append(oh.make_tensor_value_info("sh1", onnx.TensorProto.INT64, shape=(4,)))
        inputs.append(
            oh.make_tensor_value_info("y1", onnx.TensorProto.FLOAT, shape=("a", 2, 3, 4))
        )
        nodes.append(oh.make_node("Expand", ["Y", "sh1"], ["y1"]))
        nodes.append(oh.make_node("Expand", ["Y", "sh1"], ["y2"]))
        outputs.append(
            oh.make_tensor_value_info("y2", onnx.TensorProto.FLOAT, shape=("a", 2, 3, 4))
        )
        outputs.append(
            oh.make_tensor_value_info("y1", onnx.TensorProto.FLOAT, shape=("a", 2, 3, 4))
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

        print("DOT-SECTION", to_dot(model))

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

        opset_imports = [
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", 1, 3, 4))
        )
        inputs.append(oh.make_tensor_value_info("sh1", onnx.TensorProto.INT64, shape=(4,)))
        inputs.append(
            oh.make_tensor_value_info("y1", onnx.TensorProto.FLOAT, shape=("a", 2, 3, 4))
        )
        nodes.append(oh.make_node("Expand", ["Y", "sh1"], ["y1"]))
        nodes.append(oh.make_node("Identity", ["y1"], ["y2"]))
        outputs.append(
            oh.make_tensor_value_info("y2", onnx.TensorProto.FLOAT, shape=("a", 2, 3, 4))
        )
        outputs.append(
            oh.make_tensor_value_info("y1", onnx.TensorProto.FLOAT, shape=("a", 2, 3, 4))
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

        print("DOT-SECTION", to_dot(model))
    """

    @classmethod
    def _cmp(cls, n1: NodeProto, n2: NodeProto) -> bool:
        "Compares two nodes and say if they are the same."
        assert make_idn(n1) != make_idn(n2), f"Two nodes are the same not identical copies {n1}"
        if n1.input != n2.input:
            return False
        if len(n1.output) != len(n2.output):
            return False
        if len(n1.attribute) != len(n2.attribute):
            return False
        for att1, att2 in zip(n1.attribute, n2.attribute):
            if att1.name != att2.name:
                return False
            if att1.type != att2.type:
                return False
            if att1.SerializeToString() != att2.SerializeToString():
                return False
        return True

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        for i in range(len(node.output)):
            next_nodes = g.next_nodes(node.output[i])
            if len(next_nodes) <= 1:
                continue
            res = self._match_with_nodes(g, node, next_nodes)
            if res is not None:
                return res
        return self.none()

    def _match_with_nodes(self, g, node, next_nodes):
        if len(next_nodes) == 2:
            n1, n2 = next_nodes
            if n1.op_type != n2.op_type or n1.op_type == "Identity":
                return self.none()
            if not self._cmp(n1, n2):
                return self.none(node, inspect.currentframe().f_lineno)
            nodes = [n1, n2]
        else:
            cp = {}
            for n in next_nodes:
                if n.op_type == "Identity":
                    continue
                if n.op_type in cp:
                    cp[n.op_type].append(n)
                else:
                    cp[n.op_type] = [n]
            nodes = []
            for v in cp.values():
                if len(v) <= 1:
                    continue
                if len(v) == 2:
                    n1, n2 = v
                    if not self._cmp(n1, n2):
                        continue
                    nodes.extend([n1, n2])
                    continue
                enough = False
                for i in range(len(v) - 1):
                    for j in range(i + 1, len(v)):
                        if self._cmp(v[i], v[j]):
                            nodes.extend([v[i], v[j]])
                            enough = True
                            break
                    if enough:
                        break
            if len(nodes) == 0:
                return self.none(node, inspect.currentframe().f_lineno)

        for i in range(0, len(nodes), 2):
            n1, n2 = nodes[i : i + 2]
            assert len(n1.output) > 0, "A node should not have no output in this pattern."
            assert (
                not g.has_type(n1.output[0])
                or not g.has_type(n2.output[0])
                or g.get_type(n1.output[0]) == g.get_type(n2.output[0])
            ), (
                f"Nodes n1 and n2 have different output type for outputs "
                f"{n1.output[0]!r}, {n2.output[0]}, and types "
                f"{g.get_type(n1.output[0])} != {g.get_type(n2.output[0])})"
            )
        assert "Identity" not in set(n.op_type for n in nodes), (
            f"Identity nodes should be covered by this pattern "
            f"{set(n.op_type for n in nodes)}, type={node.op_type!r}, "
            f"name={node.name!r}."
        )
        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        *nodes: NodeProto,
    ) -> List[NodeProto]:
        """
        The function receives pairs of nodes. We replace every odd node
        by an identity node.
        """
        assert (
            len(nodes) % 2 == 0
        ), f"Expecting an even number of nodes but len(nodes) == {len(nodes)}"
        new_nodes = []
        for i in range(0, len(nodes), 2):
            new_nodes.append(nodes[i])

            for o1, o2 in zip(nodes[i].output, nodes[i + 1].output):
                if not o1 and not o2:
                    continue
                new_nodes.append(
                    g.make_node(
                        "Identity",
                        [o1],
                        [o2],
                        name=f"{self.__class__.__name__}--{nodes[i+1].name}",
                        doc_string=nodes[i + 1].doc_string,
                    )
                )
        return new_nodes


class SameChildrenFromInputPattern(SameChildrenPattern):
    """
    Checks there is no duplicated node doing the same than another
    one beside and taking a model input as input.

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

        opset_imports = [
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("xy1", onnx.TensorProto.FLOAT16, shape=("a", 2, 3, 4))
        )
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", 2, 3, 4))
        )
        nodes.append(oh.make_node("Cast", ["X"], ["xy1"], to=10))
        nodes.append(oh.make_node("Cast", ["X"], ["xy2"], to=10))
        outputs.append(
            oh.make_tensor_value_info("xy1", onnx.TensorProto.FLOAT16, shape=("a", 2, 3, 4))
        )
        outputs.append(
            oh.make_tensor_value_info("xy2", onnx.TensorProto.FLOAT16, shape=("a", 2, 3, 4))
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

        print("DOT-SECTION", to_dot(model))

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

        opset_imports = [
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("xy1", onnx.TensorProto.FLOAT16, shape=("a", 2, 3, 4))
        )
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", 2, 3, 4))
        )
        nodes.append(oh.make_node("Cast", ["X"], ["xy1"], to=10))
        nodes.append(oh.make_node("Identity", ["xy1"], ["xy2"]))
        outputs.append(
            oh.make_tensor_value_info("xy1", onnx.TensorProto.FLOAT16, shape=("a", 2, 3, 4))
        )
        outputs.append(
            oh.make_tensor_value_info("xy2", onnx.TensorProto.FLOAT16, shape=("a", 2, 3, 4))
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

        print("DOT-SECTION", to_dot(model))
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not node.input:
            return self.none()
        node_before = g.node_before(node.input[0])
        if node_before is not None:
            return self.none()
        next_nodes = g.next_nodes(node.input[0])
        if len(next_nodes) <= 1:
            return self.none()
        return self._match_with_nodes(g, node, next_nodes)


class ShapeBasedSameChildrenPattern(PatternOptimization):
    """
    Checks there is no duplicated node doing the same than another one beside.
    :class:`experimental_experiment.xoptim.patterns.onnx_any.SameChildrenPattern`
    checks it is exactly the same.
    This one assumes it is exactly the same in some cases such
    expand (X, sh1) = expand(X, sh2) if the output shapes are the same.

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

        opset_imports = [
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("y1", onnx.TensorProto.FLOAT, shape=("a", 2, 3, 4))
        )
        inputs.append(oh.make_tensor_value_info("sh1", onnx.TensorProto.INT64, shape=(4,)))
        inputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", 1, 3, 4))
        )
        nodes.append(oh.make_node("Expand", ["Y", "sh1"], ["y1"]))
        nodes.append(oh.make_node("Expand", ["Y", "sh1"], ["y2"]))
        outputs.append(
            oh.make_tensor_value_info("y1", onnx.TensorProto.FLOAT, shape=("a", 2, 3, 4))
        )
        outputs.append(
            oh.make_tensor_value_info("y2", onnx.TensorProto.FLOAT, shape=("a", 2, 3, 4))
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

        print("DOT-SECTION", to_dot(model))

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

        opset_imports = [
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("y1", onnx.TensorProto.FLOAT, shape=("a", 2, 3, 4))
        )
        inputs.append(oh.make_tensor_value_info("sh1", onnx.TensorProto.INT64, shape=(4,)))
        inputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", 1, 3, 4))
        )
        nodes.append(oh.make_node("Expand", ["Y", "sh1"], ["y1"]))
        nodes.append(oh.make_node("Identity", ["y1"], ["y2"]))
        outputs.append(
            oh.make_tensor_value_info("y1", onnx.TensorProto.FLOAT, shape=("a", 2, 3, 4))
        )
        outputs.append(
            oh.make_tensor_value_info("y2", onnx.TensorProto.FLOAT, shape=("a", 2, 3, 4))
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

        print("DOT-SECTION", to_dot(model))
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"Expand", "Reshape"} or node.domain != "":
            return self.none()
        next_nodes = g.next_nodes(node.input[0])
        if len(next_nodes) <= 1:
            return self.none()

        op_types = {n.op_type for n in next_nodes}

        for op_type in ["Expand", "Reshape"]:
            if op_type in op_types:
                selected = [
                    n
                    for n in next_nodes
                    if n.op_type == op_type
                    and n.input[0] == node.input[0]
                    and g.has_shape(n.output[0])
                ]
                if len(selected) < 2:
                    continue
                shapes = [g.get_shape(n.output[0]) for n in selected]
                if len(set(shapes)) == 1:
                    return MatchResult(self, selected, self.apply)
        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        *nodes: NodeProto,
    ) -> List[NodeProto]:
        """
        The function receives multiple nodes of the same type.
        We keep one and replace the other by an Identity node.
        """
        new_nodes = [
            nodes[0],
            *[
                g.make_node(
                    "Identity",
                    [nodes[0].output[0]],
                    [n.output[0]],
                    name=f"{self.__class__.__name__}--{n.name}",
                )
                for n in nodes[1:]
            ],
        ]
        return new_nodes


class IdentityPattern(PatternOptimization):
    """
    Replaces operator such as
    Div(X, 1), Mul(X, 1), Add(X, 0), Sub(X, 0), Transpose(X, [0, 1, 2, ...])
    by identity nodes.

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

        opset_imports = [
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("x3", onnx.TensorProto.FLOAT, shape=("a", "b", "c"))
        )
        nodes.append(oh.make_node("Transpose", ["x3"], ["Y"], perm=[0, 1, 2]))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "b", "c"))
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

        print("DOT-SECTION", to_dot(model))

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

        opset_imports = [
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("x3", onnx.TensorProto.FLOAT, shape=("a", "b", "c"))
        )
        nodes.append(oh.make_node("Identity", ["x3"], ["Y"]))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "b", "c"))
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

        print("DOT-SECTION", to_dot(model))
    """

    @classmethod
    def _any_value_to_scalar(cls, cst):
        try:
            return float(cst)
        except TypeError:
            return complex(cst)

    @classmethod
    def _has_unique_value(cls, cst):
        return cst.min() == cst.max()

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            node.op_type
            not in {"Add", "Mul", "Div", "Sub", "Transpose", "Slice", "And", "Or", "Expand"}
            or node.domain != ""
        ):
            return self.none()

        if node.op_type == "Slice":
            if len(node.input) == 5:
                steps = node.input[4]
                if (
                    not g.is_constant(steps)
                    or g.get_computed_constant(steps) is None
                    or set(g.get_computed_constant(steps)) != {1}
                ):
                    return self.none(node, inspect.currentframe().f_lineno)
            starts, ends = node.input[1:3]
            if (
                not g.is_constant(starts)
                or g.get_computed_constant(starts) is None
                or set(g.get_computed_constant(starts)) != {0}
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            if (
                not g.is_constant(ends)
                or g.get_computed_constant(ends) is None
                or set(g.get_computed_constant(ends))
                != {9223372036854775807}  # this a value used by torch
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [node], self.apply, insert_at=node)

        if node.op_type == "Transpose":
            perm = list(g.get_attribute(node, "perm").ints)
            expected = list(range(len(perm)))
            if perm != expected:
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [node], self.apply, insert_at=node)

        if node.op_type == "Expand":
            if not g.is_constant(node.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            value = g.get_computed_constant(node.input[1])
            if value is None:
                return self.none(node, inspect.currentframe().f_lineno)
            with g.builder.maybe_disable_fake_tensor_mode():
                unique = set(value)
            if unique != {1}:
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [node], self.apply, insert_at=node)

        assert len(node.input) == 2, (
            f"Unexpected number of inputs {len(node.input)} "
            f"for node type {node.op_type!r} and name {node.name!r}, "
            f"node.input={node.input}"
        )
        if g.is_constant(node.input[1]):
            if g.has_rank(node.input[1]) and g.get_rank(node.input[1]) > 1:
                # No need to go further.
                return self.none(node, inspect.currentframe().f_lineno)
            shape = g.get_constant_shape(node.input[1], exc=False)
            if shape is None:
                return self.none(node, inspect.currentframe().f_lineno)
            if shape in (tuple(), (1,)):
                # simple case
                if not g.is_constant_scalar(node.input[1]):
                    return self.none(node, inspect.currentframe().f_lineno)
                cst = g.get_constant_scalar(node.input[1])
                val = self._any_value_to_scalar(cst)
                if val == 0:
                    if node.op_type in {"Add", "Sub", "Or"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val is False:
                    if node.op_type in {"Or"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val == 1:
                    if node.op_type in {"Mul", "Div", "And"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val is True:
                    if node.op_type in {"And"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
            elif len(shape) == 1 and node.op_type in {"Add", "Mul", "Sub", "Div", "And", "Or"}:
                # less simple case, the tensor is multiplied on its last dimension.
                cst = g.get_computed_constant(node.input[1])
                if cst is None:
                    return self.none(node, inspect.currentframe().f_lineno)
                if not self._has_unique_value(cst):
                    return self.none(node, inspect.currentframe().f_lineno)
                unique = cst[0]
                if not g.has_shape(node.input[0]):
                    return self.none(node, inspect.currentframe().f_lineno)
                shape = g.get_shape(node.input[0])
                if shape[-1] != cst.shape[0]:
                    return self.none(node, inspect.currentframe().f_lineno)
                if node.op_type in {"Add", "Sub", "And"} and unique != 0:
                    return self.none(node, inspect.currentframe().f_lineno)
                if node.op_type in {"And"} and unique is not True:
                    return self.none(node, inspect.currentframe().f_lineno)
                if node.op_type in {"Mul", "Div", "Or"} and unique != 1:
                    return self.none(node, inspect.currentframe().f_lineno)
                if node.op_type in {"Or"} and unique is not False:
                    return self.none(node, inspect.currentframe().f_lineno)
                return MatchResult(self, [node], self.apply, insert_at=node)
        elif g.is_constant(node.input[0]):
            if g.has_rank(node.input[0]) and g.get_rank(node.input[0]) > 1:
                # No need to go further.
                return self.none(node, inspect.currentframe().f_lineno)
            shape = g.get_constant_shape(node.input[0], exc=False)
            if shape is None:
                return self.none(node, inspect.currentframe().f_lineno)
            if shape in (tuple(), (1,)):
                # simple case
                if not g.is_constant_scalar(node.input[0]):
                    return self.none(node, inspect.currentframe().f_lineno)
                cst = g.get_constant_scalar(node.input[0])
                val = self._any_value_to_scalar(cst)
                if val == 0:
                    if node.op_type in {"Add", "Or"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val is False:
                    if node.op_type in {"Or"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val == 1:
                    if node.op_type in {"Mul", "And"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val is True:
                    if node.op_type in {"And"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)

        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        name = node.input[
            (
                1
                if g.is_constant(node.input[0])
                and g.has_shape(node.input[0])
                and g.get_shape(node.input[0]) in {(), (1,)}
                else 0
            )
        ]
        return [
            g.make_node(
                "Identity",
                [name],
                [node.output[0]],
                name=f"{self.__class__.__name__}--{node.name}",
            )
        ]


class ShapeBasedIdentityPattern(PatternOptimization):
    """
    If a slice leads to the same shape and the step is 1 then it is identity.
    In some cases, just known the same is enough to replace them.

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

        opset_imports = [
            oh.make_opsetid("", 18),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = [onh.from_array(np.array([0], dtype=np.int64), "zero")]
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a",)))
        nodes.append(
            oh.make_node(
                "Shape",
                ["X"],
                ["N"],
                start=0,
                end=1,
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["zero"],
                value=onh.from_array(np.array([0], dtype=np.int64), name="value"),
            )
        )
        nodes.append(oh.make_node("Slice", ["X", "zero", "N", "zero"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a",)))
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))

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

        opset_imports = [
            oh.make_opsetid("", 18),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a",)))
        nodes.append(oh.make_node("Identity", ["X"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a",)))
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"Transpose", "Slice"} or node.domain != "":
            return self.none()

        if node.op_type == "Slice":
            if len(node.input) == 5:
                steps = node.input[4]
                if (
                    not g.is_constant(steps)
                    or g.get_computed_constant(steps) is None
                    or set(g.get_computed_constant(steps)) != {1}
                ):
                    return self.none(node, inspect.currentframe().f_lineno)

            if not g.has_shape(node.input[0]) or not g.has_shape(node.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)

            shape_i = g.get_shape_renamed(node.input[0])
            shape_o = g.get_shape_renamed(node.output[0])
            if shape_i == shape_o:
                return MatchResult(self, [node], self.apply, insert_at=node)

        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "Identity",
                [node.input[0]],
                [node.output[0]],
                name=f"{self.__class__.__name__}--{node.name}",
            )
        ]


class SwapUnaryPattern(PatternOptimization):
    """
    Tries to move computation nodes before any transpose or reshape.
    That works for unary operator or equivalent to that.

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
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b", "c", "d"))
        )
        inputs.append(oh.make_tensor_value_info("cst", onnx.TensorProto.FLOAT, shape=(1,)))
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["cst"],
                value=onh.from_array(np.array([2.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(oh.make_node("Transpose", ["X"], ["xt"], perm=[0, 2, 1, 3]))
        nodes.append(oh.make_node("Mul", ["xt", "cst"], ["Y"]))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "c", "b", "d"))
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

        print("DOT-SECTION", to_dot(model))

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
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b", "c", "d"))
        )
        inputs.append(oh.make_tensor_value_info("cst", onnx.TensorProto.FLOAT, shape=(1,)))
        nodes.append(oh.make_node("Mul", ["X", "cst"], ["SwapUnaryPattern--xt"]))
        nodes.append(
            oh.make_node("Transpose", ["SwapUnaryPattern--xt"], ["Y"], perm=[0, 2, 1, 3])
        )
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "c", "b", "d"))
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

        print("DOT-SECTION", to_dot(model))
    """

    _unary_types = unary_like_op_types()
    _binary_types_scalar_cst = {"Mul", "Add", "Div", "Sub"}

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            node.op_type not in {"Transpose", "Reshape", "Squeeze", "Unsqueeze"}
            or node.domain != ""
        ):
            return self.none()
        if not g.has_rank(node.input[0]) or g.get_rank(node.input[0]) == 0:
            # the unary or binary op changes the shape
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node.output[0]) or node.op_type in {"Squeeze", "Unsqueeze"}:
            return self.none(node, inspect.currentframe().f_lineno)
        next_node = g.next_node(node.output[0])
        if next_node.op_type not in self._unary_types and (
            next_node.op_type not in self._binary_types_scalar_cst
            or not g.has_shape(next_node.input[1])
            or g.get_shape(next_node.input[1]) != (1,)
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply)

    def apply(
        self, g: "GraphBuilder", node: NodeProto, unary_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        temp_name = g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")
        res = [
            g.make_node(
                unary_node.op_type,
                [node.input[0], *unary_node.input[1:]],
                [temp_name],
                name=f"{self.__class__.__name__}--{unary_node.name}",
            ),
            g.make_node(
                node.op_type,
                [temp_name, *node.input[1:]],
                [unary_node.output[0]],
                name=f"{self.__class__.__name__}--{node.name}",
            ),
        ]
        if unary_node.attribute:
            res[0].attribute.extend(unary_node.attribute)
        if node.attribute:
            res[1].attribute.extend(node.attribute)
        return res
