import inspect
from typing import List, Optional
from onnx import NodeProto, TensorProto
from ...helpers import size_type, is_float_type
from ..patterns_api import MatchResult, PatternOptimization


class CastPattern(PatternOptimization):
    """Checks that a Cast is really needed."""

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Cast" or node.domain != "":
            return self.none()

        if not g.has_type(node.input[0]):
            itype = g.try_infer_type(node.input[0], exc=False)
            if itype == 0:
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            itype = g.get_type(node.input[0])

        att = g.get_attribute(node, "to")

        if att.i != itype:
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


class CastCastPattern(PatternOptimization):
    """
    Checks that two consecutive cast can be avoided.

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
            oh.make_tensor_value_info("x1", onnx.TensorProto.FLOAT16, shape=("b", "c"))
        )
        nodes.append(make_node_extended("Cast", ["x1"], ["x2"], to=1))
        nodes.append(make_node_extended("Cast", ["x2"], ["Y"], to=1))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("b", "c")))
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
            oh.make_tensor_value_info("x1", onnx.TensorProto.FLOAT16, shape=("b", "c"))
        )
        nodes.append(make_node_extended("Cast", ["x1"], ["Y"], to=1))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("b", "c")))
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

    def _one_cast(self, t1: int, t2: int, t3: int) -> int:
        if t1 == t3:
            if t2 == t1:
                return t1
            if t1 in {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16}:
                if t2 in {TensorProto.FLOAT, TensorProto.DOUBLE}:
                    return t1
        elif t3 == t2:
            return t2
        elif t1 == t2:
            return t3
        return 0

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Cast" or node.domain != "":
            return self.none()
        cast_before = g.node_before(node.input[0])
        if cast_before is None or cast_before.op_type != "Cast" or cast_before.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_type(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        itype = g.get_type(cast_before.input[0])
        middle_type = g.get_attribute(cast_before, "to").i
        final_type = g.get_attribute(node, "to").i
        if not self._one_cast(itype, middle_type, final_type):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(
            self,
            [cast_before, node],
            self.apply,
            insert_at=None if g.is_used_more_than_once(node.input[0]) else node,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        cast1: NodeProto,
        cast2: NodeProto,
    ) -> List[NodeProto]:
        itype = g.get_type(cast1.input[0])
        middle_type = g.get_attribute(cast1, "to").i
        final_type = g.get_attribute(cast2, "to").i
        one_type = self._one_cast(itype, middle_type, final_type)
        extend = [cast1] if g.is_used_more_than_once(cast1.output[0]) else []
        if one_type == itype:
            return [
                *extend,
                g.make_node(
                    "Identity",
                    cast1.input,
                    cast2.output,
                    name=f"{self.__class__.__name__}--{cast2.name}",
                    doc_string=cast2.doc_string,
                ),
            ]
        return [
            *extend,
            g.make_node(
                "Cast",
                cast1.input,
                cast2.output,
                to=one_type,
                name=f"{self.__class__.__name__}--{cast2.name}",
                doc_string=cast2.doc_string,
            ),
        ]


class CastCastBinaryPattern(PatternOptimization):
    """
    Moves two cast operators beyond a binary operator
    The cast must cast from a float type to another float type.

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
        inputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", 4)))
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", 4)))
        nodes.append(make_node_extended("Cast", ["X"], ["xc"], to=10))
        nodes.append(make_node_extended("Cast", ["Y"], ["yc"], to=10))
        nodes.append(make_node_extended("Add", ["xc", "yc"], ["Z"]))
        outputs.append(oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT16, shape=("a", 4)))
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
        inputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", 4)))
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", 4)))
        nodes.append(make_node_extended("Add", ["X", "Y"], ["add-X"]))
        nodes.append(make_node_extended("Cast", ["add-X"], ["Z"], to=10))
        outputs.append(oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT16, shape=("a", 4)))
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

    _dtypes_allowed = {
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
    }

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"Add", "Div", "Mul", "Sub"} or node.domain != "":
            return self.none()

        if g.is_used_more_than_once(node.input[0]) or g.is_used_more_than_once(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_type(node.input[0]) or not g.has_type(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        dtype_left, dtype_right = g.get_type(node.input[0]), g.get_type(node.input[1])
        if dtype_left not in self._dtypes_allowed or dtype_right not in self._dtypes_allowed:
            return self.none(node, inspect.currentframe().f_lineno)

        left, right = g.node_before(node.input[0]), g.node_before(node.input[1])
        if left is None or left.op_type != "Cast" or left.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if right is None or right.op_type != "Cast" or right.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        itype = (
            g.get_type(left.input[0])
            if g.has_type(left.input[0])
            else (g.get_type(right.input[0]) if g.has_type(right.input[0]) else 0)
        )
        if itype == 0:
            return self.none(node, inspect.currentframe().f_lineno)

        dtype_left, dtype_right = g.get_type(left.input[0]), g.get_type(right.input[0])
        if dtype_left not in self._dtypes_allowed or dtype_right not in self._dtypes_allowed:
            return self.none(node, inspect.currentframe().f_lineno)

        # We also need to check the precision is not lowered.
        # At this stage dtype_left == dtype_right otherwise ONNX would complain.
        if size_type(dtype_left) > size_type(itype):
            # The precision is higher for the computation. Let's not do that.
            return self.none(node, inspect.currentframe().f_lineno)
        if is_float_type(dtype_left) != is_float_type(itype):
            # float, int changes, let's avoid that as well
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [left, right, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        left: NodeProto,
        right: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        to = g.get_attribute(left, "to")

        new_node = g.make_node(
            node.op_type,
            [left.input[0], right.input[0]],
            name=f"{self.__class__.__name__}--{node.name}",
        )
        cast_node = g.make_node(
            "Cast",
            new_node.output,
            node.output,
            to=to.i,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node, cast_node]


class CastOpCastPattern(PatternOptimization):
    """
    Removes two cast surrounding another operator.

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
        inputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, shape=("a", "b"))
        )
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b")))
        nodes.append(make_node_extended("Cast", ["Y"], ["yc"], to=1))
        nodes.append(make_node_extended("Add", ["X", "yc"], ["zc"]))
        nodes.append(make_node_extended("Cast", ["zc"], ["Z"], to=10))
        outputs.append(
            oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT16, shape=("a", "b"))
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
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, shape=("a", "b"))
        )
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b")))
        nodes.append(make_node_extended("Cast", ["X"], ["CastOpCastPattern--zc"], to=10))
        nodes.append(make_node_extended("Add", ["CastOpCastPattern--zc", "Y"], ["Z"]))
        outputs.append(
            oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT16, shape=("a", "b"))
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

    _dtypes_allowed = {
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
    }

    _unary_ops = {"MulSigmoid", "Neg", "Sigmoid", "Softmax"}
    _binary_ops = {"Add", "Sub", "Mul", "Div"}
    _other_ops = {"SoftmaxGrad"}

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            (node.op_type not in self._binary_ops or node.domain != "")
            and node.op_type not in self._other_ops
            and node.op_type not in self._unary_ops
        ):
            return self.none()
        if "ComputationCastOpCastPattern--" in node.name:
            return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(node.input[0]) or (
            len(node.input) > 1 and g.is_used_more_than_once(node.input[1])
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        cast_out_node = next_nodes[0]
        if cast_out_node.op_type != "Cast" or cast_out_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        cast_in_left = g.node_before(node.input[0])
        cast_in_right = g.node_before(node.input[1]) if len(node.input) > 1 else None
        if "Cast" not in (
            "" if cast_in_left is None else cast_in_left.op_type,
            "" if cast_in_right is None else cast_in_right.op_type,
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        if cast_out_node is None and (
            cast_in_left is None
            or cast_in_left.op_type != "Cast"
            or cast_in_right is None
            or cast_in_right.op_type != "Cast"
        ):
            # Then we only allow this if the computation type is lower precision.
            compute_type = g.get_type(node.output[0])
            before_type = g.get_type((cast_in_left or cast_in_right).input[0])
            if not (
                compute_type == TensorProto.FLOAT
                and before_type in (TensorProto.FLOAT16, TensorProto.BFLOAT16)
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            if not is_float_type(compute_type) and is_float_type(before_type):
                # The intent is something else.
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            compute_type = g.get_type(node.output[0])
            other_type = (
                g.get_type(cast_out_node.output[0])
                if cast_out_node
                else (g.get_type((cast_in_left or cast_in_right).input[0]))
            )
            if not is_float_type(compute_type) and is_float_type(other_type):
                # The intent is something else.
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self,
            [
                (
                    cast_in_left
                    if cast_in_left is not None and cast_in_left.op_type == "Cast"
                    else None
                ),
                (
                    cast_in_right
                    if cast_in_right is not None and cast_in_right.op_type == "Cast"
                    else None
                ),
                node,
                cast_out_node,
            ],
            self.apply,
            insert_at=node,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        cast_in_left: NodeProto,
        cast_in_right: NodeProto,
        node: NodeProto,
        cast_out_node: NodeProto,
    ) -> List[NodeProto]:
        # to = g.get_attribute(cast_in_left or cast_in_right, "to").i
        to_out = g.get_attribute(cast_out_node, "to").i
        left_input = None
        right_input = None
        new_nodes = []

        if cast_in_left is None:
            left_input = g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")
            new_nodes.append(
                g.make_node(
                    "Cast",
                    [node.input[0]],
                    [left_input],
                    to=to_out,
                    name=f"{self.__class__.__name__}--CastL",
                )
            )
        else:
            left_input = cast_in_left.input[0]

        if cast_in_right is None and len(node.input) > 1:
            right_input = g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")
            new_nodes.append(
                g.make_node(
                    "Cast",
                    [node.input[1]],
                    [right_input],
                    to=to_out,
                    name=f"{self.__class__.__name__}--CastR",
                )
            )
        else:
            right_input = None if cast_in_right is None else cast_in_right.input[0]

        new_node = g.make_node(
            node.op_type,
            [left_input] if right_input is None else [left_input, right_input],
            cast_out_node.output,
            domain=node.domain,
            name=f"{self.__class__.__name__}--{node.name}",
        )
        if node.attribute:
            new_node.attribute.extend(node.attribute)
        new_nodes.append(new_node)

        if g.is_used_more_than_once(node.output[0]):
            final_cast = g.make_node(
                "Cast",
                [new_node.output[0]],
                [node.output[0]],
                to=g.get_type(node.output[0]),
                name=f"{self.__class__.__name__}--{node.name}",
            )
            new_nodes.append(final_cast)

        return new_nodes


class ComputationCastOpCastPattern(PatternOptimization):
    """
    Changes the computation type to make it faster if one of the inputs
    was just casted before.

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
        inputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, shape=("a", "b"))
        )
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b")))
        nodes.append(make_node_extended("Cast", ["Y"], ["yc"], to=1))
        nodes.append(make_node_extended("Add", ["X", "yc"], ["Z"]))
        outputs.append(oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, shape=("a", "b")))
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
        inputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, shape=("a", "b"))
        )
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b")))
        nodes.append(
            make_node_extended("Cast", ["X"], ["ComputationCastOpCastPattern--X"], to=10)
        )
        nodes.append(
            make_node_extended(
                "Add",
                ["ComputationCastOpCastPattern--X", "Y"],
                ["ComputationCastOpCastPattern--Z"],
            )
        )
        nodes.append(
            make_node_extended("Cast", ["ComputationCastOpCastPattern--Z"], ["Z"], to=1)
        )
        outputs.append(oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, shape=("a", "b")))
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

    _dtypes_allowed = {
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
    }

    _binary_ops = {"Add", "Sub", "Mul", "Div"}

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in self._binary_ops or node.domain != "":
            return self.none()

        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])
        type_left = "" if node_left is None else node_left.op_type
        type_right = "" if node_right is None else node_right.op_type
        if not ((type_left == "Cast") ^ (type_right == "Cast")):
            # only one cast allowed
            return self.none(node, inspect.currentframe().f_lineno)

        if type_left == "Cast":
            node_right = None
        else:
            node_left = None
        node_before = node_left or node_right
        if not g.has_type(node.output[0]) or not g.has_type(node_before.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        output_type = g.get_type(node.output[0])
        before_type = g.get_type(node_before.input[0])
        if not (
            output_type == TensorProto.FLOAT
            and before_type in (TensorProto.FLOAT16, TensorProto.BFLOAT16)
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(node_before.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        op_types = [n.op_type for n in next_nodes]
        if "Cast" in op_types:
            return self.none(node, inspect.currentframe().f_lineno)

        # At this stage, we know the computation type is float and one input
        # has a lower type precision. Let's change it.
        return MatchResult(self, [node_left, node_right, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_left: Optional[NodeProto],
        node_right: Optional[NodeProto],
        node: NodeProto,
    ) -> List[NodeProto]:
        to_type = g.get_type(node.output[0])

        inputs = []
        if node_left is None:
            before_type = g.get_type(node_right.input[0])
            name = g.unique_name(f"{self.__class__.__name__}--{node.input[0]}")
            cast_node = g.make_node(
                "Cast",
                [node.input[0]],
                [name],
                to=before_type,
                name=f"{self.__class__.__name__}--{node.name}",
            )
            inputs = [name, node_right.input[0]]
        else:
            before_type = g.get_type(node_left.input[0])
            name = g.unique_name(f"{self.__class__.__name__}--{node.input[1]}")
            cast_node = g.make_node(
                "Cast",
                [node.input[1]],
                [name],
                to=before_type,
                name=f"{self.__class__.__name__}--{node.name}",
            )
            inputs = [node_left.input[0], name]

        name = g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")
        new_node = g.make_node(
            node.op_type,
            inputs,
            [name],
            domain=node.domain,
            name=f"{self.__class__.__name__}--{node.name}",
        )
        final_cast = g.make_node(
            "Cast",
            [name],
            [node.output[0]],
            to=to_type,
            name=f"{self.__class__.__name__}--{node.name}",
        )
        return [cast_node, new_node, final_cast]
