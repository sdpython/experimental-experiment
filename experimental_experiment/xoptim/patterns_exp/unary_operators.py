import inspect
from typing import List, Optional
from onnx import NodeProto, TensorProto
from ..patterns_api import MatchResult, PatternOptimization


class TransposeCastPattern(PatternOptimization):
    """
    Replaces Cast + Transpose or Transpose + Cast into
    Transpose2DCast16 or Transpose2DCastFP32 depending on the output type.

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
            oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b")))
        nodes.append(make_node_extended("Transpose", ["X"], ["xt"], perm=[1, 0]))
        nodes.append(make_node_extended("Cast", ["xt"], ["Y"], to=10))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, shape=("b", "a"))
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
        from onnx_array_api.translate_api.make_helper import make_node_extended

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b")))
        nodes.append(
            make_node_extended(
                "Transpose2DCastFP16", ["X"], ["Y"], domain="onnx_extended.ortops.optim.cuda"
            )
        )
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, shape=("b", "a"))
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

    _allowed_types = (TensorProto.FLOAT, TensorProto.FLOAT16)

    def __init__(self, verbose: int = 0, priority: int = 3):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type != "Transpose" or node.domain != "":
            return self.none()

        perm = list(g.get_attribute(node, "perm").ints)
        if perm != [1, 0]:
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_type(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        if g.get_type(node.input[0]) not in self._allowed_types:
            return self.none(node, inspect.currentframe().f_lineno)

        cast_node_before = g.node_before(node.input[0])
        if (
            cast_node_before is None
            or cast_node_before.op_type != "Cast"
            or cast_node_before.domain != ""
            or g.is_used_more_than_once(node.input[0])
            or not g.has_type(cast_node_before.input[0])
            or g.get_type(cast_node_before.input[0]) not in self._allowed_types
        ):
            cast_node_before = None

        if cast_node_before is not None:
            return MatchResult(self, [cast_node_before, node, None], self.apply, insert_at=node)

        cast_node_after = g.next_nodes(node.output[0])
        if (
            len(cast_node_after) != 1
            or cast_node_after[0].op_type != "Cast"
            or cast_node_after[0].domain != ""
            or g.is_used_more_than_once(node.output[0])
            or not g.has_type(cast_node_after[0].output[0])
            or g.get_type(cast_node_after[0].output[0]) not in self._allowed_types
        ):
            cast_node_after = None

        if cast_node_after is not None:
            return MatchResult(self, [None, node, cast_node_after[0]], self.apply, insert_at=node)

        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        cast_node_before: Optional[NodeProto],
        node: NodeProto,
        cast_node_after: Optional[NodeProto],
    ) -> List[NodeProto]:
        out_type = (
            g.get_type(node.output[0])
            if cast_node_after is None
            else g.get_type(cast_node_after.output[0])
        )

        if out_type == TensorProto.FLOAT:
            suffix = "32"
        elif out_type == TensorProto.FLOAT16:
            suffix = "16"
        else:
            raise AssertionError(f"out_type={out_type} must be in {self._allowed_types}")

        new_node = g.make_node(
            f"Transpose2DCastFP{suffix}",
            node.input if cast_node_before is None else cast_node_before.input,
            node.output if cast_node_after is None else cast_node_after.output,
            domain="onnx_extended.ortops.optim.cuda",
            name=f"{self.__class__.__name__}--{node.name}",
        )
        return [new_node]
