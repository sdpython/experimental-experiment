import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class SimpleRotaryPattern(PatternOptimization):
    """
    Replaces Split Neg Concat by SimpleRotary.

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
        inputs.append(oh.make_tensor_value_info("splits", onnx.TensorProto.INT64, shape=(2,)))
        inputs.append(
            oh.make_tensor_value_info(
                "X", onnx.TensorProto.FLOAT, shape=("UNKNOWNDIM", "UNKNOWNDIM1")
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["splits"],
                value=onh.from_array(np.array([4, 4], dtype=np.int64), name="value"),
            )
        )
        nodes.append(make_node_extended("Split", ["X", "splits"], ["s1", "s2"], axis=-1))
        nodes.append(make_node_extended("Neg", ["s2"], ["ns2"]))
        nodes.append(make_node_extended("Concat", ["ns2", "s1"], ["Y"], axis=-1))
        outputs.append(
            oh.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, shape=("UNKNOWNDIM2", "UNKNOWNDIM3")
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
        inputs.append(oh.make_tensor_value_info("splits", onnx.TensorProto.INT64, shape=(2,)))
        inputs.append(
            oh.make_tensor_value_info(
                "X", onnx.TensorProto.FLOAT, shape=("UNKNOWNDIM", "UNKNOWNDIM1")
            )
        )
        nodes.append(
            make_node_extended(
                "Rotary",
                ["X", "splits"],
                ["Y"],
                domain="onnx_extended.ortops.optim.cuda",
                side="right",
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, shape=("UNKNOWNDIM2", "UNKNOWNDIM3")
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

        print("DOT-SECTION", to_dot(model))
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type != "Split" or node.domain != "":
            return self.none()

        if not g.has_rank(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        axis = g.get_attribute(node, "axis", exc=False)
        axis = 0 if axis is None else axis.i
        rk = g.get_rank(node.input[0])
        if axis < 0:
            axis += rk
        if axis != rk - 1:
            return self.none(node, inspect.currentframe().f_lineno)

        if len(node.input) == 2:
            cst = g.get_computed_constant(node.input[1])
            if cst.dtype != np.int64 or cst.shape != (2,) or cst[0] != cst[1]:
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            att = g.get_attribute(node, "num_outputs", exc=False)
            if att is None or att.i != 2:
                return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(node.output[0]) or g.is_used_more_than_once(node.output[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        left_node = g.next_node(node.output[0])
        right_node = g.next_node(node.output[1])
        if left_node.op_type != "Neg" and right_node.op_type != "Neg":
            return self.none(node, inspect.currentframe().f_lineno)
        if left_node.op_type != "Concat" and right_node.op_type != "Concat":
            return self.none(node, inspect.currentframe().f_lineno)

        if left_node.op_type == "Neg":
            inputs = [node.output[1], left_node.output[0]]
            neg_node = left_node
            concat_node = right_node
        else:
            inputs = [right_node.output[0], node.output[0]]
            neg_node = right_node
            concat_node = left_node
        if inputs != list(concat_node.input):
            return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(neg_node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        axis_ = g.get_attribute(concat_node, "axis", exc=False).i
        if axis_ < 0:
            axis_ += rk
        if axis != axis_:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, neg_node, concat_node], self.apply, insert_at=concat_node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        split_node: NodeProto,
        neg_node: NodeProto,
        concat_node: NodeProto,
    ) -> List[NodeProto]:
        side = "right" if neg_node.input[0] == split_node.output[1] else "left"
        new_node = g.make_node(
            "Rotary",
            split_node.input,
            concat_node.output,
            side=side,
            name=f"{self.__class__.__name__}--{neg_node.name}",
            domain="onnx_extended.ortops.optim.cuda",
        )
        return [new_node]
