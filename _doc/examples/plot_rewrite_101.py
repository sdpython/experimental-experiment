"""
.. _l-plot-rewrite-101:

=========================
101: Onnx Model Rewriting
=========================

This example shows how to rewrite a graph using a pattern.

A model
=======
"""

from typing import List, Optional
import onnx.helper as oh
from onnx import NodeProto, TensorProto
from experimental_experiment.helpers import pretty_onnx
from onnx_array_api.plotting.graphviz_helper import plot_dot
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.xoptim import EasyPatternOptimization


proto = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Sigmoid", ["Y"], ["sy"]),
            oh.make_node("Mul", ["Y", "sy"], ["ysy"]),
            oh.make_node("Mul", ["X", "ysy"], ["final"]),
        ],
        "nd",
        [
            oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, "b", "c"]),
            oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["a", "b", "c"]),
        ],
        [oh.make_tensor_value_info("final", TensorProto.FLOAT, ["a", "b", "c"])],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=9,
)


print(pretty_onnx(proto))

# %%
# And visually.

plot_dot(proto)

# %%
# The pattern
# ===========


class MulMulSigmoidPattern(EasyPatternOptimization):
    def match_pattern(self, g: GraphBuilder, X, Y):
        return g.op.Mul(X, g.op.Mul(Y, g.op.Sigmoid(Y)))

    def apply_pattern(self, g: GraphBuilder, X, Y):
        return g.anyop.MulMulSigmoid(X, Y, domain="onnx_extended.ortops.optim.cuda")


# %%
# Optimization
# ============

gr = GraphBuilder(
    proto,
    infer_shapes_options=True,
    optimization_options=OptimizationOptions(
        patterns=[MulMulSigmoidPattern(verbose=1)],
        verbose=1,  # a higher value increases the verbosity when optimizations for patterns
    ),
)

new_proto = gr.to_onnx()
print(pretty_onnx(new_proto))

# %%
# And visually.

plot_dot(new_proto)

# %%
# Filtering
# =========
#
# Let's assume now we want to apply the pattern only when the
# shapes are identical.


class MulMulSigmoidPattern2(EasyPatternOptimization):
    def match_pattern(self, g: GraphBuilder, X, Y):
        return g.op.Mul(X, g.op.Mul(Y, g.op.Sigmoid(Y)))

    def apply_pattern(self, g: GraphBuilder, X, Y):
        return g.anyop.MulMulSigmoid(X, Y, domain="onnx_extended.ortops.optim.cuda")

    def validate_mapping(
        self,
        g: GraphBuilder,
        deleted_nodes: List[NodeProto],
        pattern_nodes: Optional[List[NodeProto]] = None,
    ) -> bool:
        for node in deleted_nodes:
            if (
                node.op_type == "Mul"
                and g.has_shape(node.input[0])
                and g.has_shape(node.input[1])
            ):
                sh1 = g.get_shape(node.input[0])
                sh2 = g.get_shape(node.input[1])
                if sh1 != sh2:
                    if self.verbose > 0:
                        print(
                            f"[MulMulSigmoidPattern2.validate_mapping] "
                            f"match not valid because shapes are different"
                            f"{node.input[0]}:{sh1} != {node.input[1]}:{sh2}"
                        )
                    return False
        return True


gr = GraphBuilder(
    proto,
    infer_shapes_options=True,
    optimization_options=OptimizationOptions(
        patterns=[MulMulSigmoidPattern2(verbose=1)],
        verbose=0,
    ),
)

new_proto = gr.to_onnx()
print(pretty_onnx(new_proto))
