"""
.. _l-plot-shape_inference-201:

===========================
201: Better shape inference
===========================


A simple model
==============
"""

import numpy as np
import onnx
import onnx.helper as oh
import onnx.shape_inference as osh
from onnx.reference import ReferenceEvaluator
from experimental_experiment.xshape.shape_builder_impl import BasicShapeBuilder

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Concat", ["X", "Y"], ["xy"], axis=1),
            oh.make_node("Split", ["xy"], ["S1", "S2"], axis=1, num_outputs=2),
            oh.make_node("Concat", ["S2", "S1"], ["zs"], axis=1),
            oh.make_node("Relu", ["zs"], ["Z"]),
        ],
        "dummy",
        [
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["a", "b"]),
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["a", "c"]),
        ],
        [oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, ["a", "e"])],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=9,
)

feeds = dict(X=np.random.rand(3, 4).astype(np.float32), Y=np.random.rand(3, 6).astype(np.float32))
ref = ReferenceEvaluator(model)
expected = ref.run(None, feeds)

# %%
# Classic Shape Inference
# =======================

model2 = osh.infer_shapes(model)

for info in model2.graph.value_info:
    t = info.type.tensor_type
    shape = tuple(d.dim_param or d.dim_value for d in t.shape.dim)
    print(f"{info.name}: {t.elem_type}:{shape}")

# %%
# Basic Shape Inference
# =====================
builder = BasicShapeBuilder()
builder.run_model(model)
builder.update_shapes(model)

for info in model.graph.value_info:
    t = info.type.tensor_type
    shape = tuple(d.dim_param or d.dim_value for d in t.shape.dim)
    print(f"{info.name}: {t.elem_type}:{shape}")
