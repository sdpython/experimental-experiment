"""
=======================================
Playground for big optimization pattern
=======================================

# %%
# Write the code produing the model
# =================================
"""
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx_array_api.translate_api import translate
from experimental_experiment.xbuilder.reverse_graph_builder import to_graph_pattern_matching

onx = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Cast", ["v0_0"], ["x1"], to=onnx.TensorProto.FLOAT),
            oh.make_node("Cast", ["v0_0"], ["x2"], to=onnx.TensorProto.FLOAT),
            oh.make_node("Flatten", ["x1"], ["f1"], axis=0),
            oh.make_node("Flatten", ["x2"], ["f2"], axis=0),
            oh.make_node("Concat", ["f1", "i1"], ["c1"], axis=1),
            oh.make_node("Concat", ["f2", "i2"], ["c2"], axis=1),
            oh.make_node("Reshape", ["c1", "s1"], ["m1"]),
            oh.make_node("Reshape", ["c2", "s2"], ["m2"]),
            oh.make_node("MatMul", ["m1", "m2"], ["mm"]),
            oh.make_node("Identity", ["mm"], ["output"]),
        ],
        "nd",
        [oh.make_tensor_value_info("v0_0", onnx.TensorProto.DOUBLE, [5])],
        [oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 3, 3, 3])],
        [
            onh.from_array(np.zeros((1, 49)).astype(np.float32), name="i1"),
            onh.from_array(np.zeros((1, 4)).astype(np.float32), name="i2"),
            onh.from_array(np.array([2, 3, 3, 3], dtype=np.int64), name="s1"),
            onh.from_array(np.array([3, 3], dtype=np.int64), name="s2"),
        ],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=9,
)
print(translate(onx, api="onnx", use_random_weights=True))

# %%
# Pattern Matching
# ================

pattern = to_graph_pattern_matching(onx)
print(pattern)
