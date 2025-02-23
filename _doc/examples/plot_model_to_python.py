"""
=======================================
Playground for big optimization pattern
=======================================

# %%
# Write the code produing the model
# =================================
"""

import onnx
from onnx_array_api.translate_api import translate
from experimental_experiment.xbuilder.reverse_graph_builder import to_graph_pattern_matching

model_name = "/home/xadupre/examples/issues/unfused_Attention.onnx"

onx = onnx.load(model_name)
print(translate(onx, api="onnx", use_random_weights=True))

# %%
# Pattern Matching
# ================

pattern = to_graph_pattern_matching(onx)
print(pattern)
