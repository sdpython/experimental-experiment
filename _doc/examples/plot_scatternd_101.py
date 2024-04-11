"""
==============
101: ScatterND
==============

How to parallelize something like the following?
"""

import numpy as np
import onnx.helper as oh
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot


model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node(
                "ScatterND", ["X", "indices", "updates"], ["Y"], reduction="add"
            )
        ],
        "g",
        [
            oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", "b"]),
            oh.make_tensor_value_info("indices", TensorProto.INT64, ["i", "j", "k"]),
            oh.make_tensor_value_info("updates", TensorProto.FLOAT, ["i", "j", "k"]),
        ],
        [oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["a", "b"])],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=9,
)

print(onnx_simple_text_plot(model))


shape = (5, 7)
X = np.zeros(shape, dtype=np.float32)
indices = np.zeros((2, 10, 1)).astype(np.int64)
updates = np.ones((2, 10, 7)).astype(np.float32)
feeds = {"X": X, "indices": indices, "updates": updates}


##########################################
# Let's see the evaluation by the ReferenceEvaluator.


def _scatter_nd_impl(data, indices, updates, reduction=None):  # type: ignore
    output = np.copy(data)
    for i in np.ndindex(indices.shape[:-1]):
        print(f"updates for i={i}, indices={indices[i]}, updates={updates[i]}")
        if reduction == "add":
            output[tuple(indices[i])] += updates[i]
        elif reduction == "mul":
            output[tuple(indices[i])] *= updates[i]
        elif reduction == "max":
            output[tuple(indices[i])] = np.maximum(output[indices[i]], updates[i])
        elif reduction == "min":
            output[tuple(indices[i])] = np.minimum(output[indices[i]], updates[i])
        else:
            output[tuple(indices[i])] = updates[i]
    return output


class ScatterND(OpRun):
    def _run(self, data, indices, updates, reduction=None):  # type: ignore
        y = _scatter_nd_impl(data, indices, updates, reduction=reduction)
        return (y,)


ref = ReferenceEvaluator(model, new_ops=[ScatterND])


got = ref.run(None, feeds)[0]
print(got)
