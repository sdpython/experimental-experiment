import unittest
from typing import Optional
import numpy as np
from onnx import TensorProto, helper as oh, numpy_helper as onh
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xbuilder.graph_builder import GraphBuilder, OptimizationOptions
from experimental_experiment.xoptim.unfused import unfused_nodes

TFLOAT = TensorProto.FLOAT
TFLOAT16 = TensorProto.FLOAT16
TINT64 = TensorProto.INT64
_mkv_ = oh.make_tensor_value_info


class TestXoptimUnfused(ExtTestCase):
    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def _get_model_redln(self, dtype=None, axis=-1, fixed=True):
        itype = TFLOAT if dtype in (None, np.float32) else TensorProto.FLOAT16
        axis_ = [axis] if axis in (-1, 1) else [0, 1]
        return oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("ReduceMean", ["X", "axis"], ["mean"], keepdims=1),
                    oh.make_node("Sub", ["X", "mean"], ["xc"]),
                    oh.make_node("Pow", ["xc", "two"], ["x2"]),
                    oh.make_node("ReduceMean", ["x2", "axis"], ["mean2"], keepdims=1),
                    oh.make_node("Sqrt", ["mean2"], ["mean2s"]),
                    oh.make_node("Div", ["xc", "mean2s"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", itype, [2, 3] if fixed else ["a", "b"])],
                [_mkv_("Y", itype, [2, 3] if fixed else ["a", "b"])],
                [
                    onh.from_array(np.array(axis_, dtype=np.int64), name="axis"),
                    onh.from_array(np.array([2], dtype=np.float32).T, name="two"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 20)],
            ir_version=10,
        )

    def test_layer_normalization_simple(self):
        dtype = np.float32
        model = self._get_model_redln(dtype=dtype, axis=0, fixed=False)
        set_types0 = set(n.op_type for n in model.graph.node)
        self.assertEqual(set_types0, {"Pow", "Sub", "Sqrt", "Div", "ReduceMean"})

        feeds = {"X": (self._range(2, 3) + np.array([1, 2, 3])).astype(dtype)}

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["LayerNormalization"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        set_types = set(n.op_type for n in opt_onx.graph.node)
        self.assertEqual(set_types, {"LayerNormalization", "Shape", "ConstantOfShape"})

        ref1 = ExtendedReferenceEvaluator(model)
        refo = ExtendedReferenceEvaluator(opt_onx)
        self.assertEqualArray(ref1.run(None, feeds)[0], refo.run(None, feeds)[0], atol=1e-6)

        for node in opt_onx.graph.node:
            if node.op_type == "LayerNormalization":
                input_names = list(node.input)
                output_names = list(node.output)

        nodes = unfused_nodes(model, input_names, output_names)
        all_in, all_out = set(), set()
        for node in opt_onx.graph.node:
            all_in |= set(node.input)
            all_out |= set(node.output)
        self.assertSetContained(set(input_names), all_in)
        self.assertSetContained(set(output_names), all_out)
        set_types = set(n.op_type for n in nodes)
        self.assertNotIn("LayerNormalization", set_types)
        self.assertEqual(set_types, {"Div", "ReduceMean", "Sub", "Pow", "Sqrt"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
