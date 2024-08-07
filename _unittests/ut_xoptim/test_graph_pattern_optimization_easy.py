"""
Use:

::

    LOG_PATTERN_OPTIMIZE=10 python _unittests/ut_xoptim/test_graph_pattern_optimization.py -k test_rotary_concat_part_plug 
"""

import unittest
import numpy as np
import onnx
from onnx import (
    TensorProto,
    helper as oh,
    numpy_helper as onh,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)


TFLOAT = TensorProto.FLOAT
TFLOAT16 = TensorProto.FLOAT16
TINT64 = TensorProto.INT64


class TestGraphPatternOptimizationEasy(ExtTestCase):

    def _get_random_inputs(self, model: onnx.ModelProto):
        feeds = {}
        for i in model.graph.input:
            ish = tuple(i.type.tensor_type.shape.dim)
            # Creates an input tensor with a dimension defined by the onnx model
            # or equals to i + 2 with i being the dimension index.
            # The tensor is kept small to make the test fast.
            shape = tuple(
                (d.dim_value if d.dim_value > 0 else i + 2) for i, d in enumerate(ish)
            )
            if i.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
                feeds[i.name] = np.random.randn(*shape).astype(np.float32)
            elif i.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
                feeds[i.name] = np.random.randn(*shape).astype(np.float16)
            elif i.type.tensor_type.elem_type == onnx.TensorProto.INT64:
                feeds[i.name] = np.arange(0, shape[0]).astype(np.int64)
            else:
                raise AssertionError(f"Not implemented for input {i}")
        return feeds

    def _check_model(
        self,
        model: onnx.ModelProto,
        optimized_model: onnx.ModelProto,
        feeds=None,
        atol: float = 0.0,
        rtol: float = 1e-7,
        use_ort: bool = False,
    ):
        # import onnxruntime does not because of the subfolder named onnxruntime
        import onnxruntime

        if not feeds:
            feeds = self._get_random_inputs(model)

        if use_ort:
            cls = lambda onx: onnxruntime.InferenceSession(  # noqa: E731
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        else:
            cls = lambda onx: ExtendedReferenceEvaluator(onx, verbose=0)  # noqa: E731
        ref = cls(model)
        opt = cls(optimized_model)
        expected = ref.run(None, feeds)
        got = opt.run(None, feeds)
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)

    def test_softmax_cross_entropy_loss_cast(self):
        models = [
            oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node("Equal", ["I", "B"], ["eq1"]),
                        oh.make_node("Not", ["eq1"], ["neq1"]),
                        oh.make_node("Where", ["neq1", "I", "zeroi"], ["ind"]),
                        oh.make_node("Unsqueeze", ["ind", "one"], ["flat_ind"]),
                        oh.make_node("LogSoftmax", ["X"], ["logX"], axis=1),
                        oh.make_node(
                            "GatherElements", ["logX", "flat_ind"], ["gx"], axis=1
                        ),
                        oh.make_node("Squeeze", ["gx", "one"], ["flat_gx"]),
                        oh.make_node("Neg", ["flat_gx"], ["neg_gx"]),
                        oh.make_node("Where", ["neq1", "neg_gx", "zerof"], ["w2"]),
                        oh.make_node("Cast", ["w2"], ["w2f"], to=TFLOAT),
                        oh.make_node("Cast", ["neq1"], ["neq1f"], to=TFLOAT),
                        oh.make_node(
                            "ReduceSum",
                            ["w2f"],
                            ["red1"],
                            keepdims=0,
                            noop_with_empty_axes=0,
                        ),
                        oh.make_node(
                            "ReduceSum",
                            ["neq1f"],
                            ["red2"],
                            keepdims=0,
                            noop_with_empty_axes=0,
                        ),
                        oh.make_node("Cast", ["red1"], ["red1_16"], to=TFLOAT16),
                        oh.make_node("Cast", ["red2"], ["red2_16"], to=TFLOAT16),
                        oh.make_node("Div", ["red1_16", "red2_16"], ["Y"]),
                    ],
                    "name",
                    [
                        oh.make_tensor_value_info("X", TFLOAT16, ["A", "B"]),
                        oh.make_tensor_value_info("I", TINT64, ["C"]),
                    ],
                    [oh.make_tensor_value_info("Y", TFLOAT16, [])],
                    [
                        onh.from_array(np.array([-100], dtype=np.int64), name="B"),
                        onh.from_array(np.array([1], dtype=np.int64), name="one"),
                        onh.from_array(np.array([0], dtype=np.float16), name="zerof"),
                        onh.from_array(np.array([0], dtype=np.int64), name="zeroi"),
                    ],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
            ),
        ]

        for model in models:
            gr = GraphBuilder(
                model,
                infer_shapes=True,
                optimization_options=OptimizationOptions(
                    patterns=["SoftmaxCrossEntropyLossCast"], verbose=0
                ),
            )
            opt_onx = gr.to_onnx(optimize=True)
            self.assertIn(
                "SoftmaxCrossEntropyLoss", set(n.op_type for n in opt_onx.graph.node)
            )
            self.assertEqual(0, len(opt_onx.graph.initializer))
            self._check_model(model, opt_onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
