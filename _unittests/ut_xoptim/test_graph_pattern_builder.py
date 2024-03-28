import os
import time
import unittest
import numpy as np
from onnx import TensorProto, load
from onnx.checker import check_model
import onnx.helper as oh
from onnx.reference.op_run import OpRun
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xoptim import EasyPatternOptimization
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)

T = str
TFLOAT = TensorProto.FLOAT


class TestGraphPatternBuilder(ExtTestCase):

    def _range(self, *shape, bias: float = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def test_graph_pattern_builder(self):

        class AddAddPattern(EasyPatternOptimization):
            """
            Replaces ConstantOfShape + ScatterND with ScatterNDOfShape (com.domain).
            """

            @classmethod
            def match_pattern(self, g: "GraphBuilder", x: T, y: T, z: T):  # noqa: F821
                """
                Builds the pattern to match.
                """
                tmp = g.op.Add(x, y)
                return g.op.Add(tmp, z)

            @classmethod
            def apply_pattern(cls, g: "GraphBuilder", x: T, y: T, z: T):  # noqa: F821
                """
                Builds the pattern to match.
                """
                return g.anyop.AddAdd(x, y, z, domain="ZZZ")

        class AddAdd(OpRun):
            op_domain = "ZZZ"

            def _run(self, x, y, z):
                return (x + y + z,)

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["x", "y"], ["gggg"]),
                    oh.make_node("Add", ["gggg", "z"], ["final"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("x", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("y", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("z", TFLOAT, [None, None]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        check_model(model)
        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=[AddAddPattern(verbose=0)],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["AddAdd"],
            [n.op_type for n in opt_onx.graph.node],
        )

        feeds = {
            "x": self._range(5, 6),
            "y": self._range(5, 6),
            "z": self._range(5, 6),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        self.assertEqual(0, len(opt_onx.graph.initializer))
        check_model(opt_onx)
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("ZZZ", opsets)
        self.assertEqual(opsets["ZZZ"], 1)

        ref2 = ExtendedReferenceEvaluator(opt_onx, new_ops=[AddAdd])
        got = ref2.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_graph_pattern_builder_multi_outputs(self):

        class AddAddAddAddPattern(EasyPatternOptimization):
            """
            Replaces ConstantOfShape + ScatterND with ScatterNDOfShape (com.domain).
            """

            @classmethod
            def match_pattern(
                self, g: "GraphBuilder", x: T, y: T, w: T, z: T
            ):  # noqa: F821
                """
                Builds the pattern to match.
                """
                tmp = g.op.Add(x, y)
                tmp2 = g.op.Add(tmp, w)
                r1 = g.op.Add(tmp, z)
                return tmp2, r1

            @classmethod
            def apply_pattern(
                cls, g: "GraphBuilder", x: T, y: T, w: T, z: T
            ):  # noqa: F821
                """
                Builds the pattern to match.
                """
                return g.anyop.AddAddAddAdd(x, y, w, z, domain="ZZZ", outputs=2)

        class AddAddAddAdd(OpRun):
            op_domain = "ZZZ"

            def _run(self, x, y, w, z):
                return (x + y + w, x + y + z)

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["x", "y"], ["gggg"]),
                    oh.make_node("Add", ["gggg", "w"], ["f1"]),
                    oh.make_node("Add", ["gggg", "z"], ["f2"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("x", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("y", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("z", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("w", TFLOAT, [None, None]),
                ],
                [
                    oh.make_tensor_value_info("f1", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("f2", TFLOAT, [None, None]),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        check_model(model)
        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=[AddAddAddAddPattern(verbose=0)],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["AddAddAddAdd"],
            [n.op_type for n in opt_onx.graph.node],
        )

        feeds = {
            "x": self._range(5, 6),
            "y": self._range(5, 6),
            "w": self._range(5, 6),
            "z": self._range(5, 6),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        self.assertEqual(0, len(opt_onx.graph.initializer))
        check_model(opt_onx)
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("ZZZ", opsets)
        self.assertEqual(opsets["ZZZ"], 1)

        ref2 = ExtendedReferenceEvaluator(opt_onx, new_ops=[AddAddAddAdd])
        got = ref2.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_rotary_emb(self):
        # The test work on a model if it has the expected name.
        # A dummy model is used if not present (not implemented yet).
        import torch

        class RotaryEmbeddingPattern(EasyPatternOptimization):
            """
            Fusion for Rotary.
            """

            @classmethod
            def match_pattern(cls, g, x, pos_ids, axis):
                # original code: the code does verifies the constant yet
                # unsqueeze = op.Unsqueeze(x, [1])
                op = g.op

                unsqueeze = op.Unsqueeze(x, axis)
                cast = op.Cast(unsqueeze, to=TensorProto.FLOAT)

                matmul = op.MatMul(pos_ids, cast)
                transpose = op.Transpose(matmul)
                concattraining = g.anyop.ConcatTraining(
                    transpose, transpose, domain="com.microsoft"
                )

                sin = op.Sin(concattraining)
                cast1 = op.Cast(sin, to=TensorProto.FLOAT)
                cos = op.Cos(concattraining)
                cast2 = op.Cast(cos, to=TensorProto.FLOAT)
                return cast1, cast2

            @classmethod
            def validate_mapping(
                cls,
                g,
                deleted_nodes,
                added_nodes,
            ) -> bool:
                # If some pattern needs to be rejected.
                return True

            @classmethod
            def apply_pattern(cls, g, x, pos_ids, axis):
                op = g.op
                cos_cache = torch.randn(256, 256).to(torch.float16)
                sin_cache = torch.randn(256, 256).to(torch.float16)
                return op.RotaryEmbedding(
                    x, pos_ids, cos_cache, sin_cache, domain="com.microsoft"
                )

        model = "gemma_optimized_pre_grad_training_2.onnx"
        if not os.path.exists(model):
            raise unittest.SkipTest(f"{model!r} is missing")

        begin = time.perf_counter()
        onx = load(model)
        gr = GraphBuilder(
            onx,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=[RotaryEmbeddingPattern(verbose=10)],
                verbose=10,
            ),
        )
        if __name__ == "__main__":
            print(f"Loading done in {time.perf_counter() - begin}s")

        # from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
        # print(onnx_simple_text_plot(onx))

        opt_onx = gr.to_onnx(optimize=True)

        begin = time.perf_counter()
        opt_onx = gr.optimize()
        if __name__ == "__main__":
            print(f"Matching done in {time.perf_counter() - begin}s")

        begin = time.perf_counter()
        opt_onx = gr.to_onnx(optimize=False)
        if __name__ == "__main__":
            print(f"Building done in {time.perf_counter() - begin}s")

        begin = time.perf_counter()
        buffer = opt_onx.SerializeToString()
        with open(f"{model}.opt.onnx", "wb") as f:
            f.write(buffer)
        if __name__ == "__main__":
            print(f"Saving done in {time.perf_counter() - begin}s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
