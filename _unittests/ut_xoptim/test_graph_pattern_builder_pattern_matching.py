import os
import time
import unittest
from typing import List, Optional
import numpy as np
from onnx import NodeProto, TensorProto, load
from onnx.checker import check_model
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx.reference.op_run import OpRun
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xoptim import (
    EasyPatternOptimization,
    make_pattern_from_onnx,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.xoptim import GraphBuilderPatternOptimization

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

            def match_pattern(self, g: GraphBuilder, x: T, y: T, z: T):
                """
                Builds the pattern to match.
                """
                tmp = g.op.Add(x, y)
                return g.op.Add(tmp, z)

            def apply_pattern(self, g: GraphBuilder, x: T, y: T, z: T):
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

            def match_pattern(self, g: GraphBuilder, x: T, y: T, w: T, z: T):
                """
                Builds the pattern to match.
                """
                tmp = g.op.Add(x, y)
                tmp2 = g.op.Add(tmp, w)
                r1 = g.op.Add(tmp, z)
                return tmp2, r1

            def apply_pattern(self, g: GraphBuilder, x: T, y: T, w: T, z: T):
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

    def test_rotary_embedding(self):
        # The test work on a model if it has the expected name.
        # A dummy model is used if not present (not implemented yet).

        class RotaryEmbeddingPattern(EasyPatternOptimization):
            """
            Fusion for Rotary.
            """

            def match_pattern(
                self, g, x: "INT64", pos_ids: "FLOAT", axis: "INT64"  # noqa: F821
            ):
                # original code: the code does verifies the constant yet
                # unsqueeze = op.Unsqueeze(x, [1])
                op = g.op

                unsqueeze = op.Unsqueeze(x, axis)
                cast = op.Cast(unsqueeze, to=TensorProto.FLOAT)

                matmul = op.MatMul(pos_ids, cast)
                transpose = op.Transpose(matmul)
                output, length = g.anyop.ConcatTraining(
                    transpose,
                    transpose,
                    domain="com.microsoft",
                    outputs=2,
                )

                sin = op.Sin(output)
                cast1 = op.Cast(sin, to=TensorProto.FLOAT)
                cos = op.Cos(output)
                cast2 = op.Cast(cos, to=TensorProto.FLOAT)
                return cast1, cast2

            def validate_mapping(
                self,
                g,
                deleted_nodes,
                pattern_nodes,
            ) -> bool:
                # If some pattern needs to be rejected.
                return True

            def apply_pattern(
                self, g, x: "INT64", pos_ids: "FLOAT", axis: "INT64"  # noqa: F821
            ):
                op = g.op
                cos_cache = op.Constant(
                    value=onh.from_array(np.random.rand(256, 256).astype(np.float16))
                )
                sin_cache = op.Constant(
                    value=onh.from_array(np.random.rand(256, 256).astype(np.float16))
                )
                return g.anyop.RotaryEmbedding(
                    x,
                    pos_ids,
                    cos_cache,
                    sin_cache,
                    domain="com.microsoft",
                    outputs=2,
                )

        def do():
            rot = RotaryEmbeddingPattern()
            g = GraphBuilderPatternOptimization(
                GraphBuilder({"": 18, "com.microsoft": 1}, verbose=10), verbose=10
            )
            pat = rot._build_pattern(g, rot.match_pattern)
            onx = pat.builder.to_onnx(optimize=False)
            gr = GraphBuilder(
                onx,
                infer_shapes=False,
                optimization_options=OptimizationOptions(
                    patterns=[RotaryEmbeddingPattern(verbose=10)],
                    verbose=10,
                ),
            )
            opt_onx = gr.optimize()
            opt_onx = gr.to_onnx(optimize=False)
            return opt_onx

        opt_onx, out, _ = self.capture(do)
        self.assertIn("[RotaryEmbedding", out)

        expected = ["RotaryEmbedding"]
        self.assertEqual(expected, [n.op_type for n in opt_onx.graph.node])

    def test_rotary_embedding_same_axis(self):
        # The test work on a model if it has the expected name.
        # A dummy model is used if not present (not implemented yet).

        class RotaryEmbeddingPattern(EasyPatternOptimization):
            """
            Fusion for Rotary.
            """

            def match_pattern(
                self, g, x: "INT64", pos_ids: "FLOAT", axis: "INT64"  # noqa: F821
            ):
                # original code: the code does verifies the constant yet
                # unsqueeze = op.Unsqueeze(x, [1])
                op = g.op

                unsqueeze = op.Unsqueeze(x, axis)
                cast = op.Cast(unsqueeze, to=TensorProto.FLOAT)

                matmul = op.MatMul(pos_ids, cast)
                transpose = op.Transpose(matmul)
                output, length = g.anyop.ConcatTraining(
                    transpose,
                    transpose,
                    domain="com.microsoft",
                    outputs=2,
                )

                sin = op.Sin(output)
                cast1 = op.Cast(sin, to=TensorProto.FLOAT)
                cos = op.Cos(output)
                cast2 = op.Cast(cos, to=TensorProto.FLOAT)
                unsq1 = op.Unsqueeze(cast1, axis)
                unsq2 = op.Unsqueeze(cast2, axis)
                return unsq1, unsq2

            def validate_mapping(
                self,
                g,
                deleted_nodes,
                pattern_nodes,
            ) -> bool:
                # If some pattern needs to be rejected.
                return True

            def apply_pattern(
                self, g, x: "INT64", pos_ids: "FLOAT", axis: "INT64"  # noqa: F821
            ):
                op = g.op
                cos_cache = op.Constant(
                    value=onh.from_array(np.random.rand(256, 256).astype(np.float16))
                )
                sin_cache = op.Constant(
                    value=onh.from_array(np.random.rand(256, 256).astype(np.float16))
                )
                return g.anyop.RotaryEmbedding(
                    x,
                    pos_ids,
                    cos_cache,
                    sin_cache,
                    domain="com.microsoft",
                    outputs=2,
                )

        def do():
            rot = RotaryEmbeddingPattern()
            g = GraphBuilderPatternOptimization(
                GraphBuilder({"": 18, "com.microsoft": 1}, verbose=10), verbose=10
            )
            pat = rot._build_pattern(g, rot.match_pattern)
            onx = pat.builder.to_onnx(optimize=False)
            gr = GraphBuilder(
                onx,
                infer_shapes=False,
                optimization_options=OptimizationOptions(
                    patterns=[RotaryEmbeddingPattern(verbose=10)],
                    verbose=10,
                ),
            )
            opt_onx = gr.optimize()
            opt_onx = gr.to_onnx(optimize=False)
            return opt_onx

        opt_onx, out, _ = self.capture(do)
        self.assertIn("[RotaryEmbedding", out)

        expected = ["RotaryEmbedding"]
        self.assertEqual(expected, [n.op_type for n in opt_onx.graph.node])

    def test_rotary_emb_file(self):
        # The test work on a model if it has the expected name.
        # A dummy model is used if not present (not implemented yet).
        import torch

        class RotaryEmbeddingPattern(EasyPatternOptimization):
            """
            Fusion for Rotary.
            """

            def match_pattern(self, g, x, pos_ids, axis):
                # original code: the code does verifies the constant yet
                # unsqueeze = op.Unsqueeze(x, [1])
                op = g.op

                unsqueeze = op.Unsqueeze(x, axis)
                cast = op.Cast(unsqueeze, to=TensorProto.FLOAT)

                matmul = op.MatMul(pos_ids, cast)
                transpose = op.Transpose(matmul)
                output, length = g.anyop.ConcatTraining(
                    transpose, transpose, domain="com.microsoft", outputs=2
                )

                sin = op.Sin(output)
                cast1 = op.Cast(sin, to=TensorProto.FLOAT)
                cos = op.Cos(output)
                cast2 = op.Cast(cos, to=TensorProto.FLOAT)
                unsq1 = op.Unsqueeze(cast1, axis)
                unsq2 = op.Unsqueeze(cast2, axis)
                return unsq1, unsq2

            def validate_mapping(
                self,
                g,
                deleted_nodes,
                pattern_nodes,
            ) -> bool:
                # If some pattern needs to be rejected.
                return True

            def apply_pattern(self, g, x, pos_ids, axis):
                op = g.op
                cos_cache = torch.randn(256, 256).to(torch.float16)
                sin_cache = torch.randn(256, 256).to(torch.float16)
                return op.RotaryEmbedding(
                    x, pos_ids, cos_cache, sin_cache, domain="com.microsoft", outputs=2
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
                patterns=[RotaryEmbeddingPattern(verbose=0)],
                remove_identity=False,
                verbose=0,
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
        op_types = set(node.op_type for node in opt_onx.graph.node)
        self.assertIn("RotaryEmbedding", op_types)

    def test_graph_pattern_builder_onnx(self):

        class AddAdd(OpRun):
            op_domain = "ZZZ"

            def _run(self, x, y, z):
                return (x + y + z,)

        model_match = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy"]),
                    oh.make_node("Add", ["xy", "Z"], ["F"]),
                ],
                "match",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [None]),
                    oh.make_tensor_value_info("Y", TFLOAT, [None]),
                    oh.make_tensor_value_info("Z", TFLOAT, [None]),
                ],
                [oh.make_tensor_value_info("F", TFLOAT, [None])],
            )
        )

        model_apply = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("AddAdd", ["X", "Y", "Z"], ["F"], domain="ZZZ"),
                ],
                "apply",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [None]),
                    oh.make_tensor_value_info("Y", TFLOAT, [None]),
                    oh.make_tensor_value_info("Z", TFLOAT, [None]),
                ],
                [oh.make_tensor_value_info("F", TFLOAT, [None])],
            )
        )

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
                patterns=[
                    make_pattern_from_onnx(
                        "AddAddPattern", model_match, model_apply, verbose=0
                    )
                ],
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

    def test_validate_mapping(self):

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
                [
                    oh.make_tensor_value_info(
                        "final", TensorProto.FLOAT, ["a", "b", "c"]
                    )
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        class MulMulSigmoidPattern(EasyPatternOptimization):

            def match_pattern(self, g: GraphBuilder, X, Y):
                return g.op.Mul(X, g.op.Mul(Y, g.op.Sigmoid(Y)))

            def apply_pattern(self, g: GraphBuilder, X, Y):
                return g.anyop.MulMulSigmoid(
                    X, Y, domain="onnx_extended.ortops.optim.cuda"
                )

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
                            return False
                return True

        gr = GraphBuilder(
            proto,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=[MulMulSigmoidPattern(verbose=1)],
                verbose=0,
            ),
        )

        new_proto = gr.to_onnx()
        self.assertEqual(len(new_proto.graph.node), len(proto.graph.node))

    def test_graph_pattern_builder_multi_outputs_slice(self):

        class SliceSplitPattern(EasyPatternOptimization):

            def match_pattern(
                self,
                g: GraphBuilder,
                x: "FLOAT",  # noqa: F821
                b0: "INT64",  # noqa: F821
                e0: "INT64",  # noqa: F821
                a0: "INT64",  # noqa: F821
                b1: "INT64",  # noqa: F821
                e1: "INT64",  # noqa: F821
                a1: "INT64",  # noqa: F821
            ):
                return g.op.Slice(x, b0, e0, a0), g.op.Slice(x, b1, e1, a1)

            def apply_pattern(self, g: GraphBuilder, x: T, b0, e0, a0, b1, e1, a1):
                return g.op.Split(x, axis=-1, num_outputs=2)

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Slice", ["X", "zero", "half", "axis"], ["spl1"]),
                    oh.make_node("Slice", ["X", "half", "last", "axis"], ["spl2"]),
                ],
                "name",
                [oh.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4, 6])],
                [
                    oh.make_tensor_value_info("spl1", TensorProto.FLOAT, [3, 2, 3]),
                    oh.make_tensor_value_info("spl2", TensorProto.FLOAT, [3, 2, 3]),
                ],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([3], dtype=np.int64), name="half"),
                    onh.from_array(np.array([6], dtype=np.int64), name="last"),
                    onh.from_array(np.array([2], dtype=np.int64), name="axis"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )

        check_model(model)
        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=[SliceSplitPattern(verbose=0)],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Split"], [n.op_type for n in opt_onx.graph.node])

        feeds = {"X": self._range(3, 2, 6) * (3 * 2 * 6)}
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        self.assertEqual(0, len(opt_onx.graph.initializer))
        check_model(opt_onx)
        self.assertEqual([], [i.name for i in opt_onx.graph.initializer])
        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        self.assertEqual(len(expected), len(got))
        self.assertEqualArray(expected[0], got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
