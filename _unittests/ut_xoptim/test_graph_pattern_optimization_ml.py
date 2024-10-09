import unittest
import numpy as np
from onnx import TensorProto, helper as oh, numpy_helper as onh
from onnx.checker import check_model
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator

TARGET_OPSET = 18


class TestGraphPatternOptimizationMl(ExtTestCase):
    def test_tree_ensemble_regressor_mul(self):
        rule = "BRANCH_LEQ"

        targets = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        node1 = oh.make_node(
            "TreeEnsembleRegressor",
            ["X"],
            ["Ym"],
            domain="ai.onnx.ml",
            n_targets=1,
            aggregate_function="SUM",
            base_values=None,
            nodes_falsenodeids=[4, 3, 0, 0, 0, 2, 0, 4, 0, 0],
            nodes_featureids=[0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
            nodes_hitrates=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            nodes_missing_value_tracks_true=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            nodes_modes=[
                rule,
                rule,
                "LEAF",
                "LEAF",
                "LEAF",
                rule,
                "LEAF",
                rule,
                "LEAF",
                "LEAF",
            ],
            nodes_nodeids=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            nodes_treeids=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            nodes_truenodeids=[1, 2, 0, 0, 0, 1, 0, 3, 0, 0],
            nodes_values=[
                0.26645058393478394,
                0.6214364767074585,
                0.0,
                0.0,
                0.0,
                -0.7208403944969177,
                0.0,
                -0.5592705607414246,
                0.0,
                0.0,
            ],
            post_transform="NONE",
            target_ids=[0, 0, 0, 0, 0, 0],
            target_nodeids=[2, 3, 4, 1, 3, 4],
            target_treeids=[0, 0, 0, 1, 1, 1],
            target_weights=targets,
        )
        graph = oh.make_graph(
            [node1, oh.make_node("Mul", ["Ym", "cst"], ["Y"])],
            "ml",
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 3])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 1])],
            [onh.from_array(np.array([0.4], dtype=np.float32), name="cst")],
        )
        onx = oh.make_model_gen_version(
            graph,
            opset_imports=[
                oh.make_opsetid("", TARGET_OPSET),
                oh.make_opsetid("ai.onnx.ml", 4),
            ],
        )
        check_model(onx)

        gr = GraphBuilder(
            onx,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["TreeEnsembleRegressorMul"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["TreeEnsembleRegressor"], [n.op_type for n in opt_onx.graph.node])

        feeds = {"X": (np.arange(9).reshape((3, 3)) / 10).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(onx)
        expected = ref.run(None, feeds)[0]

        sess = ExtendedReferenceEvaluator(opt_onx)
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_tree_ensemble_regressor_mul_as_tensor(self):
        from onnxruntime import InferenceSession

        rule = "BRANCH_LEQ"

        targets = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        node1 = oh.make_node(
            "TreeEnsembleRegressor",
            ["X"],
            ["Ym"],
            domain="ai.onnx.ml",
            n_targets=1,
            aggregate_function="SUM",
            base_values=None,
            nodes_falsenodeids=[4, 3, 0, 0, 0, 2, 0, 4, 0, 0],
            nodes_featureids=[0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
            nodes_hitrates=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            nodes_missing_value_tracks_true=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            nodes_modes=[
                rule,
                rule,
                "LEAF",
                "LEAF",
                "LEAF",
                rule,
                "LEAF",
                rule,
                "LEAF",
                "LEAF",
            ],
            nodes_nodeids=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            nodes_treeids=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            nodes_truenodeids=[1, 2, 0, 0, 0, 1, 0, 3, 0, 0],
            nodes_values=[
                0.26645058393478394,
                0.6214364767074585,
                0.0,
                0.0,
                0.0,
                -0.7208403944969177,
                0.0,
                -0.5592705607414246,
                0.0,
                0.0,
            ],
            post_transform="NONE",
            target_ids=[0, 0, 0, 0, 0, 0],
            target_nodeids=[2, 3, 4, 1, 3, 4],
            target_treeids=[0, 0, 0, 1, 1, 1],
            target_weights_as_tensor=onh.from_array(
                np.array(targets, dtype=np.float32), name="target_weights_as_tensor"
            ),
        )
        graph = oh.make_graph(
            [node1, oh.make_node("Mul", ["Ym", "cst"], ["Y"])],
            "ml",
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 3])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 1])],
            [onh.from_array(np.array([0.4], dtype=np.float32), name="cst")],
        )
        onx = oh.make_model_gen_version(
            graph,
            opset_imports=[
                oh.make_opsetid("", TARGET_OPSET),
                oh.make_opsetid("ai.onnx.ml", 4),
            ],
        )
        check_model(onx)

        gr = GraphBuilder(
            onx,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["TreeEnsembleRegressorMul"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["TreeEnsembleRegressor"], [n.op_type for n in opt_onx.graph.node])

        feeds = {"X": (np.arange(9).reshape((3, 3)) / 10).astype(np.float32)}
        ref = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)[0]

        sess = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
