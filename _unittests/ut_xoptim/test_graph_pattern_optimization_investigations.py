import os
import unittest
import numpy as np
import onnx
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.xoptim import get_pattern_list


class TestGraphPatternOptimizationInvestigation(ExtTestCase):
    def _range(self, *shape, bias: float = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def _get_model(self, name: str) -> onnx.ModelProto:
        p = os.path.join(os.path.dirname(__file__), "..", "ut_xbuilder", "data", name)
        if not os.path.exists(p):
            p = os.path.join(os.path.dirname(__file__), "data", name)
        self.assertExists(p)
        return onnx.load(p)

    def test_get_pattern_list(self):
        res = get_pattern_list("investigation")
        names = set(r.__class__.__name__ for r in res)
        self.assertIn("BinaryInvestigation", names)

    def test_binary_ops(self):
        model = self._get_model("noopt-llama-custom__1.onnx")

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["BinaryInvestigation"], verbose=1
            ),
        )
        opt_onx, out, err = self.capture(lambda: gr.to_onnx(optimize=True))
        self.assertIn("[BinaryInvestigation] Mul(2x2x1024x512, 1x1x1024x512)", out)
        self.assertGreater(len(model.graph.node), len(opt_onx.graph.node))

    def test_dump_applied_patterns(self):
        model = self._get_model("noopt-llama-custom__1.onnx")

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=["RotaryConcatPart"],
                dump_applied_patterns="test_dump_applied_patterns",
                verbose=10,
            ),
        )
        opt_onx, out, err = self.capture(lambda: gr.to_onnx(optimize=True))
        self.assertIn("save", out)
        assert os.path.exists(
            "test_dump_applied_patterns"
        ), "folder 'test_dump_applied_patterns' not found"
        assert os.listdir("test_dump_applied_patterns"), (
            f"No file found in 'test_dump_applied_patterns': "
            f"{os.listdir('test_dump_applied_patterns')}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
