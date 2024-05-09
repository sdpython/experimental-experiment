import os
import unittest
from typing import List
from onnx import ModelProto, TensorProto, load as load_onnx
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xoptim import PatternOptimization
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.xoptim.patterns import get_default_patterns

T = str
TFLOAT = TensorProto.FLOAT


class TestGraphPatternDynamic(ExtTestCase):

    def _get_model(self, name: str, skip=False) -> ModelProto:
        if os.path.exists(name):
            return load_onnx(name)
        p = os.path.join(os.path.dirname(__file__), "..", "ut_xbuilder", "data", name)
        if not os.path.exists(p):
            p = os.path.join(os.path.dirname(__file__), "data", name)
        if skip and not os.path.exists(p):
            raise unittest.SkipTest(f"Unable to find {p!r}.")
        self.assertExists(p)
        return load_onnx(p)

    def test_graph_default_forward(self):
        static_model = self._get_model("shape-dort-static-llama-custom__0.onnx")
        dynamic_model = self._get_model("shape-dort-dynamic-llama-custom__0.onnx")
        patterns = get_default_patterns()
        self._check_models_patterns(static_model, dynamic_model, patterns)

    def _check_models_patterns(
        self,
        model1: ModelProto,
        model2: ModelProto,
        patterns: List[PatternOptimization],
    ):
        for i in range(len(patterns)):
            opts = OptimizationOptions(patterns=patterns[: i + 1], verbose=0)
            gr1 = GraphBuilder(model1, infer_shapes=True, optimization_options=opts)
            stat1 = gr1.optimize()
            gr2 = GraphBuilder(model2, infer_shapes=True, optimization_options=opts)
            stat2 = gr2.optimize()
            print(stat1)
            print(stat2)
            self.assertEqual(stat1, stat2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
