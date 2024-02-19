import os
import unittest
import onnx
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_exp.graph_builder import GraphBuilder


class TestGraphPatternOptimization(ExtTestCase):
    def _get_model(self, name: str) -> onnx.ModelProto:
        p = os.path.join(os.path.dirname(__file__), "data", name)
        self.assertExists(p)
        return onnx.load(p)

    def test_unsqueeze_unsqueeze(self):
        origin = self._get_model("dort-c-custom__0.onnx")
        before = [node for node in origin.graph.node if node.op_type == "Unsqueeze"]
        gr = GraphBuilder(origin)
        res, out, err = self.capture(lambda: gr.optimize_with_patterns(verbose=10))
        self.assertEmpty(err)
        self.assertEmpty(res)
        self.assertIn("[GraphBuilderPatternOptimization.optimize] done after", out)
        self.assertIn("UnsqueezeUnsqueezePattern", out)

        onx = gr.to_onnx(optimize=False)
        after = [node for node in onx.graph.node if node.op_type == "Unsqueeze"]
        self.assertEqual(len(after), len(before) - 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
