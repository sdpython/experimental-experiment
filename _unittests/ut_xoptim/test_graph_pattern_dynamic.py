import os
import pprint
import unittest
from typing import List, Optional
from onnx import ModelProto, TensorProto, load as load_onnx, save as save_onnx
from experimental_experiment.ext_test_case import ExtTestCase, hide_stdout
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

    @hide_stdout()
    def test_check_one(self):
        static_model = self._get_model("shape-dort-static-llama-custom__0.onnx")
        dynamic_model = self._get_model("shape-dort-dynamic-llama-custom__0.onnx")
        opts = OptimizationOptions(patterns="ReshapeMatMulReshape", verbose=10)
        gr1 = GraphBuilder(
            static_model, infer_shapes_options=True, optimization_options=opts, verbose=0
        )
        gr1.optimize()
        gr1 = GraphBuilder(
            dynamic_model, infer_shapes_options=True, optimization_options=opts, verbose=0
        )
        gr1.optimize()

    def test_graph_default_forward_single(self):
        static_model = self._get_model("shape-dort-static-llama-custom__0.onnx")
        dynamic_model = self._get_model("shape-dort-dynamic-llama-custom__0.onnx")
        patterns = get_default_patterns()
        self._check_models_patterns(
            static_model,
            dynamic_model,
            patterns,
            False,
            exceptions=[
                "ExpandPattern",
                "ReshapeMatMulReshapePattern",
                "Reshape2Of3Pattern",
                "MatMulReshape2Of3Pattern",
            ],
        )

    def test_graph_default_backward_single(self):
        static_model = self._get_model("shape-dort-static-llama-custom__1.onnx")
        dynamic_model = self._get_model("shape-dort-dynamic-llama-custom__1.onnx")
        patterns = get_default_patterns()
        self._check_models_patterns(
            static_model,
            dynamic_model,
            patterns,
            False,
            # Remaining cases if constrainsts are not handled properly
            exceptions=["Reshape2Of3Pattern", "MatMulReshape2Of3Pattern"],
        )

    def test_graph_default_forward_cumulative(self):
        static_model = self._get_model("shape-dort-static-llama-custom__0.onnx")
        dynamic_model = self._get_model("shape-dort-dynamic-llama-custom__0.onnx")
        patterns = get_default_patterns()
        self._check_models_patterns(
            static_model,
            dynamic_model,
            patterns,
            True,
            exceptions=[
                "ExpandPattern",
                "Reshape2Of3Pattern",
                "ReshapeMatMulReshapePattern",
            ],
        )

    def test_graph_default_backward_cumulative(self):
        static_model = self._get_model("shape-dort-static-llama-custom__1.onnx")
        dynamic_model = self._get_model("shape-dort-dynamic-llama-custom__1.onnx")
        patterns = get_default_patterns()
        self._check_models_patterns(
            static_model,
            dynamic_model,
            patterns,
            True,
            exceptions=[
                "SwitchOrderBinaryPattern",
                # needs to handle constraints for this
                "Reshape2Of3Pattern",
                "MatMulReshape2Of3Pattern",
                "TransposeReshapeMatMulPattern",
            ],
            name="test_graph_default_backward_cumulative",
        )

    def _check_models_patterns(
        self,
        model1: ModelProto,
        model2: ModelProto,
        patterns: List[PatternOptimization],
        cumulative: bool,
        exceptions: List[str],
        name: Optional[str] = None,
    ):
        avoid_patterns = {"apply_ShapeBasedExpandSwapPattern", "apply_ReshapePattern"}
        model1s = model1.SerializeToString()
        model2s = model2.SerializeToString()
        self.assertNotEqual(len(model1.graph.node), len(model2.graph.node))
        for i in range(len(patterns)):
            model1 = ModelProto()
            model1.ParseFromString(model1s)
            model2 = ModelProto()
            model2.ParseFromString(model2s)
            with self.subTest(i=i, cumulative=cumulative, pattern=patterns[i]):
                opts = OptimizationOptions(
                    patterns=patterns[: i + 1] if cumulative else patterns[i : i + 1],
                    verbose=0,
                )
                gr1 = GraphBuilder(
                    model1, infer_shapes_options=True, optimization_options=opts
                )
                gr1._check([], step="T1")
                stat1 = gr1.optimize()
                gr2 = GraphBuilder(
                    model2, infer_shapes_options=True, optimization_options=opts
                )
                gr2._check([], step="T2")
                stat2 = gr2.optimize()
                pat = patterns[i]
                prefix = f"apply_{pat.__class__.__name__}"
                app1 = [
                    s
                    for s in stat1
                    if s["pattern"].startswith(prefix) and s["pattern"] not in avoid_patterns
                ]
                app2 = [
                    s
                    for s in stat2
                    if s["pattern"].startswith(prefix) and s["pattern"] not in avoid_patterns
                ]
                if pat.__class__.__name__ in exceptions:
                    assert len(app1) > 0, f"Issue with pattern {patterns[i]} and app1={app1}"
                    continue
                if len(app1) > len(app2):
                    if name:
                        save_onnx(
                            gr2.to_onnx(optimize=False),
                            f"{name}_{pat.__class__.__name__}.onnx",
                        )
                    raise AssertionError(
                        f"Discrepancies static/dynamic: i={i}, "
                        f"len(app1)={len(app1)}, len(app2)={len(app2)}, "
                        f"pattern={patterns[i]}\n-----\n"
                        f"#applied static={len(app1)}, #applied dynamic={len(app2)}"
                        f"\n---------\n{pprint.pformat(app1)}"
                        f"\n--------\n{pprint.pformat(app2)}"
                        f"\n------------\n{gr2.get_debug_msg()}"
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
