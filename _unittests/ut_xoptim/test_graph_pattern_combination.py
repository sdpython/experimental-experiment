import os
import unittest
import numpy as np
from onnx import (
    ModelProto,
    TensorProto,
    helper as oh,
    numpy_helper as onh,
    load as load_onnx,
)
from onnx.checker import check_model
from onnx.shape_inference import infer_shapes
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_onnxruntime_training,
)
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)

TFLOAT = TensorProto.FLOAT


class TestGraphPatternCombination(ExtTestCase):

    @classmethod
    def setUpClass(cls):
        for name in [
            "dump_bug_test_pattern_combination_false.onnx",
            "dump_bug_test_pattern_combination.onnx",
        ]:
            if os.path.exists(name):
                os.remove(name)

    def _fix_shape(self, onx):
        skip_names = set()
        for node in onx.graph.node:
            if node.op_type in {"SequenceConstruct", "SequenceAt"}:
                skip_names |= set(node.output)

        new_shapes = []
        for sh in onx.graph.value_info:
            if sh.name in skip_names:
                continue
            if sh.type.tensor_type.elem_type != 0:
                new_shapes.append(sh)
        del onx.graph.value_info[:]
        onx.graph.value_info.extend(new_shapes)

    def _check_ort_cpu_or_cuda(self, onx):

        def cl(text):
            return (
                text.replace("\n", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
            )

        def s(cond):
            if not cond:
                with open("dump_bug_test_pattern_combination_false.onnx", "wb") as f:
                    f.write(onx.SerializeToString())
            return cond

        for i in onx.graph.input:
            assert s(i.type.tensor_type.elem_type != 0), f"Input {i.name!r} has no type"
        for i in onx.graph.output:
            assert s(
                i.type.tensor_type.elem_type != 0
            ), f"Output {i.name!r} has no type"

        skip_names = set()
        for node in onx.graph.node:
            if node.op_type in {"SequenceConstruct", "SequenceAt"}:
                skip_names |= set(node.output)

        for sh in onx.graph.value_info:
            if sh.name in skip_names:
                continue
            assert s(
                sh.type.tensor_type.elem_type != 0
            ), f"Result {sh.name!r} has no type"

        # check_model(onx)
        infer_shapes(onx)

        import onnxruntime
        from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument

        opsets = {d.domain: d.version for d in onx.opset_import}
        options = onnxruntime.SessionOptions()
        providers = ["CPUExecutionProvider"]
        if "onnx_extended.ortops.optim.cuda" in opsets:
            try:
                from onnx_extended.ortops.optim.cuda import get_ort_ext_libs
            except ImportError:
                raise unittest.SkipTest("onnx_extended not installed.")

            options.register_custom_ops_library(get_ort_ext_libs()[0])
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "onnx_extended.ortops.optim.cpu" in opsets:
            try:
                from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
            except ImportError:
                raise unittest.SkipTest("onnx_extended not installed.")

            options.register_custom_ops_library(get_ort_ext_libs()[0])

        try:
            onnxruntime.InferenceSession(
                onx.SerializeToString(), options, providers=providers
            )
        except (Fail, InvalidArgument) as e:
            err = []
            rows = []
            for i in onx.graph.input:
                rows.append(f"input-: {i.name!r} {cl(str(i.type))}")
                if i.type.tensor_type.elem_type == 0:
                    err.append(f"ERR:input-: {i.name!r} {cl(str(i.type))}")
            for i in onx.graph.output:
                rows.append(f"output: {i.name!r} {cl(str(i.type))}")
                if i.type.tensor_type.elem_type == 0:
                    err.append(f"ERR:output: {i.name!r} {cl(str(i.type))}")
            for i in onx.graph.value_info:
                rows.append(f"shape-: {i.name!r} {cl(str(i.type))}")
                if i.type.tensor_type.elem_type == 0:
                    err.append(f"ERR:shape-: {i.name!r} {cl(str(i.type))}")
            msg = "\n".join(err + rows)

            with open("dump_bug_test_pattern_combination.onnx", "wb") as f:
                f.write(onx.SerializeToString())
            raise AssertionError(msg) from e

    def _get_model(self, name: str) -> ModelProto:
        p = os.path.join(os.path.dirname(__file__), "..", "ut_xbuilder", "data", name)
        if not os.path.exists(p):
            p = os.path.join(os.path.dirname(__file__), "data", name)
        self.assertExists(p)
        return load_onnx(p)

    def _range(self, *shape, bias: float = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def test_reshape_matmul_reshape_static(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [32, 128]),
                    oh.make_tensor_value_info("Y", TFLOAT, [3, 5, 128, 64]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 32, 64])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(
                        np.array([1, 32, 128], dtype=np.int64), name="shape1"
                    ),
                    onh.from_array(
                        np.array([15, 128, 64], dtype=np.int64), name="shape2"
                    ),
                    onh.from_array(
                        np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"
                    ),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=[
                    "Cast",
                    "ReshapeMatMulReshape",
                    "UnsqueezeUnsqueeze",
                    "MatMulReshape2Of3",
                    "ReshapeReshape",
                ],
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reshape_matmul_reshape_dynamic_1(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["D32", "D128"]),
                    oh.make_tensor_value_info(
                        "Y", TFLOAT, ["batch", "channel", "D128", "D64"]
                    ),
                ],
                [
                    oh.make_tensor_value_info(
                        "Z", TFLOAT, ["batch", "channel", "D32", "64"]
                    )
                ],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(
                        np.array([1, 32, 128], dtype=np.int64), name="shape1"
                    ),
                    onh.from_array(
                        np.array([15, 128, 64], dtype=np.int64), name="shape2"
                    ),
                    onh.from_array(
                        np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"
                    ),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=[
                    "Cast",
                    "ReshapeMatMulReshape",
                    "UnsqueezeUnsqueeze",
                    "MatMulReshape2Of3",
                    "ReshapeReshape",
                ],
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reshape_matmul_reshape_dynamic_2(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["D32", "D128"]),
                    oh.make_tensor_value_info(
                        "Y", TFLOAT, ["batch", "channel", "D128", "D64"]
                    ),
                ],
                [
                    oh.make_tensor_value_info(
                        "Z", TFLOAT, ["batch", "channel", "D32", "64"]
                    )
                ],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(
                        np.array([1, 32, 128], dtype=np.int64), name="shape1"
                    ),
                    onh.from_array(
                        np.array([15, 128, 64], dtype=np.int64), name="shape2"
                    ),
                    onh.from_array(
                        np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"
                    ),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            optimization_options=OptimizationOptions(
                patterns=[
                    "Cast",
                    "ReshapeMatMulReshape",
                    "UnsqueezeUnsqueeze",
                    "MatMulReshape2Of3",
                    "ReshapeReshape",
                ]
            ),
            infer_shapes=True,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_reshape_matmul_reshape_keep_intermediate(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [32, 128]),
                    oh.make_tensor_value_info("Y", TFLOAT, [3, 5, 128, 64]),
                ],
                [
                    oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 32, 64]),
                    oh.make_tensor_value_info("xm1", TFLOAT, [1, 32, 128]),
                ],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(
                        np.array([1, 32, 128], dtype=np.int64), name="shape1"
                    ),
                    onh.from_array(
                        np.array([15, 128, 64], dtype=np.int64), name="shape2"
                    ),
                    onh.from_array(
                        np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"
                    ),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            optimization_options=OptimizationOptions(
                patterns=[
                    "Cast",
                    "ReshapeMatMulReshape",
                    "UnsqueezeUnsqueeze",
                    "MatMulReshape2Of3",
                    "ReshapeReshape",
                ],
                verbose=10,
            ),
            infer_shapes=True,
        )
        s = str(gr.optimization_options)
        self.assertIn("OptimizationOptions(", s)
        self.assertIn("CastPattern", s)
        opt_onx, out, _ = self.capture(lambda: gr.to_onnx(optimize=True))
        self.assertIn("remove_initializer:shape2", out)
        self.assertEqual(
            ["Unsqueeze", "Reshape", "MatMul"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(2, len(opt_onx.graph.initializer))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_simplified_only(self):
        for model in ["noopt-llama-custom__0.onnx"]:
            onx = self._get_model(model)
            with self.subTest(model=model):
                gr = GraphBuilder(
                    onx,
                    optimization_options=OptimizationOptions(
                        patterns=["SimplifiedLayerNormalization"],
                        verbose=0,
                    ),
                    infer_shapes=True,
                )
                onx = gr.to_onnx(optimize=True)
                types = set([n.op_type for n in onx.graph.node])
                self.assertIn("SimplifiedLayerNormalization", types)
                self._check_ort_cpu_or_cuda(onx)

    @requires_onnxruntime_training()
    def test_simplified_all_but_transpose_reshape_matmul(self):
        self._simplified_with_all({"TransposeReshapeMatMulPattern"})

    @requires_onnxruntime_training()
    def test_simplified_all_but_sub1_mul(self):
        self._simplified_with_all({"Sub1MulPattern"})

    @requires_onnxruntime_training()
    def test_simplified_with_all(self):
        self._simplified_with_all({})

    def _simplified_with_all(self, disabled):
        for model in [
            "noopt-llama-custom__1.onnx",
            "noopt-llama-custom__0.onnx",
            "noopt-phi-custom__0.onnx",
            "noopt-phi-custom__1.onnx",
            "llama_forward.onnx",
            "llama_forward.onnx",
            "opt-llama-custom-forward.onnx",
            "dort-llama-llama-ort_1.onnx",
            "dort-llama2-llama-ort+_1.onnx",
            "opt-llama-custom-backward.onnx",
        ]:
            options = OptimizationOptions(
                patterns="default+onnxruntime",
                verbose=0,
                verifies=False,
            )
            options.patterns = [
                p for p in options.patterns if p.__class__.__name__ not in disabled
            ]
            onx = self._get_model(model)
            self._fix_shape(onx)
            self._check_ort_cpu_or_cuda(onx)
            with self.subTest(model=model):
                gr = GraphBuilder(
                    onx,
                    optimization_options=options,
                    infer_shapes=True,
                )
                onx = gr.to_onnx(optimize=True)
                self._check_ort_cpu_or_cuda(onx)

    def test_study(self):
        # model = "dort-llama-llama-ort_1.onnx"
        model = "dort-model-llama-custom__1.onnx"
        enabled = {
            "AddMulPattern",
        }
        enabled = {}
        disabled = {}
        options = OptimizationOptions(
            patterns="default+onnxruntime+experimental",
            verbose=0,
            verifies=False,
            dump_applied_patterns="dump_applied_pattern",
            processor="CPU,CUDA",
        )
        options.patterns = [
            p
            for p in options.patterns
            if (not enabled or p.__class__.__name__ in enabled)
            and p.__class__.__name__ not in disabled
        ]
        assert options.patterns, "Pattern is empty."
        if __name__ == "__main__":
            options.verbose = 1 if len(options.patterns) > 2 else 10
            print(f"-- patterns={[c.__class__.__name__ for c in options.patterns]}")
            print(f"-- verbose={options.verbose}")
        for p in options.patterns:
            p.verbose = options.verbose
        onx = self._get_model(model)
        self._fix_shape(onx)
        gr = GraphBuilder(
            onx,
            optimization_options=options,
            infer_shapes=True,
        )
        new_onx = gr.to_onnx(optimize=True)
        with open(f"test_study_{model}", "wb") as f:
            f.write(onx.SerializeToString())

        from onnxruntime.capi.onnxruntime_pybind11_state import NotImplemented

        try:
            self._check_ort_cpu_or_cuda(onx)
        except NotImplemented as e:
            raise unittest.SkipTest(f"missing extension: {e}")
        self._check_ort_cpu_or_cuda(new_onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
