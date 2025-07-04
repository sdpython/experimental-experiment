"""
::

    clear&&python _unittests/ut_xoptim/test_graph_pattern_combination.py -k study
"""

import os
import unittest
import sys
from typing import Optional
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
from onnx.onnx_cpp2py_export.shape_inference import InferenceError
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    requires_onnxruntime_training,
    has_onnxruntime_training,
)
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)

TFLOAT = TensorProto.FLOAT


def cuda_recent_enough():
    import torch

    try:
        v = torch.version.cuda
    except ImportError:
        return False
    return v != "11.8"


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

    def _check_ort_cpu_or_cuda(self, onx, model=None):
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
            assert s(
                i.type.tensor_type.elem_type != 0
            ), f"Model={model!r}, Input {i.name!r} has no type"
        for i in onx.graph.output:
            assert s(
                i.type.tensor_type.elem_type != 0
            ), f"Model={model!r}, Output {i.name!r} has no type"

        skip_names = set()
        for node in onx.graph.node:
            if node.op_type in {"SequenceConstruct", "SequenceAt"}:
                skip_names |= set(node.output)

        for sh in onx.graph.value_info:
            if sh.name in skip_names:
                continue
            assert s(
                sh.type.tensor_type.elem_type != 0
            ), f"Model={model!r}, Result {sh.name!r} has no type"

        # check_model(onx)
        try:
            infer_shapes(onx)
        except InferenceError as e:
            if "Cannot infer type and shape for node name" not in str(e):
                raise

        import onnxruntime
        from onnxruntime.capi.onnxruntime_pybind11_state import (
            Fail,
            InvalidArgument,
            RuntimeException,
        )

        opsets = {d.domain: d.version for d in onx.opset_import}
        options = onnxruntime.SessionOptions()
        providers = ["CPUExecutionProvider"]
        if "onnx_extended.ortops.optim.cuda" in opsets:
            try:
                from onnx_extended.ortops.optim.cuda import get_ort_ext_libs
            except ImportError:
                raise unittest.SkipTest("onnx_extended not installed.")  # noqa: B904

            options.register_custom_ops_library(get_ort_ext_libs()[0])
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "onnx_extended.ortops.optim.cpu" in opsets:
            try:
                from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
            except ImportError:
                raise unittest.SkipTest("onnx_extended not installed.")  # noqa: B904

            options.register_custom_ops_library(get_ort_ext_libs()[0])

        try:
            onnxruntime.InferenceSession(onx.SerializeToString(), options, providers=providers)
        except (Fail, InvalidArgument, RuntimeException) as e:
            if "com.microsoft:SoftmaxGrad(-1) is not a registered" in str(e):
                raise unittest.SkipTest(  # noqa: B904
                    f"onnxruntime-training is needed due to {e}"
                )
            if "No such file or directory" in str(e):
                raise unittest.SkipTest(  # noqa: B904
                    f"onnxruntime-training is needed due to {e}"
                )
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
            raise AssertionError(f"Model={model!r}\n{msg}") from e

    def _get_model(self, name: str, skip=False) -> ModelProto:
        if os.path.exists(name):
            return load_onnx(name)
        p = os.path.join(os.path.dirname(__file__), "..", "ut_xbuilder", "data", name)
        if not os.path.exists(p):
            p = os.path.join(os.path.dirname(__file__), "data", name)
        if skip and not os.path.exists(p):
            raise unittest.SkipTest(f"Unable to find {p!r}.")
        self.assertExists(p)
        return load_onnx(p, load_external_data=False)

    def _range(self, *shape, bias: Optional[float] = None):
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
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
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
        self.assertEqual(["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node])
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
                [oh.make_tensor_value_info("Z", TFLOAT, ["batch", "channel", "D32", "64"])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
                ],
            )
        )
        check_model(model)
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
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
        self.assertEqual(["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node])
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
                [oh.make_tensor_value_info("Z", TFLOAT, ["batch", "channel", "D32", "64"])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
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
            infer_shapes_options=True,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Unsqueeze", "MatMul"], [n.op_type for n in opt_onx.graph.node])
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
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
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
            infer_shapes_options=True,
        )
        s = str(gr.optimization_options)
        self.assertIn("OptimizationOptions(", s)
        self.assertIn("CastPattern", s)
        opt_onx, out, _ = self.capture(lambda: gr.to_onnx(optimize=True))
        self.assertIn("remove_initializer 3:3/6:shape2", out)
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
                    infer_shapes_options=True,
                )
                onx = gr.to_onnx(optimize=True)
                types = set(n.op_type for n in onx.graph.node)
                self.assertIn("SimplifiedLayerNormalization", types)
                self._check_ort_cpu_or_cuda(onx)

    @requires_onnxruntime_training()
    def test_simplified_with_all_default(self):
        self._simplified_with_all({}, experimental=False, check_ort=cuda_recent_enough())

    def test_simplified_with_all_experimental(self):
        self._simplified_with_all({}, experimental=True, check_ort=cuda_recent_enough())

    def test_position_issue1(self):
        self._simplified_with_all(
            {},
            experimental=True,
            check_ort=cuda_recent_enough(),
            models_list=["dort-llama-llama-ort_1.onnx"],
        )

    def test_position_issue2(self):
        self._simplified_with_all(
            {},
            experimental=True,
            check_ort=cuda_recent_enough(),
            models_list=["dort-llama2-llama-ort+_1.onnx"],
        )

    def _simplified_with_all(
        self, disabled, experimental=False, check_ort=True, models_list=None
    ):
        import torch

        for model in models_list or [
            "noopt-llama-custom__1.onnx",
            "noopt-llama-custom__0.onnx",
            "noopt-phi-custom__0.onnx",
            "noopt-phi-custom__1.onnx",
            "llama_forward.onnx",
            "llama_forward.onnx",
            "opt-llama-custom-forward.onnx",
            # "dort-llama-llama-ort_1.onnx",
            # "dort-llama2-llama-ort+_1.onnx",
            "opt-llama-custom-backward.onnx",
            "dort_forward.onnx",
            "dort_backward.onnx",
            "dort-model-llama-ort+_0.onnx",
            "dort-model-llama-ort+_1.onnx",
            "dort-model-llama-ort+_1_split.onnx",
            "em_llama_custom_static_fp32_cuda_large_h6_58fa9.onnx.2.onnx",
            "em_phi_custom_static_fp32_cuda_large_h6_58fa9.onnx.2.onnx",
        ]:
            if model in {
                "noopt-llama-custom__1.onnx",
                "noopt-phi-custom__1.onnx",
                "opt-llama-custom-backward.onnx",
            } and sys.platform in {
                "win32",
                "darwin",
            }:
                # Fatal error: com.microsoft:SoftmaxGrad(-1) is not a registered function/op
                continue
            options = OptimizationOptions(
                patterns=(
                    "default+onnxruntime+experimental"
                    if experimental
                    else "default+onnxruntime"
                ),
                verbose=0,
                verifies=False,
                processor="CPU,CUDA" if torch.cuda.device_count() > 0 else "CPU",
                constant_folding=False,
            )
            options.patterns = [
                p for p in options.patterns if p.__class__.__name__ not in disabled
            ]
            onx = self._get_model(model)
            self._fix_shape(onx)
            if check_ort:
                self._check_ort_cpu_or_cuda(onx, model=model)
            gr = GraphBuilder(
                onx,
                optimization_options=options,
                infer_shapes_options=True,
            )
            new_onx = None
            try:
                new_onx = gr.to_onnx(optimize=True)
            except AssertionError as e:
                if model in {
                    "dort-llama-llama-ort_1.onnx",
                    "dort-llama2-llama-ort+_1.onnx",
                } and (
                    "Node at position 29 cannot be moved." in str(e)
                    or "Node at position 31 cannot be moved." in str(e)
                ):
                    raise unittest.SkipTest(  # noqa: B904
                        "Algorithm inserting nodes is still not perfect"
                    )
                raise AssertionError(f"Model {model!r} failed.")  # noqa: B904
            assert new_onx is not None, f"Model {model!r} was not optimized."
            op_types = [n.op_type for n in new_onx.graph.node]
            if experimental and "ScatterND" in op_types:
                if (
                    torch.cuda.device_count() > 0
                    or len([n for n in op_types if n == "ScatterND"]) > 1
                ):
                    self.dump_onnx(f"dump_{model}", new_onx)
                    raise AssertionError(f"Model {model!r} has ScatterND.")
            if check_ort and has_onnxruntime_training():
                self._check_ort_cpu_or_cuda(new_onx, model=model)

    @skipif_ci_windows("crash")
    def test_study(self):
        """
        clear&&python _unittests/ut_xoptim/test_graph_pattern_combination.py -k study
        """
        model = "llama_layer_norm.onnx"
        enabled = {
            "SimplifiedLayerNormalizationPattern",
        }
        # enabled = {}
        disabled = {}
        options = OptimizationOptions(
            patterns="default+onnxruntime+experimental",
            verbose=0,
            verifies=False,
            dump_applied_patterns="dump_applied_pattern",
            max_iter=len(enabled) + 1 if enabled else -1,
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
            options.verbose = 1 if len(options.patterns) > 3 else 10
            print(f"### __name__={__name__!r}")
            print(f"-- patterns={[c.__class__.__name__ for c in options.patterns]}")
            print(f"-- verbose={options.verbose}")
        for p in options.patterns:
            p.verbose = options.verbose
        onx = self._get_model(model, skip=True)
        self._fix_shape(onx)

        def do():
            gr = GraphBuilder(
                onx,
                optimization_options=options,
                infer_shapes_options=False,
                verbose=2 if __name__ == "__main__" else 0,
            )
            return gr.to_onnx(optimize=True, large_model=False)

        # from onnx_array_api.profiling import profile, profile2graph
        # ps = profile(do)[0]
        # root, nodes = profile2graph(ps, clean_text=lambda x: x.split("/")[-1])
        # text = root.to_text()
        # print(text)

        new_onx = do()

        with open(f"test_study_{os.path.split(model)[-1]}", "wb") as f:
            f.write(new_onx.SerializeToString())

        from onnxruntime.capi.onnxruntime_pybind11_state import (
            Fail,
            NotImplemented,
            RuntimeException,
        )

        try:
            self._check_ort_cpu_or_cuda(onx)
        except NotImplemented as e:
            raise unittest.SkipTest(f"missing extension: {e}")  # noqa: B904
        except Fail as e:
            if "com.microsoft:SoftmaxGrad(-1) is not a registered function" in str(e):
                unittest.SkipTest(f"onnxruntime-training is needed for {e}")
            raise
        except RuntimeException as e:
            if (
                "Invalid fd was supplied" in str(e)
                or "cannot get file size" in str(e)
                or "No such file or directory" in str(e)
            ):
                raise unittest.SkipTest(f"missing extension: {e}")  # noqa: B904
            raise
        self._check_ort_cpu_or_cuda(new_onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
