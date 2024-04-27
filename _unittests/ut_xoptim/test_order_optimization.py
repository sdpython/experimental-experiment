import os
import unittest
import numpy as np
from onnx import ModelProto, TensorProto, helper as oh, load as load_onnx
from onnx.checker import check_model
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    ignore_warnings,
    requires_onnxruntime_training,
)
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.xoptim import OrderAlgorithm

TFLOAT = TensorProto.FLOAT
TFLOAT16 = TensorProto.FLOAT16


class TestGraphOrderOptimization(ExtTestCase):

    def _range(self, *shape, bias: float = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def _get_model(self, name: str, skip=False) -> ModelProto:
        p = os.path.join(os.path.dirname(__file__), "..", "ut_xbuilder", "data", name)
        if not os.path.exists(p):
            p = os.path.join(os.path.dirname(__file__), "data", name)
        if skip and not os.path.exists(p):
            raise unittest.SkipTest(f"Unable to find {p!r}.")
        self.assertExists(p)
        return load_onnx(p)

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

    @ignore_warnings(RuntimeWarning)
    @hide_stdout()
    def test_arandom_order(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy1"]),
                    oh.make_node("Mul", ["X", "Y"], ["xy2"]),
                    oh.make_node("Sub", ["X", "Y"], ["xy3"]),
                    oh.make_node("Div", ["X", "Y"], ["xy4"]),
                    oh.make_node("Add", ["xy1", "xy2"], ["xy10"]),
                    oh.make_node("Mul", ["xy1", "xy2"], ["xy12"]),
                    oh.make_node("Sub", ["xy3", "xy4"], ["xy13"]),
                    oh.make_node("Div", ["xy3", "xy4"], ["xy14"]),
                    oh.make_node("Add", ["xy10", "xy12"], ["r20"]),
                    oh.make_node("Add", ["xy13", "xy14"], ["r21"]),
                    oh.make_node("Add", ["r21", "r20"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [4, 4]),
                    oh.make_tensor_value_info("Y", TFLOAT, [4, 4]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [4, 4])],
            )
        )
        check_model(model)
        op_types = [n.op_type for n in model.graph.node]

        verbose = 10
        gr = GraphBuilder(
            model,
            infer_shapes=True,
            optimization_options=OptimizationOptions(
                patterns=None, verbose=verbose, order=OrderAlgorithm.RANDOM
            ),
            verbose=0,
        )

        feeds = {"X": self._range(4, 4), "Y": self._range(4, 4)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        opt_onx = gr.to_onnx(optimize=True)
        new_op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertEqual(len(op_types), len(new_op_types))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @requires_onnxruntime_training()
    def test_order_bigger_model(self):
        for model in [
            # "noopt-phi-custom__1.onnx",
            # "noopt-llama-custom__1.onnx",
            # "noopt-llama-custom__0.onnx",
            # "noopt-phi-custom__0.onnx",
            "llama_forward.onnx",
            "llama_forward.onnx",
            "opt-llama-custom-forward.onnx",
            "dort-llama-llama-ort_1.onnx",
            "dort-llama2-llama-ort+_1.onnx",
            "opt-llama-custom-backward.onnx",
        ]:
            # print(f"[test_order_bigger_model] {model!r}")
            onx = self._get_model(model)
            self._fix_shape(onx)
            self._check_ort_cpu_or_cuda(onx)

            # print(f"[test_order_bigger_model] starts optimize")
            with self.subTest(model=model):
                gr = GraphBuilder(
                    onx,
                    infer_shapes=False,
                    optimization_options=OptimizationOptions(
                        patterns=None, verbose=0, order=OrderAlgorithm.RANDOM
                    ),
                    verbose=0,
                )
                onx = gr.to_onnx(optimize=True)
                self._check_ort_cpu_or_cuda(onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
