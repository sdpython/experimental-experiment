import unittest
import os
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_cuda,
    requires_onnxruntime,
)


class TestIssuesOnnxruntime2024(ExtTestCase):

    @requires_cuda()
    @requires_onnxruntime("1.21")
    def test_ort_optimization(self):
        # issue https://github.com/microsoft/onnxruntime/issues/23143

        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        import onnxruntime as ort
        import numpy as np
        from experimental_experiment.reference import ExtendedReferenceEvaluator

        proto_simple = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cast", ["v0_0"], ["x1"], to=onnx.TensorProto.FLOAT),
                    oh.make_node("Cast", ["v0_0"], ["x2"], to=onnx.TensorProto.FLOAT),
                    oh.make_node("Flatten", ["x1"], ["f1"], axis=0),
                    oh.make_node("Flatten", ["x2"], ["f2"], axis=0),
                    oh.make_node("Concat", ["f1", "i1"], ["c1"], axis=1),
                    oh.make_node("Concat", ["f2", "i2"], ["c2"], axis=1),
                    oh.make_node("Reshape", ["c1", "s1"], ["m1"]),
                    oh.make_node("Reshape", ["c2", "s2"], ["m2"]),
                    oh.make_node("MatMul", ["m1", "m2"], ["mm"]),
                    oh.make_node("Identity", ["mm"], ["output"]),
                ],
                "nd",
                [oh.make_tensor_value_info("v0_0", onnx.TensorProto.DOUBLE, [5])],
                [oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 3, 3, 3])],
                [
                    onh.from_array(np.zeros((1, 49)).astype(np.float32), name="i1"),
                    onh.from_array(np.zeros((1, 4)).astype(np.float32), name="i2"),
                    onh.from_array(np.array([2, 3, 3, 3], dtype=np.int64), name="s1"),
                    onh.from_array(np.array([3, 3], dtype=np.int64), name="s2"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        # not optimized
        input_data = {"v0_0": np.arange(5).astype(np.float64)}
        proto_issue = onnx.load(
            os.path.join(os.path.dirname(__file__), "data", "inconsis3.onnx")
        )
        for i, proto in enumerate([proto_simple, proto_issue]):
            sessopts = ort.SessionOptions()
            sessopts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            sessopts.optimized_model_filepath = self.get_dump_file(
                f"test_ort_optimization_disabled_{i}.onnx"
            )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            original_session = ort.InferenceSession(
                proto.SerializeToString(), sessopts, providers=providers
            )
            output_names = ["output"]
            original_result = original_session.run(output_names, input_data)

            # optimized
            sessopts2 = ort.SessionOptions()
            sessopts2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sessopts2.optimized_model_filepath = self.get_dump_file(
                f"test_ort_optimization_enabled_{i}.onnx"
            )
            original_session2 = ort.InferenceSession(
                proto.SerializeToString(), sessopts2, providers=providers
            )
            original_result2 = original_session2.run(output_names, input_data)

            ref = ExtendedReferenceEvaluator(proto, verbose=10)
            onnx_results = ref.run(output_names, input_data)
            # fails here
            np.testing.assert_allclose(onnx_results[0], original_result[0])
            np.testing.assert_allclose(onnx_results[0], original_result2[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
