import unittest
import os
from experimental_experiment.ext_test_case import ExtTestCase, requires_onnxruntime


class TestIssuesOnnxruntime2025(ExtTestCase):

    @requires_onnxruntime("1.21")
    def test_ort_optimization_23199(self):
        # issue https://github.com/microsoft/onnxruntime/issues/23199

        import onnx
        import onnxruntime as ort
        import numpy as np
        from experimental_experiment.reference import ExtendedReferenceEvaluator

        # not optimized
        input_data = {"v5_0": np.random.rand(55, 7, 1, 40).astype(np.float32)}
        proto_issue = onnx.load(
            os.path.join(os.path.dirname(__file__), "data", "inconsis20730.onnx")
        )
        for i, proto in enumerate([proto_issue]):
            sessopts = ort.SessionOptions()
            sessopts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            sessopts.optimized_model_filepath = file_model1 = (
                f"test_ort_optimization_23199_disabled_{i}.onnx"
            )
            providers = [
                "CPUExecutionProvider"
            ]  # ["CUDAExecutionProvider", "CPUExecutionProvider"]
            original_session = ort.InferenceSession(
                proto.SerializeToString(), sessopts, providers=providers
            )
            output_names = ["output"]
            original_result = original_session.run(output_names, input_data)

            # optimized
            sessopts2 = ort.SessionOptions()
            sessopts2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sessopts2.optimized_model_filepath = file_model2 = (
                f"test_ort_optimization_23199_enabled_{i}.onnx"
            )
            original_session2 = ort.InferenceSession(
                proto.SerializeToString(), sessopts2, providers=providers
            )
            original_result2 = original_session2.run(output_names, input_data)

            ref = ExtendedReferenceEvaluator(proto, verbose=10)
            onnx_results = ref.run(output_names, input_data)

            # before comparing the final results, let's compare the intermediate results
            from onnx_array_api.reference import compare_onnx_execution

            model1 = onnx.load(file_model1)
            model2 = onnx.load(file_model2)
            res1, res2, align, dc = compare_onnx_execution(
                model1,
                model2,
                inputs=[input_data[k.name] for k in model1.graph.input],
                verbose=1,
                raise_exc=True,
                cls=ExtendedReferenceEvaluator,
            )
            # for r in res2:
            #    r.name = clean_name(r.name)
            text = dc.to_str(res1, res2, align, column_size=90)
            print("-----------------------------")
            print(text)
            print("-----------------------------")

            # fails here
            self.assertEqual(len(onnx_results), len(original_result))
            self.assertEqual(len(onnx_results), len(original_result2))
            for i in range(len(onnx_results)):
                with self.subTest(i=i, name=output_names[i]):
                    np.testing.assert_allclose(onnx_results[i], original_result[i])
                    np.testing.assert_allclose(onnx_results[i], original_result2[i])


if __name__ == "__main__":
    unittest.main(verbosity=2)
