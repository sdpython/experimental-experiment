import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_cuda,
    long_test,
)
from experimental_experiment.torch_bench.big_models.try_phi35_mini_instruct import (
    get_model_inputs,
)


class TestPhi35MiniInstruct(ExtTestCase):
    @long_test()
    @hide_stdout()
    @requires_cuda(memory=24)
    def test_get_model_inputs(self):
        model_fct, inputs = get_model_inputs(device="cuda", dtype="float16", verbose=1)
        model = model_fct()
        model(*inputs)

    @long_test()
    @requires_cuda(memory=24)
    def test_export_export(self):
        import torch

        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        export_program = torch.export.export(model, inputs, strict=False)
        self.assertNotEmpty(export_program)

    @long_test()
    @requires_cuda(memory=24)
    def test_export_onnx_dynamo(self):
        import torch

        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        torch.onnx.export(model, inputs, "test_phi35_mini_instruct_dynamo.onnx", dynamo=True)

    @long_test()
    @requires_cuda(memory=24)
    def test_export_onnx_dynamo_auto(self):
        import torch

        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="auto")
        model = model_fct()
        torch.onnx.export(
            model, inputs, "test_phi35_mini_instruct_dynamo_auto.onnx", dynamo=True
        )

    @long_test()
    @requires_cuda(memory=24)
    def test_export_onnx_custom_float16(self):
        from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
        from experimental_experiment.xbuilder import OptimizationOptions

        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        onx = to_onnx(
            model,
            inputs,
            large_model=True,
            verbose=1,
            options=OptimizationOptions(max_iter=10),
            export_options=ExportOptions(strict=False, decomposition=True),
        )
        onx.save("test_phi35_mini_instruct_custom_f16.onnx", all_tensors_to_one_file=True)

    @long_test()
    @requires_cuda(memory=24)
    def test_export_onnx_custom_auto(self):
        from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
        from experimental_experiment.xbuilder import OptimizationOptions

        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="auto")
        model = model_fct()
        onx = to_onnx(
            model,
            inputs,
            large_model=True,
            verbose=1,
            options=OptimizationOptions(max_iter=10),
            export_options=ExportOptions(strict=False, decomposition_table="all"),
        )
        onx.save("test_phi35_mini_instruct_custom_auto.onnx", all_tensors_to_one_file=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
