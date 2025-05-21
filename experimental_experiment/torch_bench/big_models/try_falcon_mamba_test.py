import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_cuda,
    long_test,
)
from onnx_diagnostic.torch_export_patches import torch_export_patches
from experimental_experiment.torch_bench.big_models.try_falcon_mamba import (
    load_model,
    demo_model,
    get_model_inputs,
)


class TestFalconMamba(ExtTestCase):
    @long_test()
    @hide_stdout()
    @requires_cuda(memory=24)
    def test_demo_float16(self):
        tokenizer, model = load_model(
            device="cuda", dtype="float16", verbose=1, load_tokenizer=True
        )
        demo_model(tokenizer, model, verbose=1)

    @hide_stdout()
    @requires_cuda(memory=24)
    def test_get_model_inputs(self):
        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        model(*inputs)

    @long_test()
    @requires_cuda(memory=24)
    def test_export_export(self):
        import torch

        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        with torch_export_patches():
            export_program = torch.export.export(model, inputs)
            self.assertNotEmpty(export_program)

    @long_test()
    @requires_cuda(memory=24)
    def test_export_onnx_dynamo(self):
        import torch

        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        with torch_export_patches():
            torch.onnx.export(model, inputs, "test_falcon_mamba_onnx_dynamo.onnx", dynamo=True)

    @long_test()
    @requires_cuda(memory=24)
    def test_export_onnx_custom(self):
        from experimental_experiment.torch_interpreter import to_onnx
        from experimental_experiment.xbuilder import OptimizationOptions

        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        with torch_export_patches():
            onx = to_onnx(
                model,
                inputs,
                large_model=True,
                verbose=1,
                options=OptimizationOptions(max_iter=10),
            )
            onx.save("test_falcon_mamba_custom.onnx", all_tensors_to_one_file=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
