import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_cuda,
    long_test,
)
from experimental_experiment.torch_interpreter.onnx_export_errors import (
    bypass_export_some_errors,
)
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
        with bypass_export_some_errors():
            export_program = torch.export.export(model, inputs)
            print(export_program.graph)
            self.assertNotEmpty(export_program)

    @long_test()
    @requires_cuda(memory=24)
    def test_export_onnx_dynamo(self):
        import torch

        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        with bypass_export_some_errors():
            torch.onnx.export(model, inputs, "test_falcon_mamba_onnx_dynamo.onnx", dynamo=True)

    @long_test()
    @requires_cuda(memory=24)
    def test_export_custom(self):
        from experimental_experiment.torch_interpreter import to_onnx

        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        with bypass_export_some_errors():
            onx = to_onnx(model, inputs, large_model=True)
            onx.save("test_falcon_mamba_custom.onnx", all_tensors_to_one_file=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
