import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_cuda,
    long_test,
)
from experimental_experiment.torch_bench.big_models.try_flux_transformer import (
    get_model_inputs,
)


class TestFluxTransformer(ExtTestCase):
    @long_test()
    @hide_stdout()
    @requires_cuda(memory=31)
    def test_get_model_inputs(self):
        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        model(**inputs)

    @long_test()
    @requires_cuda(memory=31)
    def test_export_onnx_dynamo(self):
        import torch

        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        torch.onnx.export(model, (inputs,), dynamo=True)

    @long_test()
    @requires_cuda(memory=31)
    def test_export_custom(self):
        from experimental_experiment.torch_interpreter import to_onnx

        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        to_onnx(model, inputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
