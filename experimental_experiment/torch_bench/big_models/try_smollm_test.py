import unittest
from experimental_experiment.ext_test_case import ExtTestCase, requires_cuda, hide_stdout
from experimental_experiment.torch_bench.big_models.try_smollm import (
    load_model,
    demo_model,
    get_model_inputs,
)


class TestSmolLM(ExtTestCase):
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
        model_fct, inputs = get_model_inputs(device="cuda", dtype="float16", verbose=1)
        model = model_fct()
        model(*inputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
