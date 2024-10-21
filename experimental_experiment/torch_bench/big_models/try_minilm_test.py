import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_cuda,
    long_test,
)
from experimental_experiment.torch_bench.big_models.try_minilm import (
    load_model,
    demo_model,
    get_model_inputs,
)


class TestMiniLM(ExtTestCase):
    @long_test()
    @hide_stdout()
    @requires_cuda(memory=1)
    def test_demo_float16(self):
        tokenizer, model = load_model(
            device="cuda", dtype="float16", verbose=1, load_tokenizer=True
        )
        demo_model(tokenizer, model, verbose=1)

    @long_test()
    @hide_stdout()
    @requires_cuda(memory=1)
    def test_get_model_inputs(self):
        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        model(**inputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
