import itertools
import unittest
from experimental_experiment.ext_test_case import ExtTestCase, skipif_ci_windows
from experimental_experiment.torch_bench._dort_cmd_common import (
    create_configuration_for_benchmark,
    create_compiled_model,
)
from experimental_experiment.torch_models.llama_helper import get_llama_model


class TestDortCmdCommond(ExtTestCase):
    def test_create_configuration_for_benchmark(self):
        for config in {"small", "large", "medium"}:
            d = create_configuration_for_benchmark("llama", config)
            self.assertIsInstance(d, dict)
        self.assertRaise(
            lambda: create_configuration_for_benchmark("llama", "f"), ValueError
        )

    @skipif_ci_windows("dynamo not supported")
    def test_get_model(self):
        cf = create_configuration_for_benchmark("llama", "small")
        model, example_args_collection = get_llama_model(**cf)
        for bck in {"ort", "custom", "plug", "debug", "inductor", "eager", "dynger"}:
            cp = create_compiled_model(
                model,
                backend=bck,
                use_dynamic=False,
                target_opset=18,
                verbose=0,
                enable_pattern=[],
                disable_pattern=[],
            )
            self.assertNotEmpty(cp)
        self.assertRaise(
            lambda: create_compiled_model(
                model,
                backend="bck",
                use_dynamic=False,
                target_opset=18,
                verbose=0,
                enable_pattern=[],
                disable_pattern=[],
            ),
            ValueError,
        )

    def test_create_many_configurations(self):
        for model, impl, mask in itertools.product(
            ["llama", "mistral", "phi"], ["eager", "sdpa"], [False, True]
        ):
            with self.subTest(model=model, impl=impl, mask=mask):
                d = create_configuration_for_benchmark(
                    model="llama", config="medium", implementation=impl, with_mask=mask
                )
                self.assertNotEmpty(d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
