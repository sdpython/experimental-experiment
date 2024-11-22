import unittest
import numpy as np
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    skipif_ci_windows,
)


class TestIssuesPytorch2024Export(ExtTestCase):

    @skipif_ci_windows("not working")
    @requires_torch("2.7")
    def test_index_put_none_decompositions(self):
        # see issue https://github.com/pytorch/pytorch/issues/141336
        import torch

        class UpdateModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.params = torch.zeros((4, 4, 10))

            def forward(self, update, index1, index2):
                copy = self.params.clone()
                copy[index1, torch.tensor([1, 2], dtype=torch.int64), index2] = update
                return copy

        model = UpdateModel()

        update = (torch.arange(2) + 10).reshape((2,)).to(torch.float32)
        index1 = torch.from_numpy(np.array([1, 2])).to(torch.int64)
        index2 = torch.from_numpy(np.array([7, 8])).to(torch.int64)
        model(update, index1, index2)

        ep = torch.export.export(model, (update, index1, index2))
        # print(ep.graph)
        ep.run_decompositions()  # Fails here

    def test_mistral_nousers(self):
        import torch
        import transformers

        config = transformers.MistralConfig(
            hidden_size=32,
            num_hidden_layers=2,
            vocab_size=99,
            intermediate_size=16,
            max_position_embeddings=512,
            num_attention_heads=2,
            num_key_value_heads=2,
            sliding_window=4096,
        )
        config._attn_implementation = "eager"
        model = transformers.MistralModel(config)
        shape = (1, 30)
        input_ids = torch.randint(0, 99, shape).to(torch.int64)
        attention_mask = torch.ones(shape)
        expected = model(input_ids, attention_mask)
        ep = torch.export.export(model, (input_ids, attention_mask))
        assert "[num_users=0]" not in str(ep.graph), f"One output is unused:\n{ep.graph}"
        mod = ep.module()
        got = mod((input_ids, attention_mask))
        torch.testing.assert_close(got, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
