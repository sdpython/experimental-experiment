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
        import onnx
        import torch
        import transformers
        import onnxruntime as ort
        from experimental_experiment.torch_interpreter import to_onnx, ExportOptions

        def assert_close(actual, desired):
            if isinstance(desired, torch.Tensor):
                torch.testing.assert_close(actual, desired)
            elif isinstance(desired, tuple):
                assert isinstance(actual, tuple)
                assert len(actual) == len(desired)
                for a, d in zip(actual, desired):
                    torch.testing.assert_close(a, d)
            else:
                raise NotImplementedError(f"Not implemented for class {type(desired)}")

        def flatten(obj):
            if isinstance(obj, torch.Tensor):
                return obj
            if isinstance(obj, tuple):
                res = []
                for o in obj:
                    if isinstance(o, torch.Tensor):
                        res.append(o)
                    else:
                        res.extend(flatten(o))
                return tuple(res)
            else:
                raise NotImplementedError(f"Not implemented for class {type(obj)}")

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

        # assert "[num_users=0]" not in str(ep.graph), f"One output is unused:\n{ep.graph}"
        mod = ep.module()
        # got = mod(input_ids, attention_mask)
        # assert_close(got.to_tuple(), expected.to_tuple())

        expected2 = model(input_ids, attention_mask * 0)
        got2 = mod(input_ids, attention_mask * 0)
        assert_close(got2.to_tuple(), expected2.to_tuple())
        # assert_close(expected2.to_tuple(), expected.to_tuple())

        onx = to_onnx(
            model,
            (input_ids, attention_mask),
            export_options=ExportOptions(aten_as_function=True),
            optimize=False,
        )
        onnx.save(onx, "test_mistral_nousers_aten.onnx")
        sess = ort.InferenceSession(
            "test_mistral_nousers_aten.onnx", providers=["CPUExecutionProvider"]
        )
        got = sess.run(
            None, {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()}
        )
        assert_close(tuple(torch.from_numpy(t) for t in got), flatten(expected.to_tuple()))

        got = sess.run(
            None,
            {"input_ids": input_ids.numpy(), "attention_mask": (attention_mask * 0).numpy()},
        )
        assert_close(tuple(torch.from_numpy(t) for t in got), flatten(expected2.to_tuple()))

        # ep = torch.onnx.export(mod, (input_ids, attention_mask), dynamo=True)
        # ep.optimize()
        # onnx.save(ep.model_proto, "test_mistral_nousers.onnx")
        # sess = ort.InferenceSession(
        #     "test_mistral_nousers.onnx", providers=["CPUExecutionProvider"]
        # )
        # got = sess.run(
        #     None, {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()}
        # )
        # assert_close(tuple(torch.from_numpy(t) for t in got), flatten(expected.to_tuple()))

    def test_inplace_affectation(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                x = x.clone()
                x[:, :2] = x[:, :2] * 2
                return x

        model = Model()
        x = torch.ones((4, 4))
        ep = torch.export.export(model, (x,))
        ep.recompile()
        torch.testing.assert_close(
            model(x), ep.module()(x)
        )  # this test should fail but it does not.
        print(ep.graph)
        """
        graph():
            %x : [num_users=1] = placeholder[target=x]
            %clone : [num_users=3] = call_function[target=torch.ops.aten.clone.default]
                (args = (%x,), kwargs = {})
            %slice_1 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%clone, 0, 0, 9223372036854775807), kwargs = {})
            %slice_2 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%slice_1, 1, 0, 2), kwargs = {})
            %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor]
                (args = (%slice_2, 2), kwargs = {})
            %slice_3 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%clone, 0, 0, 9223372036854775807), kwargs = {})
            %slice_4 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%slice_3, 1, 0, 2), kwargs = {})
            %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default]
                (args = (%slice_4, %mul), kwargs = {})
            return (clone,)  <---- This is wrong.
        """


if __name__ == "__main__":
    unittest.main(verbosity=2)
