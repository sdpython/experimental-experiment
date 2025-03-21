import pprint
import unittest
import pandas
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_cuda,
    never_test,
    requires_torch,
    requires_transformers,
    long_test,
)
from experimental_experiment.torch_models import flatten_outputs
from experimental_experiment.helpers import string_type


class TestChronosModelHelper(ExtTestCase):
    def setUp(self):
        import torch

        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        import torch

        super().tearDown()
        torch._dynamo.reset()

    @never_test()
    @requires_cuda()
    def test_spy_chronos_t5_tiny(self):
        import torch
        from chronos import ChronosPipeline

        MODEL_NAME = "amazon/chronos-t5-tiny"

        pipeline = ChronosPipeline.from_pretrained(
            MODEL_NAME,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        inputs_iteration = []
        print(
            "test_spy_chronos_t5_tiny",
            type(pipeline),
            type(pipeline.model),
            type(pipeline.model.model),
        )
        pprint.pprint(pipeline.model.config)

        def rewrite_forward(f, *args, **kwargs):
            print(
                f"------------- test_spy_chronos_t5_tiny -- iteration {len(inputs_iteration)}"
            )
            print(f"args: {string_type(args, with_shape=True, with_min_max=True)}")
            print(f"kwargs: {string_type(kwargs, with_shape=True, with_min_max=True)}")
            inputs_iteration.append((args, kwargs))
            if len(inputs_iteration) > 5:
                raise unittest.SkipTest(
                    f"Not necessary to go beyond {len(inputs_iteration)} iterations."
                )
            res = f(*args, **kwargs)
            print(f"res: {string_type(res, with_shape=True, with_min_max=True)}")
            return res

        model_forward = pipeline.model.forward
        pipeline.model.forward = lambda *args, f=model_forward, **kwargs: rewrite_forward(
            f, *args, **kwargs
        )

        df = pandas.read_csv(
            "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
        )

        # context must be either a 1D tensor, a list of 1D tensors,
        # or a left-padded 2D tensor with batch as the first dimension
        df["#Passengersopcy"] = df["#Passengers"]
        context = torch.tensor(df[["#Passengers", "#Passengersopcy"]].values.T)
        prediction_length = 104
        forecast = pipeline.predict(
            context, prediction_length
        )  # shape [num_series, num_samples, prediction_length]

        if __name__ == "__main__":
            print(f"test_spy_chronos_t5_tiny={forecast}")

        pipeline.model.forward = model_forward

        # does not work
        # torch.export.export(pipeline.model, inputs_iteration[0][0])

    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @requires_torch("2.6")  # torch.export.Dim.DYNAMIC
    @requires_transformers("4.49.9999")
    @long_test()
    def test_a_get_chronos_t5_tiny_fixed_nospy(self):
        import torch
        from experimental_experiment.torch_models.chronos_model_helper import (
            get_chronos_t5_tiny,
        )

        data = get_chronos_t5_tiny(
            batch_size=2, common_dynamic_shapes=True, fixed_prediction_length=17
        )
        model, model_inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        # Does not work with something elkse than a constant.
        expected = list(flatten_outputs(model(**model_inputs)))
        self.assertNotEmpty(expected)
        # import torch.export._draft_export
        # draft_export fails at line: transformers/generation/utils.py", line 3336
        # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        # RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
        # print("--------------")
        # ep, report = torch.export._draft_export.draft_export(
        #   model, (), model_inputs, dynamic_shapes=ds, strict=False)
        # )
        # print(report)
        # print(ep)
        # print("--------------")
        # patches should not be needed after this PR:
        # https://github.com/huggingface/transformers/pull/36652
        ep = torch.export.export(model, (), model_inputs, dynamic_shapes=ds, strict=False)
        assert ep


if __name__ == "__main__":
    unittest.main(verbosity=2)
