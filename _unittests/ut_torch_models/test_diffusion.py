import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    require_diffusers,
    skipif_ci_windows,
    long_test,
)
from experimental_experiment.torch_models import flatten_outputs


class TestDiffusion(ExtTestCase):

    @require_diffusers("0.30.0")
    @skipif_ci_windows("crashing")
    @long_test()
    def test_get_stable_diffusion_2_unet(self):
        # import torch
        from experimental_experiment.torch_models.diffusion_model_helper import (
            get_stable_diffusion_2_unet,
        )

        model, model_inputs = get_stable_diffusion_2_unet(
            overwrite=True,
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            inputs_as_tuple=True,
        )
        expected = list(flatten_outputs(model(*model_inputs)))
        self.assertNotEmpty(expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
