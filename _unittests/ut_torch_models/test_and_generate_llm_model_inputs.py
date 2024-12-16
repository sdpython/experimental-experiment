import copy
import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
    hide_stdout,
    requires_cuda,
)
from experimental_experiment.torch_models.phi3_helper import has_phi3
from experimental_experiment.torch_models.dummy_inputs import generate_dummy_inputs
from experimental_experiment.torch_models.dummy_inputs.llm_dummy_inputs import (
    restore_dummy_inputs_for_phi35_vision_instruct,
)
from experimental_experiment.mini_onnx_builder import create_input_tensors_from_onnx_model
from experimental_experiment.helpers import string_type
from experimental_experiment.torch_models.llm_model_helper import LLMInputKind


class TestLlmModelInputs(ExtTestCase):

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings((UserWarning, FutureWarning))
    @requires_cuda()
    @hide_stdout()
    def test_generate_dummy_inputs_no_images(self):
        # This test is used to generate dummy inputs.
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_vision_instruct,
        )

        device = "cuda"

        filenames = generate_dummy_inputs(
            "microsoft/Phi-3.5-vision-instruct",
            num_hidden_layers=1,
            device=device,
            n_iterations=2,
            with_images=False,
            prefix="test_",
            verbose=1,
        )
        for f in filenames:
            with self.subTest(filename=f):
                self.assertExists(f)

                model, *_ = get_phi35_vision_instruct(
                    num_hidden_layers=1, common_dynamic_shapes=True
                )
                model = model.to(device)
                args, kwargs = create_input_tensors_from_onnx_model(
                    f, device=device, engine="onnxruntime"
                )
                text = string_type((args, kwargs), with_shape=True, with_min_max=True)
                self.assertIn("input_ids", text)
                model(*args, **kwargs)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings((UserWarning, FutureWarning))
    @requires_cuda()
    @hide_stdout()
    def test_generate_dummy_inputs_with_images(self):
        # This test is used to generate dummy inputs.
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_vision_instruct,
        )

        device = "cuda"

        filenames = generate_dummy_inputs(
            "microsoft/Phi-3.5-vision-instruct",
            num_hidden_layers=1,
            device=device,
            n_iterations=2,
            prefix="test_",
            verbose=1,
            with_images=True,
        )
        for f in filenames:
            with self.subTest(filename=f):
                self.assertExists(f)

                model, *_ = get_phi35_vision_instruct(
                    num_hidden_layers=1, common_dynamic_shapes=True
                )
                model = model.to(device)
                args, kwargs = create_input_tensors_from_onnx_model(
                    f, device=device, engine="onnxruntime"
                )
                model(*args, **kwargs)

    def test_restore_dummy_inputs_no_images(self):
        for it in range(0, 2):
            with self.subTest(iteration=it):
                dummies = restore_dummy_inputs_for_phi35_vision_instruct(
                    num_hidden_layers=1, with_images=False, n_iteration=it
                )
                self.assertIsInstance(dummies, tuple)
                self.assertEqual(len(dummies), 2)
                self.assertIsInstance(dummies[0], tuple)
                self.assertIsInstance(dummies[1], dict)
                for k, v in dummies[1].items():
                    if k == "past_key_values":
                        self.assertIn("DynamicCache", str(type(v)))

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings((UserWarning, FutureWarning))
    @requires_cuda()
    @hide_stdout()
    def test_restore_dummy_inputs_no_images_and_check(self):
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_vision_instruct,
        )

        device = "cuda"
        model, *_ = get_phi35_vision_instruct(num_hidden_layers=1, common_dynamic_shapes=True)
        model = model.to(device)
        for it in range(0, 2):
            with self.subTest(iteration=it):
                args, kwargs = restore_dummy_inputs_for_phi35_vision_instruct(
                    num_hidden_layers=1, with_images=False, n_iteration=it, device=device
                )
                model(*args, **kwargs)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings((UserWarning, FutureWarning))
    @requires_cuda()
    @hide_stdout()
    def test_get_dummy_inputs_and_check(self):
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_vision_instruct,
        )

        input_names = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "pixel_values",
            "image_sizes",
            "labels",
            "use_cache",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
        ]

        device = "cuda"
        for it in range(2):
            with self.subTest(iteration=it):
                model, kwargs, _ = get_phi35_vision_instruct(
                    num_hidden_layers=1,
                    input_cache=it == 1,
                    device=device,
                    input_kind=LLMInputKind.input_ids
                    | LLMInputKind.position_ids
                    | LLMInputKind.attention_mask
                    | LLMInputKind.past_key_values,
                    common_dynamic_shapes=True,
                )
                model(**copy.deepcopy(kwargs))
                model(*[kwargs.get(n, None) for n in input_names])

    def test_restore_dummy_inputs_with_images(self):
        for it in range(0, 2):
            with self.subTest(iteration=it):
                dummies = restore_dummy_inputs_for_phi35_vision_instruct(
                    num_hidden_layers=1, with_images=True, n_iteration=it
                )
                self.assertIsInstance(dummies, tuple)
                self.assertEqual(len(dummies), 2)
                self.assertIsInstance(dummies[0], tuple)
                self.assertIsInstance(dummies[1], dict)
                for k, v in dummies[1].items():
                    if k == "past_key_values":
                        self.assertIn("DynamicCache", str(type(v)))

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings((UserWarning, FutureWarning))
    @requires_cuda()
    @hide_stdout()
    def test_restore_dummy_inputs_with_images_and_check(self):
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_vision_instruct,
        )

        device = "cuda"
        model, _ = get_phi35_vision_instruct(num_hidden_layers=1)
        model = model.to(device)
        for it in range(0, 2):
            with self.subTest(iteration=it):
                args, kwargs = restore_dummy_inputs_for_phi35_vision_instruct(
                    num_hidden_layers=1, with_images=True, n_iteration=it, device=device
                )
                model(*args, **kwargs)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings((UserWarning, FutureWarning))
    @requires_cuda()
    @hide_stdout()
    def test_get_dummy_inputs_with_imgaes_and_check(self):
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_vision_instruct,
        )

        device = "cuda"
        for it in range(2):
            with self.subTest(iteration=it):
                model, kwargs = get_phi35_vision_instruct(
                    num_hidden_layers=1,
                    input_cache=it == 1,
                    device=device,
                    input_kind=LLMInputKind.ALL,
                )
                model(**kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
