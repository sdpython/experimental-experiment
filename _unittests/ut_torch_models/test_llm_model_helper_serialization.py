import copy
import unittest
import onnx
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_cuda,
    skipif_ci_windows,
    long_test,
    requires_torch,
)
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_models import flatten_outputs
from experimental_experiment.torch_models.phi3_helper import has_phi3
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
from experimental_experiment.helpers import string_type, max_diff


class TestLlmModelHelperSerialization(ExtTestCase):
    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @skipif_ci_windows("not supported")
    @long_test()
    def test_get_phi35_mini_instruct(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_mini_instruct,
        )

        model, model_inputs = get_phi35_mini_instruct(num_hidden_layers=1)
        expected = list(flatten_outputs(model(**model_inputs)))
        # torch.onnx.export(
        #    model,
        #    (model_inputs["input_ids"], model_inputs["attention_mask"]),
        #    "test_get_phi35_mini_instruct_onnx_dynamo.onnx",
        #    dynamo=True,
        # )
        onx = to_onnx(
            model,
            None,  # args
            model_inputs,  # kwargs
            large_model=True,
            verbose=0,
            options=OptimizationOptions(max_iter=10),
            export_options=ExportOptions(strict=False, decomposition_table="all"),
        )
        filename = "test_phi35_mini_instruct_custom.onnx"
        onx.save(filename, all_tensors_to_one_file=True)
        import onnxruntime

        sess = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])
        got = sess.run(None, {k: v.numpy() for k, v in model_inputs.items()})
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            self.assertEqualArray(a, b, atol=1e-5)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @skipif_ci_windows("not supported")
    @requires_cuda()
    @long_test()
    def test_get_phi35_mini_instruct_cuda(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_mini_instruct,
        )

        model, model_inputs = get_phi35_mini_instruct(num_hidden_layers=1)
        model = model.to("cuda")
        model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}
        expected = list(flatten_outputs(model(**model_inputs)))
        onx = to_onnx(
            model,
            None,  # args
            model_inputs,  # kwargs
            large_model=True,
            verbose=0,
            options=OptimizationOptions(max_iter=10),
            export_options=ExportOptions(strict=False, decomposition_table="all"),
        )
        filename = "test_phi35_mini_instruct_custom_cuda.onnx"
        onx.save(filename, all_tensors_to_one_file=True)
        import onnxruntime

        sess = onnxruntime.InferenceSession(
            filename, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        got = sess.run(None, {k: v.cpu().numpy() for k, v in model_inputs.items()})
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            self.assertEqualArray(a, b, atol=1e-2)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @skipif_ci_windows("not supported")
    @requires_cuda()
    @long_test()
    def test_get_phi35_mini_instruct_cuda_modules(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_mini_instruct,
        )

        model, model_inputs = get_phi35_mini_instruct(num_hidden_layers=1)
        model = model.to("cuda")
        model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}
        expected = list(flatten_outputs(model(**model_inputs)))
        onx = to_onnx(
            model,
            None,  # args
            model_inputs,  # kwargs
            large_model=True,
            verbose=0,
            options=OptimizationOptions(max_iter=10),
            export_options=ExportOptions(strict=False, decomposition_table="all"),
            export_modules_as_functions=True,
            inline=False,  # function do not retain shape information
        )
        filename = "test_phi35_mini_instruct_custom_cuda_modules.onnx"
        onx.save(filename, all_tensors_to_one_file=True)
        import onnxruntime

        sess = onnxruntime.InferenceSession(
            filename, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        got = sess.run(None, {k: v.cpu().numpy() for k, v in model_inputs.items()})
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            self.assertEqualArray(a, b, atol=1e-2)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @skipif_ci_windows("not supported")
    @requires_cuda()
    @long_test()
    def test_get_phi35_mini_instruct_cuda_modules_dynshapes(self):
        import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_mini_instruct,
        )

        model, model_inputs = get_phi35_mini_instruct(num_hidden_layers=1, batch=2)
        model = model.to("cuda")
        model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}
        expected = list(flatten_outputs(model(**model_inputs)))
        dims = {
            0: torch.export.Dim("batch", min=1, max=128),
            1: torch.export.Dim("seq", min=1, max=512) * 8 - 2,
        }
        dyn_shapes = {"input_ids": dims, "attention_mask": dims}
        onx, builder = to_onnx(
            model,
            None,  # args
            model_inputs,  # kwargs
            large_model=True,
            verbose=0,
            options=OptimizationOptions(max_iter=10),
            export_options=ExportOptions(strict=False, decomposition_table="all"),
            export_modules_as_functions=True,
            inline=False,  # function do not retain shape information
            dynamic_shapes=dyn_shapes,
            return_builder=True,
        )
        text = builder.pretty_text()
        filename = "test_phi35_mistruct_custom_cuda_mod_dynshapes.onnx"
        with open("test_get_phi35_mini_instruct_custom_cuda_modules_dynshapes.txt", "w") as f:
            f.write(text)
        onx.save(filename, all_tensors_to_one_file=True)
        onx.inline = True
        filename = "test_phi35_mistruct_custom_cuda_mod_dynshapes.inline.onnx"
        onx.save(filename, all_tensors_to_one_file=True)

        new_onx = onnx.load(filename, load_external_data=True)
        del new_onx.graph.value_info[:]
        new_onx = onnx.shape_inference.infer_shapes(new_onx)
        filename = "test_phi35_mistruct_custom_cuda_mod_dynshapes.inline.reshaped.onnx"
        onnx.save(new_onx, filename, save_as_external_data=True, all_tensors_to_one_file=True)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            filename, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        got = sess.run(None, {k: v.cpu().numpy() for k, v in model_inputs.items()})
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            self.assertEqualArray(a, b, atol=1e-2)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @requires_torch("2.6", "bug")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @long_test()
    def test_get_phi35_mini_instruct_auto(self):
        import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_mini_instruct,
        )
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )

        with bypass_export_some_errors():
            data = get_phi35_mini_instruct(num_hidden_layers=1)
            model, model_inputs = data["model"], data["inputs"]
            model = model
            with torch.autocast(device_type="cpu", dtype=torch.float16):
                onx = to_onnx(
                    model,
                    None,  # args
                    model_inputs,  # kwargs
                    large_model=True,
                    verbose=0,
                    options=OptimizationOptions(max_iter=10),
                    export_options=ExportOptions(strict=False, decomposition_table="all"),
                )
            filename = "test_phi35_mini_instruct_custom_auto.onnx"
            onx.save(filename, all_tensors_to_one_file=True)
            m = onnx.load(filename, load_external_data=False)
            self.assertEqual(
                set(i.type.tensor_type.elem_type for i in m.graph.output),
                {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16},
            )

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @requires_torch("2.7", "no decompositions leads to inplace functions")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @long_test()
    def test_get_phi35_mini_instruct_no_decomposition(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_mini_instruct,
        )

        model, model_inputs = get_phi35_mini_instruct(num_hidden_layers=1)
        expected = list(flatten_outputs(model(**model_inputs)))
        onx = to_onnx(
            model,
            None,  # args
            model_inputs,  # kwargs
            large_model=True,
            verbose=0,
            options=OptimizationOptions(max_iter=10),
            export_options=ExportOptions(strict=False),
        )
        filename = "test_phi35_mini_instruct_no_decomposition_custom.onnx"
        onx.save(filename, all_tensors_to_one_file=True)
        import onnxruntime

        sess = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])
        got = sess.run(None, {k: v.numpy() for k, v in model_inputs.items()})
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            self.assertEqualArray(a, b, atol=1e-4)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @long_test()
    def test_get_ai21_jamba_15_mini(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_ai21_jamba_15_mini,
        )

        data = get_ai21_jamba_15_mini(num_hidden_layers=1)
        model, model_inputs = data["model"], data["inputs"]
        expected = list(flatten_outputs(model(**model_inputs)))
        self.assertNotEmpty(expected)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @long_test()
    def test_get_falcon_mamba_7b(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_falcon_mamba_7b,
        )

        data = get_falcon_mamba_7b(num_hidden_layers=1)
        model, model_inputs = data["model"], data["inputs"]
        expected = list(flatten_outputs(model(**model_inputs)))
        self.assertNotEmpty(expected)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @long_test()
    def test_get_all_mini_ml_l6_v1(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_all_mini_ml_l6_v1,
        )

        data = get_all_mini_ml_l6_v1(num_hidden_layers=1)
        model, model_inputs = data["model"], data["inputs"]
        expected = list(flatten_outputs(model(**model_inputs)))
        self.assertNotEmpty(expected)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @requires_torch("2.6")  # torch.export.Dim.DYNAMIC
    @long_test()
    def test_get_smollm_1_7b(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_smollm_1_7b,
        )

        data = get_smollm_1_7b(num_hidden_layers=1)
        model, model_inputs = data["model"], data["inputs"]
        expected = list(flatten_outputs(model(**model_inputs)))
        self.assertNotEmpty(expected)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @long_test()
    def test_get_phi3_vision_128k_instruct(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi3_vision_128k_instruct,
        )

        try:
            model, model_inputs = get_phi3_vision_128k_instruct(num_hidden_layers=1)
        except ImportError as e:
            raise unittest.SkipTest(f"missing file: {e}")
        expected = list(flatten_outputs(model(**model_inputs)))
        self.assertNotEmpty(expected)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @long_test()
    def test_get_phi35_mini_instruct_cache_export(self):
        import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_mini_instruct,
        )
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )

        model, model_inputs = get_phi35_mini_instruct(num_hidden_layers=1)

        with bypass_export_some_errors():
            exported_program = torch.export.export(model, tuple(), model_inputs, strict=False)
        self.assertNotEmpty(exported_program)

    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @unittest.skip("Wrong setting for the images")
    @long_test()
    def test_get_llama32_9b_vision(self):
        import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_llama32_9b_vision,
        )
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )

        model, model_inputs = get_llama32_9b_vision(
            num_hidden_layers=1,
            num_global_layers=1,
            rope_scaling={
                "type": "default",
                "rope_type": "default",
                "mrope_section": [2, 1, 1],
            },
        )
        with bypass_export_some_errors():
            exported_program = torch.export.export(model, tuple(), model_inputs, strict=False)
        self.assertNotEmpty(exported_program)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @long_test()
    def test_get_phi35_vision_instruct(self):
        import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_vision_instruct,
        )
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )

        data = get_phi35_vision_instruct(num_hidden_layers=1)
        model, model_inputs = data["model"], data["inputs"]

        with bypass_export_some_errors(
            patch_transformers=True, replace_dynamic_cache=True
        ) as modificator:
            exported_program = torch.export.export(model, tuple(), model_inputs, strict=True)
            onx = to_onnx(model, tuple(), modificator(model_inputs))
            onnx.save(onx, "test_get_phi35_vision_instruct.onnx")
        self.assertNotEmpty(exported_program)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @long_test()
    def test_get_phi35_vision_instruct_input_kind(self):
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_vision_instruct,
            LLMInputKind,
        )

        data = get_phi35_vision_instruct(
            num_hidden_layers=1, input_kind=LLMInputKind.input_ids, common_dynamic_shapes=True
        )
        model, model_inputs, dyn_shapes = data["model"], data["inputs"], data["dynamic_shapes"]
        self.assertEqual(list(model_inputs), ["input_ids"])
        self.assertEqual(list(dyn_shapes), ["input_ids"])

        data = get_phi35_vision_instruct(
            num_hidden_layers=1,
            input_kind=LLMInputKind.input_ids | LLMInputKind.position_ids,
            common_dynamic_shapes=True,
        )
        model, model_inputs, dyn_shapes = data["model"], data["inputs"], data["dynamic_shapes"]
        self.assertEqual(list(model_inputs), ["input_ids", "position_ids"])
        self.assertEqual(list(dyn_shapes), ["input_ids", "position_ids"])

        model, model_inputs, dyn_shapes = get_phi35_vision_instruct(
            num_hidden_layers=1, input_kind=LLMInputKind.ALL, common_dynamic_shapes=True
        )
        self.assertEqual(
            list(model_inputs),
            [
                "input_ids",
                "position_ids",
                "attention_mask",
                "past_key_values",
                "pixel_values",
                "image_sizes",
            ],
        )
        self.assertEqual(
            list(dyn_shapes),
            [
                "input_ids",
                "position_ids",
                "attention_mask",
                "past_key_values",
                "pixel_values",
                "image_sizes",
            ],
        )

    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @skipif_ci_windows("not supported")
    @requires_torch("2.6")  # for torch.export.Dim.DYNAMIC
    @long_test()
    def test_get_phi2(self):
        import torch
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )
        from experimental_experiment.torch_models.llm_model_helper import get_phi2

        for n_iter in [1, 0]:
            for ds in [True, False]:
                with self.subTest(input_cache=n_iter, ds=ds):
                    if ds:
                        res = get_phi2(
                            num_hidden_layers=1,
                            input_cache=n_iter == 1,
                            common_dynamic_shapes=ds,
                            intermediate_size=5120,
                            # hidden_size=1280,
                            batch_size=2,
                        )
                        model, model_inputs, dyn_shapes = (
                            res["model"],
                            res["inputs"],
                            res["dynamic_shapes"],
                        )
                        model(**copy.deepcopy(model_inputs))
                        with bypass_export_some_errors(
                            patch_transformers=True, replace_dynamic_cache=True
                        ) as modificator:
                            torch.export.export(
                                model,
                                (),
                                kwargs=modificator(model_inputs),
                                dynamic_shapes=dyn_shapes,
                                strict=False,
                            )
                        self.assertIn("dict(", string_type(model_inputs))
                    else:
                        res = get_phi2(
                            num_hidden_layers=1,
                            intermediate_size=5120,
                            # hidden_size=1280,
                            input_cache=n_iter == 1,
                            common_dynamic_shapes=ds,
                            batch_size=2,
                        )
                        model, model_inputs = res["model"], res["inputs"]
                        model(**copy.deepcopy(model_inputs))
                        with bypass_export_some_errors(
                            patch_transformers=True, replace_dynamic_cache=True
                        ) as modificator:
                            torch.export.export(
                                model, (), modificator(model_inputs), strict=False
                            )

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @requires_torch("2.6")  # torch.export.Dim.DYNAMIC
    @long_test()
    def test_get_smollm_1_7b_cache(self):
        import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_smollm_1_7b,
        )
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )

        data = get_smollm_1_7b(
            batch_size=2, num_hidden_layers=1, input_cache=True, common_dynamic_shapes=True
        )
        model, model_inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        expected = list(flatten_outputs(model(**model_inputs)))
        self.assertNotEmpty(expected)
        with bypass_export_some_errors(replace_dynamic_cache=True):
            torch.export.export(model, (), model_inputs, dynamic_shapes=ds, strict=False)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @requires_torch("2.6")  # torch.export.Dim.DYNAMIC
    # @long_test(): let's keep this test to avoid any regression.
    def test_get_phi4_export(self):
        import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi4,
        )
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )

        data = get_phi4(
            batch_size=2, num_hidden_layers=1, input_cache=True, common_dynamic_shapes=True
        )
        model, model_inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        expected = list(flatten_outputs(model(**model_inputs)))
        self.assertNotEmpty(expected)
        with bypass_export_some_errors(
            replace_dynamic_cache=True, catch_constraints=False
        ) as modificator:
            model_inputs = modificator(model_inputs)
            torch.export.export(model, (), model_inputs, dynamic_shapes=ds, strict=False)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @requires_torch("2.6")  # torch.export.Dim.DYNAMIC
    @long_test()
    def test_get_phi4_onnx(self):
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi4,
        )
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )

        data = get_phi4(
            batch_size=2, num_hidden_layers=1, input_cache=True, common_dynamic_shapes=True
        )
        model, model_inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        expected = list(flatten_outputs(model(**model_inputs)))
        self.assertNotEmpty(expected)
        with bypass_export_some_errors(
            replace_dynamic_cache=True, catch_constraints=False
        ) as modificator:
            model_inputs = modificator(model_inputs)
            to_onnx(
                model,
                (),
                model_inputs,
                dynamic_shapes=ds,
                filename="test_get_phi4_onnx.onnx",
                export_options=ExportOptions(strict=False),
                large_model=True,
            )

    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @skipif_ci_windows("not supported")
    @requires_torch("2.6")  # for torch.export.Dim.DYNAMIC
    def test_phi2_output_order_export(self):
        import torch
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )
        from experimental_experiment.torch_models.llm_model_helper import get_phi2

        res = get_phi2(
            num_hidden_layers=2,
            input_cache=True,
            common_dynamic_shapes=True,
            intermediate_size=5120,
            # hidden_size=1280,
            batch_size=2,
        )
        model, model_inputs, dyn_shapes = (res["model"], res["inputs"], res["dynamic_shapes"])
        expected = model(**copy.deepcopy(model_inputs))
        with bypass_export_some_errors(
            patch_transformers=True, replace_dynamic_cache=True
        ) as modificator:
            modified_inputs = modificator(model_inputs)
            ep = torch.export.export(
                model,
                (),
                kwargs=modified_inputs,
                dynamic_shapes=dyn_shapes,
                strict=False,
            )
            mod = ep.module()
            got = mod(**modified_inputs)

        # We check that should be the same order.
        self.assertEqualAny(expected, got)
        flatten_expected = torch.utils._pytree.tree_flatten(expected)[0]
        flatten_got = torch.utils._pytree.tree_flatten(got)[0]
        self.assertEqualAny(flatten_expected, flatten_got)
        diff = max_diff(expected, got)
        self.assertLess(diff["abs"], 1e-5)
        diff = max_diff(expected, flatten_got)
        self.assertLess(diff["abs"], 1e-5)

    @ignore_warnings("TracerWarning")
    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("not supported")
    @requires_torch("2.6")  # for torch.export.Dim.DYNAMIC
    def test_phi2_output_order_onnx_dynamo(self):
        import torch
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )
        from experimental_experiment.torch_models.llm_model_helper import get_phi2

        res = get_phi2(
            num_hidden_layers=2,
            input_cache=True,
            common_dynamic_shapes=True,
            intermediate_size=5120,
            # hidden_size=1280,
            batch_size=2,
        )
        model, model_inputs, dyn_shapes = (res["model"], res["inputs"], res["dynamic_shapes"])
        expected = model(**copy.deepcopy(model_inputs))
        with bypass_export_some_errors(
            patch_transformers=True, replace_dynamic_cache=True
        ) as modificator:
            modified_inputs = modificator(model_inputs)
            ep = torch.onnx.export(
                model,
                (),
                kwargs=modified_inputs,
                dynamic_shapes=dyn_shapes,
                dynamo=True,
            )
            flatten_inputs = torch.utils._pytree.tree_flatten(model_inputs)[0]
            flatten_expected = torch.utils._pytree.tree_flatten(expected)[0]

        from onnxruntime import InferenceSession

        sess = InferenceSession(
            ep.model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = dict(
            zip(
                [i.name for i in sess.get_inputs()],
                [t.detach().cpu().numpy() for t in flatten_inputs],
            )
        )
        got = sess.run(None, feeds)
        self.assertEqualAny([_.detach().numpy() for _ in flatten_expected], got, atol=1e-5)
        diff = max_diff(flatten_expected, got)
        self.assertLess(diff["abs"], 1e-5)
        diff = max_diff(expected, got)
        self.assertLess(diff["abs"], 1e-5)

    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @skipif_ci_windows("not supported")
    @requires_torch("2.6")  # for torch.export.Dim.DYNAMIC
    def test_phi2_output_order_custom(self):
        import torch
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )
        from experimental_experiment.torch_models.llm_model_helper import get_phi2

        res = get_phi2(
            num_hidden_layers=2,
            input_cache=True,
            common_dynamic_shapes=True,
            intermediate_size=5120,
            # hidden_size=1280,
            batch_size=2,
        )
        model, model_inputs, dyn_shapes = (res["model"], res["inputs"], res["dynamic_shapes"])
        expected = model(**copy.deepcopy(model_inputs))
        with bypass_export_some_errors(
            patch_transformers=True, replace_dynamic_cache=True
        ) as modificator:
            flatten_inputs = torch.utils._pytree.tree_flatten(model_inputs)[0]
            modified_inputs = modificator(model_inputs)
            onx = to_onnx(
                model,
                (),
                kwargs=modified_inputs,
                dynamic_shapes=dyn_shapes,
            )
            flatten_expected = torch.utils._pytree.tree_flatten(expected)[0]

        from onnxruntime import InferenceSession

        sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        feeds = dict(
            zip(
                [i.name for i in sess.get_inputs()],
                [t.detach().cpu().numpy() for t in flatten_inputs],
            )
        )
        got = sess.run(None, feeds)
        self.assertEqualAny([_.detach().numpy() for _ in flatten_expected], got, atol=1e-5)
        diff = max_diff(flatten_expected, got)
        self.assertLess(diff["abs"], 1e-5)
        diff = max_diff(expected, got)
        self.assertLess(diff["abs"], 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
