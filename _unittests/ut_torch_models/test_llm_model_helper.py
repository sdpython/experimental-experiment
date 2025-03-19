import copy
import unittest
import onnx
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_cuda,
    skipif_ci_windows,
    long_test,
    never_test,
    requires_torch,
    requires_transformers,
)
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_models import flatten_outputs
from experimental_experiment.torch_models.phi3_helper import has_phi3
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
from experimental_experiment.helpers import string_type


class TestLlmModelHelper(ExtTestCase):
    def setUp(self):
        import torch

        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        import torch

        super().tearDown()
        torch._dynamo.reset()

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @skipif_ci_windows("not supported")
    @long_test()
    @unittest.skip("can't evaluate if seq_len > original_max_position_embeddings:")
    def test_get_phi35_mini_instruct(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_mini_instruct,
        )
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )

        data = get_phi35_mini_instruct(num_hidden_layers=1)
        model, model_inputs = data["model"], data["inputs"]
        expected = list(flatten_outputs(model(**model_inputs)))
        # torch.onnx.export(
        #    model,
        #    (model_inputs["input_ids"], model_inputs["attention_mask"]),
        #    "test_get_phi35_mini_instruct_onnx_dynamo.onnx",
        #    dynamo=True,
        # )
        with bypass_export_some_errors():
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
    @unittest.skip("can't evaluate if seq_len > original_max_position_embeddings:")
    def test_get_phi35_mini_instruct_cuda(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_mini_instruct,
        )
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )

        data = get_phi35_mini_instruct(num_hidden_layers=1)
        model, model_inputs = data["model"], data["inputs"]
        model = model.to("cuda")
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to("cuda")
        model.model.rotary_emb.original_inv_freq = model.model.rotary_emb.original_inv_freq.to(
            "cuda"
        )
        model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}
        if "past_key_values" in model_inputs:
            cache = model_inputs["past_key_values"]
            cache.key_cache = [t.to("cuda") for t in cache.key_cache]
            cache.value_cache = [t.to("cuda") for t in cache.value_cache]
        expected = list(flatten_outputs(model(**model_inputs)))
        with bypass_export_some_errors():
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
    @unittest.skip("can't evaluate if seq_len > original_max_position_embeddings:")
    def test_get_phi35_mini_instruct_cuda_modules(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_mini_instruct,
        )
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )

        data = get_phi35_mini_instruct(num_hidden_layers=1)
        model, model_inputs = data["model"], data["inputs"]
        model = model.to("cuda")
        model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to("cuda")
        model.model.rotary_emb.original_inv_freq = model.model.rotary_emb.original_inv_freq.to(
            "cuda"
        )
        model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}
        if "past_key_values" in model_inputs:
            cache = model_inputs["past_key_values"]
            cache.key_cache = [t.to("cuda") for t in cache.key_cache]
            cache.value_cache = [t.to("cuda") for t in cache.value_cache]
        expected = list(flatten_outputs(model(**model_inputs)))
        with bypass_export_some_errors():
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
    @unittest.skip("can't evaluate if seq_len > original_max_position_embeddings:")
    def test_get_phi35_mini_instruct_cuda_modules_dynshapes(self):
        import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_mini_instruct,
        )

        data = get_phi35_mini_instruct(num_hidden_layers=1, batch_size=2, input_cache=False)
        model, model_inputs = data["model"], data["inputs"]
        model = model.to("cuda")
        model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}
        expected = model(**model_inputs)
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
        filename = self.get_dump_file(
            "test_phi35_mistruct_custom_cuda_mod_dynshapes.inline.reshaped.onnx"
        )
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
    @unittest.skip("can't evaluate if seq_len > original_max_position_embeddings:")
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
    @unittest.skip("can't evaluate if seq_len > original_max_position_embeddings:")
    def test_get_phi35_mini_instruct_no_decomposition(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_mini_instruct,
        )

        data = get_phi35_mini_instruct(num_hidden_layers=1, input_cache=False)
        model, model_inputs = data["model"], data["inputs"]
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
    @unittest.skip("Uses HybridMambaAttentionDynamicCache in 4.48.0")
    def test_get_ai21_jamba_15_mini(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_ai21_jamba_15_mini,
        )

        data = get_ai21_jamba_15_mini(num_hidden_layers=1)
        model, model_inputs = data["model"], data["inputs"]
        print(model_inputs)
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
    @unittest.skip("can't evaluate if seq_len > original_max_position_embeddings:")
    def test_get_phi35_mini_instruct_cache_export(self):
        import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi35_mini_instruct,
        )
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )

        data = get_phi35_mini_instruct(num_hidden_layers=1)
        model, model_inputs = data["model"], data["inputs"]

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
            onnx.save(onx, self.get_dump_file("test_get_phi35_vision_instruct.onnx"))
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
        _model, model_inputs, dyn_shapes = (
            data["model"],
            data["inputs"],
            data["dynamic_shapes"],
        )
        self.assertEqual(list(model_inputs), ["input_ids"])
        self.assertEqual(list(dyn_shapes), ["input_ids"])

        data = get_phi35_vision_instruct(
            num_hidden_layers=1,
            input_kind=LLMInputKind.input_ids | LLMInputKind.position_ids,
            common_dynamic_shapes=True,
        )
        self.assertEqual(list(model_inputs), ["input_ids", "position_ids"])
        self.assertEqual(list(dyn_shapes), ["input_ids", "position_ids"])

        data = get_phi35_vision_instruct(
            num_hidden_layers=1, input_kind=LLMInputKind.ALL, common_dynamic_shapes=True
        )
        _model, model_inputs, dyn_shapes = (
            data["model"],
            data["inputs"],
            data["dynamic_shapes"],
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
    def test_b_get_phi4_export(self):
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

    @never_test()
    def test_spy_tiny_llm(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        MODEL_NAME = "arnir0/Tiny-LLM"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        inputs_iteration = []

        def rewrite_forward(f, *args, **kwargs):
            print(f"------------- test_spy_tiny_llm -- iteration {len(inputs_iteration)}")
            print(f"args: {string_type(args, with_shape=True, with_min_max=True)}")
            print(f"kwargs: {string_type(kwargs, with_shape=True, with_min_max=True)}")
            inputs_iteration.append((args, kwargs))
            if len(inputs_iteration) > 5:
                raise unittest.SkipTest(
                    f"Not necessary to go beyond {len(inputs_iteration)} iterations."
                )
            return f(*args, **kwargs)

        model_forward = model.forward
        model.forward = lambda *args, f=model_forward, **kwargs: rewrite_forward(
            f, *args, **kwargs
        )

        def generate_text(
            prompt, model, tokenizer, max_length=512, temperature=1, top_k=50, top_p=0.95
        ):
            inputs = tokenizer.encode(prompt, return_tensors="pt")

            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        prompt = (
            "According to all known laws of aviation, "
            "there is no way a bee should be able to fly."
        )

        generated_text = generate_text(prompt, model, tokenizer)
        if __name__ == "__main__":
            print(f"test_spy_tiny_llm={generated_text}")
        model.forward = model_forward

    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @requires_torch("2.6")  # torch.export.Dim.DYNAMIC
    @requires_transformers("4.49.9999")
    def test_a_get_tiny_llm_default_rope(self):
        """Somehow putting this test after test_get_phi4_export makes it fail."""
        import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_tiny_llm,
        )
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )

        data = get_tiny_llm(
            batch_size=2, num_hidden_layers=1, input_cache=True, common_dynamic_shapes=True
        )
        model, model_inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        expected = list(flatten_outputs(model(**model_inputs)))
        self.assertNotEmpty(expected)
        with bypass_export_some_errors(replace_dynamic_cache=False):
            torch.export.export(model, (), model_inputs, dynamic_shapes=ds, strict=False)

    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @requires_transformers("4.51")  # handle dynamic rope
    def test_a_get_tiny_llm_dynamic_rope(self):
        import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_tiny_llm,
        )
        from experimental_experiment.torch_interpreter.onnx_export_errors import (
            bypass_export_some_errors,
        )

        data = get_tiny_llm(
            batch_size=2,
            num_hidden_layers=1,
            input_cache=True,
            common_dynamic_shapes=True,
            dynamic_rope=True,
        )
        model, model_inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        expected = list(flatten_outputs(model(**model_inputs)))
        self.assertNotEmpty(expected)
        with bypass_export_some_errors(replace_dynamic_cache=False):
            torch.export.export(model, (), model_inputs, dynamic_shapes=ds, strict=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
