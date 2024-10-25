import unittest
import onnx
from experimental_experiment.ext_test_case import ExtTestCase, ignore_warnings, requires_cuda
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_models import flatten_outputs
from experimental_experiment.torch_models.phi3_helper import has_phi3
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


class TestLlm(ExtTestCase):
    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    def test_get_phi_35_mini_instruct(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi_35_mini_instruct,
        )

        model, model_inputs = get_phi_35_mini_instruct(num_hidden_layers=1)
        expected = list(flatten_outputs(model(**model_inputs)))
        # torch.onnx.export(
        #    model,
        #    (model_inputs["input_ids"], model_inputs["attention_mask"]),
        #    "test_get_phi_35_mini_instruct_onnx_dynamo.onnx",
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
        filename = "test_get_phi_35_mini_instruct_custom.onnx"
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
    @requires_cuda()
    def test_get_phi_35_mini_instruct_cuda(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi_35_mini_instruct,
        )

        model, model_inputs = get_phi_35_mini_instruct(num_hidden_layers=1)
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
        filename = "test_get_phi_35_mini_instruct_custom_cuda.onnx"
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
    def test_get_phi_35_mini_instruct_auto(self):
        import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi_35_mini_instruct,
        )

        model, model_inputs = get_phi_35_mini_instruct(num_hidden_layers=1)
        model = model
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            onx = to_onnx(
                model,
                None,  # args
                model_inputs,  # kwargs
                large_model=True,
                verbose=0,
                options=OptimizationOptions(max_iter=10),
                export_options=ExportOptions(strict=False, decomposition_table="all"),
            )
        filename = "test_get_phi_35_mini_instruct_custom_auto.onnx"
        onx.save(filename, all_tensors_to_one_file=True)
        m = onnx.load(filename, load_external_data=False)
        self.assertEqual(
            set(i.type.tensor_type.elem_type for i in m.graph.output),
            {onnx.TensorProto.BFLOAT16},
        )

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    def test_get_phi_35_mini_instruct_no_decomposition(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi_35_mini_instruct,
        )

        model, model_inputs = get_phi_35_mini_instruct(num_hidden_layers=1)
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
        filename = "test_get_phi_35_mini_instruct_no_decomposition_custom.onnx"
        onx.save(filename, all_tensors_to_one_file=True)
        import onnxruntime

        sess = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])
        got = sess.run(None, {k: v.numpy() for k, v in model_inputs.items()})
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            self.assertEqualArray(a, b, atol=1e-4)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    def test_get_ai21_jamba_15_mini(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_ai21_jamba_15_mini,
        )

        model, model_inputs = get_ai21_jamba_15_mini(num_hidden_layers=1)
        expected = list(flatten_outputs(model(**model_inputs)))
        self.assertNotEmpty(expected)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    def test_get_falcon_mamba_7b(self):
        # import torch
        from experimental_experiment.torch_models.llm_model_helper import (
            get_falcon_mamba_7b,
        )

        model, model_inputs = get_falcon_mamba_7b(num_hidden_layers=1)
        expected = list(flatten_outputs(model(**model_inputs)))
        self.assertNotEmpty(expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
