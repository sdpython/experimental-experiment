"""
Note:
::

    transformers/modeling_attn_mask_utils.py:172:
            mask = mask.masked_fill(context_mask, torch.finfo(dtype).min)
"""

import unittest
import torch
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_cuda,
    long_test,
)
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_bench.big_models.try_phi35_vision import (
    get_model_inputs,
)


class TestPhi35Vision(ExtTestCase):
    @long_test()
    @hide_stdout()
    @requires_cuda(memory=24)
    def test_get_model_inputs(self):
        model_fct, inputs = get_model_inputs(device="cuda", dtype="float16", verbose=1)
        model = model_fct()
        model(*inputs)

    @long_test()
    @requires_cuda(memory=24)
    def test_export_export(self):
        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        with torch.no_grad():
            model(*inputs)
            export_program = torch.export.export(model, inputs, strict=False)
        self.assertNotEmpty(export_program)

    @long_test()
    @requires_cuda(memory=24)
    def test_export_onnx_dynamo_f16(self):
        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        torch.onnx.export(model, inputs, "test_phi35_vision_dynamo_f16.onnx", dynamo=True)

    @long_test()
    @requires_cuda(memory=24)
    def test_export_onnx_dynamo_auto(self):
        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="auto")
        model = model_fct()
        torch.onnx.export(model, inputs, "test_phi35_vision_dynamo_auto.onnx", dynamo=True)

    @long_test()
    @requires_cuda(memory=24)
    def test_export_onnx_custom_f16(self):

        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        onx = to_onnx(
            model,
            inputs,
            large_model=True,
            verbose=1,
            options=OptimizationOptions(max_iter=10),
            export_options=ExportOptions(strict=False),
        )
        onx.save("test_phi35_vision_custom_f16.onnx", all_tensors_to_one_file=True)

    @long_test()
    @requires_cuda(memory=24)
    def test_export_onnx_custom_f16_submodule(self):

        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="float16")
        model = model_fct()
        with torch.no_grad():
            model(*inputs)
            onx = to_onnx(
                model,
                inputs,
                large_model=True,
                verbose=1,
                options=OptimizationOptions(max_iter=10),
                export_options=ExportOptions(strict=False),
                export_modules_as_functions=True,
            )
            onx.save(
                "test_phi35_vision_custom_f16_submodule.onnx", all_tensors_to_one_file=True
            )

    @long_test()
    @requires_cuda(memory=24)
    def test_export_onnx_custom_auto(self):
        model_fct, inputs = get_model_inputs(verbose=1, device="cuda", dtype="auto")
        model = model_fct()
        onx = to_onnx(
            model,
            inputs,
            large_model=True,
            verbose=1,
            options=OptimizationOptions(max_iter=10),
            export_options=ExportOptions(strict=False),  # , decomposition_table="all"),
        )
        onx.save("test_phi35_vision_custom_auto.onnx", all_tensors_to_one_file=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
