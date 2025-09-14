import os
import unittest
import warnings
from onnx.reference import ReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
)
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.helpers import pretty_onnx


class TestOnnxExportLarge(ExtTestCase):

    def return_module_cls_relu(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch
            from torch import nn

        class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv1 = nn.Conv2d(1, 128, 5)

            def forward(self, x):
                return torch.relu(self.conv1(x))

        input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)
        return MyModel(), input_tensor

    def export_utils(
        self,
        prefix,
        model,
        *args,
        remove_unused=False,
        constant_folding=True,
        verbose=0,
        rename_input=True,
        expected_weights=None,
    ):
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        names = []
        name = os.path.join(prefix, "large.onnx")
        if os.path.exists(name):
            os.remove(name)
        large_onx = to_onnx(
            model,
            tuple(args),
            input_names=["input"] if rename_input else None,
            options=OptimizationOptions(
                remove_unused=remove_unused,
                constant_folding=constant_folding,
                verbose=verbose,
                patterns=None,
            ),
            verbose=verbose,
            large_model=True,
        )
        if expected_weights is not None:
            assert len(large_onx.model_proto.graph.initializer) == expected_weights, (
                f"The model has {len(large_onx.model_proto.graph.initializer)} "
                f"initiliazers, expecting {expected_weights}, inputs are "
                f"{[_.name for _ in large_onx.model_proto.graph.input]}."
            )
        large_onx.save(name)
        names.append(name)
        return names

    def check_model_ort(self, name):
        from onnxruntime import InferenceSession

        if isinstance(name, str):
            try:
                InferenceSession(name, providers=["CPUExecutionProvider"])
            except Exception as e:
                import onnx

                raise AssertionError(  # noqa: B904
                    f"onnxruntime cannot load the model "
                    f"due to {e}\n{pretty_onnx(onnx.load(name))}"
                )
            return
        try:
            InferenceSession(name.SerializeToString(), providers=["CPUExecutionProvider"])
        except Exception as e:
            raise AssertionError(  # noqa: B904
                f"onnxruntime cannot load the model due to {e}\n{pretty_onnx(name)}"
            )

    @skipif_ci_windows("torch dynamo not supported on windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_export_as_large_model(self):
        model, input_tensor = self.return_module_cls_relu()
        names = self.export_utils("test_export_as_large_model", model, input_tensor)
        x = input_tensor.numpy()
        results = []
        for name in names:
            ref = ReferenceEvaluator(name)
            results.append(ref.run(None, {"input": x})[0])
            self.check_model_ort(name)
        self.assertEqualArray(results[0], results[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
