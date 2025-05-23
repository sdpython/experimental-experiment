import unittest
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_torch,
    requires_onnxruntime_training,
    has_cuda,
    skipif_ci_windows,
)
from experimental_experiment.torch_models.llama_helper import get_llama_model
from experimental_experiment.torch_models.mistral_helper import get_mistral_model
from experimental_experiment.torch_test_helper import export_to_onnx, check_model_ort
from experimental_experiment.torch_bench._dort_cmd_common import create_compiled_model
from experimental_experiment.torch_models.training_helper import train_loop
from experimental_experiment.torch_interpreter import ExportOptions


class TestEdMistral(ExtTestCase):
    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings(DeprecationWarning)
    @requires_torch("2.6")
    def test_mistral_export_rename(self):
        import torch

        model, input_tensors = get_mistral_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)
        with torch.no_grad():
            try:
                ret = export_to_onnx(
                    model,
                    *input_tensors,
                    rename_inputs=True,
                    export_options=ExportOptions(decomposition_table="default"),
                )
            except RuntimeError as e:
                if "cannot mutate tensors with frozen storage" in str(e):
                    raise unittest.SkipTest(  # noqa: B904
                        "cannot mutate tensors with frozen storag"
                    )
                raise
        onx = ret["proto"]
        xp = [x.numpy() for x in input_tensors]
        feeds = {f"input{i}": x for i, x in enumerate(xp)}
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, feeds)
        self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)
        if has_cuda():
            sess = check_model_ort(onx, providers="cuda")
            results = sess.run(None, feeds)
            self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=5e-3)

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings(DeprecationWarning)
    @requires_torch("2.6")
    def test_mistral_export_norename(self):
        import torch

        model, input_tensors = get_mistral_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)
        with torch.no_grad():
            try:
                ret = export_to_onnx(
                    model,
                    *input_tensors,
                    rename_inputs=False,
                    export_options=ExportOptions(decomposition_table="default"),
                )
            except RuntimeError as e:
                if "cannot mutate tensors with frozen storage" in str(e):
                    raise unittest.SkipTest(  # noqa: B904
                        "cannot mutate tensors with frozen storag"
                    )
                raise
        onx = ret["proto"]
        names = [i.name for i in onx.graph.input]
        xp = [x.numpy() for x in input_tensors]
        feeds = dict(zip(names, xp))
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, feeds)
        self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)
        if has_cuda():
            sess = check_model_ort(
                onx, providers="cuda", dump_file="test_mistral_export_norename.onnx"
            )
            results = sess.run(None, feeds)
            self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-2)

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch(
        "2.8",
        "AssertionError: original output #6 is None or "
        "issue with torch.ops.prims.convert_element_type.default",
    )
    def test_mistral_cort_static(self):
        model, input_tensors = get_mistral_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)

        compiled_model, storage = create_compiled_model(
            model,
            backend="debug",
            use_dynamic=False,
            target_opset=18,
            verbose=0,
            return_storage=True,
            rename_inputs=True,
        )
        results = compiled_model(*input_tensors)
        self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)
        instances = storage["instance"]
        self.assertEqual(len(instances), 1)  # forward

        train_loop(model, *input_tensors)
        train_loop(compiled_model, *input_tensors)
        instances = storage["instance"]
        self.assertEqual(len(instances), 2)  # forward + backward

        if __name__ == "__main__":
            for i, inst in enumerate(instances):
                self.dump_onnx(f"test_mistral_cort_static_{i}.onnx", inst["onnx"])

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch(
        "2.8",
        "AssertionError: original output #6 is None or "
        "issue with torch.ops.prims.convert_element_type.default",
    )
    def test_mistral_cort_static_norename(self):
        model, input_tensors = get_mistral_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)

        compiled_model, storage = create_compiled_model(
            model,
            backend="debug",
            use_dynamic=False,
            target_opset=18,
            verbose=0,
            return_storage=True,
            rename_inputs=False,
        )
        results = compiled_model(*input_tensors)
        self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)
        instances = storage["instance"]
        self.assertEqual(len(instances), 1)  # forward

        train_loop(model, *input_tensors)
        train_loop(compiled_model, *input_tensors)
        instances = storage["instance"]
        self.assertEqual(len(instances), 2)  # forward + backward

        if __name__ == "__main__":
            for i, inst in enumerate(instances):
                self.dump_onnx(f"test_mistral_cort_static_{i}_norename.onnx", inst["onnx"])

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_onnxruntime_training(True)
    def test_mistral_cort_dynamic_simple(self):
        model, input_tensors = get_mistral_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)

        compiled_model = create_compiled_model(
            model,
            # with backend="ort", it fails in onnxscript
            backend="custom",
            use_dynamic=True,
            target_opset=18,
            verbose=0,
            return_storage=False,
            rename_inputs=True,
            dump_prefix=(
                "test_mistral_cort_dynamic_simple" if __name__ == "__main__" else None
            ),
        )
        results = compiled_model(*input_tensors)
        self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)

        train_loop(model, *input_tensors)
        train_loop(compiled_model, *input_tensors)

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.7", "AssertionError: original output #6 is None")
    @requires_onnxruntime_training(True)
    def test_mistral_cort_dynamic_norename(self):
        model, input_tensors = get_mistral_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)

        compiled_model = create_compiled_model(
            model,
            backend="custom",
            use_dynamic=True,
            target_opset=18,
            verbose=0,
            return_storage=False,
            rename_inputs=False,
        )
        results = compiled_model(*input_tensors)
        self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)

        train_loop(model, *input_tensors)
        train_loop(compiled_model, *input_tensors)

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_onnxruntime_training(True)
    def test_mistral_cort_dynamic_norename_custom(self):
        model, input_tensors = get_llama_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)

        compiled_model, storage = create_compiled_model(
            model,
            backend="custom",
            use_dynamic=True,
            target_opset=18,
            verbose=0,
            return_storage=True,
            rename_inputs=False,
            dump_prefix="test_mistral_cort_dynamic_norename_custom",
        )
        results = compiled_model(*input_tensors)
        self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)
        instances = storage["instance"]
        # self.assertEqual(len(instances), 1)  # forward

        train_loop(model, *input_tensors)
        train_loop(compiled_model, *input_tensors)
        instances = storage["instance"]
        self.assertEqual(len(instances), 2)  # forward + backward

        if __name__ == "__main__":
            for i, inst in enumerate(instances):
                self.dump_onnx(
                    f"test_mistral_cort_dynamic_{i}_norename_custom.onnx", inst["onnx"]
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
