import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_torch,
    requires_onnxruntime_training,
    has_cuda,
    skipif_ci_windows,
)
from experimental_experiment.torch_models.phi_helper import get_phi_model
from experimental_experiment.torch_bench._dort_cmd_common import create_compiled_model
from experimental_experiment.torch_models.training_helper import (
    train_loop,
    train_loop_mixed_precision,
)


class TestEdPhi(ExtTestCase):
    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_phi_cort_static_not_mixed(self):
        model, input_tensors = get_phi_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)

        compiled_model, storage = create_compiled_model(
            model,
            backend="debug",
            use_dynamic=False,
            target_opset=18,
            verbose=0,  # (0, 10),
            return_storage=True,
            rename_inputs=False,
            # dump_prefix="test_phi_cort_static",
            # disable_pattern="MatMulReshape2Of3",
            optimize=True,
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
                self.dump_onnx(f"test_phi_cort_static_{i}.onnx", inst["onnx"])

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning, RuntimeWarning))
    @requires_torch("2.4")
    @unittest.skipIf(not has_cuda(), reason="requires cuda")
    def test_phi_cort_static_mixed_debug(self):
        import torch

        model, input_tensors = get_phi_model()
        model = model.to("cuda")
        input_tensors = [tuple([i.to("cuda") for i in inp]) for inp in input_tensors]
        input_tensors = input_tensors[0]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            torch.cuda.synchronize()
            expected = model(*input_tensors)
            torch.cuda.synchronize()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            compiled_model, storage = create_compiled_model(
                model,
                backend="debug",
                use_dynamic=False,
                target_opset=18,
                verbose=0,  # (0, 10),
                return_storage=True,
                rename_inputs=False,
                # dump_prefix="test_phi",
                optimize=True,
            )
            torch.cuda.synchronize()
            results = compiled_model(*input_tensors)
            torch.cuda.synchronize()
        self.assertEqualArray(expected[0].detach().cpu().numpy(), results[0], atol=1e-2)
        instances = storage["instance"]
        self.assertEqual(len(instances), 1)  # forward

        train_loop_mixed_precision(model, *input_tensors)
        train_loop_mixed_precision(compiled_model, *input_tensors)
        instances = storage["instance"]
        self.assertEqual(len(instances), 2)  # forward + backward

        if __name__ == "__main__":
            for i, inst in enumerate(instances):
                self.dump_onnx(f"test_phi_cort_static_mixed_{i}.onnx", inst["onnx"])

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    @unittest.skipIf(not has_cuda(), reason="requires cuda")
    def test_phi_cort_static_mixed_ort(self):
        import torch

        model, input_tensors = get_phi_model()
        model = model.to("cuda")
        input_tensors = [tuple([i.to("cuda") for i in inp]) for inp in input_tensors]
        input_tensors = input_tensors[0]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            torch.cuda.synchronize()
            expected = model(*input_tensors)
            torch.cuda.synchronize()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            compiled_model, storage = create_compiled_model(
                model,
                backend="custom",
                use_dynamic=False,
                target_opset=18,
                verbose=0,  # (0, 10),
                return_storage=True,
                rename_inputs=False,
                dump_prefix="test_phi_ort",
                optimize=True,
            )
            torch.cuda.synchronize()
            results = compiled_model(*input_tensors)
            torch.cuda.synchronize()
        self.assertEqualArray(expected[0].detach().cpu().numpy(), results[0], atol=1e-2)
        instances = storage["instance"]
        self.assertEqual(len(instances), 1)  # forward

        train_loop_mixed_precision(model, *input_tensors)
        train_loop_mixed_precision(compiled_model, *input_tensors)
        instances = storage["instance"]
        self.assertEqual(len(instances), 2)  # forward + backward

        if __name__ == "__main__":
            for i, inst in enumerate(instances):
                self.dump_onnx(f"test_phi_cort_static_mixed_{i}.onnx", inst["onnx"])

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_onnxruntime_training(True)
    def test_phi_cort_dynamic(self):
        model, input_tensors = get_phi_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)

        compiled_model, storage = create_compiled_model(
            model,
            backend="custom",
            use_dynamic=True,
            target_opset=18,
            verbose=0,  # (0, 10),
            return_storage=True,
            rename_inputs=False,
            # dump_prefix="test_phi",
            optimize=True,
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
                self.dump_onnx(f"test_phi_cort_dynamic_{i}.onnx", inst["onnx"])

    """
    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.3", "AssertionError: original output #6 is None")
    def test_phi_cort_static(self):
        model, input_tensors = get_phi_model()
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
                self.dump_onnx(f"test_phi_cort_static_{i}.onnx", inst["onnx"])

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.3", "AssertionError: original output #6 is None")
    def test_phi_cort_static_norename(self):
        model, input_tensors = get_phi_model()
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
                self.dump_onnx(
                    f"test_phi_cort_static_{i}_norename.onnx", inst["onnx"]
                )

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.3", "AssertionError: original output #6 is None")
    def test_phi_cort_dynamic(self):
        model, input_tensors = get_phi_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)

        compiled_model, storage = create_compiled_model(
            model,
            backend="debug",
            use_dynamic=True,
            target_opset=18,
            verbose=0,
            return_storage=True,
            rename_inputs=True,
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
                self.dump_onnx(f"test_phi_cort_dynamic_{i}.onnx", inst["onnx"])

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.3", "AssertionError: original output #6 is None")
    def test_phi_cort_dynamic_norename(self):
        model, input_tensors = get_phi_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)

        compiled_model, storage = create_compiled_model(
            model,
            backend="debug",
            use_dynamic=True,
            target_opset=18,
            verbose=0,
            return_storage=True,
            rename_inputs=False,
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
                    f"test_phi_cort_dynamic_{i}_norename.onnx", inst["onnx"]
                )

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.3", "AssertionError: original output #6 is None")
    def test_phi_cort_dynamic_norename_custom(self):
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
                    f"test_phi_cort_dynamic_{i}_norename_custom.onnx", inst["onnx"]
                )
        """


if __name__ == "__main__":
    unittest.main(verbosity=2)
