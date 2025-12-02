import os
import unittest
import onnx
from experimental_experiment.ext_test_case import ExtTestCase, requires_torch, never_test
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


class TestIssuesPytorch2025Export(ExtTestCase):

    @requires_torch("2.6")
    def test_pads_with_constant_1(self):
        import torch

        def dummy_function(idx, x_len):
            # [1, 2, 3] becomes [1, 2, 3, x_len]
            return torch.nn.functional.pad(idx, (0, 1), value=x_len)

        class Model(torch.nn.Module):
            def forward(self, x, y):
                padded = dummy_function(x, y.shape[0])
                return torch.arange(padded.max())

        model = Model()
        inputs = (
            (torch.arange(3) + 1).to(torch.int64),
            torch.tensor([0, 5], dtype=torch.int64),
        )
        model(*inputs)

        AUTO = torch.export.Dim.AUTO
        ep = torch.export.export(model, inputs, dynamic_shapes={"x": {0: AUTO}, "y": {0: AUTO}})

        epo = torch.onnx.export(ep, dynamo=True)
        epo.optimize()
        epo.save("test_pads_with_constant_1.onnx")
        onx = to_onnx(ep)
        onnx.save(onx, self.get_dump_file("test_pads_with_constant_1.custom.onnx"))
        onnx.checker.check_model(onx)

    @requires_torch("2.6")
    def test_pads_with_constant_2(self):
        import torch

        def dummy_function(idx, x_len):
            # [1, 2, 3] becomes [1, 2, 3, x_len]
            return torch.cat(
                [idx, torch.tensor([x_len], dtype=torch.int64)],
                dim=0,
            )

        class Model(torch.nn.Module):
            def forward(self, x, y):
                padded = dummy_function(x, y.shape[0])
                return torch.arange(padded.max())

        model = Model()
        inputs = (
            (torch.arange(3) + 1).to(torch.int64),
            torch.tensor([0, 5], dtype=torch.int64),
        )
        model(*inputs)

        AUTO = torch.export.Dim.AUTO
        ep = torch.export.export(model, inputs, dynamic_shapes={"x": {0: AUTO}, "y": {0: AUTO}})

        epo = torch.onnx.export(ep, dynamo=True)
        epo.optimize()
        epo.save(self.get_dump_file("test_pads_with_constant_2.onnx"))
        onx = to_onnx(ep)
        onnx.save(onx, self.get_dump_file("test_pads_with_constant_2.custom.onnx"))
        onnx.checker.check_model(onx)

    @requires_torch("2.10")
    def test_multinomial(self):
        # https://github.com/pytorch/pytorch/issues/149048
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.multinomial(x, y.shape[0])

        model = Model()
        inputs = (
            torch.tensor([[4, 5], [6, 7]], dtype=torch.float32),
            torch.tensor([0, 5], dtype=torch.int64),
        )
        model(*inputs)

        DYNAMIC = torch.export.Dim.DYNAMIC
        ep = torch.export.export(
            model, inputs, dynamic_shapes={"x": {0: DYNAMIC, 1: DYNAMIC}, "y": {0: DYNAMIC}}
        )
        epo = torch.onnx.export(ep, dynamo=True)
        epo.optimize()
        epo.save(self.get_dump_file("test_multinomial.onnx"))
        onx = to_onnx(ep)
        onnx.save(onx, self.get_dump_file("test_multinomial.custom.onnx"))
        onnx.checker.check_model(onx)

    # @requires_torch("2.9")
    def test_infer_size_no11_check(self):
        # related to https://github.com/pytorch/pytorch/issues/143495
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, y):
                i = torch.nonzero(x)
                j = torch.nonzero(y)
                torch._check(i.shape[0] == j.shape[0])
                je = j.expand(i.shape)
                return i + je

        model = Model()
        inputs = (
            torch.tensor([[0, 5], [6, 7], [8, 0]], dtype=torch.float32),
            torch.tensor([[5, 0], [6, 7], [0, 8]], dtype=torch.float32),
        )
        model(*inputs)

        DYNAMIC = torch.export.Dim.DYNAMIC
        ep = torch.export.export(
            model,
            inputs,
            dynamic_shapes={"x": {0: DYNAMIC, 1: DYNAMIC}, "y": {0: DYNAMIC, 1: DYNAMIC}},
        )
        assert ep

    @never_test()
    def test_issue_torchvision_166163(self):
        # https://github.com/pytorch/pytorch/issues/166163
        import torch
        import torchvision

        def assert_discrepancies(model_name, model, example_inputs):
            onnx_inputs = [tensor.numpy() for tensor in example_inputs]
            ort_session = self.make_inference_session(model_name)

            onnxruntime_input = {i.name: v for i, v in zip(ort_session.get_inputs(), onnx_inputs)}

            onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
            self.assertEqual(len(onnxruntime_outputs), 1)
            expected = model(*example_inputs)
            self.assertEqualArray(expected, onnxruntime_outputs[0], atol=1e-5)

        models = [
            # torchvision.models.convnext_tiny,
            # torchvision.models.efficientnet_b0,
            torchvision.models.mobilenet_v2,
            # torchvision.models.resnet18,
            # torchvision.models.squeezenet1_0,
            # torchvision.models.vgg11,
        ]
        for cl in models:
            m = cl(weights=None).eval()
            input = torch.rand((3, 224, 224))  # Example input for ResNet
            input_batch = torch.stack([input, input])
            print(f"-- {self.string_type(input_batch, with_shape=True)}")
            prefix = self.get_dump_file(f"test_issue_torchvision_166163_{m.__class__.__name__}")

            # torch-script
            # print("-- export with script")
            # torch.onnx.export(m, (input_batch,), f"{prefix}.script.onnx", dynamo=False)
            # assert_discrepancies(f"{prefix}.script.onnx", m, (input_batch,))
            print("-- export with export")
            ep = torch.export.export(
                m,
                (input_batch,),  # dynamic_shapes=({0: torch.export.Dim.DYNAMIC},)
            )
            with open(f"{prefix}.ep.txt", "w") as f:
                f.write(str(ep))
            self.assertEqualArray(m(input_batch), ep.module()(input_batch))

            print("-- export with custom")
            model_name = f"{prefix}.custom.onnx"
            to_onnx(
                m,
                (input_batch,),
                dynamic_shapes=({0: "dim_x"},),
                export_options=ExportOptions(save_ep=(f"{prefix}.ep", 2**30)),
                filename=model_name,
            )
            torch.save((input_batch,), f"{prefix}.inputs.pt")
            ep = torch.export.load(f"{prefix}.ep.ep.pt2")
            self.assertEqualArray(m(input_batch), ep.module()(input_batch))
            assert_discrepancies(model_name, m, (input_batch,))

            # onnx-dynamo
            print("-- export with dynamo")
            onnx_program = torch.onnx.export(
                m, (input_batch,), dynamic_shapes=({0: "dim_x"},), dynamo=True
            )

            onnx_program.save(f"{prefix}.dynamo.onnx")
            assert_discrepancies(f"{prefix}.dynamo.onnx", m, (input_batch,))

    @never_test()
    def test_issue_169178_resnet18(self):
        import torch
        import torchvision
        from torchvision.models import resnet18, ResNet18_Weights
        import onnxruntime
        import numpy as np
        from pathlib import Path
        from tqdm import tqdm

        def export(exporter: str, model_path: str):
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            model.eval()
            preprocess = ResNet18_Weights.DEFAULT.transforms()

            batch_size = 128
            dummy_input = torch.rand(batch_size, 3, 224, 224)
            dummy_input = preprocess(dummy_input)

            additional_kwargs = {}
            if exporter != "script":
                additional_kwargs["dynamic_shapes"] = ({0: "batch_size"},)
            else:
                additional_kwargs["dynamic_axes"] = {"image": {0: "batch"}}

            if exporter == "dynamo":
                onnx_program = torch.onnx.export(
                    model,
                    dummy_input,
                    export_params=True,
                    opset_version=18,
                    dynamo=True,
                    input_names=["image"],
                    output_names=["predictions"],
                    report=True,
                    **additional_kwargs,
                )
                onnx_program.save(model_path)
            elif exporter == "custom":
                from onnx_diagnostic.export.api import to_onnx

                to_onnx(
                    model,
                    (dummy_input,),
                    filename=model_path,
                    target_opset=18,
                    input_names=["image"],
                    output_names=["predictions"],
                    exporter="custom",
                    verbose=1,
                    dynamic_shapes=({0: "batch_size"},),
                )
            else:
                torch.onnx.export(
                    model,
                    dummy_input,
                    model_path,
                    export_params=True,
                    opset_version=18,
                    dynamo=False,
                    input_names=["image"],
                    output_names=["predictions"],
                    **additional_kwargs,
                )

        def validate(data_path: Path, *onnx_model_paths: str):
            dataset = torchvision.datasets.ImageFolder(
                data_path / "imagenet" / "val", ResNet18_Weights.DEFAULT.transforms()
            )
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

            sesss = [
                onnxruntime.InferenceSession(p, providers=["CPUExecutionProvider"])
                for p in onnx_model_paths
            ]

            dataset_size = len(data_loader)
            print(f"Dataset Size: {dataset_size}")
            errors = []
            ok = []
            for i, (images, _target) in tqdm(
                enumerate(data_loader), total=dataset_size, desc="Validation"
            ):
                predictions = []
                scores = []
                for sess in sesss:
                    output = sess.run([], {"image": images.numpy()})[0]
                    predictions.append(np.argmax(output, axis=1).item())
                    scores.append(np.max(output, axis=1).item())

                unique = set(predictions)
                if len(unique) != 1:
                    errors.append(
                        f"-- disagreement on image {i}, "
                        f"predictions={predictions}, scores={scores}"
                    )
                else:
                    ok.append(f"-- image {i}: {scores}")

            if errors:
                print("\n".join(errors))
            else:
                print("\n".join(ok[:100]))
            assert not errors, f"Issues with the following {len(errors)}"

        def main():
            RESNET18_DYNAMO_FALSE = "resnet18_dynamo_false.onnx"
            RESNET18_DYNAMO_TRUE = "resnet18_dynamo_true.onnx"
            RESNET18_CUSTOM = "resnet18_custom.onnx"
            DATASET_PATH = Path("/home/xadupre/examples/images")  # imagenet/val

            if not os.path.exists(RESNET18_DYNAMO_FALSE) or not os.path.exists(
                RESNET18_DYNAMO_TRUE
            ):
                print("========== Export ==========")
                export("script", model_path=RESNET18_DYNAMO_FALSE)
                export("dynamo", model_path=RESNET18_DYNAMO_TRUE)
                export("custom", model_path=RESNET18_CUSTOM)

            print("========== Validation ==========")
            validate(DATASET_PATH, RESNET18_DYNAMO_FALSE, RESNET18_DYNAMO_TRUE, RESNET18_CUSTOM)

        main()

    @never_test()
    def test_issue_resnet50_168224(self):
        import torch
        import onnxruntime
        from torchvision import models
        from onnx_diagnostic.helpers import max_diff, string_type

        example_inputs = (torch.randn(1, 3, 224, 224),)
        onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]
        torch_model = models.resnet50(pretrained=True).eval()
        print(f"Input length: {len(onnx_inputs)} {string_type(onnx_inputs, with_shape=True)}")

        for exporter in ["script", "onnx-dynamo", "custom"]:
            model_path = self.get_dump_file(f"test_issue_resnet50_168224.{exporter}.onnx")
            if exporter == "onnx-dynamo":
                torch.onnx.export(torch_model, example_inputs, model_path, dynamo=True)
            elif exporter == "script":
                torch.onnx.export(torch_model, example_inputs, model_path, dynamo=False)
            else:
                from onnx_diagnostic.export.api import to_onnx

                to_onnx(
                    torch_model,
                    example_inputs,
                    filename=model_path,
                    target_opset=18,
                    input_names=["image"],
                    output_names=["predictions"],
                    exporter="custom",
                    verbose=0,
                    dynamic_shapes=({0: "batch_size"},),
                )

            ort_session = onnxruntime.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )

            onnxruntime_input = {
                input_arg.name: input_value
                for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)
            }

            onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]
            torch_outputs = torch_model(*example_inputs)
            assert len(torch_outputs) == len(onnxruntime_outputs)
            for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
                diff = max_diff(torch_output, torch.tensor(onnxruntime_output), hist=[0.1])
                print("--", diff)
                torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))
            print("PyTorch and ONNX Runtime output matched!")
            print(f"Output length: {len(onnxruntime_outputs)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
