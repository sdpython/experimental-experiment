import itertools
import unittest
from typing import Optional
import numpy as np
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_cuda,
    requires_torch,
)


class TestOnnxExportComplex(ExtTestCase):
    @ignore_warnings((UserWarning, DeprecationWarning))
    @requires_cuda()
    def test_export_polar(self):
        import torch

        class Neuron(torch.nn.Module):
            def forward(self, x, angle):
                return torch.polar(x, angle)

        model, x, angle = Neuron(), torch.rand(4, 4), torch.rand(4, 4)
        expected = model(x, angle)
        onx = to_onnx(model, (x, angle))
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.numpy(), "angle": angle.numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

    def _range(self, *shape, bias: Optional[float] = None):
        import torch

        n = int(np.prod(shape))
        x = torch.arange(n).to(torch.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).to(torch.float32)

    @ignore_warnings(UserWarning)
    @requires_torch("2.7")
    def test_fft_simple_1(self):
        import torch

        for n, dim, norm, opset in itertools.product(
            [None, 3, 8], [-1, 0], [None, "forward", "ortho", "backward"], [20, 17]
        ):
            with self.subTest(n=n, dim=dim, norm=norm):

                class Model(torch.nn.Module):
                    def __init__(self, n, dim, norm):
                        super().__init__()
                        self.n = n
                        self.dim = dim
                        self.norm = norm

                    def forward(self, x):
                        return torch.fft.fft(x, n=self.n, dim=self.dim, norm=self.norm).abs()

                model = Model(n, dim, norm)
                x = self._range(4, 5, bias=1)
                x[:, 2] += torch.tensor(np.arange(x.shape[0]).tolist())
                expected = model(x)
                DYN = torch.export.Dim.DYNAMIC
                ep = torch.export.export(model, (x,), dynamic_shapes=({0: DYN, 1: DYN},))
                # print(ep)
                ep = ep.run_decompositions()
                assert ep
                # print(ep)

                # No decomposition.
                onx = to_onnx(
                    model,
                    (x,),
                    dynamic_shapes=({0: "batch", 1: "length"},),
                    target_opset=opset,
                )
                self.assertEqual(onx.opset_import[0].domain, "")
                self.assertEqual(onx.opset_import[0].version, opset)
                # self.print_model(onx)
                ref = ExtendedReferenceEvaluator(onx, verbose=0)
                got = ref.run(None, {"x": x.numpy()})
                self.assertEqualArray(expected, got[0], atol=1e-5)

                # With decomposition.
                onx = to_onnx(
                    model,
                    (x,),
                    dynamic_shapes=({0: "batch", 1: "length"},),
                    export_options=ExportOptions("all"),
                    target_opset=opset,
                )
                # self.print_model(onx)
                ref = ExtendedReferenceEvaluator(onx)
                got = ref.run(None, {"x": x.numpy()})
                self.assertEqualArray(expected, got[0], atol=1e-5)

    @ignore_warnings(UserWarning)
    @requires_torch("2.7")
    def test_fft_simple_2(self):
        import torch

        for n, dim, norm, opset in itertools.product(
            [None, [3, 2], [8, 7]], [0, 1], [None, "forward", "ortho"], [20, 17]
        ):
            with self.subTest(n=n, dim=dim, norm=norm):

                class Model(torch.nn.Module):
                    def __init__(self, s, dim, norm):
                        super().__init__()
                        self.s = s
                        self.dim = dim
                        self.norm = norm

                    def forward(self, x):
                        return torch.fft.fft2(
                            x, s=self.s, dim=(self.dim, self.dim + 1), norm=self.norm
                        ).abs()

                model = Model(n, dim, norm)
                x = self._range(4, 5, 5, bias=1)
                expected = model(x)
                DYN = torch.export.Dim.DYNAMIC
                ep = torch.export.export(model, (x,), dynamic_shapes=({0: DYN, 1: DYN, 2: DYN},))
                # print(ep)
                ep = ep.run_decompositions()
                assert ep
                # print(ep)

                # No decomposition.
                onx = to_onnx(
                    model,
                    (x,),
                    dynamic_shapes=({0: "batch", 1: "length"},),
                    target_opset=opset,
                )
                self.assertEqual(onx.opset_import[0].domain, "")
                self.assertEqual(onx.opset_import[0].version, opset)
                # self.print_model(onx)
                ref = ExtendedReferenceEvaluator(onx, verbose=0)
                got = ref.run(None, {"x": x.numpy()})
                self.assertEqualArray(expected, got[0], atol=1e-5)

                # With decomposition.
                onx = to_onnx(
                    model,
                    (x,),
                    dynamic_shapes=({0: "batch", 1: "length"},),
                    export_options=ExportOptions("all"),
                    target_opset=opset,
                )
                # self.print_model(onx)
                ref = ExtendedReferenceEvaluator(onx)
                got = ref.run(None, {"x": x.numpy()})
                self.assertEqualArray(expected, got[0], atol=1e-5)

    @ignore_warnings(UserWarning)
    @requires_torch("2.7")
    def test_ifft_simple_1(self):
        import torch

        for n, dim, norm, opset in itertools.product(
            [None, 3, 8], [-1, 0], [None, "backward", "forward", "ortho"], [20, 17]
        ):
            with self.subTest(n=n, dim=dim, norm=norm):

                class Model(torch.nn.Module):
                    def __init__(self, n, dim, norm):
                        super().__init__()
                        self.n = n
                        self.dim = dim
                        self.norm = norm

                    def forward(self, x):
                        return torch.fft.ifft(x, n=self.n, dim=self.dim, norm=self.norm).abs()

                model = Model(n, dim, norm)
                x = self._range(4, 5, bias=1)
                x[:, 2] += torch.tensor(np.arange(x.shape[0]).tolist())
                expected = model(x)
                DYN = torch.export.Dim.DYNAMIC
                ep = torch.export.export(model, (x,), dynamic_shapes=({0: DYN, 1: DYN},))
                # print(ep)
                ep = ep.run_decompositions()
                assert ep
                # print(ep)

                # No decomposition.
                onx = to_onnx(
                    model,
                    (x,),
                    dynamic_shapes=({0: "batch", 1: "length"},),
                    target_opset=opset,
                )
                self.assertEqual(onx.opset_import[0].domain, "")
                self.assertEqual(onx.opset_import[0].version, opset)
                # self.print_model(onx)
                ref = ExtendedReferenceEvaluator(onx, verbose=0)
                got = ref.run(None, {"x": x.numpy()})
                self.assertEqualArray(expected, got[0], atol=1e-5)

                # With decomposition.
                onx = to_onnx(
                    model,
                    (x,),
                    dynamic_shapes=({0: "batch", 1: "length"},),
                    export_options=ExportOptions("all"),
                    target_opset=opset,
                )
                # self.print_model(onx)
                ref = ExtendedReferenceEvaluator(onx)
                got = ref.run(None, {"x": x.numpy()})
                self.assertEqualArray(expected, got[0], atol=1e-5)

    @ignore_warnings(UserWarning)
    @requires_torch("2.7")
    def test_ifft_simple_2(self):
        import torch

        for n, dim, norm, opset in itertools.product(
            [None, [3, 2], [8, 7]], [0, 1], [None, "forward", "ortho", "backward"], [20, 17]
        ):
            with self.subTest(n=n, dim=dim, norm=norm):

                class Model(torch.nn.Module):
                    def __init__(self, s, dim, norm):
                        super().__init__()
                        self.s = s
                        self.dim = dim
                        self.norm = norm

                    def forward(self, x):
                        return torch.fft.ifft2(
                            x, s=self.s, dim=(self.dim, self.dim + 1), norm=self.norm
                        ).abs()

                model = Model(n, dim, norm)
                x = self._range(4, 5, 5, bias=1)
                expected = model(x)
                DYN = torch.export.Dim.DYNAMIC
                ep = torch.export.export(model, (x,), dynamic_shapes=({0: DYN, 1: DYN, 2: DYN},))
                # print(ep)
                ep = ep.run_decompositions()
                assert ep
                # print(ep)

                # No decomposition.
                onx = to_onnx(
                    model,
                    (x,),
                    dynamic_shapes=({0: "batch", 1: "length"},),
                    target_opset=opset,
                )
                self.assertEqual(onx.opset_import[0].domain, "")
                self.assertEqual(onx.opset_import[0].version, opset)
                # self.print_model(onx)
                ref = ExtendedReferenceEvaluator(onx, verbose=0)
                got = ref.run(None, {"x": x.numpy()})
                self.assertEqualArray(expected, got[0], atol=1e-5)

                # With decomposition.
                onx = to_onnx(
                    model,
                    (x,),
                    dynamic_shapes=({0: "batch", 1: "length"},),
                    export_options=ExportOptions("all"),
                    target_opset=opset,
                )
                # self.print_model(onx)
                ref = ExtendedReferenceEvaluator(onx)
                got = ref.run(None, {"x": x.numpy()})
                self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
