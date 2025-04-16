import itertools
import unittest
from experimental_experiment.ext_test_case import ExtTestCase, ignore_warnings
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


class TestOnnxExportControlFlow(ExtTestCase):
    @ignore_warnings(UserWarning)
    def test_scan_1(self):
        import torch

        def add(carry: torch.Tensor, y: torch.Tensor):
            next_carry = carry + y
            return [next_carry, next_carry]

        class ScanModel(torch.nn.Module):
            def forward(self, x):
                init = torch.zeros_like(x[0])
                carry, out = torch.ops.higher_order.scan(add, [init], [x], [])
                return carry

        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        model = ScanModel()
        expected = model(x)
        self.assertEqualArray(expected, x.sum(axis=0))
        self.assertNotEmpty(torch.export.export(model, (x,), strict=False).graph)

        for optimize in [False, True]:
            with self.subTest(optimize=optimize):
                onx = to_onnx(
                    model,
                    (x,),
                    optimize=optimize,
                    export_options=ExportOptions(decomposition_table="default"),
                )
                names = [(f.domain, f.name) for f in onx.functions]
                self.assertEqual(len(names), len(set(names)))

                ref = ExtendedReferenceEvaluator(onx)

                for _x in (-x, x):
                    expected = model(_x)
                    feeds = {"x": _x.detach().numpy()}
                    got = ref.run(None, feeds)
                    self.assertEqualArray(expected, got[0], atol=1e-5)

                import onnxruntime

                sess = onnxruntime.InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                for _x in (-x, x):
                    expected = model(_x)
                    feeds = {"x": _x.detach().numpy()}
                    got = sess.run(None, feeds)
                    self.assertEqualArray(expected, got[0], atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_scan_2(self):
        import torch

        def add(
            carry1: torch.Tensor, carry2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor
        ):
            next_carry1 = carry1 + y1
            next_carry2 = carry2 * y2
            return [next_carry1, next_carry2, next_carry1, next_carry2]

        class ScanModel(torch.nn.Module):
            def forward(self, x):
                init1 = torch.zeros_like(x[0])
                init2 = torch.ones_like(x[0])
                carry1, carry2, out1, out2 = torch.ops.higher_order.scan(
                    add, [init1, init2], [x, x * 2], additional_inputs=[]
                )
                return carry1, carry2, out1, out2

        x = torch.tensor([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32)
        model = ScanModel()
        expected = model(x)
        self.assertEqualArray(expected[0], x.sum(axis=0))
        self.assertNotEmpty(torch.export.export(model, (x,), strict=False).graph)

        for optimize in [False, True]:
            with self.subTest(optimize=optimize):
                onx = to_onnx(
                    model,
                    (x,),
                    optimize=optimize,
                    export_options=ExportOptions(decomposition_table="default", strict=False),
                )
                names = [(f.domain, f.name) for f in onx.functions]
                self.assertEqual(len(names), len(set(names)))

                ref = ExtendedReferenceEvaluator(onx)

                for _x in (-x, x):
                    expected = model(_x)
                    feeds = {"x": _x.detach().numpy()}
                    got = ref.run(None, feeds)
                    for e, g in zip(expected, got):
                        self.assertEqualArray(e, g, atol=1e-5)

                import onnxruntime

                sess = onnxruntime.InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                for _x in (-x, x):
                    expected = model(_x)
                    feeds = {"x": _x.detach().numpy()}
                    got = sess.run(None, feeds)
                    for e, g in zip(expected, got):
                        self.assertEqualArray(e, g, atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_scan_cdist_carry(self):
        import torch

        def dist(carry: torch.Tensor, x: torch.Tensor):
            sub = carry - x.reshape((1, -1))
            sq = sub * sub
            rd = sq.sum(axis=1) ** 0.5
            # clone --> UnsupportedAliasMutationException:
            # Combine_fn might be aliasing the input!
            return [carry.clone(), rd]

        class ScanModel(torch.nn.Module):
            def forward(self, x):
                carry, out = torch.ops.higher_order.scan(dist, [x], [x], additional_inputs=[])
                return out

        x = torch.tensor([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32)
        model = ScanModel()
        expected = model(x)
        self.assertEqual(expected.shape, (3, 3))
        self.assertEqualArray(expected, torch.cdist(x, x))
        self.assertNotEmpty(torch.export.export(model, (x,), strict=False).graph)

        for optimize in [False, True]:
            with self.subTest(optimize=optimize):
                onx = to_onnx(
                    model,
                    (x,),
                    optimize=optimize,
                    export_options=ExportOptions(decomposition_table="default", strict=False),
                )
                names = [(f.domain, f.name) for f in onx.functions]
                self.assertEqual(len(names), len(set(names)))

                ref = ExtendedReferenceEvaluator(onx)

                for _x in (-x, x):
                    expected = model(_x)
                    feeds = {"x": _x.detach().numpy()}
                    got = ref.run(None, feeds)
                    self.assertEqualArray(expected, got[0], atol=1e-5)

                import onnxruntime

                sess = onnxruntime.InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                for _x in (-x, x):
                    expected = model(_x)
                    feeds = {"x": _x.detach().numpy()}
                    got = sess.run(None, feeds)
                    self.assertEqualArray(expected, got[0], atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_scan_cdist_add(self):
        import torch

        def dist(unused: torch.Tensor, x: torch.Tensor, samex: torch.Tensor):
            sub = samex - x.reshape((1, -1))
            sq = sub * sub
            rd = torch.sqrt(sq.sum(axis=1))
            # clone --> UnsupportedAliasMutationException:
            # Combine_fn might be aliasing the input!
            return [unused.clone(), rd]

        class ScanModel(torch.nn.Module):
            def forward(self, x):
                z = torch.tensor([0], dtype=torch.float32)
                y = x.clone()
                out = torch.ops.higher_order.scan(dist, [z], [x], additional_inputs=[y])
                return out[1]

        x = torch.tensor([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32)
        model = ScanModel()
        expected = model(x)
        self.assertEqual(expected.shape, (3, 3))
        self.assertEqualArray(expected, torch.cdist(x, x))
        ep = torch.export.export(model, (x,), strict=False)
        self.assertNotEmpty(ep.graph)

        for optimize in [False, True]:
            with self.subTest(optimize=optimize):
                onx = to_onnx(
                    model,
                    (x,),
                    optimize=optimize,
                    export_options=ExportOptions(decomposition_table="default", strict=False),
                    inline=False,
                )
                with open(
                    self.get_dump_file(f"test_scan_cdist_add_{int(optimize)}.onnx"), "wb"
                ) as f:
                    f.write(onx.SerializeToString())

                names = [(f.domain, f.name) for f in onx.functions]
                self.assertEqual(len(names), len(set(names)))

                import onnxruntime

                sess = onnxruntime.InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                for _x in (-x, x):
                    expected = model(_x)
                    feeds = {"x": _x.detach().numpy()}
                    got = sess.run(None, feeds)
                    self.assertEqualArray(expected, got[0], atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_scan_cdist_dynamic(self):
        import torch

        def dist(y: torch.Tensor, scanned_x: torch.Tensor):
            sub = y - scanned_x.reshape((1, -1))
            sq = sub * sub
            rd = torch.sqrt(sq.sum(axis=1))
            # clone --> UnsupportedAliasMutationException:
            # Combine_fn might be aliasing the input!
            return [y.clone(), rd]

        class ModuleWithControlFlowLoopScan(torch.nn.Module):

            def forward(self, x, y):
                carry, out = torch.ops.higher_order.scan(dist, [y], [x], additional_inputs=[])
                return out

        x_rows = torch.export.Dim("x_rows")
        y_rows = torch.export.Dim("y_rows")
        dim = torch.export.Dim("dim")
        dyns = [
            ({0: x_rows, 1: dim}, {0: y_rows, 1: dim}),
            {"x": {0: x_rows, 1: dim}, "y": {0: y_rows, 1: dim}},
        ]
        inputs = [
            (torch.randn(3, 4), torch.randn(5, 4)),
            (torch.randn(13, 14), torch.randn(15, 14)),
        ]
        inputs.append((-inputs[0][0], -inputs[0][1]))
        model = ModuleWithControlFlowLoopScan()

        for optimize, ds in itertools.product([False, True], dyns):
            onx = to_onnx(
                model,
                inputs[0],
                optimize=optimize,
                dynamic_shapes=ds,
                export_options=ExportOptions(decomposition_table="default", strict=False),
                inline=False,
            )
            if isinstance(ds, dict):
                for i in onx.graph.input:
                    shape = i.type.tensor_type.shape
                    ods = tuple(d.dim_param for d in shape.dim)
                    self.assertEqual(ods, (f"{i.name}_rows", "dim"))
            else:
                for i in onx.graph.input:
                    shape = i.type.tensor_type.shape
                    ods = tuple(d.dim_param for d in shape.dim)
                    self.assertEqual(ods, (f"{i.name}_rows", "dim"))
            # self.print_model(onx)
            names = [(f.domain, f.name) for f in onx.functions]
            self.assertEqual(len(names), len(set(names)))
            import onnxruntime

            sess = onnxruntime.InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            ref = ExtendedReferenceEvaluator(onx)

            for xy in inputs:
                with self.subTest(optimize=optimize, ds=type(ds), xy=xy[0].shape):
                    expected = model(*xy)
                    feeds = {"x": xy[0].detach().numpy(), "y": xy[1].detach().numpy()}
                    got = ref.run(None, feeds)
                    self.assertEqualArray(expected, got[0], atol=1e-5)
                    got = sess.run(None, feeds)
                    self.assertEqualArray(expected, got[0], atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_scan_cdist_dynamic_inline(self):
        import torch

        def dist(y: torch.Tensor, scanned_x: torch.Tensor):
            sub = y - scanned_x.reshape((1, -1))
            sq = sub * sub
            rd = torch.sqrt(sq.sum(axis=1))
            # clone --> UnsupportedAliasMutationException:
            # Combine_fn might be aliasing the input!
            return [y.clone(), rd]

        class ModuleWithControlFlowLoopScan(torch.nn.Module):

            def forward(self, x, y):
                carry, out = torch.ops.higher_order.scan(dist, [y], [x], additional_inputs=[])
                return out

        x_rows = torch.export.Dim("x_rows")
        y_rows = torch.export.Dim("y_rows")
        dim = torch.export.Dim("dim")
        dyns = [
            ({0: x_rows, 1: dim}, {0: y_rows, 1: dim}),
            {"x": {0: x_rows, 1: dim}, "y": {0: y_rows, 1: dim}},
        ]
        inputs = [
            (torch.randn(3, 4), torch.randn(5, 4)),
            (torch.randn(13, 14), torch.randn(15, 14)),
        ]
        inputs.append((-inputs[0][0], -inputs[0][1]))
        model = ModuleWithControlFlowLoopScan()

        for optimize, ds in itertools.product([False, True], dyns):
            onx = to_onnx(
                model,
                inputs[0],
                optimize=optimize,
                dynamic_shapes=ds,
                export_options=ExportOptions(decomposition_table="default", strict=False),
                inline=True,
            )
            if isinstance(ds, dict):
                for i in onx.graph.input:
                    shape = i.type.tensor_type.shape
                    ods = tuple(d.dim_param for d in shape.dim)
                    self.assertEqual(ods, (f"{i.name}_rows", "dim"))
            else:
                for i in onx.graph.input:
                    shape = i.type.tensor_type.shape
                    ods = tuple(d.dim_param for d in shape.dim)
                    self.assertEqual(ods, (f"{i.name}_rows", "dim"))
            # self.print_model(onx)
            names = [(f.domain, f.name) for f in onx.functions]
            self.assertEqual(len(names), len(set(names)))
            import onnxruntime

            sess = onnxruntime.InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            ref = ExtendedReferenceEvaluator(onx)

            for xy in inputs:
                with self.subTest(optimize=optimize, ds=type(ds), xy=xy[0].shape):
                    expected = model(*xy)
                    feeds = {"x": xy[0].detach().numpy(), "y": xy[1].detach().numpy()}
                    got = ref.run(None, feeds)
                    self.assertEqualArray(expected, got[0], atol=1e-5)
                    got = sess.run(None, feeds)
                    self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
