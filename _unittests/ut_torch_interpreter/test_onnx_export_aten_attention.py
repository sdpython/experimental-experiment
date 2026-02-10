import unittest
from experimental_experiment.ext_test_case import ExtTestCase, has_onnxruntime
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


class TestOnnxExportAtenAttention(ExtTestCase):

    def test_scaled_dot_product_attention_not_causal(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, query, key, value):
                res = torch.nn.functional.scaled_dot_product_attention(query, key, value)
                return res.transpose(0, 1)

        model = Model()
        device = "cpu"
        query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        inputs = (query, key, value)
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float16)
        ds1 = {0: "batch", 1: "seq_length", 2: "cache_length", 3: "last_dim"}
        ds = (ds1, ds1, ds1)

        options = ExportOptions(aten_as_function=False)
        onx = to_onnx(model, inputs, dynamic_shapes=ds, export_options=options)
        self.assertNotIn(
            "aten_scaled_dot_product_attention_default",
            {n.op_type for n in onx.graph.node},
        )
        feeds = dict(zip(["query", "key", "value"], [x.detach().cpu().numpy() for x in inputs]))
        # self.dump_onnx("test_scaled_dot_product_attention_not_causal.onnx", onx)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-3)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_scaled_dot_product_attention_causal(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, query, key, value):
                res = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, is_causal=True
                )
                return res.transpose(0, 1)

        model = Model()
        device = "cpu"
        query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        inputs = (query, key, value)
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float16)
        self.assertEqual(expected.shape, (8, 32, 128, 64))
        ds1 = {0: "batch", 1: "seq_length", 2: "cache_length", 3: "last_dim"}
        ds = (ds1, ds1, ds1)

        options = ExportOptions(aten_as_function=False)
        onx = to_onnx(model, inputs, dynamic_shapes=ds, export_options=options)
        self.assertNotIn(
            "aten_scaled_dot_product_attention_default",
            [n.op_type for n in onx.graph.node],
        )
        feeds = dict(zip(["query", "key", "value"], [x.detach().cpu().numpy() for x in inputs]))
        # self.dump_onnx("test_scaled_dot_product_attention_causal.onnx", onx)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-3)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_scaled_dot_product_attention_function_1(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, query, key, value):
                res = torch.nn.functional.scaled_dot_product_attention(query, key, value)
                return res.transpose(0, 1)

        model = Model()
        device = "cpu"
        query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        inputs = (query, key, value)
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float16)
        self.assertEqual(expected.shape, (8, 32, 128, 64))
        ds1 = {0: "batch", 1: "seq_length", 2: "cache_length", 3: "last_dim"}
        ds = (ds1, ds1, ds1)

        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=ds,
            inline=False,
            export_options=ExportOptions(
                aten_as_function={"aten.scaled_dot_product_attention.default"}
            ),
        )
        self.assertEqual(
            ["aten_scaled_dot_product_attention_default", "Transpose"],
            [n.op_type for n in onx.graph.node],
        )
        feeds = dict(zip(["query", "key", "value"], [x.detach().cpu().numpy() for x in inputs]))
        self.dump_onnx("test_scaled_dot_product_attention_function_1.onnx", onx)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-3)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_scaled_dot_product_attention_function_2(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, query, key, value):
                res1 = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, is_causal=True
                )
                res2 = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, is_causal=False
                )
                res3 = torch.nn.functional.scaled_dot_product_attention(
                    query, value, key, is_causal=True
                )
                return res1.transpose(0, 1) + res2.transpose(0, 1) + res3.transpose(0, 1)

        model = Model()
        device = "cpu"
        query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        inputs = (query, key, value)
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float16)
        self.assertEqual(expected.shape, (8, 32, 128, 64))
        ds1 = {0: "batch", 1: "seq_length", 2: "cache_length", 3: "last_dim"}
        ds = (ds1, ds1, ds1)

        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=ds,
            export_options=ExportOptions(
                aten_as_function={"aten.scaled_dot_product_attention.default"}
            ),
        )
        self.dump_onnx("test_scaled_dot_product_attention_function_2.onnx", onx)
        self.assertEqual(
            [
                "aten_scaled_dot_product_attention_default",
                "aten_scaled_dot_product_attention_default_l2l",
                "aten_scaled_dot_product_attention_default",
                "Transpose",
                "Transpose",
                "Add",
                "Transpose",
                "Add",
            ],
            [n.op_type for n in onx.graph.node],
        )
        self.assertIn("aten", set(n.domain for n in onx.graph.node))
        for node in onx.graph.node:
            if node.domain == "aten":
                keys = [p.key for p in node.metadata_props]
                self.assertEqual(
                    [
                        "aten_name",
                        "args",
                        "kwargs",
                        "module[0]",
                        "intypes",
                        "outtypes",
                        "inshapes",
                        "outshapes",
                    ],
                    keys,
                )
        for f in onx.functions:
            keys = [p.key for p in f.metadata_props]
            self.assertEqual(["inline"], keys)
            values = [p.value for p in f.metadata_props]
            self.assertEqual(["0"], values)
        feeds = dict(zip(["query", "key", "value"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_group_norm_opset_17_21(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.group_norm(x, 4)

        inputs = (torch.randn(2, 32, 64, 128, dtype=torch.float32),)
        model = Model()
        expected = model(*inputs)
        ds = ({3: "last"},)
        for opset in [17, 21]:
            with self.subTest(opset=opset):
                onx = to_onnx(model, inputs, dynamic_shapes=ds, target_opset=opset)
                self.dump_onnx(f"test_group_norm_opset_{opset}.onnx", onx)
                self.assertEqual(
                    ["Shape", "Reshape", "InstanceNormalization", "Reshape"],
                    [n.op_type for n in onx.graph.node],
                )
                self.assertEqual(
                    ("", opset), (onx.opset_import[0].domain, onx.opset_import[0].version)
                )

                feeds = dict(zip(["x"], [x.detach().cpu().numpy() for x in inputs]))
                ref = ExtendedReferenceEvaluator(onx)
                got = ref.run(None, feeds)[0]
                self.assertEqualArray(expected, got, atol=1e-4)

                import onnxruntime

                sess = onnxruntime.InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                got = sess.run(None, feeds)[0]
                self.assertEqualArray(expected, got, atol=1e-4)

    def test_scaled_dot_product_attention_4D_18_23(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, query, key, value):
                return torch.nn.functional.scaled_dot_product_attention(query, key, value)

        query = torch.rand(32, 8, 128, 64, dtype=torch.float32)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float32)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float32)
        inputs = (query, key, value)
        model = Model()
        expected = model(*inputs)
        ds1 = {0: "batch", 2: "cache_length", 3: "last_dim"}
        ds = (ds1, ds1, ds1)
        for opset in [24, 18]:
            with self.subTest(opset=opset):
                onx = to_onnx(
                    model,
                    inputs,
                    dynamic_shapes=ds,
                    target_opset=opset,
                    export_options=(
                        ExportOptions(
                            aten_as_function={"aten.scaled_dot_product_attention.default"}
                        )
                        if opset < 24
                        else None
                    ),
                )
                self.dump_onnx(f"test_scaled_dot_product_attention_{opset}.onnx", onx)
                if opset >= 24:
                    self.assertEqual(
                        ["Attention"],
                        [n.op_type for n in onx.graph.node],
                    )
                else:
                    self.assertEqual(
                        ["aten_scaled_dot_product_attention_default"],
                        [n.op_type for n in onx.graph.node],
                    )
                self.assertEqual(
                    ("", opset), (onx.opset_import[0].domain, onx.opset_import[0].version)
                )

                feeds = dict(
                    zip(["query", "key", "value"], [x.detach().cpu().numpy() for x in inputs])
                )
                ref = ExtendedReferenceEvaluator(onx)
                got = ref.run(None, feeds)[0]
                self.assertEqualArray(expected, got, atol=1e-2)

                import onnxruntime

                if not has_onnxruntime("1.23"):
                    raise unittest.SkipTest("onnxruntime 1.23 is required")

                try:
                    sess = onnxruntime.InferenceSession(
                        onx.SerializeToString(), providers=["CPUExecutionProvider"]
                    )
                except Exception as e:
                    if "till opset 23" in str(e):
                        raise unittest.SkipTest("onnxruntime does not support opset > 23 yet")
                got = sess.run(None, feeds)[0]
                self.assertEqualArray(expected, got, atol=1e-2)

    def test_enable_gqa_in_attention(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        model = Model()
        query = torch.randn(2, 4, 8, 16)
        key = torch.randn(2, 2, 8, 16)
        value = torch.randn(2, 2, 8, 16)
        inputs = (query, key, value)
        dynamic_shapes = ({0: "batch", 2: "seq"}, {0: "batch", 2: "seq"}, {0: "batch", 2: "seq"})

        for opset in [18, 24]:
            with self.subTest(opset=opset):
                onx = to_onnx(
                    model,
                    inputs,
                    dynamic_shapes=dynamic_shapes,
                    target_opset=opset,
                    export_options=(
                        ExportOptions(
                            aten_as_function={"aten.scaled_dot_product_attention.default"}
                        )
                        if opset < 24
                        else None
                    ),
                )
                self.dump_onnx(f"test_enable_gqa_in_attention.{opset}.onnx", onx)
                sess = self._check_with_ort(onx)
                expected = model(*inputs)
                feeds = dict(zip("qkv", [i.detach().numpy() for i in inputs]))
                got = sess.run(None, feeds)
                self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
