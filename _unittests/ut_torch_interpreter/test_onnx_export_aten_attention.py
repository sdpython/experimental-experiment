import unittest
from experimental_experiment.ext_test_case import ExtTestCase
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
        feeds = dict(
            zip(["query", "key", "value"], [x.detach().cpu().numpy() for x in inputs])
        )
        # self.dump_onnx("test_scaled_dot_product_attention_not_causal.onnx", onx)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-3)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-3)

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
        feeds = dict(
            zip(["query", "key", "value"], [x.detach().cpu().numpy() for x in inputs])
        )
        # self.dump_onnx("test_scaled_dot_product_attention_causal.onnx", onx)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-3)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-3)

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

        onx = to_onnx(model, inputs, dynamic_shapes=ds, inline=False)
        self.assertEqual(
            ["aten_scaled_dot_product_attention_default", "Transpose"],
            [n.op_type for n in onx.graph.node],
        )
        feeds = dict(
            zip(["query", "key", "value"], [x.detach().cpu().numpy() for x in inputs])
        )
        self.dump_onnx("test_scaled_dot_product_attention_function_1.onnx", onx)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-3)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-3)

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

        onx = to_onnx(model, inputs, dynamic_shapes=ds)
        self.dump_onnx("test_scaled_dot_product_attention_function_2.onnx", onx)
        self.assertEqual(
            [
                "aten_scaled_dot_product_attention_default",
                "aten_scaled_dot_product_attention_default__v2",
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
                self.assertEqual(["aten_name", "args", "kwargs"], keys)
        for f in onx.functions:
            keys = [p.key for p in f.metadata_props]
            self.assertEqual(["inline"], keys)
            values = [p.value for p in f.metadata_props]
            self.assertEqual(["0"], values)
        feeds = dict(
            zip(["query", "key", "value"], [x.detach().cpu().numpy() for x in inputs])
        )
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_group_norm_opset_17_18_21(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.group_norm(x, 4)

        inputs = (torch.randn(1, 4, 4, 4, dtype=torch.float32),)
        model = Model()
        expected = model(*inputs)
        ds = ({3: "last"},)
        for opset in [17, 18, 21]:
            with self.subTest(opset=opset):
                onx = to_onnx(model, inputs, dynamic_shapes=ds, target_opset=opset)
                self.dump_onnx(f"test_group_norm_opset_{opset}.onnx", onx)
                if opset >= 18:
                    self.assertEqual(
                        ["GroupNormalization"], [n.op_type for n in onx.graph.node]
                    )
                else:
                    self.assertEqual(
                        ["Reshape", "InstanceNormalization", "Shape", "Reshape"],
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

    def test_scaled_dot_product_attention_18_23(self):
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
        for opset in [23, 18]:
            with self.subTest(opset=opset):
                onx = to_onnx(model, inputs, dynamic_shapes=ds, target_opset=opset)
                self.dump_onnx(f"scaled_dot_product_attention_{opset}.onnx", onx)
                if opset >= 23:
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

                sess = onnxruntime.InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                got = sess.run(None, feeds)[0]
                self.assertEqualArray(expected, got, atol=1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
