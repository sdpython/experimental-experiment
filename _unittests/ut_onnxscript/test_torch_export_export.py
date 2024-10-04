import unittest
import torch
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    skipif_ci_apple,
    requires_torch,
)


class TestTorchExportExport(ExtTestCase):

    @skipif_ci_windows("not available on Windows")
    @skipif_ci_apple("not able to fix it")
    @requires_torch("2.5")
    def test_scaled_dot_product_attention_export_issue(self):

        class DummyModel(torch.nn.Module):
            def __init__(self, enable_math: bool):
                super().__init__()
                self.enable_math = False

            def forward(self, query, key, value):
                res = torch.nn.functional.scaled_dot_product_attention(query, key, value)
                rest = res.transpose(0, 1)
                final = rest.view(8, 32, 128 * 64)
                return final

        model = DummyModel(False)
        device = "cpu"

        query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        expected = model(query, key, value)
        self.assertEqual(expected.dtype, torch.float16)
        self.assertEqual(expected.shape, (8, 32, 8192))

        cpl = torch.compile(model)
        new_output = cpl(query, key, value)
        self.assertEqual(new_output.dtype, torch.float16)
        self.assertEqual(new_output.shape, (8, 32, 8192))
        self.assertEqualArray(expected, new_output)

        export = torch.export.export(model, (query, key, value))
        module = export.module()
        got = module(query, key, value)
        self.assertEqualArray(expected, got)

        # Fails here due (pytorch.issue)
        # Cannot view a tensor with shape torch.Size([8, 32, 128, 64]) and strides
        # (64, 512, 16384, 1) as a tensor with shape (8, 32, 8192)
        # export.run_decompositions()

        # Let's rewrite the model by inserting a node flatten between transpose and view.

        def transform(
            m: torch.nn.Module, tracer_class: type = torch.fx.Tracer
        ) -> torch.nn.Module:
            modified = False
            graph = tracer_class().trace(m)
            for node in graph.nodes:
                if node.op != "call_method" or node.target != "transpose":
                    continue
                insert = False
                for user in node.users:
                    if user.op == "call_method" and user.target == "view":
                        insert = True
                        break
                if not insert:
                    continue

                modified = True
                with graph.inserting_after(node):
                    new_node = graph.call_method("flatten", args=(node,))
                    node.replace_all_uses_with(new_node)
                    # new_node is replaced as well so we manually revert the replacement
                    new_node.update_arg(0, node)
                    node.users = {new_node: None}

            if not modified:
                # No need to rewrite.
                return None

            graph.lint()
            return torch.fx.GraphModule(m, graph)

        rewritten_model = transform(model)
        self.assertNotEmpty(rewritten_model)

        # new check
        export = torch.export.export(rewritten_model, (query, key, value))

        self.assertIn("aten.clone.default", str(export.graph))
        module = export.module()
        got = module(query, key, value)
        self.assertEqualArray(expected, got)
        export.run_decompositions()


if __name__ == "__main__":
    unittest.main(verbosity=2)
