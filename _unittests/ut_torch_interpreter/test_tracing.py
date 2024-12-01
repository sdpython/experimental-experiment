import unittest
import torch
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_interpreter.tracing import (
    CustomTracer,
    CustomProxy,
    _len,
    _isinstance,
)


class TestTracing(ExtTestCase):

    def test__proxy(self):
        graph = torch.fx.Graph()
        node = graph.create_node("placeholder", "tx", args=(), kwargs={}, name="txn")
        tr = CustomTracer()
        tr.graph = torch.fx.Graph(tracer_cls=CustomTracer)
        x = CustomProxy(node, tr)
        i = _len(x)
        self.assertIsInstance(i, CustomProxy)

    def test_tracing_abs(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.abs(x)

        model = Model()
        x = torch.ones((4, 4))
        expected = model(x)
        graph = CustomTracer().trace(model)
        mod = torch.fx.GraphModule(model, graph)
        got = mod(x)
        self.assertEqualArray(expected, got)

    def test_tracing_inplace_add(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                xc = x.clone()
                xc += 2
                return xc + 3

        model = Model()
        x = torch.ones((4, 4))
        expected = model(x)
        graph = CustomTracer().trace(model)
        self.assertNotIn("add_]", str(graph))
        mod = torch.fx.GraphModule(model, graph)
        got = mod(x)
        self.assertEqualArray(expected, got)

    def test_tracing_inplace_add_(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                xc = x.clone()
                xc.add_(2)
                return xc + 3

        model = Model()
        x = torch.ones((4, 4))
        expected = model(x)
        graph = CustomTracer().trace(model)
        self.assertIn("add_]", str(graph))
        mod = torch.fx.GraphModule(model, graph)
        self.assertIn("add_]", str(mod.graph))
        got = mod(x)
        self.assertEqualArray(expected, got)

    def test_tracing_inplace_add_users(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                xc = x.clone()
                xc.add_(2)
                return xc + 3

        model = Model()
        x = torch.ones((4, 4))
        expected = model(x)
        graph = CustomTracer().trace(model, remove_inplace=False)
        self.assertEqual(
            len(
                [node for node in graph.nodes if len(node.users) == 0 and node.op != "output"]
            ),
            1,
        )
        self.assertIn("(%clone, 3)", str(graph))
        graph = CustomTracer().trace(model, remove_inplace=True)
        self.assertEmpty(
            [node for node in graph.nodes if len(node.users) == 0 and node.op != "output"]
        )
        self.assertNotIn("(%clone, 3)", str(graph))
        mod = torch.fx.GraphModule(model, graph)
        self.assertNotIn("(%clone, 3)", str(mod.graph))
        got = mod(x)
        self.assertEqualArray(expected, got)

    def test_tracing_inplace_add_mul_users(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                xc = x.clone()
                xc.add_(2)
                xc.add_(5)
                return xc + 3

        model = Model()
        x = torch.ones((4, 4))
        expected = model(x)
        graph = CustomTracer().trace(model, remove_inplace=False)
        self.assertEqual(
            len(
                [node for node in graph.nodes if len(node.users) == 0 and node.op != "output"]
            ),
            2,
        )
        self.assertIn("(%clone, 3)", str(graph))
        graph = CustomTracer().trace(model, remove_inplace=True)
        self.assertEmpty(
            [node for node in graph.nodes if len(node.users) == 0 and node.op != "output"]
        )
        self.assertNotIn("(%clone, 3)", str(graph))
        mod = torch.fx.GraphModule(model, graph)
        self.assertNotIn("(%clone, 3)", str(mod.graph))
        got = mod(x)
        self.assertEqualArray(expected, got)

    def test_tracing_inplace_setitem(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                xc = x.clone()
                y = xc[:, :2] * 2
                xc[:, :2] = y
                return xc + 3

        model = Model()
        x = torch.ones((4, 4))
        expected = model(x)
        graph = CustomTracer().trace(model)
        # print(graph)
        self.assertIn("operator.setitem", str(graph))
        mod = torch.fx.GraphModule(model, graph)
        # print(mod.graph)
        got = mod(x)
        self.assertEqualArray(expected, got)

    def test_tracing_isinstance(self):
        class Model(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x, lx):
                if _isinstance(lx, list):
                    return torch.sigmoid(self.linear(x)) + lx
                t = lx[0] * lx[1].sum(axis=1, keepdim=True)
                return torch.sigmoid(self.linear(x)) - t

        model = Model()
        self.assertRaise(
            lambda: CustomTracer().trace(model),
            RuntimeError,
            "Unable to know if cls is from type",
        )

    def test_tracing_len(self):
        class Model(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x, lx: list):
                t = lx[0] * lx[1].sum(axis=1, keepdim=True)
                llx = _len(lx)
                tn = t / llx
                return torch.sigmoid(self.linear(x)) - tn

        model = Model()
        inputs = (
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(4) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(4 * 2) + 10).reshape((-1, 2)).to(torch.float32),
            ],
        )
        graph = CustomTracer().trace(model)
        mod = torch.fx.GraphModule(model, graph)
        expected = model(*inputs)
        got = mod(*inputs)
        self.assertEqualArray(expected, got)

    def test_tracing_inplace_setitem_ellipsis(self):
        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.params = torch.zeros((1, 8192, 4), dtype=torch.float32)

            def forward(self, index, update):
                copy = self.params.clone()
                copy[..., index] = update
                return copy

        model = Model()
        inputs = (
            (torch.tensor([0, 3, 2, 1], dtype=torch.int64)),
            (torch.arange(4 * 8192) + 10).reshape((-1, 4)).to(torch.float32),
        )
        expected = model(*inputs)
        graph = CustomTracer().trace(model)
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
