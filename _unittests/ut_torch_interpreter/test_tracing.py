import copy
import operator
import unittest
import torch
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_interpreter.tracing import (
    CustomTracer,
    CustomProxy,
    _len,
    _isinstance,
    setitem_with_transformation,
)


class TestTracing(ExtTestCase):

    def test_tracing_simple_proxy(self):
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
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_tracing_list_variable_length(self):
        class Model(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x, lx: list):
                t = torch.cat(lx, axis=1).sum(axis=1, keepdim=True)
                return torch.sigmoid(self.linear(x)) - self.buff + t

        model = Model()
        inputs = (
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(4) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(4 * 2) + 10).reshape((-1, 2)).to(torch.float32),
            ],
        )
        expected = model(*inputs)
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_tracing_setitem_mask(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                mask = x.to(bool)
                x[mask] = 2
                return x

        inputs = (torch.randn((2, 3, 3)),)
        model = Model()
        expected = model(*inputs)
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_tracing_cond(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x)

                def false_fn(x):
                    return torch.cos(x)

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        inputs = (torch.rand(5, 3),)
        model = Model()
        expected = model(*inputs)
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_index_Tensor_copy_0(self):
        class Model(torch.nn.Module):
            def forward(self, x, sumx):
                K_33 = x.clone()
                K_33[2:-2, 2:-2, :-1] = sumx[None, :, None]
                K_33[2:-2, 2:-2, -1] = 0.0
                return K_33

        inputs = (
            (torch.arange(7 * 9 * 11) + 10).reshape((7, 9, 11)).to(torch.float32),
            torch.arange(5).to(torch.float32),
        )
        model = Model()
        expected = model(*copy.deepcopy(inputs))
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        self.assertIn(operator.setitem, {n.target for n in graph.nodes})
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_index_Tensor_copy_1(self):
        class Model(torch.nn.Module):
            def forward(self, x, sumx):
                K_33 = x.clone()
                K_33[2:-2, 2:-2, :] = sumx[None, :, None]
                return K_33

        inputs = (
            (torch.arange(7 * 9 * 11) + 10).reshape((7, 9, 11)).to(torch.float32),
            torch.arange(5).to(torch.float32),
        )
        model = Model()
        expected = model(*copy.deepcopy(inputs))
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        self.assertIn(operator.setitem, {n.target for n in graph.nodes})
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_index_Tensor_copy_2(self):
        class Model(torch.nn.Module):
            def forward(self, x, sumx):
                K_33 = x.clone()
                K_33[2:-2, 2:-2, :-1] = sumx[None, :, None]
                return torch.abs(K_33)

        inputs = (
            (torch.arange(7 * 9 * 11) + 10).reshape((7, 9, 11)).to(torch.float32),
            torch.arange(5).to(torch.float32),
        )
        model = Model()
        expected = model(*copy.deepcopy(inputs))
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        self.assertIn(operator.setitem, {n.target for n in graph.nodes})
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_index_Tensor_copy_3(self):
        class Model(torch.nn.Module):
            def forward(self, x, sumx):
                K_33 = x.clone()
                K_33[2:-2, 2:-2, :-1] = sumx[None, :, None]
                e = torch.abs(K_33)
                K_33[2:-2, 2:-2, -1] = 0.0
                return K_33 + e

        inputs = (
            (torch.arange(7 * 9 * 11) + 10).reshape((7, 9, 11)).to(torch.float32),
            torch.arange(5).to(torch.float32),
        )
        model = Model()
        expected = model(*copy.deepcopy(inputs))
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        self.assertIn(operator.setitem, {n.target for n in graph.nodes})
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_index_Tensor_copy_exp(self):
        class Model(torch.nn.Module):
            def forward(self, x, sumx):
                K_33 = x.clone()
                torch.exp_(K_33[2:-2, 2:-2, :-1])
                return K_33

        inputs = (
            (torch.arange(7 * 9 * 11) + 10).reshape((7, 9, 11)).to(torch.float32),
            torch.arange(5).to(torch.float32),
        )
        model = Model()
        expected = model(*copy.deepcopy(inputs))
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        self.assertIn(setitem_with_transformation, {n.target for n in graph.nodes})
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    @unittest.skip("TODO: fix it")
    def test_tracing_fixed_list_with_none(self):
        class Model(torch.nn.Module):

            def forward(self, lx):
                x = lx[0]
                if lx[1] is not None:
                    x += lx[1]
                if lx[2] is not None:
                    x += lx[2]
                return x

            _inputs = [
                ([torch.rand((4, 4)), torch.rand((4, 4)), None],),
                ([torch.rand((4, 4)), torch.rand((4, 4)), torch.rand((4, 4))],),
            ]

        inputs = Model._inputs
        model = Model()
        graph = CustomTracer().trace(model)
        for inp in inputs:
            # print(torch.export.export(model, inp).graph)
            expected = model(*inp)
            mod = torch.fx.GraphModule(model, graph)
            got = mod(*inp)
            self.assertEqualArray(expected, got)

    @unittest.skip("TODO: fix it")
    def test_tracing_int_shape(self):
        class Model(torch.nn.Module):
            @staticmethod
            def add_one(dim: int) -> int:
                return dim + 1

            def forward(self, x):
                y = torch.ones((x.shape[0], x.shape[1] + 1))
                return y

            _inputs = [(torch.rand((4, 4)),), (torch.rand((5, 5)),)]
            _dynamic = {"x": {0: torch.export.Dim("dx"), 1: torch.export.Dim("dy")}}

        inputs = Model._inputs
        model = Model()
        graph = CustomTracer().trace(model, dynamic_shapes=Model._dynamic)
        for inp in inputs:
            expected = model(*inp)
            mod = torch.fx.GraphModule(model, graph)
            got = mod(*inp)
            self.assertEqualArray(expected, got)

    @unittest.skip("TODO: fix it")
    def test_tracing_function_int_shape(self):
        class Model(torch.nn.Module):
            @staticmethod
            def add_one(dim: int) -> int:
                return dim + 1

            def forward(self, x):
                dy1 = Model.add_one(x.shape[1])
                y = torch.ones((x.shape[0], dy1))
                return y

            _inputs = [(torch.rand((4, 4)),), (torch.rand((5, 5)),)]
            _dynamic = {"x": {0: torch.export.Dim("dx"), 1: torch.export.Dim("dy")}}

        inputs = Model._inputs
        model = Model()
        graph = CustomTracer().trace(model, dynamic_shapes=Model._dynamic)
        for inp in inputs:
            expected = model(*inp)
            mod = torch.fx.GraphModule(model, graph)
            got = mod(*inp)
            self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
