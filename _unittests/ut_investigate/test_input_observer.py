import unittest
import torch
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.investigate.input_observer import (
    InputObserver,
    infer_dynamic_dimensions,
)


class TestTracingVeector(ExtTestCase):
    def test_infer_dynamic_dimensions(self):
        self.assertEqual([2], infer_dynamic_dimensions([(1, 2, 3), (1, 2, 4)]))
        self.assertEqual([0, 2], infer_dynamic_dimensions([(1, 2, 3), (2, 2, 4)]))

    def test_io_captured_args(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = [
            (torch.randn((5, 6)), torch.randn((1, 6))),
            (torch.randn((7, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((7, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(({0: cst, 1: cst}, {1: cst}), observer.infer_dynamic_shapes())

    def test_io_captured_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6))),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7))),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8))),
            dict(x=torch.randn((7, 9)), y=torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(**kwargs) for kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(dict(x={0: cst, 1: cst}, y={1: cst}), observer.infer_dynamic_shapes())

    def test_io_captured_kwargs_bool(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, add=True):
                if add:
                    return x + y
                return x - y

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6)), add=False),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7)), add=False),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8)), add=False),
            dict(x=torch.randn((7, 9)), y=torch.randn((1, 9)), add=False),
        ]

        model = Model()
        expected = [model(**kwargs) for kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(dict(x={0: cst, 1: cst}, y={1: cst}), observer.infer_dynamic_shapes())

    def test_io_captured_args_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, z=None, w=None):
                r = x + y
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((1, 6))),
                dict(z=torch.randn((5, 6)), w=torch.randn((1, 6))),
            ),
            (
                (torch.randn((6, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((6, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((7, 8)), torch.randn((1, 8))),
                dict(z=torch.randn((7, 8)), w=torch.randn((1, 8))),
            ),
            (
                (torch.randn((8, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((8, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(),
        )

    def test_io_captured_optional_args(self):
        class Model(torch.nn.Module):
            def forward(self, x, y=None):
                if y is None:
                    return x
                return x - y

        inputs = [
            (torch.randn((5, 6)),),
            (torch.randn((6, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((8, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(({0: cst, 1: cst}, {1: cst}), observer.infer_dynamic_shapes())

    def test_io_captured_optional_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y=None):
                if y is None:
                    return x
                return x - y

        inputs = [
            dict(x=torch.randn((5, 6))),
            dict(x=torch.randn((6, 7)), y=torch.randn((1, 7))),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8))),
            dict(x=torch.randn((8, 9)), y=torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(**kwargs) for kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(dict(x={0: cst, 1: cst}, y={1: cst}), observer.infer_dynamic_shapes())

    def test_io_captured_optional_args_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y=None, z=None, w=None):
                r = x + y if y is not None else x
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)),),
                dict(w=torch.randn((1, 6))),
            ),
            (
                (torch.randn((6, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((6, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((7, 8)), torch.randn((1, 8))),
                dict(z=torch.randn((7, 8)), w=torch.randn((1, 8))),
            ),
            (
                (torch.randn((8, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((8, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(),
        )

    def test_io_captured_not_supported_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x=None, y=None):
                if y is None:
                    return x
                if x is None:
                    return y
                return x - y

        inputs = [
            dict(x=torch.randn((5, 6))),
            dict(y=torch.randn((1, 7))),
            dict(y=torch.randn((1, 7))),
            dict(y=torch.randn((1, 7))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        with self.assertRaises(RuntimeError):
            observer.infer_dynamic_shapes()


if __name__ == "__main__":
    unittest.main(verbosity=2)
