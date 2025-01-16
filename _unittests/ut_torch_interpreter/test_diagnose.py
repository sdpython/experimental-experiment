import unittest
from typing import Optional
from experimental_experiment.ext_test_case import ExtTestCase, hide_stdout, requires_torch
from experimental_experiment.torch_interpreter.diagnose import (
    infer_shape_type_from_execution,
    CustomOpStrategy,
)


class TestDiagnose(ExtTestCase):
    @requires_torch("2.6")
    @hide_stdout()
    def test_infer_shape_type_from_execution_args(self):
        import torch

        class MA(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class MM(torch.nn.Module):
            def forward(self, x, y):
                return x * y

        class MASMM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ma = MA()
                self.mm = MM()

            def forward(self, x, y, z):
                return self.ma(x, y) - self.mm(y, z)

        class Big(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ma = MA()
                self.masmm = MASMM()

            def forward(self, x):
                return self.ma(x, self.masmm(x, x, x))

        big = Big()
        x = torch.randn((5, 6))
        y = big(x)
        self.assertNotEmpty(y)

        inputs = [
            ((torch.randn((5, 6)),), {}),
            ((torch.randn((6, 6)),), {}),
        ]

        diag = infer_shape_type_from_execution(big, inputs, verbose=1)
        pretty = diag.pretty_text(with_dynamic_shape=True)
        self.assertIn("DS=", pretty)

    @requires_torch("2.6")
    @hide_stdout()
    def test_infer_shape_type_from_execution_kwargs(self):
        import torch

        class MA(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class MM(torch.nn.Module):
            def forward(self, x, y):
                return x * y

        class MASMM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ma = MA()
                self.mm = MM()

            def forward(self, x, y, z):
                return self.ma(x, y=y) - self.mm(y, y=z)

        class Big(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ma = MA()
                self.masmm = MASMM()

            def forward(self, x):
                return self.ma(x, y=self.masmm(x, x, x))

        big = Big()
        x = torch.randn((5, 6))
        y = big(x)
        self.assertNotEmpty(y)

        inputs = [
            ((torch.randn((5, 6)),), {}),
            ((torch.randn((6, 6)),), {}),
        ]

        diag = infer_shape_type_from_execution(big, inputs, verbose=1)
        pretty = diag.pretty_text(with_dynamic_shape=True)
        self.assertIn("DS=", pretty)

    @requires_torch("2.6")
    @hide_stdout()
    def test_infer_shape_type_from_execution_phi2(self):
        from experimental_experiment.torch_models.llm_model_helper import get_phi2

        res = get_phi2(
            num_hidden_layers=2,
            input_cache=True,
            common_dynamic_shapes=True,
            intermediate_size=5120,
            batch_size=2,
        )
        model, _inputs, _inputs2, ds = (
            res["model"],
            res["inputs"],
            res["inputs2"],
            res["dynamic_shapes"],
        )
        inputs = [_inputs, _inputs2]

        diag = infer_shape_type_from_execution(model, inputs, verbose=2)
        pretty = diag.pretty_text(with_dynamic_shape=True)
        self.assertIn("DS=", pretty)
        args, ds_found = diag.guess_dynamic_shapes()
        self.assertEqual(args, tuple())
        self.assertEqual(set(ds), set(ds_found))

        def _check(v1, v2):
            if isinstance(v1, dict):
                self.assertIsInstance(v2, dict)
                self.assertEqual(set(v1), set(v2))
                return
            if isinstance(v1, list):
                self.assertIsInstance(v2, list)
                self.assertEqual(len(v1), len(v2))
                for a, b in zip(v1, v2):
                    _check(a, b)
                return
            raise AssertionError(f"unexpected type {type(v1)}")

        for k in ds:
            _check(ds[k], ds_found[k])

    @requires_torch("2.6")
    @hide_stdout()
    def test_infer_shape_type_from_execution_export(self):
        import torch

        class MA(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class MM(torch.nn.Module):
            def forward(self, x, y):
                return x * y

        class MASMM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ma = MA()
                self.mm = MM()

            def forward(self, x, y, z):
                return self.ma(x, y) - self.mm(y, z)

        class Big(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ma = MA()
                self.masmm = MASMM()

            def forward(self, x):
                return self.ma(x, self.masmm(x, x, x))

        big = Big()
        x = torch.randn((5, 6))
        y = big(x)
        self.assertNotEmpty(y)

        inputs = [
            ((torch.randn((5, 6)),), {}),
            ((torch.randn((6, 6)),), {}),
        ]

        diag = infer_shape_type_from_execution(big, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
        )
        self.assertNotEmpty(ep)
        assert hasattr(diag, "fx"), "No exported program found in diag."
        atts = [k for k in dir(diag) if k.startswith("exporter")]
        self.assertEqual(set(atts), {"exporter_discs", "exporter_outputs", "exporter_status"})

    @requires_torch("2.6")
    @hide_stdout()
    def test_infer_shape_type_from_execution_args_to_kwargs(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, **kwargs):
                return kwargs["x"].abs()

        model = Model()
        x = torch.randn((5, 6))
        y = model(x=x)
        self.assertNotEmpty(y)

        inputs = [
            (tuple(), {"x": x}),
            (tuple(), {"x": torch.randn((6, 6))}),
        ]

        diag = infer_shape_type_from_execution(model, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
        )
        self.assertNotEmpty(ep)
        assert hasattr(diag, "fx"), "No exported program found in diag."
        atts = [k for k in dir(diag) if k.startswith("exporter")]
        self.assertEqual(set(atts), {"exporter_discs", "exporter_outputs", "exporter_status"})
        ds = diag.guess_dynamic_shapes()
        self.assertEqual(ds, (tuple(), {"x": {0: torch.export.Dim.DYNAMIC}}))
        _a, _kw, ds = diag._move_to_kwargs(*diag.inputs[0], ds)
        self.assertEqual(ds, (tuple(), {"kwargs": {"x": {0: torch.export.Dim.DYNAMIC}}}))

    @requires_torch("2.6")
    def test_infer_shape_type_from_execution_piece_try_no_weight(self):
        import torch

        class SubModel(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x):
                return self.sub(x, x * x)

        model = Model()
        x = torch.randn((5, 6))
        ds = {"x": {0: torch.export.Dim.DYNAMIC}}
        ep = torch.export.export(model, (x,), dynamic_shapes=ds)
        self.assertNotEmpty(ep)

        _forward = model.sub.forward

        def _sub_forward_(x, y):
            return _forward(x, y)

        def _symbolic_forward(x, y):
            return torch.empty_like(x)

        schema_str = "(Tensor x, Tensor y) -> Tensor"
        custom_def = torch.library.CustomOpDef(
            "test_diag_lib", "SubModel_forward", schema_str, _sub_forward_
        )
        custom_def.register_kernel("cpu")(_sub_forward_)
        custom_def._abstract_fn = _symbolic_forward

        def _new_forward(x, y):
            return torch.ops.test_diag_lib.SubModel_forward(x, y)

        model.sub.forward = _new_forward
        ep = torch.export.export(model, (x,), dynamic_shapes=ds)
        model.sub.forward = _forward
        self.assertIn("torch.ops.test_diag_lib.SubModel_forward", str(ep))

    @requires_torch("2.6")
    def test_infer_shape_type_from_execution_piece_try_no_weight_args(self):
        import torch

        class SubModel(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x):
                return self.sub(x, x * x)

        model = Model()
        x = torch.randn((5, 6))
        ds = {"x": {0: torch.export.Dim.DYNAMIC}}
        ep = torch.export.export(model, (x,), dynamic_shapes=ds)
        self.assertNotEmpty(ep)

        _forward = model.sub.forward

        def _sub_forward_(*args, _call_=_forward, **kwargs):
            return _call_(*args, **kwargs)

        def _symbolic_forward(*args, **kwargs):
            return torch.empty_like(args[0])

        schema_str = "(Tensor x, Tensor y) -> Tensor"
        custom_def = torch.library.CustomOpDef(
            "test_diag_lib", "SubModelK_forward", schema_str, _sub_forward_
        )
        custom_def.register_kernel("cpu")(_sub_forward_)
        custom_def._abstract_fn = _symbolic_forward

        def _new_forward(*args, _name="SubModelK_forward", **kwargs):
            f = getattr(torch.ops.test_diag_lib, _name)
            return f(*args, **kwargs)

        model.sub.forward = _new_forward
        ep = torch.export.export(model, (x,), dynamic_shapes=ds)
        model.sub.forward = _forward
        self.assertIn("torch.ops.test_diag_lib.SubModelK_forward", str(ep))

    @requires_torch("2.6")
    def test_infer_shape_type_from_execution_piece_try_weight(self):
        import torch

        class SubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn((1, 6)))

            def forward(self, x, y):
                return x * self.weight - y

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x):
                return self.sub(x, x * x)

        model = Model()
        x = torch.randn((5, 6))
        ds = {"x": {0: torch.export.Dim.DYNAMIC}}
        ep = torch.export.export(model, (x,), dynamic_shapes=ds)
        self.assertNotEmpty(ep)

        _forward = model.sub.forward

        def _sub_forward_(x, y, _call=_forward):
            return _call(x, y)

        def _symbolic_forward(x, y):
            return torch.empty_like(x)

        schema_str = "(Tensor x, Tensor y) -> Tensor"
        custom_def = torch.library.CustomOpDef(
            "test_diag_lib", "SubModelWK_forward", schema_str, _sub_forward_
        )
        custom_def.register_kernel("cpu")(_sub_forward_)
        custom_def._abstract_fn = _symbolic_forward

        def _new_forward(x, y, _name="SubModelWK_forward"):
            f = getattr(torch.ops.test_diag_lib, _name)
            return f(x, y)

        model.sub.forward = _new_forward
        ep = torch.export.export(model, (x,), dynamic_shapes=ds)
        model.sub.forward = _forward
        self.assertIn("torch.ops.test_diag_lib.SubModelWK_forward", str(ep))

    @requires_torch("2.6")
    def test_infer_shape_type_from_execution_piece_try_weight_args(self):
        import torch

        class SubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn((1, 6)))

            def forward(self, x, y):
                return x * self.weight - y

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x):
                return self.sub(x, x * x)

        model = Model()
        x = torch.randn((5, 6))
        ds = {"x": {0: torch.export.Dim.DYNAMIC}}
        ep = torch.export.export(model, (x,), dynamic_shapes=ds)
        self.assertNotEmpty(ep)

        _forward = model.sub.forward

        def _sub_forward_(*args, _call_=_forward, **kwargs):
            return _call_(*args, **kwargs)

        def _symbolic_forward(*args):
            return torch.empty_like(args[0])

        schema_str = "(Tensor x, Tensor y) -> Tensor"
        custom_def = torch.library.CustomOpDef(
            "test_diag_lib", "SubModelKAW_forward", schema_str, _sub_forward_
        )
        custom_def.register_kernel("cpu")(_sub_forward_)
        custom_def._abstract_fn = _symbolic_forward

        def _new_forward(*args, _name="SubModelKAW_forward", **kwargs):
            f = getattr(torch.ops.test_diag_lib, _name)
            return f(*args, **kwargs)

        model.sub.forward = _new_forward
        ep = torch.export.export(model, (x,), dynamic_shapes=ds)
        model.sub.forward = _forward
        self.assertIn("torch.ops.test_diag_lib.SubModelKAW_forward", str(ep))
        self.assertIn('sub_model_kaw_forward: "f32[s0, 6]"', str(ep))

    @requires_torch("2.6")
    @hide_stdout()
    def test_infer_shape_type_from_execution_piece_all(self):
        import torch

        class SubModel(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x):
                return self.sub(x, x * x)

        model = Model()
        x = torch.randn((5, 6))
        y = model(x)
        self.assertNotEmpty(y)

        inputs = [
            ((torch.randn((5, 6)),), {}),
            ((torch.randn((6, 6)),), {}),
        ]

        diag = infer_shape_type_from_execution(model, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=1,
            replace_by_custom_op=CustomOpStrategy.ALWAYS,
            quiet=10,
        )
        self.assertNotEmpty(ep)
        assert hasattr(diag, "fx"), "No exported program found in diag."
        atts = [k for k in dir(diag) if k.startswith("exporter")]
        self.assertEqual(set(atts), {"exporter_discs", "exporter_outputs", "exporter_status"})
        self.assertIn("torch.ops.diag_lib.C__main__.default", str(ep))
        self.assertNotEmpty(diag.forward_custom_op_schema)
        self.assertNotEmpty(diag.children[0].forward_custom_op_schema)

    @requires_torch("2.6")
    @hide_stdout()
    def test_export_piece_auto(self):
        import torch

        class SubModelFail(torch.nn.Module):
            def forward(self, x):
                if x.sum() > 0:
                    return x
                return -x

        class SubModel(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()
                self.subfail = SubModelFail()

            def forward(self, x):
                return self.sub(x, x * x) + self.subfail(x)

        model = Model()
        x = torch.randn((5, 6))
        y = model(x)
        self.assertNotEmpty(y)

        inputs = [
            ((torch.randn((5, 6)),), {}),
            ((torch.randn((6, 6)),), {}),
        ]

        diag = infer_shape_type_from_execution(model, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=1,
            replace_by_custom_op=CustomOpStrategy.ONLY_IF_FAILING,
            quiet=1,
        )
        self.assertNotEmpty(ep)
        assert hasattr(diag, "fx"), "No exported program found in diag."
        atts = [k for k in dir(diag) if k.startswith("exporter")]
        self.assertEqual(set(atts), {"exporter_discs", "exporter_outputs", "exporter_status"})
        self.assertIn("torch.ops.diag_lib.CC__main___subfail.default", str(ep))
        self.assertNotEmpty(diag.children[0].forward_custom_op_schema)
        self.assertNotEmpty(diag.children[1].forward_custom_op_schema)
        report = diag.get_export_status()
        self.assertIn("OK with children as custom ops", report)

    @requires_torch("2.6")
    @hide_stdout()
    def test_export_piece_none(self):
        import torch

        def memo(x: torch.Tensor, y: Optional[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
            pass

        sch = torch.library.infer_schema(memo, mutates_args=())
        self.assertEqual(sch, "(Tensor x, Tensor? y, Tensor z) -> Tensor")

        class SubModelFail(torch.nn.Module):
            def forward(self, x, y, z):
                if y is None:
                    if x.sum() > 0:
                        return x
                    return -x
                return y + z

        class SubModel(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()
                self.subfail = SubModelFail()

            def forward(self, x):
                z = x * x
                return self.sub(x, z) + self.subfail(x, None, z)

        model = Model()
        x = torch.randn((5, 6))
        y = model(x)
        self.assertNotEmpty(y)

        inputs = [
            ((torch.randn((5, 6)),), {}),
            ((torch.randn((6, 6)),), {}),
        ]

        diag = infer_shape_type_from_execution(model, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=1,
            replace_by_custom_op=CustomOpStrategy.ONLY_IF_FAILING,
            quiet=1,
        )
        self.assertNotEmpty(ep)
        report = diag.get_export_status()
        self.assertIn("OK with children as custom ops", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
