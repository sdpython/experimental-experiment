import unittest
from typing import List, Optional
from experimental_experiment.ext_test_case import ExtTestCase, hide_stdout, requires_torch
from experimental_experiment.helpers import string_type
from experimental_experiment.torch_interpreter.piece_by_piece import (
    trace_execution_piece_by_piece,
    CustomOpStrategy,
    StatusExport,
    StatusExportCode,
)
from experimental_experiment.torch_interpreter.piece_by_piece_serialize import (
    choose_kwargs_for_dynamic_shapes,
    extract_names_from_schema,
    serialize_args,
)


class TestPieceByPiece(ExtTestCase):
    def test_name_status_export(self):
        st = StatusExport(StatusExportCode.NONE)
        self.assertEqual(st.status.name, "NONE")
        st = StatusExport(StatusExportCode.OK)
        self.assertEqual(st.status.name, "OK")
        st = StatusExport(StatusExportCode.OK | StatusExportCode.CHILDC)
        self.assertEqual(st.status.name, "OK_CHILDC")
        st = StatusExport(StatusExportCode.OK | StatusExportCode.CUSTOM)
        self.assertEqual(st.status.name, "OK_CUSTOM")
        st = StatusExport(StatusExportCode.FAIL)
        self.assertEqual(st.status.name, "FAIL")
        st = StatusExport(StatusExportCode.FAIL | StatusExportCode.CHILDC)
        self.assertEqual(st.status.name, "FAIL_CHILDC")
        st = StatusExport(StatusExportCode.FAIL | StatusExportCode.CUSTOM)
        self.assertEqual(st.status.name, "FAIL_CUSTOM")
        #
        st = StatusExportCode.FAIL
        self.assertEqual(st.name, "FAIL")
        st = st.remove(StatusExportCode.FAIL)
        self.assertEqual(st.name, "NONE")

    @requires_torch("2.6")
    @hide_stdout()
    def test_trace_execution_piece_by_piece_args(self):
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

        diag = trace_execution_piece_by_piece(big, inputs, verbose=1)
        pretty = diag.pretty_text(with_dynamic_shape=True)
        self.assertIn("DS=", pretty)

    @requires_torch("2.6")
    @hide_stdout()
    def test_trace_execution_piece_by_piece_kwargs(self):
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

        diag = trace_execution_piece_by_piece(big, inputs, verbose=1)
        pretty = diag.pretty_text(with_dynamic_shape=True)
        self.assertIn("DS=", pretty)

    @requires_torch("2.6")
    @hide_stdout()
    def test_trace_execution_piece_by_piece_phi2(self):
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

        diag = trace_execution_piece_by_piece(model, inputs, verbose=2)
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
    def test_trace_execution_piece_by_piece_export(self):
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

        diag = trace_execution_piece_by_piece(big, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
        )
        self.assertIsInstance(ep, StatusExport)
        assert hasattr(diag, "fx"), "No exported program found in diag."
        atts = [k for k in dir(diag) if k.startswith("exporter")]
        self.assertEqual(set(atts), {"exporter_discs", "exporter_outputs", "exporter_status"})

    @requires_torch("2.6")
    @hide_stdout()
    def test_trace_execution_piece_by_piece_args_to_kwargs(self):
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

        diag = trace_execution_piece_by_piece(model, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            quiet=0,
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
    @hide_stdout()
    def test_trace_execution_piece_by_piece_args_not_to_kwargs(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x=None, **kwargs):
                return x.abs()

        model = Model()
        x = torch.randn((5, 6))
        y = model(x=x)
        self.assertNotEmpty(y)

        inputs = [
            (tuple(), {"x": x}),
            (tuple(), {"x": torch.randn((6, 6))}),
        ]

        diag = trace_execution_piece_by_piece(model, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            quiet=0,
        )
        self.assertNotEmpty(ep)
        assert hasattr(diag, "fx"), "No exported program found in diag."
        atts = [k for k in dir(diag) if k.startswith("exporter")]
        self.assertEqual(set(atts), {"exporter_discs", "exporter_outputs", "exporter_status"})
        ds = diag.guess_dynamic_shapes()
        self.assertEqual(ds, (tuple(), {"x": {0: torch.export.Dim.DYNAMIC}}))
        _a, _kw, ds = diag._move_to_kwargs(*diag.inputs[0], ds)
        self.assertEqual(ds, (tuple(), {"x": {0: torch.export.Dim.DYNAMIC}}))

    @requires_torch("2.6")
    @hide_stdout()
    def test_trace_execution_piece_by_piece_args_or_not_to_kwargs(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, y=None, **kwargs):
                return x.abs() + torch.exp(y) + torch.cos(kwargs["z"])

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((5, 6))
        z = torch.randn((5, 6))
        w = model(x, y=y, z=z)
        self.assertNotEmpty(w)

        inputs = [
            ((x,), {"y": y, "z": z}),
            ((torch.randn((6, 6)),), {"y": torch.randn((6, 6)), "z": torch.randn((6, 6))}),
        ]

        diag = trace_execution_piece_by_piece(model, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            quiet=0,
        )
        self.assertNotEmpty(ep)
        assert hasattr(diag, "fx"), "No exported program found in diag."
        atts = [k for k in dir(diag) if k.startswith("exporter")]
        self.assertEqual(set(atts), {"exporter_discs", "exporter_outputs", "exporter_status"})
        ds = diag.guess_dynamic_shapes()
        self.assertEqual(
            ds,
            (
                ({0: torch.export.Dim.DYNAMIC},),
                {"y": {0: torch.export.Dim.DYNAMIC}, "z": {0: torch.export.Dim.DYNAMIC}},
            ),
        )
        _a, _kw, ds = diag._move_to_kwargs(*diag.inputs[0], ds)
        self.assertEqual(
            ds,
            (
                tuple(),
                {
                    "x": {0: torch.export.Dim.DYNAMIC},
                    "y": {0: torch.export.Dim.DYNAMIC},
                    "kwargs": {"z": {0: torch.export.Dim.DYNAMIC}},
                },
            ),
        )

    @requires_torch("2.6")
    def test_trace_execution_piece_by_piece_piece_try_no_weight(self):
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
    def test_trace_execution_piece_by_piece_piece_try_no_weight_args(self):
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
    def test_trace_execution_piece_by_piece_piece_try_weight(self):
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
    def test_trace_execution_piece_by_piece_piece_try_weight_args(self):
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
    def test_trace_execution_piece_by_piece_piece_all(self):
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

        diag = trace_execution_piece_by_piece(model, inputs)
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
        self.assertIn("torch.ops.diag_lib.C_Model.default", str(ep.exported))
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

        diag = trace_execution_piece_by_piece(model, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=2,
            replace_by_custom_op=CustomOpStrategy.ONLY_IF_FAILING,
            quiet=1,
        )
        self.assertIsInstance(ep, StatusExport)
        self.assertIsInstance(ep.exported, torch.export.ExportedProgram)
        assert hasattr(diag, "fx"), "No exported program found in diag."
        atts = [k for k in dir(diag) if k.startswith("exporter")]
        self.assertEqual(set(atts), {"exporter_discs", "exporter_outputs", "exporter_status"})
        self.assertIn("torch.ops.diag_lib.C_Model_subfail.default", str(ep.exported))
        self.assertNotEmpty(diag.children[0].forward_custom_op_schema)
        self.assertNotEmpty(diag.children[1].forward_custom_op_schema)
        report = diag.get_export_report()
        self.assertIn("OK_CHILDC", report)

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

        diag = trace_execution_piece_by_piece(model, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            replace_by_custom_op=CustomOpStrategy.ONLY_IF_FAILING,
            quiet=1,
        )
        self.assertNotEmpty(ep)
        report = diag.get_export_report()
        self.assertIn("OK_CHILDC", report)

    @requires_torch("2.6")
    @hide_stdout()
    def test_export_piece_dynamic_cache(self):
        import torch
        import transformers

        def memo(
            x: torch.Tensor, y: List[torch.Tensor], z: List[torch.Tensor]
        ) -> List[torch.Tensor]:
            pass

        sch = torch.library.infer_schema(memo, mutates_args=())
        self.assertEqual(sch, "(Tensor x, Tensor[] y, Tensor[] z) -> Tensor[]")

        class SubModelCache(torch.nn.Module):
            def forward(self, cache):
                d = cache.__class__()
                d.update(cache.key_cache[0] + 1, cache.value_cache[0] + 2, 0)
                return d

        class SubModel(torch.nn.Module):
            def forward(self, x, cache):
                return x + cache.key_cache[0] + cache.value_cache[0]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()
                self.subcache = SubModelCache()

            def forward(self, x, cache):
                return self.sub(x, self.subcache(cache))

        cache = transformers.cache_utils.DynamicCache()
        cache.update(torch.ones((5, 6)), torch.ones((5, 6)) + 2, 0)
        model = Model()
        x = torch.randn((5, 6))
        y = model(x, cache)
        self.assertNotEmpty(y)

        cache2 = transformers.cache_utils.DynamicCache()
        cache2.update(torch.ones((6, 6)), torch.ones((6, 6)) + 2, 0)

        inputs = [
            ((torch.randn((5, 6)), cache), {}),
            ((torch.randn((6, 6)), cache2), {}),
        ]

        expected_dyn_shapes = "(({0: DYN}, [[{0: DYN}], [{0: DYN}]]), {})"
        diag = trace_execution_piece_by_piece(model, inputs)
        dyn_shapes = diag.guess_dynamic_shapes()
        got = str(dyn_shapes).replace("<_DimHint.DYNAMIC: 3>", "DYN")
        self.assertEqual(expected_dyn_shapes, got)

        expected = [
            ((0, torch.float32),),
            ((0, torch.float32),),
            ((0, torch.float32), (0, torch.float32)),
        ]
        c_schema = [
            "(Tensor x, Tensor cache_n2_0, Tensor cache_n2_1) -> Tensor",
            "(Tensor x, Tensor cache_n2_0, Tensor cache_n2_1) -> Tensor",
            "(Tensor cache_n2_0, Tensor cache_n2_1) -> Tensor[]",
        ]
        for iexp, obj, esch in zip(expected, diag, c_schema):
            mapping = obj.build_shape_mapping_indices()
            self.assertEqual(iexp, mapping)
            sch = obj.build_c_schema()
            self.assertEqual(esch, sch)

        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            replace_by_custom_op=CustomOpStrategy.ALWAYS,
            quiet=0,
            bypass_kwargs=dict(patch_transformers=True, replace_dynamic_cache=True),
        )
        self.assertNotEmpty(ep)
        assert hasattr(diag, "fx"), "No exported program found in diag."
        atts = [k for k in dir(diag) if k.startswith("exporter")]
        self.assertEqual(set(atts), {"exporter_discs", "exporter_outputs", "exporter_status"})

        c_schema = [
            "torch.ops.diag_lib.C_Model.default",
            "torch.ops.diag_lib.C_Model_sub.default",
            "torch.ops.diag_lib.C_Model_subcache.default",
        ]
        for obj, esch in zip(diag, c_schema):
            ep = obj.fx
            self.assertIn(esch, str(ep))

    @requires_torch("2.6")
    @hide_stdout()
    def test_trace_execution_piece_by_piece_piece_local(self):
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

        diag = trace_execution_piece_by_piece(model, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
            quiet=0,
        )
        self.assertNotEmpty(ep)
        assert hasattr(diag, "fx"), "No exported program found in diag."
        atts = [k for k in dir(diag) if k.startswith("exporter")]
        self.assertEqual(set(atts), {"exporter_discs", "exporter_outputs", "exporter_status"})
        self.assertIn("torch.ops.diag_lib.C_Model_sub.default", str(ep.exported))
        self.assertEmpty(diag.forward_custom_op_schema)
        self.assertNotEmpty(diag.children[0].forward_custom_op_schema)

    @requires_torch("2.6")
    def test_piece_by_piece_piece_exporter_report(self):
        import torch

        class SubSubModel(torch.nn.Module):
            def forward(self, x, y):
                return x * y

        class SubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.subsub = SubSubModel()

            def forward(self, x, y):
                return self.subsub(x - y, y)

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

        diag = trace_execution_piece_by_piece(model, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=0,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
            quiet=0,
        )
        self.assertNotEmpty(ep)
        report = diag.get_export_report(exported_program=True)
        self.assertIn(
            'ep:         def forward(self, x: "f32[s0, 6]", y: "f32[s0, 6]"):', report
        )
        report = diag.get_export_report(fx=True)
        self.assertIn(
            "fx:     %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor]",
            report,
        )

    def test_extract_names_from_schema(self):
        expected = [
            ("((Tensor x, Tensor y) -> Tensor)", ["x", "y"]),
            ("((Tensor x, Tensor? y) -> Tensor)", ["x", "y"]),
        ]
        for a, b in expected:
            g = extract_names_from_schema(a)
            self.assertEqual(b, g)

    def test_serialize_args_in(self):
        import torch

        inputs_args = [((torch.randn((5, 6)), torch.randn((5, 6))), {})]

        args, kwargs = serialize_args(*inputs_args[0], schema="(Tensor x, Tensor y) -> Tensor")
        self.assertNotEmpty(args)
        self.assertEqual(kwargs, {})
        sargs = string_type(args, with_shape=True)
        self.assertEqual(sargs, "(T1s5x6,T1s5x6)")
        self.assertEqualArray(inputs_args[0][0][0], args[0])
        self.assertEqualArray(inputs_args[0][0][1], args[1])

    def test_serialize_args_in_dict(self):
        import torch

        inputs_args = [((torch.randn((5, 6)),), {"y": torch.randn((5, 6))})]

        args, kwargs = serialize_args(
            *inputs_args[0], schema="(Tensor x, Tensor y) -> Tensor", args_names=["x", "y"]
        )
        self.assertNotEmpty(args)
        self.assertEqual(kwargs, {})
        sargs = string_type(args, with_shape=True)
        self.assertEqual(sargs, "(T1s5x6,T1s5x6)")
        self.assertEqualArray(inputs_args[0][0][0], args[0])
        self.assertEqualArray(inputs_args[0][1]["y"], args[1])

    def test_serialize_args_out(self):
        import torch

        inputs_args = [((torch.randn((5, 6)), torch.randn((5, 6))), None)]

        args = serialize_args(*inputs_args[0], schema="(Tensor x, Tensor y) -> Tensor")
        self.assertIsInstance(args, tuple)
        sargs = string_type(args, with_shape=True)
        self.assertEqual(sargs, ("(T1s5x6,T1s5x6)"))
        self.assertEqualArray(inputs_args[0][0][0], args[0])
        self.assertEqualArray(inputs_args[0][0][1], args[1])

    def test_serialize_args_out1(self):
        import torch

        x = torch.randn((5, 6))
        args = serialize_args(x, None, schema="(Tensor x, Tensor y) -> Tensor")
        self.assertIsInstance(args, torch.Tensor)
        sargs = string_type(args, with_shape=True)
        self.assertEqual(sargs, "T1s5x6")
        self.assertEqualArray(x, args)

    @requires_torch("2.6")
    @hide_stdout()
    def test_piece_by_piece_piece_custom_kwargs_always(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, y=None):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((5, 6))
        z = model(x, y=y)
        self.assertNotEmpty(z)

        inputs = [
            ((torch.randn((5, 6)),), {"y": torch.randn((5, 6))}),
            ((torch.randn((6, 6)),), {"y": torch.randn((6, 6))}),
        ]

        diag = trace_execution_piece_by_piece(model, inputs)
        ds = diag.guess_dynamic_shapes()
        sds = str(ds).replace("<_DimHint.DYNAMIC: 3>", "DYN")
        self.assertEqual(sds, "(({0: DYN},), {'y': {0: DYN}})")
        choose = choose_kwargs_for_dynamic_shapes(*ds, diag.forward_positioned_parameter_names)
        schoose = str(choose).replace("<_DimHint.DYNAMIC: 3>", "DYN")
        self.assertEqual(schoose, "{'y': {0: DYN}, 'x': {0: DYN}}")
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=4,
            replace_by_custom_op=CustomOpStrategy.ALWAYS,
            quiet=0,
        )
        self.assertNotEmpty(ep)
        report = diag.get_export_report(exported_program=True)
        self.assertIn(
            'ep:         def forward(self, x: "f32[s0, 6]", y: "f32[s1, 6]"):', report
        )
        report = diag.get_export_report(fx=True)
        self.assertIn("torch.ops.diag_lib.C_Model.default", report)

    @requires_torch("2.6")
    def test_piece_by_piece_piece_custom_kwargs_local(self):
        import torch

        class SubModel(torch.nn.Module):
            def forward(self, x, y=None):
                if y is None:
                    return x**2
                return x**2 - y

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x, y=None):
                return self.sub(x, y=y * x)

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((5, 6))
        z = model(x, y=y)
        self.assertNotEmpty(z)

        inputs = [
            ((torch.randn((5, 6)),), {"y": torch.randn((5, 6))}),
            ((torch.randn((6, 6)),), {"y": torch.randn((6, 6))}),
        ]

        diag = trace_execution_piece_by_piece(model, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=0,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
            quiet=0,
        )
        self.assertNotEmpty(ep)
        report = diag.get_export_report(exported_program=True)
        self.assertIn(
            'ep:         def forward(self, x: "f32[s0, 6]", y: "f32[s0, 6]"):', report
        )
        report = diag.get_export_report(fx=True)
        self.assertIn(
            "fx:     %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor]",
            report,
        )

    @requires_torch("2.6")
    def test_piece_by_piece_piece_bool(self):
        import torch

        class SubModel(torch.nn.Module):
            def forward(self, x, y=None, square=False):
                if y is None:
                    if square:
                        return x**2
                    return torch.abs(x)
                return x**2 - y

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x, y=None):
                return self.sub(x, y=y * x, square=True)

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((5, 6))
        z = model(x, y=y)
        self.assertNotEmpty(z)

        inputs = [
            ((torch.randn((5, 6)),), {"y": torch.randn((5, 6))}),
            ((torch.randn((6, 6)),), {"y": torch.randn((6, 6))}),
        ]

        diag = trace_execution_piece_by_piece(model, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=0,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
            quiet=0,
        )
        self.assertNotEmpty(ep)
        report = diag.get_export_report(fx=True)
        self.assertIn("torch.ops.aten.pow", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
