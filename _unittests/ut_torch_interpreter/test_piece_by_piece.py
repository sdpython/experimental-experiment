import ast
import unittest
import inspect
from typing import List, Optional
import torch
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_torch,
    requires_transformers,
    ignore_warnings,
    skipif_ci_windows,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache, CacheKeyValue
from experimental_experiment.helpers import string_type
from onnx_diagnostic.torch_export_patches import (
    torch_export_patches,
    register_additional_serialization_functions,
)
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
    tree_spec_as_name,
    tree_spec_from_name,
)


def traceable_local_f(x, y):
    return x.abs() + y.abs() + 1e-5


def traceable_local_f_recursive(x, y):
    return traceable_local_f(x, y)


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

    def test_serizalize_arg_1(self):
        import torch

        x = torch.randn((5, 6))
        args, _kwargs = serialize_args((x,), {}, schema=None, args_names=["x", "flash_args"])
        st = string_type(args, with_shape=True)
        self.assertEqual(st, "(T1s5x6,)")

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
    def test_trace_execution_piece_by_piece_export(self):
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

    @requires_torch("2.12", "https://github.com/pytorch/pytorch/issues/150022")
    @hide_stdout()
    def test_trace_execution_piece_by_piece_args_to_kwargs(self):
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

    @requires_torch("2.9", "https://github.com/pytorch/pytorch/issues/150022")
    @hide_stdout()
    def test_trace_execution_piece_by_piece_args_not_to_kwargs(self):
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

    @requires_torch("2.12")
    @hide_stdout()
    def test_trace_execution_piece_by_piece_args_or_not_to_kwargs(self):
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
        self.assertInOr(
            (
                'sub_model_kaw_forward: "f32[s0, 6]"',
                'sub_model_kaw_forward: "f32[s35, 6]"',
                'sub_model_kaw_forward: "f32[s77, 6]"',
            ),
            str(ep),
        )

    @requires_torch("2.6")
    @hide_stdout()
    def test_trace_execution_piece_by_piece_piece_all(self):
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

    @requires_torch("2.7")
    @ignore_warnings(UserWarning)
    @hide_stdout()
    def test_export_piece_dynamic_cache(self):
        def memo(
            x: torch.Tensor, y: List[torch.Tensor], z: List[torch.Tensor]
        ) -> List[torch.Tensor]:
            pass

        sch = torch.library.infer_schema(memo, mutates_args=())
        self.assertEqual(sch, "(Tensor x, Tensor[] y, Tensor[] z) -> Tensor[]")

        class SubModelCache(torch.nn.Module):
            def forward(self, cache):
                d = cache.__class__()
                dc = CacheKeyValue(cache)
                d.update(dc.key_cache[0] + 1, dc.value_cache[0] + 2, 0)
                return d

        class SubModel(torch.nn.Module):
            def forward(self, x, cache):
                dc = CacheKeyValue(cache)
                return x + dc.key_cache[0] + dc.value_cache[0]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()
                self.subcache = SubModelCache()

            def forward(self, x, cache):
                return self.sub(x, self.subcache(cache))

        cache = make_dynamic_cache([(torch.ones((5, 6, 5, 6)), torch.ones((5, 6, 5, 6)) + 2)])
        model = Model()
        x = torch.randn((5, 6, 5, 6))
        y = model(x, cache)
        self.assertNotEmpty(y)

        cache2 = make_dynamic_cache([(torch.ones((6, 6, 6, 6)), torch.ones((6, 6, 6, 6)) + 2)])

        inputs = [
            ((torch.randn((5, 6, 5, 6)), cache), {}),
            ((torch.randn((6, 6, 6, 6)), cache2), {}),
        ]

        expected_dyn_shapes = "(({0: DYN, 2: DYN}, [{0: DYN, 2: DYN}, {0: DYN, 2: DYN}]), {})"
        diag = trace_execution_piece_by_piece(model, inputs)
        dyn_shapes = diag.guess_dynamic_shapes()
        got = (
            str(dyn_shapes)
            .replace("<_DimHint.DYNAMIC: 3>", "DYN")
            .replace("<_DimHintType.DYNAMIC: 3>", "DYN")
            .replace("_DimHint(type=DYN)", "DYN")
            .replace("_DimHint(type=DYN, min=None, max=None, _factory=True)", "DYN")
            .replace("DimHint(DYNAMIC)", "DYN")
        )
        print(got)
        self.assertEqual(expected_dyn_shapes, got)
        print(diag.pretty_text(with_shape=True, with_min_max=False))

        expected = [
            ((0, torch.float32, None),),
            ((0, torch.float32, None),),
            ((0, torch.float32, None), (0, torch.float32, None)),
        ]
        c_schema = [
            "(Tensor x, Tensor cache_n2_0, Tensor cache_n2_1) -> Tensor",
            "(Tensor x, Tensor cache_n2_0, Tensor cache_n2_1) -> Tensor",
            "(Tensor cache_n2_0, Tensor cache_n2_1) -> Tensor[]",
        ]
        for _iexp, obj, esch in zip(expected, diag, c_schema):
            # serialization function must be registered
            # mapping = obj.build_shape_mapping_indices()
            # self.assertEqual(iexp, mapping)
            sch = obj.build_c_schema()
            self.assertEqual(esch, sch)

        with torch_export_patches(patch_transformers=True):
            ep = diag.try_export(
                exporter="fx",
                use_dynamic_shapes=True,
                exporter_kwargs=dict(strict=False),
                verbose=0,
                replace_by_custom_op=CustomOpStrategy.ALWAYS,
                quiet=0,
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

    @requires_torch("2.7")
    @ignore_warnings(UserWarning)
    @hide_stdout()
    def test_export_piece_dynamic_cache_io(self):
        class SubModelCacheIn(torch.nn.Module):
            def forward(self, cache):
                cache = CacheKeyValue(cache)
                return cache.key_cache[0] * cache.value_cache[0]

        class SubModelCacheOut(torch.nn.Module):
            def forward(self, x, y):
                d = cache.__class__()
                d.update(x + 1, y + 2, 0)
                return d

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.subin = SubModelCacheIn()
                self.subout = SubModelCacheOut()

            def forward(self, x, y):
                cache = self.subout(x, y)
                return self.subin(cache)

        cache = make_dynamic_cache([(torch.ones((5, 6, 5, 6)), torch.ones((5, 6, 5, 6)) + 2)])
        model = Model()
        x = torch.randn((5, 6, 5, 6))
        y = torch.randn((5, 6, 5, 6))

        inputs = [
            ((x, y), {}),
            ((torch.randn((6, 6, 6, 6)), torch.randn((6, 6, 6, 6))), {}),
        ]

        expected_dyn_shapes = "(({0: DYN, 2: DYN}, {0: DYN, 2: DYN}), {})"
        diag = trace_execution_piece_by_piece(model, inputs)
        dyn_shapes = diag.guess_dynamic_shapes()
        got = (
            str(dyn_shapes)
            .replace("<_DimHint.DYNAMIC: 3>", "DYN")
            .replace("<_DimHintType.DYNAMIC: 3>", "DYN")
            .replace("_DimHint(type=DYN)", "DYN")
            .replace("_DimHint(type=DYN, min=None, max=None, _factory=True)", "DYN")
            .replace("DimHint(DYNAMIC)", "DYN")
        )
        self.assertEqual(expected_dyn_shapes, got)

        expected = [
            ((0, torch.float32),),
            ((0, torch.float32),),
            ((0, torch.float32), (0, torch.float32)),
        ]
        c_schema = [
            "(Tensor x, Tensor y) -> Tensor",
            "(Tensor cache_n2_0, Tensor cache_n2_1) -> Tensor",
            "(Tensor x, Tensor y) -> Tensor[]",
        ]
        for _iexp, obj, esch in zip(expected, diag, c_schema):
            #    mapping = obj.build_shape_mapping_indices()
            #    self.assertEqual(iexp, mapping)
            sch = obj.build_c_schema()
            self.assertEqual(esch, sch)

        with torch_export_patches(patch_transformers=True):
            ep = diag.try_export(
                exporter="fx",
                use_dynamic_shapes=True,
                exporter_kwargs=dict(strict=False),
                verbose=10,
                replace_by_custom_op=CustomOpStrategy.ALWAYS,
                quiet=0,
            )
        self.assertNotEmpty(ep)
        assert hasattr(diag, "fx"), "No exported program found in diag."
        atts = [k for k in dir(diag) if k.startswith("exporter")]
        self.assertEqual(set(atts), {"exporter_discs", "exporter_outputs", "exporter_status"})

        c_schema = [
            "torch.ops.diag_lib.C_Model.default",
            "torch.ops.diag_lib.C_Model_subin.default",
            "torch.ops.diag_lib.C_Model_subout.default",
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
    @hide_stdout()
    def test_trace_execution_piece_by_piece_piece_local_local(self):
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
        diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
            quiet=0,
        )
        ep = diag.export_local(use_dynamic_shapes=True, exporter_kwargs=dict(strict=False))
        self.assertNotEmpty(ep)

    @requires_torch("2.6")
    @hide_stdout()
    def test_to_onnx_local(self):
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
        diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
            quiet=0,
        )
        onx = diag.to_onnx_local(verbose=10)
        self.assertNotEmpty(onx)
        self.dump_onnx("test_to_onnx_local.onnx", onx)
        ref = ExtendedReferenceEvaluator(onx)
        self.assertEqualArray(y, ref.run(None, {ref.input_names[0]: x.numpy()})[0])

    @requires_torch("2.6")
    @hide_stdout()
    def test_to_onnx_local_2s(self):
        import torch

        class SubModel(torch.nn.Module):
            def forward(self, x, y):
                return x - y, x.to(torch.int64)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x):
                a, b = self.sub(x, x * x)
                return a + b.to(a.dtype) * 2

        model = Model()
        x = torch.randn((5, 6))
        y = model(x)
        self.assertNotEmpty(y)

        inputs = [
            ((torch.randn((5, 6)),), {}),
            ((torch.randn((6, 6)),), {}),
        ]

        diag = trace_execution_piece_by_piece(model, inputs)
        diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
            quiet=0,
        )
        onx = diag.to_onnx_local(verbose=10, optimize=True)
        self.dump_onnx("test_to_onnx_local_2s.onnx", onx)
        self.assertNotIn("SequenceAt", str(onx))
        self.assertNotEmpty(onx)
        ref = ExtendedReferenceEvaluator(onx, verbose=10)
        g = ref.run(None, {ref.input_names[0]: x.numpy()})
        self.assertEqualArray(y, g[0])

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
        self.assertInOr(
            (
                'ep:         def forward(self, x: "f32[s0, 6]", y: "f32[s0, 6]"):',
                'ep:         def forward(self, x: "f32[s35, 6]", y: "f32[s35, 6]"):',
                'ep:         def forward(self, x: "f32[s17, 6]", y: "f32[s17, 6]"):',
            ),
            report,
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
        sds = (
            str(ds)
            .replace("<_DimHint.DYNAMIC: 3>", "DYN")
            .replace("<_DimHintType.DYNAMIC: 3>", "DYN")
            .replace("_DimHint(type=DYN)", "DYN")
            .replace("_DimHint(type=DYN, min=None, max=None, _factory=True)", "DYN")
            .replace("DimHint(DYNAMIC)", "DYN")
        )
        self.assertEqual(sds, "(({0: DYN},), {'y': {0: DYN}})")
        choose = choose_kwargs_for_dynamic_shapes(*ds, diag.forward_positioned_parameter_names)
        schoose = (
            str(choose)
            .replace("<_DimHint.DYNAMIC: 3>", "DYN")
            .replace("<_DimHintType.DYNAMIC: 3>", "DYN")
            .replace("_DimHint(type=DYN)", "DYN")
            .replace("_DimHint(type=DYN, min=None, max=None, _factory=True)", "DYN")
            .replace("DimHint(DYNAMIC)", "DYN")
        )
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
        self.assertInOr(
            (
                'ep:         def forward(self, x: "f32[s0, 6]", y: "f32[s1, 6]"):',
                'ep:         def forward(self, x: "f32[s35, 6]", y: "f32[s14, 6]"):',
                'ep:         def forward(self, x: "f32[s77, 6]", y: "f32[s17, 6]"):',
            ),
            report,
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
        self.assertInOr(
            (
                'ep:         def forward(self, x: "f32[s0, 6]", y: "f32[s0, 6]"):',
                'ep:         def forward(self, x: "f32[s14, 6]", y: "f32[s14, 6]"):',
                'ep:         def forward(self, x: "f32[s17, 6]", y: "f32[s17, 6]"):',
            ),
            report,
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

    @requires_torch("2.6")
    @hide_stdout()
    def test_piece_by_piece_piece_dict(self):
        import torch

        class SubModel(torch.nn.Module):
            def forward(self, x, y):
                return dict(dm=x - y, da=x + y)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x, y):
                r = self.sub(x, y)
                return r["dm"].abs() + r["da"].abs()

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
            verbose=10,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
            quiet=0,
        )
        self.assertNotEmpty(ep)
        report = diag.get_export_report(fx=True)
        self.assertIn("torch.ops.aten.abs", report)
        self.assertEqual(diag.children[0].forward_expected_output_type, ["dict__2_da__dm"])

    @ignore_warnings(FutureWarning)
    def test_serialize_any(self):
        nested = [
            torch.randn((4, 5)),
            [torch.randn((7, 5)), torch.randn((8, 5))],
            {
                "a": torch.randn((14, 5)),
                "b": torch.randn((12, 5)),
                "cl": [torch.randn((11, 5))],
            },
        ]
        flat_list, tree_spec = torch.utils._pytree.tree_flatten(nested)

        self.assertEqual(len(flat_list), 6)
        unflatten = torch.utils._pytree.tree_unflatten(flat_list, tree_spec)
        self.assertEqualAny(nested, unflatten)

        # Let's get a name
        name = tree_spec_as_name(tree_spec, 6)
        _, new_spec = tree_spec_from_name(name)
        unflatten = torch.utils._pytree.tree_unflatten(flat_list, new_spec)
        self.assertEqualAny(nested, unflatten)

    @requires_transformers("4.49.999")
    def test_serialize_dynamic_cache(self):
        cache = make_dynamic_cache([(torch.randn((19, 5)), torch.randn((21, 5)))])

        nested = [
            torch.randn((4, 5)),
            [torch.randn((7, 5)), torch.randn((8, 5))],
            {
                "a": torch.randn((14, 5)),
                "cl": cache,
            },
        ]

        with torch_export_patches():
            flat_list, tree_spec = torch.utils._pytree.tree_flatten(nested)

            self.assertTrue(all(isinstance(i, torch.Tensor) for i in flat_list))
            self.assertEqual(len(flat_list), 6)
            unflatten = torch.utils._pytree.tree_unflatten(flat_list, tree_spec)
            self.assertEqualAny(nested, unflatten)

            # Let's get a name
            name = tree_spec_as_name(tree_spec, 6)
            _, new_spec = tree_spec_from_name(name)
            unflatten = torch.utils._pytree.tree_unflatten(flat_list, new_spec)
            self.assertEqualAny(nested, unflatten)

    @requires_torch("2.6")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    def test_piece_by_piece_piece_dict_list(self):
        class SubModel(torch.nn.Module):
            def forward(self, x, y):
                return dict(dm=x - y, da=[x + y, x * y])

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x, y):
                r = self.sub(x, y)
                return r["dm"].abs() + r["da"][0].abs() + r["da"][1].abs()

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
            verbose=10,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
            quiet=0,
        )
        self.assertNotEmpty(ep)
        report = diag.get_export_report(fx=True)
        self.assertIn("torch.ops.aten.abs", report)
        self.assertEqual(len(diag.children[0].forward_expected_output_type), 1)
        self.assertStartsWith("___", diag.children[0].forward_expected_output_type[0])

    @requires_torch("2.6")
    @requires_transformers("4.49.999")
    @hide_stdout()
    def test_piece_by_piece_piece_tuple_cache(self):
        class SubModel(torch.nn.Module):
            def forward(self, x, y):
                cache = make_dynamic_cache([(x + 1, y + 2)])
                return x + y, cache

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x, y):
                r, cache = self.sub(x, y)
                cache = CacheKeyValue(cache)
                return r.abs() + cache.key_cache[0].abs() + cache.value_cache[0].abs()

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
        with torch_export_patches():
            ep = diag.try_export(
                exporter="fx",
                use_dynamic_shapes=True,
                exporter_kwargs=dict(strict=False),
                verbose=10,
                replace_by_custom_op=CustomOpStrategy.LOCAL,
                quiet=0,
            )
        self.assertNotEmpty(ep)
        report = diag.get_export_report(fx=True)
        self.assertIn("torch.ops.aten.abs", report)
        self.assertEqual(
            diag.children[0].forward_expected_output_type, ["Tensor", "DynamicCache__2"]
        )

    @requires_torch("2.6")
    @hide_stdout()
    def test_piece_by_piece_piece_shape_fct(self):
        import torch

        class SubModel(torch.nn.Module):
            def forward(self, x):
                y = torch.arange(0, 16, dtype=x.dtype).reshape((1, 1, 16))
                return x.unsqueeze(dim=2) + y

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x):
                return self.sub(x).abs()

        model = Model()
        x = torch.randn((5, 6))
        z = model(x)
        self.assertNotEmpty(z)

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
        report = diag.get_export_report(exported_program=True)
        self.assertInOr(
            ('add: "f32[s0, 6, 16]"', 'add: "f32[s35, 6, 16]"', 'add: "f32[s77, 6, 16]"'),
            report,
        )
        for node in ep.exported.graph.nodes:
            if "val" in node.meta:
                last_node = node
        shape = tuple(last_node.meta["val"].shape)
        self.assertNotEqual(shape, (6, 16))
        self.assertEqualOr(str(shape), ("(s0, 6, 16)", "(s35, 6, 16)", "(s77, 6, 16)"))

    @requires_torch("2.6")
    @hide_stdout()
    def test_piece_by_piece_piece_shape_fct2(self):
        import torch

        class SubModel(torch.nn.Module):
            def forward(self, x):
                y = torch.arange(0, 16, dtype=x.dtype).reshape((1, 1, 16))
                return x.unsqueeze(dim=2) + y, [x, y]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x):
                res = self.sub(x)
                return res[0] + res[1][0].sum() + res[1][1].sum()

        model = Model()
        x = torch.randn((5, 6))
        z = model(x)
        self.assertNotEmpty(z)

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
        report = diag.get_export_report(exported_program=True)
        self.assertInOr(
            ('add: "f32[s0, 6, 16]"', 'add: "f32[s35, 6, 16]"', 'add: "f32[s77, 6, 16]"'),
            report,
        )
        for node in ep.exported.graph.nodes:
            if "val" in node.meta:
                last_node = node
        shape = tuple(last_node.meta["val"].shape)
        self.assertNotEqual(shape, (6, 16))
        self.assertEqualOr(str(shape), ("(s0, 6, 16)", "(s35, 6, 16)", "(s77, 6, 16)"))

    @requires_torch("2.6")
    @hide_stdout()
    def test_piece_by_piece_piece_custom_shape_fct(self):
        import torch

        class SubModel(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x, y):
                res = self.sub(x, y)
                return res

        model = Model()
        x = torch.randn((5, 1))
        y = torch.randn((1, 6))
        z = model(x, y)
        self.assertNotEmpty(z)

        inputs = [
            ((torch.randn((1, 6)), torch.randn((5, 1))), {}),
            ((torch.randn((1, 7)), torch.randn((6, 1))), {}),
        ]

        diag = trace_execution_piece_by_piece(model, inputs)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
            quiet=0,
            shape_functions={
                "SubModel": {
                    0: lambda *args, **kwargs: torch.empty((args[1].shape[0], args[0].shape[1]))
                }
            },
        )
        self.assertNotEmpty(ep)
        report = diag.get_export_report(exported_program=True)
        self.assertInOr(
            (
                'c_model_sub: "f32[s1, s0]"',
                'c_model_sub: "f32[s58, s16]"',
                'c_model_sub: "f32[s17, s27]"',
            ),
            report,
        )
        for node in ep.exported.graph.nodes:
            if "val" in node.meta:
                last_node = node
        shape = tuple(last_node.meta["val"].shape)
        self.assertNotEqual(shape, (6, 16))
        self.assertEqual(len(shape), 2)
        self.assertNotEqual(shape[0], shape[1])
        self.assertIsInstance(shape[0], (str, torch.SymInt))
        self.assertIsInstance(shape[1], (str, torch.SymInt))

    @requires_torch("2.7")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    def test_piece_by_piece_piece_dict_dict(self):
        import torch
        import transformers

        class SubModel(torch.nn.Module):
            def forward(
                self,
                x: Optional[torch.Tensor] = None,
                cache: Optional[transformers.cache_utils.DynamicCache] = None,
            ):
                cache = CacheKeyValue(cache)
                new_cache = make_dynamic_cache(
                    [(cache.key_cache[0] + x, cache.value_cache[0] + x)]
                )
                return dict(past_key_value=new_cache, mask=torch.ones_like(x))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(
                self,
                x: Optional[torch.Tensor] = None,
                cache: Optional[transformers.cache_utils.DynamicCache] = None,
            ):
                res = self.sub(x, cache)
                return res

        model = Model()
        x = torch.randn((5, 6))
        cache = make_dynamic_cache([(torch.randn((5, 6)), torch.randn((5, 6)))])
        z = model(x, cache)
        self.assertNotEmpty(z)
        with torch_export_patches(patch_transformers=True):
            z2 = model(x, cache)
            zp = CacheKeyValue(z["past_key_value"])
            zp2 = CacheKeyValue(z2["past_key_value"])
            self.assertEqual(len(zp.key_cache), len(zp2.key_cache))
            for i in range(len(zp.key_cache)):
                self.assertEqualArray(zp.key_cache[i], zp2.key_cache[i])
                self.assertEqualArray(zp.value_cache[i], zp2.value_cache[i])
            self.assertEqualArray(z["mask"], z2["mask"])

        cache2 = make_dynamic_cache([(torch.randn((6, 6)), torch.randn((6, 6)))])
        inputs = [
            (tuple(), dict(x=x, cache=cache)),
            (tuple(), dict(x=torch.randn((6, 6)), cache=cache2)),
        ]

        diag = trace_execution_piece_by_piece(model, inputs)
        with torch_export_patches(patch_transformers=True):
            ep = diag.try_export(
                exporter="fx",
                use_dynamic_shapes=True,
                exporter_kwargs=dict(strict=False),
                verbose=10,
                replace_by_custom_op=CustomOpStrategy.LOCAL,
                quiet=0,
            )
        self.assertNotEmpty(ep)
        report = diag.get_export_report(exported_program=True)
        self.assertInOr(
            (
                'ones_like: "f32[s0, 6]"',
                'ones_like: "f32[s35, 6]"',
                'ones_like: "f32[s13, 6]"',
                'ones_like: "f32[s60, 6]"',
            ),
            report,
        )
        for node in ep.exported.graph.nodes:
            if "val" in node.meta:
                last_node = node
        shape = tuple(last_node.meta["val"].shape)
        self.assertNotEqual(shape, (6, 16))
        self.assertEqualOr(str(shape), ("(s0, 6)", "(s32, 6)", "(s77, 6)"))

    @requires_torch("2.7")
    @hide_stdout()
    def test_piece_by_piece_piece_kwargs_local(self):
        import torch

        class SubModel(torch.nn.Module):
            def forward(self, x: Optional[torch.Tensor] = None, **flash_args):
                return x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x: Optional[torch.Tensor] = None, **flash_args):
                res = self.sub(x, **flash_args)
                return res

        model = Model()
        x = torch.randn((5, 6))
        z = model(x)
        self.assertNotEmpty(z)

        inputs = [
            (tuple(), dict(x=x)),
            (tuple(), dict(x=torch.randn((6, 6)))),
        ]

        diag = trace_execution_piece_by_piece(model, inputs)
        with register_additional_serialization_functions():
            ep = diag.try_export(
                exporter="fx",
                use_dynamic_shapes=True,
                exporter_kwargs=dict(strict=False),
                verbose=10,
                replace_by_custom_op=CustomOpStrategy.LOCAL,
                quiet=0,
            )
        self.assertNotEmpty(ep)
        report = diag.get_export_report(exported_program=True)
        self.assertInOr(
            (
                'c_model_sub: "f32[s0, 6]"',
                'c_model_sub: "f32[s32, 6]"',
                'c_model_sub: "f32[s77, 6]"',
            ),
            report,
        )
        for node in ep.exported.graph.nodes:
            if "val" in node.meta:
                last_node = node
        shape = tuple(last_node.meta["val"].shape)
        self.assertNotEqual(shape, (6, 16))
        self.assertEqualOr(str(shape), ("(s0, 6)", "(s32, 6)", "(s77, 6)"))

    @requires_torch("2.6")
    @hide_stdout()
    def test_trace_execution_piece_by_piece_method_name(self):
        import torch

        class MA(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class MM(torch.nn.Module):
            def execute(self, x, y):
                return x * y

        class MASMM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ma = MA()
                self.mm = MM()

            def forward(self, x, y, z):
                return self.ma(x, y) - self.mm.execute(y, z)

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

        diag = trace_execution_piece_by_piece(big, inputs, traced_method={MM: "execute"})
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

    @requires_torch("2.10.99")
    @hide_stdout()
    def test_trace_execution_functions(self):
        import torch

        def local_f(x, y):
            return x.abs() + y.abs() + 1e-5

        class SubModel(torch.nn.Module):
            def forward(self, x, y):
                return (x + y) / local_f(x, y) + traceable_local_f(x, y)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.subm = SubModel()

            def forward(self, x, y):
                return self.subm(x, y)

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((5, 6))
        z = model(x, y)
        self.assertNotEmpty(z)

        lines = inspect.getsource(SubModel.forward)
        parsed = ast.parse(f"if True:\n{lines}")
        names = [node.func.id for node in ast.walk(parsed) if isinstance(node, ast.Call)]
        self.assertEqual(names, ["traceable_local_f", "local_f"])

        inputs = [
            ((x, y), {}),
            ((torch.randn((6, 6)), torch.randn((6, 6))), {}),
        ]

        diag = trace_execution_piece_by_piece(model, inputs, trace_functions=True, verbose=1)
        self.assertEqual(len(diag.children), 1)
        self.assertEqual(len(diag.children[0].children), 1)
        all_diag = list(diag)
        self.assertEqual(len(all_diag), 3)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            quiet=0,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
        )
        self.assertNotEmpty(ep)

    @requires_torch("2.10.99")
    @hide_stdout()
    def test_trace_execution_recursive_functions(self):
        import torch

        def local_f(x, y):
            return x.abs() + y.abs() + 1e-5

        class SubModel(torch.nn.Module):
            def forward(self, x, y):
                return (x + y) / local_f(x, y) + traceable_local_f_recursive(x, y)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.subm = SubModel()

            def forward(self, x, y):
                return self.subm(x, y)

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((5, 6))
        z = model(x, y)
        self.assertNotEmpty(z)

        inputs = [
            ((x, y), {}),
            ((torch.randn((6, 6)), torch.randn((6, 6))), {}),
        ]

        diag = trace_execution_piece_by_piece(model, inputs, trace_functions=True, verbose=1)
        self.assertEqual(len(diag.children), 1)
        self.assertEqual(len(diag.children[0].children), 1)
        self.assertEqual(len(diag.children[0].children[0].children), 1)
        all_diag = list(diag)
        self.assertEqual(len(all_diag), 4)
        ep = diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            quiet=0,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
        )
        self.assertNotEmpty(ep)

    @requires_torch("2.6")
    @hide_stdout()
    def test_to_onnx_local_check_reference_cls_level_1(self):
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
        diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
            quiet=0,
        )
        onx = diag.to_onnx_local(
            verbose=10,
            check_conversion_cls=dict(cls=ExtendedReferenceEvaluator, atol=1e-5, rtol=1e-5),
        )
        self.assertLess(diag.onnx_discrepancies[0]["abs"], 1e-5)
        self.assertNotEmpty(onx)
        ref = ExtendedReferenceEvaluator(onx)
        self.assertEqualArray(y, ref.run(None, {ref.input_names[0]: x.numpy()})[0])

    @requires_torch("2.6")
    @hide_stdout()
    def test_to_onnx_local_check_reference_cls_level_2(self):
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
        diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
            quiet=0,
        )
        onx = diag.to_onnx_local(
            verbose=10,
            check_conversion_cls=dict(cls=ExtendedReferenceEvaluator, atol=1e-5, rtol=1e-5),
        )
        self.assertLess(diag.onnx_discrepancies[0]["abs"], 1e-5)
        self.assertNotEmpty(onx)
        ref = ExtendedReferenceEvaluator(onx)
        self.assertEqualArray(y, ref.run(None, {ref.input_names[0]: x.numpy()})[0])

    @hide_stdout()
    @skipif_ci_windows("broken")
    def test_controlflow_cond_submodule(self):
        import torch

        class Buffering:
            def __init__(self):
                self.stored = dict(inputs=[], outputs=[])

            def add_inputs(self, a):
                self.stored["inputs"].append(a)

            def add_outputs(self, a):
                self.stored["outputs"].append(a)

        class SubThen(torch.nn.Module):
            def forward(self, x):
                return x * x

        class SubElse(torch.nn.Module):
            def forward(self, x):
                return torch.abs(x)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub_then = SubThen()
                self.sub_else = SubElse()

            def forward(self, x):
                return torch.cond(x.sum() > 0, self.sub_then, self.sub_else, [x])

        model = Model()
        inputs = [
            ((torch.rand((5, 4)),), {}),
            ((torch.rand((6, 7)),), {}),
            # We need to add inputs so that the condition goes into both branches.
            ((-torch.rand((5, 4)),), {}),
            ((-torch.rand((6, 7)),), {}),
        ]
        expected = model(*inputs[0][0])

        # steal
        def _forward_(
            *args,
            _f=None,
            verbose=10,
            buffer_add_inputs=None,
            buffer_add_outputs=None,
            **kwargs,
        ):
            if not torch.compiler.is_compiling() and buffer_add_inputs:
                buffer_add_inputs(*args, **kwargs)
            res = _f(*args, **kwargs)
            if not torch.compiler.is_compiling() and buffer_add_outputs:
                buffer_add_outputs(res)
            return res

        verbose = 5
        buffer = Buffering()
        memo1 = SubThen.forward
        memo2 = SubElse.forward
        SubThen.forward = lambda *args, _f=memo1, verbose=verbose, **kwargs: _forward_(
            *args, _f=_f, buffer_add_inputs=buffer.add_inputs, verbose=verbose, **kwargs
        )
        SubElse.forward = lambda *args, _f=memo2, verbose=verbose, **kwargs: _forward_(
            *args, _f=_f, buffer_add_outputs=buffer.add_outputs, verbose=verbose, **kwargs
        )
        got = model(*inputs[0][0])
        SubThen.forward = memo1
        SubElse.forward = memo2
        self.assertEqualArray(expected, got)
        diag = trace_execution_piece_by_piece(model, inputs, verbose=10)

        diag.try_export(
            exporter="fx",
            use_dynamic_shapes=True,
            exporter_kwargs=dict(strict=False),
            verbose=10,
            replace_by_custom_op=CustomOpStrategy.LOCAL,
            quiet=0,
        )
        onx = diag.to_onnx_local(
            verbose=10,
            check_conversion_cls=dict(cls=ExtendedReferenceEvaluator, atol=1e-5, rtol=1e-5),
        )
        ref = ExtendedReferenceEvaluator(onx)
        for inp in inputs:
            expected = model(*inp[0])
            got = ref.run(None, {ref.input_names[0]: inp[0][0].numpy()})
            self.assertEqualArray(expected, got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
