import ast
import contextlib
import io
import math
import numbers
import unittest
import inspect
from typing import Any, Dict, List, Optional
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_torch,
    ignore_warnings,
)
from experimental_experiment.xbuilder import GraphBuilder
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.helpers import string_type, max_diff
from experimental_experiment.torch_interpreter.onnx_export_errors import (
    bypass_export_some_errors,
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
from experimental_experiment.torch_models.llm_model_helper import (
    get_phi35_mini_instruct,
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
        args, kwargs = serialize_args((x,), {}, schema=None, args_names=["x", "flash_args"])
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

        with register_additional_serialization_functions():
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
            "torch.ops.diag_lib.C_Model_sub.default",
            "torch.ops.diag_lib.C_Model_subcache.default",
        ]
        for obj, esch in zip(diag, c_schema):
            ep = obj.fx
            self.assertIn(esch, str(ep))

    @requires_torch("2.6")
    @hide_stdout()
    def test_export_piece_dynamic_cache_io(self):
        import torch
        import transformers

        class SubModelCacheIn(torch.nn.Module):
            def forward(self, cache):
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

        cache = transformers.cache_utils.DynamicCache()
        cache.update(torch.ones((5, 6)), torch.ones((5, 6)) + 2, 0)
        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((5, 6))

        inputs = [
            ((x, y), {}),
            ((torch.randn((6, 6)), torch.randn((6, 6))), {}),
        ]

        expected_dyn_shapes = "(({0: DYN}, {0: DYN}), {})"
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
            "(Tensor x, Tensor y) -> Tensor",
            "(Tensor cache_n2_0, Tensor cache_n2_1) -> Tensor",
            "(Tensor x, Tensor y) -> Tensor[]",
        ]
        for _iexp, obj, esch in zip(expected, diag, c_schema):
            #    mapping = obj.build_shape_mapping_indices()
            #    self.assertEqual(iexp, mapping)
            sch = obj.build_c_schema()
            self.assertEqual(esch, sch)

        with bypass_export_some_errors():
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

    def test_serialize_any(self):
        import torch

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

    def test_serialize_dynamic_cache(self):
        import torch
        import transformers

        cache = transformers.cache_utils.DynamicCache()
        cache.update(torch.randn((19, 5)), torch.randn((21, 5)), 0)

        nested = [
            torch.randn((4, 5)),
            [torch.randn((7, 5)), torch.randn((8, 5))],
            {
                "a": torch.randn((14, 5)),
                "cl": cache,
            },
        ]

        with bypass_export_some_errors():
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
    def test_piece_by_piece_piece_dict_list(self):
        import torch

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
    @hide_stdout()
    def test_piece_by_piece_piece_tuple_cache(self):
        import torch
        import transformers

        class SubModel(torch.nn.Module):
            def forward(self, x, y):
                cache = transformers.cache_utils.DynamicCache()
                cache.update(x + 1, y + 2, 0)
                return x + y, cache

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()

            def forward(self, x, y):
                r, cache = self.sub(x, y)
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
        with bypass_export_some_errors():
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
            diag.children[0].forward_expected_output_type, ["Tensor", "DynamicCache__1_1"]
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
        self.assertIn('add: "f32[s0, 6, 16]"', report)
        for node in ep.exported.graph.nodes:
            if "val" in node.meta:
                last_node = node
        shape = tuple(last_node.meta["val"].shape)
        self.assertNotEqual(shape, (6, 16))
        self.assertEqual(str(shape), "(s0, 6, 16)")

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
        self.assertIn('add: "f32[s0, 6, 16]"', report)
        for node in ep.exported.graph.nodes:
            if "val" in node.meta:
                last_node = node
        shape = tuple(last_node.meta["val"].shape)
        self.assertNotEqual(shape, (6, 16))
        self.assertEqual(str(shape), "(s0, 6, 16)")

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
                    0: lambda *args, **kwargs: torch.empty(
                        (args[1].shape[0], args[0].shape[1])
                    )
                }
            },
        )
        self.assertNotEmpty(ep)
        report = diag.get_export_report(exported_program=True)
        self.assertIn('c_model_sub: "f32[s1, s0]"', report)
        for node in ep.exported.graph.nodes:
            if "val" in node.meta:
                last_node = node
        shape = tuple(last_node.meta["val"].shape)
        self.assertNotEqual(shape, (6, 16))
        self.assertEqual(str(shape), "(s1, s0)")

    @requires_torch("2.6")
    @hide_stdout()
    def test_piece_by_piece_piece_dict_dict(self):
        import torch
        import transformers

        class SubModel(torch.nn.Module):
            def forward(
                self,
                x: Optional[torch.Tensor] = None,
                cache: Optional[transformers.cache_utils.DynamicCache] = None,
            ):
                new_cache = transformers.cache_utils.DynamicCache()
                new_cache.update(cache.key_cache[0] + x, cache.value_cache[0] + x, 0)
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
        cache = transformers.cache_utils.DynamicCache()
        cache.update(torch.randn((5, 6)), torch.randn((5, 6)), 0)
        z = model(x, cache)
        self.assertNotEmpty(z)

        cache2 = transformers.cache_utils.DynamicCache()
        cache2.update(torch.randn((6, 6)), torch.randn((6, 6)), 0)
        inputs = [
            (tuple(), dict(x=x, cache=cache)),
            (tuple(), dict(x=torch.randn((6, 6)), cache=cache2)),
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
        print(report)
        self.assertIn('ones_like: "f32[s0, 6]"', report)
        for node in ep.exported.graph.nodes:
            if "val" in node.meta:
                last_node = node
        shape = tuple(last_node.meta["val"].shape)
        self.assertNotEqual(shape, (6, 16))
        self.assertEqual(str(shape), "(s0, 6)")

    @requires_torch("2.6")
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
        self.assertIn('c_model_sub: "f32[s0, 6]"', report)
        for node in ep.exported.graph.nodes:
            if "val" in node.meta:
                last_node = node
        shape = tuple(last_node.meta["val"].shape)
        self.assertNotEqual(shape, (6, 16))
        self.assertEqual(str(shape), "(s0, 6)")

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

    @requires_torch("2.6")
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

    @requires_torch("2.6")
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

    @requires_torch("2.7")
    @hide_stdout
    def test_piece_by_piece_phi35_local(self):
        import torch

        def result_of_same_shape1(*args, **kwargs):
            "Returns the shape of one element of the cache based on the inputs."
            return torch.empty((*args[3].shape[:2], args[1].shape[1], args[3].shape[-1])).to(
                args[3].dtype
            )

        def result_of_same_shape2(*args, **kwargs):
            "Returns the shape of one element of the cache based on the inputs."
            return torch.empty((*args[0].shape[:2], 32064)).to(args[0].dtype)

        data = get_phi35_mini_instruct(num_hidden_layers=2, common_dynamic_shapes=True)
        model, inputs, inputs2 = data["model"], data["inputs"], data["inputs2"]
        diag = trace_execution_piece_by_piece(model, [inputs, inputs2], verbose=2)

        raise unittest.SkipTest(
            "Not ready yet to work: see the content of the test to understand"
        )

        """
        Class ModelOutput contains this:

        ```
        def __getitem__(self, k):
            if isinstance(k, str):
                inner_dict = dict(self.items())
                return inner_dict[k]
            else:
                return self.to_tuple()[k]
        ```

        An attribute of this class can be accessed with a string index or an integer.
        But after it is serialized, the second option is no longer available.
        This must be fixed before before able to export every submodule.
        """

        with register_additional_serialization_functions():
            ep = diag.try_export(
                exporter="fx",
                use_dynamic_shapes=True,
                exporter_kwargs=dict(strict=False),
                verbose=1,
                replace_by_custom_op=CustomOpStrategy.LOCAL,
                quiet=0,
                shape_functions={
                    "Phi3Model": {
                        1: result_of_same_shape1,
                        2: result_of_same_shape1,
                        3: result_of_same_shape1,
                        4: result_of_same_shape1,
                    },
                    "C_Phi3ForCausalLM_lm_head": {
                        0: result_of_same_shape2,
                    },
                },
            )
            self.assertNotEmpty(ep)

    @requires_torch("2.7")
    @hide_stdout
    def test_piece_by_piece_phi35_functions(self):
        import torch

        def result_of_same_shape1(*args, **kwargs):
            "Returns the shape of one element of the cache based on the inputs."
            return torch.empty((*args[3].shape[:2], args[1].shape[1], args[3].shape[-1])).to(
                args[3].dtype
            )

        def result_of_same_shape2(*args, **kwargs):
            "Returns the shape of one element of the cache based on the inputs."
            return torch.empty((*args[0].shape[:2], 32064)).to(args[0].dtype)

        data = get_phi35_mini_instruct(num_hidden_layers=2, common_dynamic_shapes=True)
        model, inputs, inputs2 = data["model"], data["inputs"], data["inputs2"]
        diag = trace_execution_piece_by_piece(
            model, [inputs, inputs2], verbose=2, trace_functions=True
        )
        report = diag.get_export_report()
        self.assertIn("mod::transformers.models.phi3.modeling_phi3", report)

        raise unittest.SkipTest(
            "Not ready yet to work: see the content of the test to understand"
        )

        """
        Class ModelOutput contains this:

        ```
        def __getitem__(self, k):
            if isinstance(k, str):
                inner_dict = dict(self.items())
                return inner_dict[k]
            else:
                return self.to_tuple()[k]
        ```

        An attribute of this class can be accessed with a string index or an integer.
        But after it is serialized, the second option is no longer available.
        This must be fixed before before able to export every submodule.
        """

        with register_additional_serialization_functions():
            ep = diag.try_export(
                exporter="fx",
                use_dynamic_shapes=True,
                exporter_kwargs=dict(strict=False),
                verbose=1,
                replace_by_custom_op=CustomOpStrategy.LOCAL,
                quiet=0,
                shape_functions={
                    "Phi3Model": {
                        1: result_of_same_shape1,
                        2: result_of_same_shape1,
                        3: result_of_same_shape1,
                        4: result_of_same_shape1,
                    },
                    "C_Phi3ForCausalLM_lm_head": {
                        0: result_of_same_shape2,
                    },
                },
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

    @ignore_warnings(UserWarning)
    @hide_stdout()
    def test_e2e_knn_imputer(self):
        import onnx
        import sklearn
        import torch
        from experimental_experiment.skl.helpers import flatnonzero, _get_weights
        from experimental_experiment.torch_interpreter import (
            make_undefined_dimension,
            Dispatcher,
        )

        class NanEuclidean(torch.nn.Module):
            def __init__(self, squared=False, copy=True):
                super().__init__()
                self.squared = squared
                self.copy = copy

            def forward(self, X, Y):
                X = X.clone()
                Y = Y.to(X.dtype).clone()

                missing_X = torch.isnan(X)
                missing_Y = torch.isnan(Y)

                # set missing values to zero
                X[missing_X] = 0
                Y[missing_Y] = 0

                # Adjust distances for missing values
                XX = X * X
                YY = Y * Y

                distances = -2 * X @ Y.T + XX.sum(1, keepdim=True) + YY.sum(1, keepdim=True).T

                distances -= XX @ missing_Y.to(X.dtype).T
                distances -= missing_X.to(X.dtype) @ YY.T

                distances = torch.clip(distances, 0, None)

                present_X = 1 - missing_X.to(X.dtype)
                present_Y = ~missing_Y
                present_count = present_X @ present_Y.to(X.dtype).T
                distances[present_count == 0] = torch.nan
                # avoid divide by zero
                present_count = torch.maximum(
                    torch.tensor([1], dtype=present_count.dtype), present_count
                )
                distances /= present_count
                distances *= X.shape[1]

                if not self.squared:
                    distances = distances.sqrt()

                return distances

        def _get_mask(X, value_to_mask):
            return (
                torch.isnan(X)
                if (  # sklearn.utils._missing.is_scalar_nan(value_to_mask)
                    not isinstance(value_to_mask, numbers.Integral)
                    and isinstance(value_to_mask, numbers.Real)
                    and math.isnan(value_to_mask)
                )
                else (value_to_mask == X)
            )

        class SubTopKIndices(torch.nn.Module):
            def forward(self, x, k):
                # torch does not like nans
                xn = torch.nan_to_num(x, nan=1.0e10)
                return torch.topk(xn, k, dim=1, largest=False, sorted=True).indices

        class SubWeightMatrix(torch.nn.Module):
            def __init__(self, weights):
                super().__init__()
                self.weights = weights

            def forward(self, donors_dist):
                weight_matrix = _get_weights(donors_dist, self.weights)
                if weight_matrix is not None:
                    weight_matrix = weight_matrix.clone()
                    weight_matrix[torch.isnan(weight_matrix)] = 0.0
                else:
                    weight_matrix = torch.ones_like(donors_dist)
                    weight_matrix[torch.isnan(donors_dist)] = 0.0
                return weight_matrix

        class SubDonorsIdx(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._topk = SubTopKIndices()

            def forward(self, dist_pot_donors, n_neighbors):
                donors_idx = self._topk(dist_pot_donors, n_neighbors)
                donors_dist = dist_pot_donors[
                    torch.arange(donors_idx.shape[0])[:, None], donors_idx
                ]
                return donors_idx, donors_dist

        class MakeNewWeights(torch.nn.Module):
            def forward(self, donors_mask, donors, weight_matrix):
                return donors_mask.to(donors.dtype) * weight_matrix.to(donors.dtype)

        class CalcImpute(torch.nn.Module):
            def __init__(self, weights):
                super().__init__()
                self._weights = SubWeightMatrix(weights)
                self._donors_idx = SubDonorsIdx()
                self._make_new_neights = MakeNewWeights()

            def _calc_impute(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
                donors_idx, donors_dist = self._donors_idx(dist_pot_donors, n_neighbors)
                weight_matrix = self._weights(donors_dist)
                donors = fit_X_col.take(donors_idx)
                donors_mask = torch.tensor([1], dtype=donors_idx.dtype) - (
                    mask_fit_X_col.take(donors_idx)
                ).to(donors_idx.dtype)

                new_weights = self._make_new_neights(donors_mask, donors, weight_matrix)

                weights_sum = new_weights.sum(axis=1, keepdim=True)
                div = torch.where(
                    weights_sum == 0, torch.tensor([1], dtype=weights_sum.dtype), weights_sum
                )
                res = (donors * new_weights).sum(axis=1, keepdim=True) / div
                return res.squeeze(dim=1).to(dist_pot_donors.dtype)

            def forward(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
                return self._calc_impute(
                    dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col
                )

        class ColProcessor(torch.nn.Module):
            def __init__(self, col, n_neighbors, weights):
                super().__init__()
                self._calc_impute = CalcImpute(weights)
                self.col = col
                self.n_neighbors = n_neighbors

            def process_one_col(
                self,
                X,
                dist_chunk,
                non_missing_fix_X,
                mask_fit_X,
                dist_idx_map,
                mask,
                row_missing_idx,
                _fit_X,
            ):
                col = self.col
                X = X.clone()
                row_missing_chunk = row_missing_idx
                col_mask = mask[row_missing_chunk, col]

                potential_donors_idx = torch.nonzero(non_missing_fix_X[:, col], as_tuple=True)[
                    0
                ]
                receivers_idx = row_missing_chunk[flatnonzero(col_mask)]
                dist_subset = dist_chunk[dist_idx_map[receivers_idx]][:, potential_donors_idx]
                all_nan_dist_mask = torch.isnan(dist_subset).all(axis=1)
                all_nan_receivers_idx = receivers_idx[all_nan_dist_mask]
                mask_ = (~mask_fit_X[:, col]).to(_fit_X.dtype)
                mask_sum = mask_.to(X.dtype).sum()
                col_sum = (_fit_X[mask_ == 1, col]).sum().to(X.dtype)
                div = torch.where(
                    mask_sum > 0, mask_sum, torch.tensor([1], dtype=mask_sum.dtype)
                )
                X[all_nan_receivers_idx, col] = col_sum / div
                receivers_idx = receivers_idx[~all_nan_dist_mask]
                dist_subset = dist_chunk[dist_idx_map[receivers_idx]][:, potential_donors_idx]
                tn = torch.tensor(self.n_neighbors)
                n_neighbors = torch.where(
                    tn < potential_donors_idx.shape[0], tn, potential_donors_idx.shape[0]
                )
                n_neighbors = torch.where(
                    n_neighbors <= 0, torch.tensor([1], dtype=n_neighbors.dtype), n_neighbors
                )
                value = self._calc_impute(
                    dist_subset,
                    n_neighbors,
                    _fit_X[potential_donors_idx, col],
                    mask_fit_X[potential_donors_idx, col],
                )
                X[receivers_idx, col] = value.to(X.dtype)
                return X

            def forward(
                self,
                X,
                dist_chunk,
                non_missing_fix_X,
                mask_fit_X,
                dist_idx_map,
                mask,
                row_missing_idx,
                _fit_X,
            ):
                return self.process_one_col(
                    X,
                    dist_chunk,
                    non_missing_fix_X,
                    mask_fit_X,
                    dist_idx_map,
                    mask,
                    row_missing_idx,
                    _fit_X,
                )

        class MakeDictIdxMap(torch.nn.Module):
            def forward(self, X, row_missing_idx):
                dist_idx_map = torch.zeros(X.shape[0], dtype=int)
                dist_idx_map[row_missing_idx] = torch.arange(row_missing_idx.shape[0])
                return dist_idx_map

        class TorchKNNImputer(torch.nn.Module):
            def __init__(self, knn_imputer):
                super().__init__()
                assert (
                    knn_imputer.metric == "nan_euclidean"
                ), f"Not implemented for metric={knn_imputer.metric!r}"
                self.dist = NanEuclidean()
                cols = []
                for col in range(knn_imputer._fit_X.shape[1]):
                    cols.append(
                        ColProcessor(col, knn_imputer.n_neighbors, knn_imputer.weights)
                    )
                self.columns = torch.nn.ModuleList(cols)
                # refactoring
                self._make_dict_idx_map = MakeDictIdxMap()
                # knn imputer
                self.missing_values = knn_imputer.missing_values
                self.n_neighbors = knn_imputer.n_neighbors
                self.weights = knn_imputer.weights
                self.metric = knn_imputer.metric
                self.keep_empty_features = knn_imputer.keep_empty_features
                self.add_indicator = knn_imputer.add_indicator
                # results of fitting
                self.indicator_ = knn_imputer.indicator_

            def _transform_indicator(self, X):
                if self.add_indicator:
                    if not hasattr(self, "indicator_"):
                        raise ValueError(
                            "Make sure to call _fit_indicator before _transform_indicator"
                        )
                    raise NotImplementedError(type(self.indicator_))
                return None

            def _concatenate_indicator(self, X_imputed, X_indicator):
                if not self.add_indicator:
                    return X_imputed
                if X_indicator is None:
                    raise ValueError(
                        "Data from the missing indicator are not provided. Call "
                        "_fit_indicator and _transform_indicator in the imputer "
                        "implementation."
                    )
                return torch.cat([X_imputed, X_indicator], dim=0)

            def transform(self, mask_fit_X, _valid_mask, _fit_X, X):
                X = X.clone()
                mask = _get_mask(X, self.missing_values)

                X_indicator = self._transform_indicator(mask)

                row_missing_idx = flatnonzero(mask[:, _valid_mask].any(axis=1))
                non_missing_fix_X = torch.logical_not(mask_fit_X)
                dist_idx_map = self._make_dict_idx_map(X, row_missing_idx)
                pairwise_distances = self.dist(X[row_missing_idx, :], _fit_X)
                for col_processor in self.columns:
                    X = col_processor(
                        X,
                        pairwise_distances,
                        non_missing_fix_X,
                        mask_fit_X,
                        dist_idx_map,
                        mask,
                        row_missing_idx,
                        _fit_X,
                    )

                if self.keep_empty_features:
                    Xc = X.clone()
                    Xc[:, ~_valid_mask] = 0
                else:
                    Xc = X[:, _valid_mask]

                return self._concatenate_indicator(Xc, X_indicator)

            def forward(self, _mask_fit_X, _valid_mask, _fit_X, X):
                return self.transform(_mask_fit_X, _valid_mask, _fit_X, X)

        def validate(size, sizey):
            X = torch.randn((size, 2))
            Y = torch.randn((sizey, 2))
            for i in range(5):
                X[i, i % 2] = torch.nan
            for i in range(4):
                Y[i + 1, i % 2] = torch.nan

            knn_imputer = sklearn.impute.KNNImputer(n_neighbors=3)
            knn_imputer.fit(X)

            model = TorchKNNImputer(knn_imputer)

            p1 = knn_imputer.transform(Y)
            p2 = model.transform(
                torch.from_numpy(knn_imputer._mask_fit_X),
                torch.from_numpy(knn_imputer._valid_mask),
                torch.from_numpy(knn_imputer._fit_X),
                Y,
            )
            d = max_diff(p1, p2)
            assert d["abs"] < 1e-5, f"Discrepancies for size={size} and sizey={sizey}, d={d}"

            p1 = knn_imputer.transform(Y[1:2])
            p2 = model.transform(
                torch.from_numpy(knn_imputer._mask_fit_X),
                torch.from_numpy(knn_imputer._valid_mask),
                torch.from_numpy(knn_imputer._fit_X),
                Y[1:2],
            )
            d = max_diff(p1, p2)
            assert d["abs"] < 1e-5, f"Discrepancies for size={size} and sizey={sizey}, d={d}"
            return knn_imputer, Y

        knn5, Y10 = validate(5, 10)
        knn50, Y40 = validate(50, 40)

        inputs = [
            (
                (
                    torch.from_numpy(knn50._mask_fit_X),
                    torch.from_numpy(knn50._valid_mask),
                    torch.from_numpy(knn50._fit_X),
                    Y40,
                ),
                {},
            ),
            (
                (
                    torch.from_numpy(knn5._mask_fit_X),
                    torch.from_numpy(knn5._valid_mask),
                    torch.from_numpy(knn5._fit_X),
                    Y10,
                ),
                {},
            ),
        ]

        trace = trace_execution_piece_by_piece(TorchKNNImputer(knn5), inputs, verbose=0)

        shape_functions = {
            "NanEuclidean": {
                0: lambda *args, **kwargs: torch.empty(
                    (args[0].shape[0], args[1].shape[0]), dtype=args[0].dtype
                )
            },
            "CalcImpute": {
                0: lambda *args, **kwargs: torch.empty(
                    (args[0].shape[0],), dtype=args[0].dtype
                )
            },
            "SubTopKIndices": {
                0: lambda *args, **kwargs: torch.empty(
                    (
                        args[0].shape[0],
                        make_undefined_dimension(min(args[0].shape[1], knn5.n_neighbors)),
                    ),
                    dtype=args[0].dtype,
                )
            },
            "SubDonorsIdx": {
                0: lambda *args, **kwargs: torch.empty(
                    (
                        args[0].shape[0],
                        make_undefined_dimension(min(args[0].shape[1], knn5.n_neighbors)),
                    ),
                    dtype=args[0].dtype,
                ),
                1: lambda *args, **kwargs: torch.empty(
                    (
                        args[0].shape[0],
                        make_undefined_dimension(min(args[0].shape[1], knn5.n_neighbors)),
                    ),
                    dtype=torch.float32,
                ),
            },
            "MakeDictIdxMap": {
                0: lambda *args, **kwargs: torch.empty(
                    (args[0].shape[0],), dtype=args[1].dtype
                ),
            },
        }

        with contextlib.redirect_stderr(io.StringIO()), bypass_export_some_errors():
            ep = trace.try_export(
                exporter="fx",
                use_dynamic_shapes=True,
                exporter_kwargs=dict(strict=False),
                replace_by_custom_op=CustomOpStrategy.LOCAL,
                verbose=0,
                shape_functions=shape_functions,
            )

        assert ep.status in (
            ep.status.OK,
            ep.status.OK_CHILDC,
        ), f"FAIL: {ep}\n-- report --\n{trace.get_export_report()}"

        for t in trace:
            if t.exporter_status.exported is None:
                print(f"[run_decompositions] {t.dot_name} - skipped")
                continue
            print(f"[run_decompositions] {t.dot_name}")
            t.exporter_status.exported = t.exporter_status.exported.run_decompositions({})

        T = str

        def onnx_topk_indices(
            g: GraphBuilder,
            sts: Optional[Dict[str, Any]],
            outputs: List[str],
            x: T,
            k: T,
            name: str = "topk",
        ):
            assert len(outputs) == 1, f"Only one output is expected but outputs={outputs}"
            unique_name = g.unique_name("unused_topk_values")
            g.op.TopK(
                x, k, name=name, outputs=[unique_name, *outputs], largest=False, sorted=True
            )
            return outputs[0]

        dispatcher = Dispatcher(
            {
                (
                    "diag_lib::C_TorchKNNImputer_columns_0___calc_impute__donors_idx__topk"
                ): onnx_topk_indices,
                (
                    "diag_lib::C_TorchKNNImputer_columns_1___calc_impute__donors_idx__topk"
                ): onnx_topk_indices,
            }
        )

        onx = trace.to_onnx_local(
            verbose=1,
            dispatcher=dispatcher,
            check_conversion_cls=dict(cls=ExtendedReferenceEvaluator, atol=1e-5, rtol=1e-5),
        )

        onnx.save(onx, "test_e2e_knn_imputer.onnx")

        def validate_onnx(size, sizey, onx, verbose: int = 1):
            X = torch.randn((size, 2))
            Y = torch.randn((sizey, 2))
            for i in range(5):
                X[i, i % 2] = torch.nan
            for i in range(4):
                Y[i + 1, i % 2] = torch.nan

            knn_imputer = sklearn.impute.KNNImputer(n_neighbors=3)
            knn_imputer.fit(X)

            model = TorchKNNImputer(knn_imputer)

            p1 = knn_imputer.transform(Y)

            model_inputs = (
                torch.from_numpy(knn_imputer._mask_fit_X),
                torch.from_numpy(knn_imputer._valid_mask),
                torch.from_numpy(knn_imputer._fit_X),
                Y,
            )
            p2 = model.transform(*model_inputs)
            d = max_diff(p1, p2)
            assert d["abs"] < 1e-5, f"Discrepancies for size={size} and sizey={sizey}, d={d}"

            input_names = [i.name for i in onx.graph.input]
            feeds = dict(zip(input_names, [t.numpy() for t in model_inputs]))

            sess = ExtendedReferenceEvaluator(onx, verbose=0)
            got = sess.run(None, feeds)
            d = max_diff(p1, got[0])
            assert (
                d["abs"] < 1e-5
            ), f"ONNX Discrepancies for size={size} and sizey={sizey}, d={d}"

            model_inputs = (
                torch.from_numpy(knn_imputer._mask_fit_X),
                torch.from_numpy(knn_imputer._valid_mask),
                torch.from_numpy(knn_imputer._fit_X),
                Y[1:2],
            )
            p1 = knn_imputer.transform(Y[1:2])
            p2 = model.transform(*model_inputs)
            d = max_diff(p1, p2)
            assert d["abs"] < 1e-5, f"Discrepancies for size={size} and sizey={sizey}, d={d}"
            feeds = dict(zip(input_names, [t.numpy() for t in model_inputs]))
            got = sess.run(None, feeds)
            d = max_diff(p1, got[0])
            assert (
                d["abs"] < 1e-5
            ), f"ONNX Discrepancies for size={size} and sizey={sizey}, d={d}"

        validate_onnx(5, 10, onx)
        validate_onnx(50, 40, onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
