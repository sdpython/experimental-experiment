from logging import getLogger
from typing import Any, Dict, List, Optional, Union
from onnx import FunctionProto, ModelProto
from onnx.defs import get_schema
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
from .ops.op_add_add_mul_mul import (
    AddAdd,
    AddMul,
    AddSharedInput,
    MulAdd,
    MulMul,
    MulSharedInput,
    MulSub,
    SubMul,
)
from .ops.op_average_pool_grad import AveragePoolGrad
from .ops.op_cast_like import CastLike_15, CastLike_19
from .ops.op_concat import Concat
from .ops.op_constant_of_shape import ConstantOfShape
from .ops.op_fused_matmul import FusedMatMul
from .ops.op_gather_grad import GatherGrad
from .ops.op_memcpy_host import MemcpyFromHost, MemcpyToHost
from .ops.op_mul_sigmoid import MulSigmoid
from .ops.op_negxplus1 import NegXplus1
from .ops.op_quick_gelu import QuickGelu
from .ops.op_replace_zero import ReplaceZero
from .ops.op_rotary import Rotary
from .ops.op_scatter_elements import ScatterElements
from .ops.op_scatternd_of_shape import MaskedScatterNDOfShape, ScatterNDOfShape
from .ops.op_simplified_layer_normalization import SimplifiedLayerNormalization
from .ops.op_slice import Slice_1, Slice_10
from .ops.op_transpose_cast import Transpose2DCastFP16, Transpose2DCastFP32
from .ops.op_tri_matrix import TriMatrix


logger = getLogger("experimental-experiment-eval")


class ExtendedReferenceEvaluator(ReferenceEvaluator):
    """
    This class replaces the python implementation by custom implementation.
    The evaluator allows to test
    scenarios outside what an onnx backend bound to the official onnx
    operators definition could do such as optimization patterns
    involving onnxruntime contrib operators.

    ::

        from experimental_experiment.reference import ExtendedReferenceEvaluator
        ref = ExtendedReferenceEvaluator(...)

    The class overloads or adds the following operators by default:

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.reference import ExtendedReferenceEvaluator

        pprint.pprint(ExtendedReferenceEvaluator.default_ops)
    """

    default_ops = [
        AddAdd,
        AddMul,
        AddSharedInput,
        AveragePoolGrad,
        Concat,
        CastLike_15,
        CastLike_19,
        ConstantOfShape,
        FusedMatMul,
        GatherGrad,
        MaskedScatterNDOfShape,
        MemcpyFromHost,
        MemcpyToHost,
        MulAdd,
        MulMul,
        MulSharedInput,
        MulSigmoid,
        MulSub,
        NegXplus1,
        QuickGelu,
        ReplaceZero,
        Rotary,
        ScatterElements,
        ScatterNDOfShape,
        SimplifiedLayerNormalization,
        Slice_1,
        Slice_10,
        SubMul,
        Transpose2DCastFP16,
        Transpose2DCastFP32,
        TriMatrix,
    ]

    @staticmethod
    def filter_ops(proto, new_ops, opsets):
        if opsets is None and isinstance(proto, (ModelProto, FunctionProto)):
            opsets = {d.domain: d.version for d in proto.opset_import}
        best = {}
        renamed = {}
        for cl in new_ops:
            if "_" not in cl.__name__:
                continue
            vers = cl.__name__.split("_")
            try:
                v = int(vers[-1])
            except ValueError:
                # not a version
                continue
            if opsets is not None and v > opsets.get(cl.op_domain, 1):
                continue
            renamed[cl.__name__] = cl
            key = cl.op_domain, "_".join(vers[:-1])
            if key not in best or best[key][0] < v:
                best[key] = (v, cl)

        modified = []
        for cl in new_ops:
            if cl.__name__ not in renamed:
                modified.append(cl)
        for k, v in best.items():
            atts = {"domain": k[0]}
            bases = (v[1],)
            if not hasattr(v[1], "op_schema"):
                atts["op_schema"] = get_schema(k[1], v[0], domain=v[1].op_domain)
            new_cl = type(k[1], bases, atts)
            modified.append(new_cl)

        new_ops = modified
        return new_ops

    def __init__(
        self,
        proto: Any,
        opsets: Optional[Dict[str, int]] = None,
        functions: Optional[List[Union[ReferenceEvaluator, FunctionProto]]] = None,
        verbose: int = 0,
        new_ops: Optional[List[OpRun]] = None,
        **kwargs,
    ):
        if new_ops is None:
            new_ops = ExtendedReferenceEvaluator.default_ops
        else:
            new_ops = new_ops.copy()
            new_ops.extend(ExtendedReferenceEvaluator.default_ops)
        new_ops = ExtendedReferenceEvaluator.filter_ops(proto, new_ops, opsets)

        ReferenceEvaluator.__init__(
            self,
            proto,
            opsets=opsets,
            functions=functions,
            verbose=verbose,
            new_ops=new_ops,
            **kwargs,
        )

    def _log(self, level: int, pattern: str, *args: List[Any]) -> None:
        if level < self.verbose:
            new_args = [self._log_arg(a) for a in args]
            print(pattern % tuple(new_args))
        else:
            logger.debug(pattern, *args)

    def run(self, *args, **kwargs):
        """
        See :meth:`onnx.reference.ReferenceEvaluator.run`.
        """
        if len(args) == 1 and isinstance(args[0], list):
            feeds = dict(zip(self.input_names, args[0]))
            return self.run(None, feeds, **kwargs)
        return ReferenceEvaluator.run(self, *args, **kwargs)
