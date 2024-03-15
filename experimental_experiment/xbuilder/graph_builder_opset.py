from functools import partial
from typing import List, Optional, Union
import numpy as np


class Opset:
    # defined for opset >= 18
    # name: number of expected outputs
    _implemented = {
        "Abs": 1,
        "Add": 1,
        "And": 1,
        "ArgMax": 1,
        "ArgMin": 1,
        "Cast": 1,
        "CastLike": 1,
        "Celu": 1,
        "Concat": 1,
        "Constant": 1,
        "ConstantOfShape": 1,
        "Div": 1,
        "Dropout": 2,
        "Elu": 1,
        "Equal": 1,
        "Exp": 1,
        "Expand": 1,
        "Flatten": 1,
        "Gather": 1,
        "GatherElements": 1,
        "GatherND": 1,
        "Gemm": 1,
        "Greater": 1,
        "GreaterOrEqual": 1,
        "Identity": 1,
        "MatMul": 1,
        "MaxPool": 2,
        "Mul": 1,
        "Less": 1,
        "LessOrEqual": 1,
        "Log": 1,
        "LogSoftmax": 1,
        "Neg": 1,
        "Not": 1,
        "Or": 1,
        "Pow": 1,
        "Range": 1,
        "Reciprocal": 1,
        "ReduceMax": 1,
        "ReduceMean": 1,
        "ReduceMin": 1,
        "ReduceSum": 1,
        "Relu": 1,
        "Reshape": 1,
        "ScatterElements": 1,
        "ScatterND": 1,
        "Shape": 1,
        "Sigmoid": 1,
        "Slice": 1,
        "Softmax": 1,
        "Sqrt": 1,
        "Squeeze": 1,
        "Sub": 1,
        "Tile": 1,
        "Transpose": 1,
        "Trilu": 1,
        "Unsqueeze": 1,
        "Where": 1,
    }

    def __init__(self, builder: "GraphBuilder"):  # noqa: F821
        self.builder = builder

    def __getattr__(self, name):
        if name in self._implemented:
            return partial(self.make_node, name)
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            raise AttributeError(
                f"Unable to access attribute {name!r}, "
                f"you can still use this operator with method 'make_node'."
            ) from e

    def make_node(
        self,
        op_type: str,
        *inputs: Optional[Union[str, List[str]]],
        outputs: Optional[Union[int, List[str], str]] = None,
        domain: str = "",
        name: Optional[str] = None,
        **kwargs,
    ):
        if outputs is None:
            outputs = self._implemented[op_type]
        if inputs is None:
            inputs = []
        new_inputs = []
        for i in inputs:
            assert not isinstance(
                i, (list, tuple)
            ), f"Wrong inputs for operator {op_type!r}: {inputs!r}"
            if isinstance(i, str):
                new_inputs.append(i)
            elif hasattr(i, "name"):
                # torch.fx.Node
                new_inputs.append(i.name)
            else:
                cst_name = self.builder.make_initializer(
                    "", i, msg=f"input {i} of op_type={op_type!r}"
                )
                new_inputs.append(cst_name)

        return self.builder.make_node(
            op_type, new_inputs, outputs=outputs, domain=domain, name=name, **kwargs
        )

    @staticmethod
    def _iaxes(op_type, axes) -> int:
        if isinstance(axes, np.ndarray):
            iaxes = axes.tolist()
        elif isinstance(axes, int):
            iaxes = [axes]
        else:
            raise RuntimeError(
                f"Unable to call {op_type} on a dynamic input axis={axes}"
            )
        return iaxes

    def ReduceMaxAnyOpset(self, *args, **kwargs):
        if len(args) == 1:
            return self.ReduceMax(*args, **kwargs)
        assert len(args) == 2, f"ReduceMaxAnyOpset expects 2 arguments not {len(args)}"
        if self.builder.main_opset >= 18:
            return self.ReduceMax(*args, **kwargs)
        return self.ReduceMax(args[0], axes=self._iaxes("ReduceMax", args[1]), **kwargs)

    def ReduceMeanAnyOpset(self, *args, **kwargs):
        if len(args) == 1:
            return self.ReduceMean(*args, **kwargs)
        assert len(args) == 2, f"ReduceMeanAnyOpset expects 2 arguments not {len(args)}"
        if self.builder.main_opset >= 18:
            return self.ReduceMean(*args, **kwargs)
        return self.ReduceMean(
            args[0], axes=self._iaxes("ReduceMean", args[1]), **kwargs
        )

    def UnsqueezeAnyOpset(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            return self.Unsqueeze(*args)
        assert len(args) == 2, f"UnsqueezeAnyOpset expects 2 arguments not {len(args)}"
        if self.builder.main_opset >= 13:
            return self.Unsqueeze(*args, **kwargs)
        return self.Unsqueeze(args[0], axes=self._iaxes("Unsqueeze", args[1]), **kwargs)

    def ReduceSumAnyOpset(self, *args, **kwargs):
        if len(args) == 1:
            return self.ReduceSum(*args, **kwargs)
        assert len(args) == 2, f"ReduceSumAnyOpset expects 2 arguments not {len(args)}"
        if self.builder.main_opset >= 13:
            return self.ReduceSum(*args, **kwargs)
        return self.ReduceSum(args[0], axes=self._iaxes("ReduceSum", args[1]), **kwargs)
