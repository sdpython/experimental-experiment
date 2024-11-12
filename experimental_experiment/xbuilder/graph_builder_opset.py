from functools import partial
from typing import List, Optional, Union
import numpy as np


class Opset:
    """
    Makes it easier to write onnx graph.
    The method name is the node type.

    :param graph_builder: the builder
    :param allow_unknown: allows unknown operators, otherwise,
        fails this class does not the expected number of outputs
    """

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
        "Cos": 1,
        "Cosh": 1,
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
        "Sin": 1,
        "Sinh": 1,
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

    def __init__(
        self,
        builder: "GraphBuilder",  # noqa: F821
        allow_unknown: bool = False,
    ):
        self.builder = builder
        self.allow_unknown = allow_unknown

    def __getattr__(self, name):
        if name in self._implemented:
            return partial(self.make_node, name)
        if name in self.__dict__:
            return self.__dict__[name]
        return partial(self._make_node, name)

    def _make_node(self, op_type, *args, outputs=None, **kwargs):
        if outputs is None:
            if op_type in self._implemented:
                outputs = self._implemented[op_type]
            elif op_type == "Split" and kwargs.get("domain", "") == "":
                assert "num_outputs" in kwargs, (
                    "Number of outputs is not implemented yet for operator "
                    f"{op_type!r} and kwargs={kwargs}"
                )
                outputs = kwargs["num_outputs"]
            else:
                # We assume there is only one outputs.
                outputs = 1
        return self.make_node(op_type, *args, outputs=outputs, **kwargs)

    def make_node(
        self,
        op_type: str,
        *inputs: Optional[Union[str, List[str]]],
        outputs: Optional[Union[int, List[str], str]] = None,
        domain: str = "",
        name: Optional[str] = None,
        **kwargs,
    ):
        assert (
            op_type != "Split" or outputs != 1
        ), f"Operator Split is useless with one output, inputs={inputs}, outputs={outputs}"
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
            elif hasattr(i, "name") and not hasattr(i, "detach"):
                # torch.fx.Node
                assert i.name is not None, f"Unexpected name for type {type(i)}"
                new_inputs.append(i.name)
            elif i is None:
                # Optional input
                new_inputs.append("")
            elif isinstance(i, np.ndarray):
                assert 0 not in i.shape, (
                    f"Not implemented for type(i)={type(i)}, i={i}, "
                    f"inputs={inputs!r}, op_type={op_type!r}, i.shape={i.shape}"
                    f"{self.builder.get_debug_msg()}"
                )
                if i.dtype == np.int64 and i.size < 16:
                    source = "Opset.make_node.1/Shape"
                elif i.size < 2:
                    source = "Opset.make_node.1/Small"
                else:
                    source = "Opset.make_node.0"
                cst_name = self.builder.make_initializer(
                    "", i, msg=f"input {i} of op_type={op_type!r}", source=source
                )
                new_inputs.append(cst_name)
            else:
                raise AssertionError(
                    f"Not implemented for type(i)={type(i)}, i={i}, "
                    f"inputs={inputs!r}, op_type={op_type!r}{self.builder.get_debug_msg()}"
                )

        assert None not in new_inputs
        return self.builder.make_node(
            op_type,
            new_inputs,
            outputs=outputs,
            domain=domain,
            name=name or f"{self.__class__.__name__}",
            **kwargs,
        )

    @staticmethod
    def _iaxes(op_type, axes) -> int:
        if isinstance(axes, np.ndarray):
            iaxes = axes.tolist()
        elif isinstance(axes, int):
            iaxes = [axes]
        else:
            raise RuntimeError(f"Unable to call {op_type} on a dynamic input axis={axes}")
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
        return self.ReduceMean(args[0], axes=self._iaxes("ReduceMean", args[1]), **kwargs)

    def ReduceSumAnyOpset(self, *args, **kwargs):
        if len(args) == 1:
            return self.ReduceSum(*args, **kwargs)
        assert len(args) == 2, f"ReduceSumAnyOpset expects 2 arguments not {len(args)}"
        if self.builder.main_opset >= 13:
            return self.ReduceSum(*args, **kwargs)
        return self.ReduceSum(args[0], axes=self._iaxes("ReduceSum", args[1]), **kwargs)

    def SqueezeAnyOpset(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            return self.Squeeze(*args)
        assert len(args) == 2, f"SqueezeAnyOpset expects 2 arguments not {len(args)}"
        if self.builder.main_opset >= 13:
            return self.Squeeze(*args, **kwargs)
        return self.Squeeze(args[0], axes=self._iaxes("Squeeze", args[1]), **kwargs)

    def UnsqueezeAnyOpset(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            return self.Unsqueeze(*args)
        assert len(args) == 2, f"UnsqueezeAnyOpset expects 2 arguments not {len(args)}"
        if self.builder.main_opset >= 13:
            return self.Unsqueeze(*args, **kwargs)
        return self.Unsqueeze(args[0], axes=self._iaxes("Unsqueeze", args[1]), **kwargs)
