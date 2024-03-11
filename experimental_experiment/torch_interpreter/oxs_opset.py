from functools import partial
from typing import Callable, Dict, List, Optional, Union


class OxsOpset:
    """
    Bridge with :epkg:`onnxscript`.

    :param builder: builder
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
        "Unsqueeze": 1,
        "Where": 1,
    }

    def __init__(self, builder: "GraphBuilder"):  # noqa: F821
        self.builder = builder
        self._submodule = None

    @property
    def submodules(self) -> Dict[str, Callable]:
        """
        Returns the submodules implementing torch functions.
        """
        if self._submodule is not None:
            return self._submodule
        from onnxscript.function_libs.torch_lib.ops import (
            core,
            fft,
            linalg,
            nested,
            nn,
            prims,
            sparse,
            special,
            vision,
        )

        subs = {
            "onnxscript.function_libs.torch_lib.ops.core": core,
            "onnxscript.function_libs.torch_lib.ops.fft": fft,
            "onnxscript.function_libs.torch_lib.ops.linalg": linalg,
            "onnxscript.function_libs.torch_lib.ops.nested": nested,
            "onnxscript.function_libs.torch_lib.ops.nn": nn,
            "onnxscript.function_libs.torch_lib.ops.prims": prims,
            "onnxscript.function_libs.torch_lib.ops.sparse": sparse,
            "onnxscript.function_libs.torch_lib.ops.special": special,
            "onnxscript.function_libs.torch_lib.ops.vision": vision,
        }
        self._submodule = subs
        return subs

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

    def IsScalar(self, name: str) -> bool:
        if self.builder.has_shape(name):
            shape = self.builder.get_shape(name)
            return shape in (tuple(), (1,))
        if self.builder.has_rank(name):
            rank = self.builder.get_rank(name)
            if rank == 0:
                return True
        raise RuntimeError(
            f"Unable to tell if {name!r} is scalar{self.builder.get_debug_msg()}"
        )

    def make_node(
        self,
        op_type: str,
        *inputs: Optional[Union[str, List[str]]],
        outputs: Optional[Union[int, List[str], str]] = None,
        domain: str = "",
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Creates a node.

        :param op_type: type
        :param inputs: inputs
        :param outputs: outputs
        :param domain: domain
        :param name: name
        :param kwargs: additional arguments
        :return: output name
        """
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
