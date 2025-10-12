from typing import Set


def element_wise_binary_op_types() -> Set[str]:
    """
    Returns the list of element-wise operators.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xshape._onnx_helper import (
            element_wise_binary_op_types,
        )
        pprint.pprint(element_wise_binary_op_types())
    """
    return {
        "Add",
        "And",
        "Div",
        "Mul",
        "Mod",
        "Or",
        "Sub",
        "Xor",
    }


def element_wise_op_cmp_types() -> Set[str]:
    """
    Returns the list of element-wise operators
    doing comparisons.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xshape._onnx_helper import element_wise_op_cmp_types
        pprint.pprint(element_wise_op_cmp_types())
    """
    return {
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
    }


def unary_like_op_types() -> Set[str]:
    """
    Returns the list of unary *like* operators.
    They do not change the shape. They may change the type.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xshape._onnx_helper import unary_like_op_types
        pprint.pprint(unary_like_op_types())
    """
    return {
        "Abs",
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "BitShift",
        "Cast",
        "CastLike",
        "Ceil",
        "Celu",
        "Clip",
        "Cos",
        "Cosh",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
        "Elu",
        "Erf",
        "Exp",
        "IsInf",
        "Log",
        "LogSoftmax",
        "Neg",
        "Not",
        "PRelu",
        "QuantizeLinear",
        "Reciprocal",
        "Relu",
        "Round",
        "Selu",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Softmax",
        "SoftmaxCrossEntropyLoss",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Tan",
        "Tanh",
        "ThresholdRelu",
    }
