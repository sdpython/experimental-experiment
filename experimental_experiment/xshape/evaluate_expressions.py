import ast
import operator
from typing import Dict

operators = {
    ast.Add: operator.add,
    ast.BitXor: lambda x, y: max(x, y),
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Mult: operator.mul,
    ast.Pow: operator.pow,
    ast.Sub: operator.sub,
    ast.USub: operator.neg,
}


def _CeilToDiv(n: int, div: int) -> int:
    return n // div if n % div == 0 else n // div + 1


def _eval(node, variables, expr):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int):
            return node.value
        raise TypeError(
            f"Unsupported constant type {node.value!r} "
            f"in expression {expr!r} with context={variables!r}"
        )
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, variables, expr)
        right = _eval(node.right, variables, expr)
        op_type = type(node.op)
        if op_type in operators:
            return operators[op_type](left, right)
        raise TypeError(
            f"Unsupported operator: {op_type!r} "
            f"in expression {expr!r} with context={variables!r}"
        )
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, variables, expr)
        op_type = type(node.op)
        if op_type in operators:
            return operators[op_type](operand)
        raise TypeError(
            f"Unsupported unary operator: {op_type!r} "
            f"in expression {expr!r} with context={variables!r}"
        )
    if isinstance(node, ast.Name):  # variable
        if node.id in variables:
            val = variables[node.id]
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"Variable {node.id!r} must be numeric "
                    f"in expression {expr!r} with context={variables!r}"
                )
            return val
        raise NameError(
            f"Unknown variable: {node.id!r} in expression {expr!r} with context={variables!r}"
        )
    if isinstance(node, ast.Call):
        # Specific function
        name = node.func.id
        assert name == "CeilToInt", f"Unable to evaluate function {name!r} in {expr!r}"
        values = [_eval(a, variables, expr) for a in node.args]
        return _CeilToDiv(*values)

    raise TypeError(
        f"Unsupported AST node: {type(node)} in expression {expr!r} with context={variables!r}"
    )


def evaluate_expression(expression: str, context: Dict[str, int]) -> int:
    """
    Evaluates an expression handling dimensions.

    .. runpython::
        :showcode:

        from experimental_experiment.xshape.evaluate_expressions import (
            evaluate_expression,
        )

        print(evaluate_expression("x+y", dict(x=3, y=5)))
    """
    if isinstance(expression, int):
        return expression
    parsed = ast.parse(expression, mode="eval")
    return _eval(parsed.body, variables=context, expr=expression)
