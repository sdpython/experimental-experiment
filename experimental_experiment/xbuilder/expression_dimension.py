import ast
from typing import Any, Dict, Optional


class Expression:
    """
    A formula using dimension.
    """

    def __init__(self, expr: str, parsed: Optional[ast.Expression] = None):
        self.expr = expr
        self.parsed = parsed

    def __repr__(self):
        return f"{self.__class__.__name__}({self.expr!r})"


def parse_expression(
    expr: str,
    context: Optional[Dict[str, Any]] = None,
    exc: bool = True,
) -> Expression:
    """
    Parses an expression involving dimensions.

    :param expr: an expression
    :param exc: raises an exception if it fails
    :param context: known variables (or dimensions)
    :return: an expression
    """
    assert isinstance(expr, str), f"Unexpected type {type(expr)} for expr={expr!r}"
    assert exc, "parse_expression not implemented when exc is False"
    if context is None:
        context = {}
    st = ast.parse(expr, mode="eval")
    for node in ast.walk(st):
        if isinstance(node, ast.Name):
            assert node.id in context, (
                f"Unable to find name {node.id!r} in expression {expr!r}, "
                f"context is {context}"
            )
    return Expression(expr, parsed=st)
