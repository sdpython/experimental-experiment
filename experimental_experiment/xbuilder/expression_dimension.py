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
    if hasattr(expr, "__sym_float__"):
        # torch.SymInt
        return parse_expression(expr.node, context=context)
    if hasattr(expr, "_expr"):
        # torch.fx.experimental.sym_node.SymNode
        return parse_expression(str(expr._expr), context=context)

    assert isinstance(
        expr, str
    ), f"Unexpected type {type(expr)} for expr={expr!r} and context={context}"
    assert exc, "parse_expression not implemented when exc is False"
    if context is None:
        context = {}
    st = ast.parse(expr, mode="eval")
    for node in ast.walk(st):
        if isinstance(node, ast.Name):
            assert node.id in context or node.id in set(
                str(d) for d in context.values()
            ), (
                f"Unable to find name {node.id!r} in expression {expr!r}, "
                f"context is {context}"
            )
    return Expression(expr, parsed=st)
