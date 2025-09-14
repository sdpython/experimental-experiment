import ast
from typing import Any, Dict, Optional


class Expression:
    """
    A formula using dimension.

    :param expr: a string
    :param parsed: parsed tree (from :func:`ast.parse`)
    """

    def __init__(self, expr: str, parsed: Optional[ast.Expression] = None):
        self.expr = expr
        self.parsed = parsed

    def __repr__(self):
        "usual"
        return f"{self.__class__.__name__}({self.expr!r})"

    def isidentifier(self):
        "Tells if this expression is a single dimension or an expression."
        return self.expr.isidentifier


def parse_expression_tokens(expr: str):
    """Extracts the token from an expression."""
    tokens = []
    try:
        st = ast.parse(expr, mode="eval")
    except SyntaxError:
        # Something went wrong. Let's skip it.
        return {expr}
    except TypeError as e:
        raise TypeError(f"Unable to compile expression {expr!r} (type is {type(expr)})") from e
    for node in ast.walk(st):
        if isinstance(node, ast.Name):
            tokens.append(node.id)
    return set(tokens)


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
        return parse_expression(expr.node, context=context, exc=exc)
    if hasattr(expr, "_expr"):
        # torch.fx.experimental.sym_node.SymNode
        return parse_expression(str(expr._expr), context=context, exc=exc)

    assert isinstance(
        expr, str
    ), f"Unexpected type {type(expr)} for expr={expr!r} and context={context}"
    if context is None:
        context = {}
    st = ast.parse(simplify_expression(expr), mode="eval")
    for node in ast.walk(st):
        if isinstance(node, ast.Name):
            if node.id in {"Max", "Min", "CeilToInt", "IntTrueDiv", "Mod"}:
                continue
            sds = []
            for d_ in context.values():
                # WrapSym
                d = d_.sym if hasattr(d_, "sym") else d_
                try:
                    sd = str(d)
                except AttributeError as e:
                    if hasattr(d, "node") and isinstance(d.node, str):
                        sd = d.node
                    else:
                        raise AssertionError(
                            f"Unable to convert type {type(d)} into string"
                        ) from e
                sds.append(sd)
            assert not exc or context is None or node.id in context or node.id in set(sds), (
                f"Unable to find name {node.id!r} from expression {expr!r}, "
                f"context is {sorted(context)}"
            )
    return Expression(expr, parsed=st)


class ExpressionSimplifier(ast.NodeVisitor):
    def __init__(self, expr: Optional[str] = None):
        self.coeffs = {}
        self.const = 0
        self.expr = expr
        self.success = True

    def get_debug_msg(self) -> str:
        if self.expr:
            return f" expression={self.expr!r}"
        return ""

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            self.visit(node.left)
            self.visit(node.right)
        elif isinstance(node.op, ast.Sub):
            self.visit(node.left)
            # negate the right side
            neg = ExpressionSimplifier()
            neg.visit(node.right)
            for v, c in neg.coeffs.items():
                if v not in self.coeffs:
                    self.coeffs[v] = 0
                self.coeffs[v] -= c
            self.const -= neg.const
        elif isinstance(node.op, ast.Mult):
            # Only support coeff * var or var * coeff
            if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Name):
                if node.right.id not in self.coeffs:
                    self.coeffs[node.right.id] = 0
                self.coeffs[node.right.id] += node.left.value
            elif isinstance(node.right, ast.Constant) and isinstance(node.left, ast.Name):
                if node.right.id not in self.coeffs:
                    self.coeffs[node.right.id] = 0
                self.coeffs[node.left.id] += node.right.value
            else:
                # unable to simplify
                self.success = False
                return
        else:
            self.success = False
            return

    def visit_Name(self, node):
        if node.id not in self.coeffs:
            self.coeffs[node.id] = 0
        self.coeffs[node.id] += 1

    def visit_Constant(self, node):
        self.const += node.value


def simplify_expression(expr: str) -> str:
    """Simplifies an expression."""
    tree = ast.parse(expr, mode="eval")
    simp = ExpressionSimplifier(expr=expr)
    simp.visit(tree.body)
    if not simp.success:
        # visit failed
        return expr

    # Rebuild result
    terms = []
    for var, coeff in simp.coeffs.items():
        if coeff == 0:
            continue
        elif coeff == 1:
            terms.append(f"+{var}")
        elif coeff == -1:
            terms.append(f"-{var}")
        else:
            terms.append(f"{'+' if coeff > 0 else ''}{coeff}*{var}")
    if simp.const != 0:
        terms.append(f"{'+' if simp.const > 0 else ''}{simp.const}")
    result = "".join(terms)
    return result[1:] if result.startswith("+") else (result if result else "0")


def simplify_two_expressions(expr1: str, expr2: str) -> str:
    """Simplifies an expression exp1 == exp2."""
    simp1 = ExpressionSimplifier()
    simp1.visit(ast.parse(expr1, mode="eval").body)
    simp2 = ExpressionSimplifier()
    simp2.visit(ast.parse(expr2, mode="eval").body)

    terms = {}
    for var, coeff in simp1.coeffs.items():
        if coeff == 0:
            continue
        if var not in terms:
            terms[var] = 0
        terms[var] += coeff
    for var, coeff in simp2.coeffs.items():
        if coeff == 0:
            continue
        if var not in terms:
            terms[var] = 0
        terms[var] -= coeff
    return {k: v for k, v in terms.items() if v != 0}


class RenameTransformer(ast.NodeTransformer):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping

    def visit_Name(self, node):
        if node.id in self.mapping:
            return ast.copy_location(ast.Name(id=self.mapping[node.id], ctx=node.ctx), node)
        return node


def rename_expression(expr: str, mapping: Dict[str, str]) -> str:
    """
    Renames variables in a Python expression using AST.

    :param expr: Python expression as string
    :param mapping: Mapping from old names to new names
    :return: rransformed expression
    """
    tree = ast.parse(expr, mode="eval")
    transformer = RenameTransformer(mapping)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree).replace(" ", "")
