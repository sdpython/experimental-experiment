import ast
from typing import Optional


class _Common:

    def __init__(self, expr: Optional[str] = None):
        self.expr = expr

    def get_debug_msg(self) -> str:
        if self.expr:
            return f" expression={self.expr!r}"
        return ""


class CommonVisitor(ast.NodeVisitor, _Common):
    def __init__(self, expr: Optional[str] = None):
        ast.NodeVisitor.__init__(self)
        _Common.__init__(self, expr)


class CommonTransformer(ast.NodeTransformer, _Common):
    def __init__(self, expr: Optional[str] = None):
        ast.NodeTransformer.__init__(self)
        _Common.__init__(self, expr)


class ExpressionSimplifier(CommonVisitor):
    """Simplifies expression such as ``2*x-x``."""

    def __init__(self, expr: Optional[str] = None):
        super().__init__(expr)
        self.coeffs = {}
        self.const = 0
        self.success = True

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
                if node.left.id not in self.coeffs:
                    self.coeffs[node.left.id] = 0
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


class SimpleSimpliflyTransformer(CommonTransformer):
    """Simplifies expressions such as ``batch^batch``, ``x+0``, ``x*1``."""

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.BitXor):
            if (
                isinstance(node.left, ast.Name)
                and isinstance(node.right, ast.Name)
                and node.left.id == node.right.id
            ):
                return node.left
        if isinstance(node.op, ast.Add):
            if isinstance(node.left, ast.Constant) and node.left.value == 0:
                return node.right
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                return node.left
        if isinstance(node.op, ast.Mult):
            if isinstance(node.left, ast.Constant) and node.left.value == 1:
                return node.right
            if isinstance(node.right, ast.Constant) and node.right.value == 1:
                return node.left
        return node


def simplify_expression(expr: str) -> str:
    """Simplifies an expression."""
    tree = ast.parse(expr, mode="eval")
    SimpleSimpliflyTransformer()
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
