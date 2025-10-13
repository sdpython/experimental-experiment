import ast
from collections import Counter
from typing import List, Optional, Tuple


def _dump_node(n: ast.AST) -> str:
    return ast.dump(n, include_attributes=False)


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


class MulDivCancellerTransformer(CommonTransformer):
    """Simplifies ``2*x//2`` into ``x``."""

    @classmethod
    def _flatten_mul_div(cls, node: ast.AST) -> Tuple[List[ast.AST], List[ast.AST]]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            lnum, lden = cls._flatten_mul_div(node.left)
            rnum, rden = cls._flatten_mul_div(node.right)
            return lnum + rnum, lden + rden
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
            lnum, lden = cls._flatten_mul_div(node.left)
            rnum, rden = cls._flatten_mul_div(node.right)
            return lnum + rden, lden + rnum
        return [node], []

    @classmethod
    def _rebuild_from_factors(cls, numer: List[ast.AST], denom: List[ast.AST]) -> ast.AST:
        def _product(factors: List[ast.AST]) -> ast.AST:
            if not factors:
                return ast.Constant(value=1)
            node = factors[0]
            for f in factors[1:]:
                node = ast.BinOp(left=node, op=ast.Mult(), right=f)
            return node

        numer_node = _product(numer)
        if not denom:
            return numer_node
        denom_node = _product(denom)
        return ast.BinOp(left=numer_node, op=ast.FloorDiv(), right=denom_node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)

        if not (isinstance(node, ast.BinOp) and (isinstance(node.op, (ast.Mult, ast.FloorDiv)))):
            return node

        numer, denom = self._flatten_mul_div(node)

        numer_keys = [_dump_node(n) for n in numer]
        denom_keys = [_dump_node(d) for d in denom]

        num_counter = Counter(numer_keys)
        den_counter = Counter(denom_keys)

        common_keys = set(num_counter.keys()) & set(den_counter.keys())
        for k in common_keys:
            cancel = min(num_counter[k], den_counter[k])
            num_counter[k] -= cancel
            den_counter[k] -= cancel

        remaining_numer = []
        needed_num = dict(num_counter)
        for n, k in zip(numer, numer_keys):
            if needed_num.get(k, 0) > 0:
                remaining_numer.append(n)
                needed_num[k] -= 1

        remaining_denom = []
        needed_den = dict(den_counter)
        for d, k in zip(denom, denom_keys):
            if needed_den.get(k, 0) > 0:
                remaining_denom.append(d)
                needed_den[k] -= 1

        new_node = self._rebuild_from_factors(remaining_numer, remaining_denom)
        return ast.copy_location(new_node, node)


class MaxToXorTransformer(CommonTransformer):
    """Replaces ``Max(a,b)`` by ``a^b``."""

    def visit_Call(self, node):
        self.generic_visit(node)

        if (
            isinstance(node.func, ast.Name)
            and node.func.id in ("max", "Max")
            and len(node.args) == 2
        ):
            a, b = node.args
            return ast.BinOp(left=a, op=ast.BitXor(), right=b)

        return node


class SimplifyParensTransformer(CommonTransformer):
    """To simplify parenthesis."""

    def visit_BinOp(self, node):
        self.generic_visit(node)
        return node

    def visit_Expr(self, node):
        return self.generic_visit(node)


class ExpressionSimplifierVisitor(CommonVisitor):
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
            neg = ExpressionSimplifierVisitor()
            neg.visit(node.right)
            if not neg.success:
                self.success = False
                return
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


def simplify_expression(expr: str) -> str:
    """Simplifies an expression."""
    tree = ast.parse(expr, mode="eval")
    transformers = [
        SimpleSimpliflyTransformer(expr=expr),
        MulDivCancellerTransformer(expr=expr),
        MaxToXorTransformer(expr=expr),
        SimplifyParensTransformer(expr=expr),
    ]
    for tr in transformers:
        tree = tr.visit(tree)
    ast.fix_missing_locations(tree.body)
    expr = ast.unparse(tree)
    simp = ExpressionSimplifierVisitor(expr=expr)
    simp.visit(tree.body)
    if not simp.success:
        return expr.replace(" ", "")
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
    simp1 = ExpressionSimplifierVisitor(expr1)
    simp1.visit(ast.parse(expr1, mode="eval").body)
    simp2 = ExpressionSimplifierVisitor(expr2)
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
