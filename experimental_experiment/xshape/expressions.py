import ast
from typing import Dict, Optional, Set


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


class ExpressionSimplifier(ast.NodeVisitor):
    """Simplifies expression such as ``2*x-x``."""

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


class SimpleSimpliflyTransformer(ast.NodeTransformer):
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


def rename_dynamic_dimensions(
    constraints: Dict[str, Set[str]], original: Set[str], ban_prefix: str = "DYN"
) -> Dict[str, str]:
    """
    Renames dynamic shapes as requested by the user. :func:`torch.export.export` uses
    many names for dynamic dimensions. When building the onnx model,
    some of them are redundant and can be replaced by the name provided by the user.

    :param constraints: exhaustive list of used name and all the values equal to it
    :param original: the names to use if possible
    :param ban_prefix: avoid any rewriting by a constant starting with this prefix
    :return: replacement dictionary
    """
    replacements = {s: s for s in original}
    all_values = set(constraints) | original

    not_done = set(constraints)
    max_iter = len(replacements)
    while not_done and max_iter > 0:
        max_iter -= 1
        for k, v in constraints.items():
            common = v & original
            if not common:
                continue
            common = sorted(common)
            by = common[0]
            if ban_prefix and by.startswith(ban_prefix):
                continue
            replacements[k] = by
            for vv in v:
                if vv not in replacements:
                    replacements[vv] = by
        not_done = all_values - set(replacements)
    return replacements


def rename_dynamic_expression(expression: str, replacements: Dict[str, str]):
    """
    Renames variables inside an expression.
    The function removes any space.

    :param expression: something like ``s15 + seq_length``
    :param replacements: replacements to make
    :return: new string
    """

    class RenameVariable(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in replacements:
                node.id = replacements[node.id]
            return node

    try:
        tree = ast.parse(expression)
    except SyntaxError:
        return expression
    transformer = RenameVariable()
    simplify = SimpleSimpliflyTransformer()
    new_tree = simplify.visit(transformer.visit(tree))
    res = ast.unparse(new_tree).replace(" ", "")
    return res


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
