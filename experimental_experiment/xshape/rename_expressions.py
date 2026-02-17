import ast
from typing import Dict, List, Optional, Set
from .simplify_expressions import SimpleSimpliflyTransformer, CommonTransformer


def parse_expression_tokens(expr: str) -> Set[str]:
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


class RenameTransformer(CommonTransformer):
    def __init__(self, mapping, expr: Optional[str] = None):
        super().__init__(expr)
        self.mapping = mapping

    def visit_Name(self, node):
        if node.id in self.mapping:
            return ast.copy_location(ast.Name(id=self.mapping[node.id], ctx=node.ctx), node)
        return node


class ReorderCommutativeOpsTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()

    def visit_BinOp(self, node: ast.BinOp):
        # First recurse into children
        self.generic_visit(node)

        # Only process + and *
        if isinstance(node.op, (ast.Add, ast.Mult)):
            operands = self._flatten(node, type(node.op))
            operands.sort(key=self._expr_key)
            return self._rebuild(operands, node.op)

        return node

    def _flatten(self, node: ast.AST, op_type) -> List[ast.AST]:
        """Flattens a chain of same-type binary operations."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, op_type):
            return self._flatten(node.left, op_type) + self._flatten(node.right, op_type)
        return [node]

    def _rebuild(self, operands: List[ast.AST], op: ast.operator) -> ast.AST:
        """Rebuilds a binary tree from sorted operands."""
        expr = operands[0]
        for operand in operands[1:]:
            expr = ast.BinOp(left=expr, op=op, right=operand)
        return expr

    def _expr_key(self, node: ast.AST) -> str:
        """Generates a sortable key for expressions."""
        return ast.unparse(node)


class RenameVariable(CommonTransformer):
    def __init__(self, mapping, expr: Optional[str] = None):
        super().__init__()
        self.mapping = mapping

    def visit_Name(self, node):
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
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
    reorder = ReorderCommutativeOpsTransformer()
    new_tree = reorder.visit(transformer.visit(tree))
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree).replace(" ", "")


def rename_dynamic_expression(expression: str, replacements: Dict[str, str]):
    """
    Renames variables inside an expression.
    The function removes any space.

    :param expression: something like ``s15 + seq_length``
    :param replacements: replacements to make
    :return: new string
    """
    try:
        tree = ast.parse(expression)
    except SyntaxError:
        return expression
    transformer = RenameVariable(replacements)
    simplify = SimpleSimpliflyTransformer()
    reorder = ReorderCommutativeOpsTransformer()
    new_tree = reorder.visit(simplify.visit(transformer.visit(tree)))
    res = ast.unparse(new_tree).replace(" ", "")
    return res


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
