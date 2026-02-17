import ast
from typing import Dict, Optional, Set
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
    """
    Renames variable names into other based on a mapping.

    :param magging: mapping
    :param expr: only use for error messages
    """

    def __init__(self, mapping: Dict[str, str], expr: Optional[str] = None):
        super().__init__(expr)
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
    transformer = RenameTransformer(replacements)
    simplify = SimpleSimpliflyTransformer()
    new_tree = simplify.visit(transformer.visit(tree))
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
