import ast


class ExtractIfElse(ast.NodeTransformer):
    def __init__(self):
        self.func_count = 0
        self.new_functions = []  # Store extracted function definitions

    def new_func_name(self):
        """Generate a new function name."""
        self.func_count += 1
        return f"branch_func{self.func_count}"

    def fix_lineno(self, node, lineno=1):
        """Recursively set lineno and col_offset for AST nodes."""
        if isinstance(node, ast.AST):
            if not hasattr(node, "lineno") or node.lineno is None:
                node.lineno = lineno
            if not hasattr(node, "col_offset") or node.col_offset is None:
                node.col_offset = 0
            for _field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        self.fix_lineno(item, lineno)
                elif isinstance(value, ast.AST):
                    self.fix_lineno(value, lineno)

    def visit_If(self, node):
        """Extract 'if' statements into separate functions, handling local variables."""
        func_name = self.new_func_name()

        # Find variables modified inside the if-else blocks
        assigned_vars = set()
        for stmt in node.body + node.orelse:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        assigned_vars.add(target.id)

        param_names = list(assigned_vars)  # List of modified variables
        args = [ast.arg(arg=var, annotation=None) for var in param_names]

        # Create the function definition for the 'if' branch
        then_func_def = ast.FunctionDef(
            name=func_name,
            args=ast.arguments(
                args=args, posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=[
                *node.body,
                ast.Return(
                    value=ast.Tuple(
                        elts=[ast.Name(id=var, ctx=ast.Load()) for var in param_names],
                        ctx=ast.Load(),
                    )
                ),
            ],
            decorator_list=[],
        )
        self.fix_lineno(then_func_def, node.lineno)
        self.new_functions.append(then_func_def)

        # Call function and assign returned values
        call_then = ast.Assign(
            targets=[
                ast.Tuple(
                    elts=[ast.Name(id=var, ctx=ast.Store()) for var in param_names],
                    ctx=ast.Store(),
                )
            ],
            value=ast.Call(
                func=ast.Name(id=func_name, ctx=ast.Load()),
                args=[ast.Name(id=var, ctx=ast.Load()) for var in param_names],
                keywords=[],
            ),
        )
        self.fix_lineno(call_then, node.lineno)
        node.body = [call_then]

        if node.orelse:
            else_func_name = self.new_func_name()
            else_func_def = ast.FunctionDef(
                name=else_func_name,
                args=ast.arguments(
                    args=args, posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[]
                ),
                body=[
                    *node.orelse,
                    ast.Return(
                        value=ast.Tuple(
                            elts=[ast.Name(id=var, ctx=ast.Load()) for var in param_names],
                            ctx=ast.Load(),
                        )
                    ),
                ],
                decorator_list=[],
            )
            self.fix_lineno(else_func_def, node.lineno)
            self.new_functions.append(else_func_def)

            call_else = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[ast.Name(id=var, ctx=ast.Store()) for var in param_names],
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Call(
                    func=ast.Name(id=else_func_name, ctx=ast.Load()),
                    args=[ast.Name(id=var, ctx=ast.Load()) for var in param_names],
                    keywords=[],
                ),
            )
            self.fix_lineno(call_else, node.lineno)
            node.orelse = [call_else]

        return node


def refactor_if_else_functions(code: str) -> str:
    """Extracts if-else branches into functions and returns modified code."""
    tree = ast.parse(code)
    transformer = ExtractIfElse()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    # Append new function definitions at the end of the script
    new_tree.body.extend(transformer.new_functions)

    # Convert the modified AST back into Python source code
    return ast.unparse(new_tree)
