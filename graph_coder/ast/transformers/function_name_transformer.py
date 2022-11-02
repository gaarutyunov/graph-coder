import ast

__fn_name__ = "__fn_name__"


class FunctionNameTransformer(ast.NodeTransformer):
    def generic_visit(self, node):
        if hasattr(node, "annotation") and isinstance(node.annotation, ast.Name):
            return node

        is_inside_fn = isinstance(node, ast.FunctionDef) or hasattr(node, __fn_name__)
        if is_inside_fn:
            fn_name = (
                node.name
                if isinstance(node, ast.FunctionDef)
                else getattr(node, __fn_name__)
            )

        def set_fn_name(val):
            if is_inside_fn:
                setattr(val, __fn_name__, fn_name)
                if isinstance(val, ast.FunctionDef) and hasattr(val, "name"):
                    val.name = f"{fn_name}.{val.name}"
                if isinstance(val, ast.arg) and not (
                    val.arg.endswith("self") or val.arg.endswith("cls")
                ):
                    val.arg = f"{fn_name}.{val.arg}"

        for ast_field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        set_fn_name(value)
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                set_fn_name(old_value)
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, ast_field)
                else:
                    setattr(node, ast_field, new_node)

        return node
