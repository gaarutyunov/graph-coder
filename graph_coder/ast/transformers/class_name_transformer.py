import ast
import typing

from graph_coder.ast.nodes.class_def import ClassDef
from graph_coder.ast.nodes.key_value import KeyValue

__class_name__ = "__class_name__"


class ClassNameTransformer(ast.NodeTransformer):
    def visit_ClassDef(self, node):
        cls = ast.Name(id=node.name + ".cls", ctx=ast.Load())
        slf = ast.Name(id=node.name + ".self", ctx=ast.Load())
        properties = []

        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign):
                kv = KeyValue(ast.Constant(stmt.target.id), slf)
                setattr(kv, __class_name__, node.name)
                stmt.target = kv
                properties.append(stmt)
                node.body.remove(stmt)

        return self.generic_visit(
            ClassDef(
                node.name,
                node.bases,
                node.keywords,
                cls,
                slf,
                properties,
                node.body,
                node.decorator_list,
            )
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> typing.Any:
        if isinstance(node.target, ast.Name):
            node.target.id = f"{getattr(node, __class_name__)}.{node.target.id}"

    def generic_visit(self, node):
        is_inside_class = isinstance(node, ClassDef) or hasattr(node, __class_name__)
        if is_inside_class:
            class_name = (
                node.name
                if isinstance(node, ClassDef)
                else getattr(node, __class_name__)
            )

        def set_class_name(val):
            if is_inside_class:
                setattr(val, __class_name__, class_name)
                if (
                    isinstance(val, ast.Name)
                    and hasattr(val, "id")
                    and val.id.startswith("self.")
                ):
                    val.id = f"{class_name}{val.id.lstrip('self')}"
                if isinstance(val, ast.FunctionDef) and hasattr(val, "name"):
                    val.name = f"{class_name}.{val.name}"
                    if len(val.args.args) > 0 and isinstance(val.args.args[0], ast.arg):
                        if val.args.args[0].arg == "self":
                            val.args.args[0].arg = f"{class_name}.self"
                        elif val.args.args[0].arg == "cls":
                            val.args.args[0].arg = f"{class_name}.cls"

        for ast_field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        set_class_name(value)
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                set_class_name(old_value)
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, ast_field)
                else:
                    setattr(node, ast_field, new_node)

        return node
