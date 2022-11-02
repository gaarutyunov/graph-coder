import ast

from graph_coder.ast.nodes.key_value import KeyValue


class KeyValuePreTransformer(ast.NodeTransformer):
    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.attr, (str, int)):
            key = ast.Constant(node.attr)
        else:
            key = node.attr
        return KeyValue(key, self.visit(node.value))

    def visit_keyword(self, node: ast.keyword):
        if isinstance(node.arg, (str, int)):
            key = ast.Constant(node.arg)
        else:
            key = node.arg
        return KeyValue(key, self.visit(node.value))

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, (ast.Attribute, ast.keyword)):
            return self.visit_Attribute(node)
        return super().generic_visit(node)
