import ast


class DocstringRemover(ast.NodeTransformer):
    def generic_visit(self, node: ast.AST) -> ast.AST:
        try:
            docstring = ast.get_docstring(node)
            if docstring is not None:
                node.body.remove(node.body[0])
        except:
            pass
        finally:
            return super().generic_visit(node)
