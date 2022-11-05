import ast

from graph_coder.ast.transformers.docstring_remover import DocstringRemover


def test_docstring_transformer():
    """Test that docstrings are removed from the AST."""
    source = '''def f():
    """This is a docstring."""
    pass
    '''

    tree = ast.parse(source)
    DocstringRemover().visit(tree)

    assert ast.get_docstring(tree.body[0]) is None