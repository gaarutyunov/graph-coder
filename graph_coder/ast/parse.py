import ast

from graph_coder.ast.context import Context
from graph_coder.ast.transformers.ast_transformer import ASTransformer
from graph_coder.ast.transformers.key_value_pre_transformer import (
    KeyValuePreTransformer,
)
from graph_coder.ast.transformers.class_name_transformer import ClassNameTransformer
from graph_coder.ast.transformers.function_name_transformer import (
    FunctionNameTransformer,
)


def code_to_graph(source: str) -> Context:
    global counter
    counter = 0

    ctx = Context(source)
    visitor = ASTransformer(ctx)

    ast_ = ast.parse(source)

    for node in ast_.body:
        ctx.depth = 0
        node = KeyValuePreTransformer().visit(node)
        node = ClassNameTransformer().visit(node)
        node = FunctionNameTransformer().visit(node)
        visitor.visit(node)

    for edge in ctx.edges:
        ctx.add_edge(edge)

    return ctx
