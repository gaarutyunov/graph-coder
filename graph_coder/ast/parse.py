import ast
import typing

from graph_coder.ast.context import Context
from graph_coder.ast.transformers.ast_transformer import ASTransformer
from graph_coder.ast.transformers.key_value_pre_transformer import (
    KeyValuePreTransformer,
)
from graph_coder.ast.transformers.class_name_transformer import ClassNameTransformer
from graph_coder.ast.transformers.function_name_transformer import (
    FunctionNameTransformer,
)


def code_to_graph(source: str) -> typing.List[Context]:
    ctxs = []

    ast_ = ast.parse(source)

    for node in ast_.body:
        src = ast.unparse(node)
        if len(src.strip()) == 0:
            continue
        ctx = Context(src)
        ctx.depth = 0
        visitor = ASTransformer(ctx)
        node = KeyValuePreTransformer().visit(node)
        node = ClassNameTransformer().visit(node)
        node = FunctionNameTransformer().visit(node)
        visitor.visit(node)

        if ctx.g.number_of_nodes() == 0:
            continue

        for edge in ctx.edges:
            ctx.add_edge(edge)

        ctxs.append(ctx)

    return ctxs
