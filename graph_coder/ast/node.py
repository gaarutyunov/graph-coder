import ast
from dataclasses import dataclass

from graph_coder.ast.node_attributes import NodeAttributes


@dataclass
class Node:
    name: str
    category: str
    depth: int
    id: int
    attr: NodeAttributes = NodeAttributes()

    @classmethod
    def from_ast_body(
        cls, id: int, body: ast.AST, depth: int, attr: NodeAttributes = NodeAttributes()
    ) -> "Node":
        if attr is None:
            attr = NodeAttributes()
        return cls(
            name=body.__class__.__name__,
            category=body.__class__.__base__.__name__,
            attr=attr,
            depth=depth,
            id=id,
        )
