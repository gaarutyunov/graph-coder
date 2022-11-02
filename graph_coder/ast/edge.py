import ast
from dataclasses import dataclass, field

from graph_coder.ast.edge_attributes import EdgeAttributes
from graph_coder.ast.node import Node


@dataclass
class Edge:
    source_left: ast.AST = None
    source_right: ast.AST = None
    relation: str = None
    attr: EdgeAttributes = field(default_factory=EdgeAttributes)
    node_left: "Node" = None
    node_right: "Node" = None

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False

        return (
            self.node_left == other.node_left and self.node_right == other.node_right
        ) or (
            self.source_left == other.source_left
            and self.source_right == other.source_right
        )
