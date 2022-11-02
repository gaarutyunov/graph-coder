import typing
from dataclasses import dataclass, field

import networkx as nx

from graph_coder.ast.node import Node
from graph_coder.ast.edge import Edge
from graph_coder.ast.vocabulary import Vocabulary


@dataclass
class Context:
    src: str
    depth: int = field(default=0)
    counter: int = field(default=0)
    g: nx.Graph = field(default_factory=nx.Graph)
    v: Vocabulary = field(default_factory=Vocabulary)
    ctx: typing.Dict[str, "Node"] = field(default_factory=dict)
    edges: typing.List["Edge"] = field(default_factory=list)

    @property
    def id(self):
        counter = self.counter
        self.counter += 1
        return counter

    def has_node(self, node: "Node") -> bool:
        return self.g.has_node(node.id)

    def add_node(self, node: "Node"):
        if self.has_node(node):
            return
        self.g.add_node(
            node.id,
            label=self.add_string(node.name),
            category=self.add_string(node.category),
            **node.attr.to_dict(self),
        )

    def add_edge(self, edge: "Edge"):
        self.g.add_edge(
            edge.node_left.id,
            edge.node_right.id,
            label=self.add_string(edge.relation),
            **edge.attr.to_dict(self),
        )

    def add_string(self, value: str) -> int:
        return self.v.add(value)

    def __setitem__(self, key: str, value: "Node"):
        self.ctx[key] = value

    def __getitem__(self, item: str) -> typing.Optional["Node"]:
        return self.ctx[item]

    def __contains__(self, item: str) -> bool:
        return item in self.ctx

    @property
    def graph_with_features(self) -> nx.Graph:
        g = nx.Graph()
        for node, attr in self.g.nodes(data=True):
            new_attr = {
                k: self.v.key_value[v] for k, v in attr.items() if v is not None
            }
            g.add_node(node, **new_attr)

        for u, v, attr in self.g.edges(data=True):
            new_attr = {
                k: self.v.key_value[v_] for k, v_ in attr.items() if v_ is not None
            }
            g.add_edge(u, v, **new_attr)

        return g
