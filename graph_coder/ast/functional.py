#  Copyright 2023 German Arutyunov
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import ast
from typing import Tuple

import networkx as nx
from astmonkey import transformers as ast_transformers

from .visitors import GraphNodeVisitor
from graph_coder.data import AstData


def graph_to_data(idx: int, graph: nx.Graph) -> AstData:
    x = []
    edge_index = []
    edge_attr = []
    graph = nx.convert_node_labels_to_integers(graph, label_attribute="label")

    for node, label in graph.nodes(data="label"):
        x.append(label)

    for u, v, label in graph.edges(data="label"):
        edge_index.append((u, v))
        edge_attr.append(label)

    return AstData(
        idx=idx,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def get_docstring(node: ast.AST) -> str:
    try:
        doc = ast.get_docstring(node)
        return doc if doc is not None else ""
    except TypeError:
        return ""


def node_to_graph(node: ast.AST) -> Tuple[ast.AST, nx.Graph]:
    node = ast_transformers.ParentChildNodeTransformer().visit(node)
    visitor = GraphNodeVisitor()
    visitor.visit(node)

    return node, visitor.graph


def parse_graph(source: str, name: str) -> Tuple[ast.AST, nx.Graph]:
    mod = ast.parse(source, filename=name, feature_version=9)
    node = mod.body[0]

    return node_to_graph(node)
