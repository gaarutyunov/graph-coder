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

import networkx as nx


class GraphNodeVisitor(ast.NodeVisitor):
    """Visitor that creates a networkx graph from an AST."""

    def __init__(self, compact: bool = False):
        self.graph = nx.Graph()
        self.compact = compact

    def visit(self, node):
        if len(node.parents) <= 1:
            self.graph.add_node(self._node(node))
        if len(node.parents) == 1:
            u, v, attr = self._edge(node)
            self.graph.add_edge(u, v, **attr)
        super(GraphNodeVisitor, self).visit(node)

    def _node(self, node):
        fields_labels = []
        if self.compact:
            return "ast.{0}".format(node.__class__.__name__)

        for field, value in ast.iter_fields(node):
            if not isinstance(value, list):
                value_label = self._node_value_label(value)
                if value_label:
                    fields_labels.append("{0}={1}".format(field, value_label))
        return "ast.{0}({1})".format(node.__class__.__name__, ", ".join(fields_labels))

    def _node_value_label(self, value):
        if not isinstance(value, ast.AST):
            return repr(value)
        elif len(value.parents) > 1:
            return self._node(value)
        return None

    def _edge(self, node):
        return (
            self._node(node.parent),
            self._node(node),
            {"label": self._edge_label(node)},
        )

    def _edge_label(self, node):
        label = node.parent_field
        if node.parent_field_index is not None:
            label += "[{0}]".format(node.parent_field_index)
        return label
