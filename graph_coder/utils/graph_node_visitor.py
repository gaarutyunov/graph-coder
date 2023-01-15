import ast

import networkx as nx


class GraphNodeVisitor(ast.NodeVisitor):
    """Visitor that creates a networkx graph from an AST."""

    def __init__(self):
        self.graph = nx.Graph()

    def visit(self, node):
        if len(node.parents) <= 1:
            self.graph.add_node(self._node(node))
        if len(node.parents) == 1:
            u, v, attr = self._edge(node)
            self.graph.add_edge(u, v, **attr)
        super(GraphNodeVisitor, self).visit(node)

    def _node(self, node):
        fields_labels = []
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
        if not node.parent_field_index is None:
            label += "[{0}]".format(node.parent_field_index)
        return label
