import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout


def draw_tree(g: nx.Graph):
    pos = graphviz_layout(g, prog="dot")
    nx.draw(g, pos, labels=dict(g.nodes.data(data="label")))
    plt.show()
