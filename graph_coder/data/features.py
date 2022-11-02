import typing

import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def data_to_node_edge_features(
    data: Data, values: typing.Iterable[str]
) -> typing.Tuple[np.ndarray, np.ndarray]:
    values_ = np.array(list(values), dtype=object)

    x_matrix = values_[data.x]
    edge_matrix = values_[data.edge_attr]

    return x_matrix, edge_matrix


def data_to_text(data: Data, values: typing.Iterable[str]) -> str:
    node_attr, edge_attr = data_to_node_edge_features(data, values)
    res = ""

    for i, (u, v) in enumerate(to_networkx(data, to_undirected="lower").edges):
        res += " ".join(node_attr[u, node_attr[u] != "[EMP]"]) + " "
        res += " ".join(edge_attr[i, :]) + " "
        res += " ".join(node_attr[v, node_attr[v] != "[EMP]"]) + " "

    return res
