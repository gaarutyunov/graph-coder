"""
Modified from https://github.com/jw9730/tokengt
"""

import torch
import numpy as np
import pyximport

from . import algos

pyximport.install(setup_args={"include_dirs": np.get_include()})


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512, concat_features: bool = False):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset.unsqueeze(1).repeat(1, x.size(-1))
    if concat_features:
        return x.reshape(x.size(0), -1)

    return x


def preprocess_item(item):
    edge_int_feature, edge_index, node_int_feature = (
        item.edge_attr,
        item.edge_index,
        item.x,
    )
    # node_data = convert_to_single_emb(node_int_feature)
    # edge_data = convert_to_single_emb(edge_int_feature)
    node_data = node_int_feature
    edge_data = edge_int_feature

    N = node_int_feature.size(0)
    dense_adj = torch.zeros([N, N], dtype=torch.bool)
    dense_adj[edge_index[0, :], edge_index[1, :]] = True
    in_degree = dense_adj.long().sum(dim=1).view(-1)
    lap_eigvec, lap_eigval = algos.lap_eig(dense_adj, N, in_degree)  # [N, N], [N,]
    lap_eigval = lap_eigval[None, :].expand_as(lap_eigvec)

    item.node_data = node_data
    item.edge_data = edge_data
    item.edge_index = edge_index
    item.in_degree = in_degree
    item.out_degree = in_degree  # for undirected graph
    item.lap_eigvec = lap_eigvec
    item.lap_eigval = lap_eigval
    return item
