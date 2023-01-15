import dataclasses

import torch
from torch.nn.utils.rnn import pad_sequence

from graph_coder.data.algos import lap_eig
from graph_coder.data.base import BaseExample

from torch.functional import F


@dataclasses.dataclass
class GraphCoderBatch:
    idx: torch.Tensor
    source: torch.Tensor
    docstring: torch.Tensor
    edge_index: torch.LongTensor
    edge_data: torch.Tensor
    node_data: torch.Tensor
    node_num: torch.LongTensor
    edge_num: torch.LongTensor
    lap_eigval: torch.Tensor
    lap_eigvec: torch.Tensor


@torch.no_grad()
def collate(batch: list[BaseExample], pad_token_id: int) -> GraphCoderBatch:
    idx = []
    edge_index = []
    edge_data = []
    node_data = []
    node_num = []
    edge_num = []
    sources = []
    docstrings = []

    max_node_length = 0
    max_edge_length = 0
    max_node_num = 0
    max_edge_num = 0
    lap_eigvals = []
    lap_eigvecs = []

    for data in batch:
        sources.append(data.source)
        docstrings.append(data.docstring)

        data = data.graph
        lap_eigval, lap_eigvec = lap_eig(data.edge_index, data.num_nodes)
        lap_eigval = lap_eigval[None, :].expand_as(lap_eigvec)
        lap_eigvals.append(lap_eigval)
        lap_eigvecs.append(lap_eigvec)

        idx.append(data.idx)
        edge_index.append(data.edge_index)

        node_data.append(data.x)
        max_node_length = max(max_node_length, data.x.size(-1))
        edge_data.append(data.edge_attr)
        max_edge_length = max(max_edge_length, data.edge_attr.size(-1))

        node_num.append(data.x.size(0))
        max_node_num = max(max_node_num, data.x.size(0))
        edge_num.append(data.edge_attr.size(0))
        max_edge_num = max(max_edge_num, data.edge_attr.size(0))

    batch = GraphCoderBatch(
        idx=torch.LongTensor(idx),
        edge_index=torch.LongTensor(torch.cat(edge_index, dim=1)),
        edge_data=torch.cat(
            [
                F.pad(
                    x,
                    (0, max_edge_length - x.size(-1), 0, max_edge_num - x.size(0)),
                    value=pad_token_id,
                )[None, ...]
                for x in edge_data
            ]
        ),
        node_data=torch.cat(
            [
                F.pad(
                    x,
                    (0, max_node_length - x.size(-1), 0, max_node_num - x.size(0)),
                    value=pad_token_id,
                )[None, ...]
                for x in node_data
            ]
        ),
        lap_eigval=torch.cat(
            [
                F.pad(x, (0, max_node_num - x.size(1)), value=float("0"))
                for x in lap_eigvals
            ]
        ),
        lap_eigvec=torch.cat(
            [
                F.pad(x, (0, max_node_num - x.size(1)), value=float("0"))
                for x in lap_eigvecs
            ]
        ),
        node_num=torch.LongTensor(node_num),
        edge_num=torch.LongTensor(edge_num),
        source=pad_sequence(sources, batch_first=True, padding_value=pad_token_id),
        docstring=pad_sequence(
            docstrings, batch_first=True, padding_value=pad_token_id
        ),
    )

    return batch
