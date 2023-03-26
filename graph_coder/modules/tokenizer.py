#  Copyright (c) Microsoft Corporation.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  Copyright 2023 German Arutyunov
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_coder.data import lap_eig


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def get_index_embed(node_id, node_mask, padded_index):
    """
    :param node_id: Tensor([sum(node_num), D])
    :param node_mask: BoolTensor([B, max_n])
    :param padded_index: LongTensor([B, T, 2])
    :return: Tensor([B, T, 2D])
    """
    b, max_n = node_mask.size()
    max_len = padded_index.size(1)
    d = node_id.size(-1)

    padded_node_id = torch.zeros(
        b, max_n, d, device=node_id.device, dtype=node_id.dtype
    )  # [B, max_n, D]
    padded_node_id[node_mask] = node_id

    padded_node_id = padded_node_id[:, :, None, :].expand(b, max_n, 2, d)
    padded_index = padded_index[..., None].expand(b, max_len, 2, d)
    index_embed = padded_node_id.gather(1, padded_index)  # [B, T, 2, D]
    index_embed = index_embed.view(b, max_len, 2 * d)
    return index_embed


@torch.no_grad()
def get_node_mask(node_num, device):
    b = len(node_num)
    max_n = max(node_num)
    node_index = torch.arange(max_n, device=device, dtype=torch.long)[None, :].expand(
        b, max_n
    )  # [B, max_n]
    node_num_ = node_num.clone().detach()[:, None]
    node_mask = torch.less(node_index, node_num_)  # [B, max_n]
    return node_mask


@torch.no_grad()
def get_random_sign_flip(eigvec, node_mask):
    b, max_n = node_mask.size()
    d = eigvec.size(1)

    sign_flip = torch.rand(b, d, device=eigvec.device, dtype=eigvec.dtype)
    sign_flip[sign_flip >= 0.5] = 1.0
    sign_flip[sign_flip < 0.5] = -1.0
    sign_flip = sign_flip[:, None, :].expand(b, max_n, d)
    sign_flip = sign_flip[node_mask]
    return sign_flip


class GraphFeatureTokenizer(nn.Module):
    """
    Compute node and edge features for each node and edge in the graph.
    """

    def __init__(
        self,
        embedding,
        lap_node_id,
        lap_node_id_k,
        lap_node_id_sign_flip,
        lap_node_id_eig_dropout,
        type_id,
        hidden_dim,
        n_layers,
    ):
        super(GraphFeatureTokenizer, self).__init__()

        self.embedding = embedding

        self.lap_node_id = lap_node_id
        self.lap_node_id_k = lap_node_id_k
        self.lap_node_id_sign_flip = lap_node_id_sign_flip

        self.type_id = type_id

        if self.lap_node_id:
            self.lap_encoder = nn.Linear(2 * lap_node_id_k, hidden_dim, bias=False)
            self.lap_eig_dropout = (
                nn.Dropout2d(p=lap_node_id_eig_dropout)
                if lap_node_id_eig_dropout > 0
                else None
            )

        if self.type_id:
            self.order_encoder = nn.Embedding(2, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def process_batch(
        self,
        node_data,
        edge_data,
        edge_index,
        node_num,
        edge_num,
        padded_node_mask,
        padded_edge_mask,
    ) -> torch.Tensor:
        node_feature = self.embedding(node_data)  # [sum(node_num), T, 1]
        assert node_feature.size(-1) == 1, "Use embedding with output dim equal to 1"
        node_feature = node_feature.squeeze(-1)  # [sum(node_num), T]

        edge_feature = self.embedding(edge_data)  # [sum(edge_num), T, 1]
        edge_feature = edge_feature.squeeze(-1)  # [sum(edge_num), T]

        seq_len = [n + e for n, e in zip(node_num, edge_num)]
        b = len(seq_len)
        d = node_feature.size(-1)
        max_len = max(seq_len)
        device = edge_index.device

        padded_feature = torch.zeros(
            b, max_len, d, device=device, dtype=node_feature.dtype
        )  # [B, T, D]
        padded_feature[padded_node_mask.bool(), :] = node_feature
        padded_feature[padded_edge_mask.bool(), :] = edge_feature

        return padded_feature

    def get_type_embed(self, padded_index):
        """
        :param padded_index: LongTensor([B, T, 2])
        :return: Tensor([B, T, D])
        """
        order = torch.eq(padded_index[..., 0], padded_index[..., 1]).long()  # [B, T]
        order_embed = self.order_encoder(order)
        return order_embed

    def handle_eigvec(self, eigvec, node_mask):
        if self.lap_node_id_sign_flip and self.training:
            sign_flip = get_random_sign_flip(eigvec, node_mask)
            eigvec = eigvec * sign_flip
        else:
            pass
        return eigvec

    def get_eigvec_embed(
        self, edge_index, node_num, edge_num, padded_index, dtype, device
    ):
        max_n = max(node_num)
        lap_eigvecs = []

        start, end = 0, 0
        for i in range(len(edge_num)):
            end = start + edge_num[i]

            edge_index_ = edge_index[:, start:end]
            _, lap_eigvec = lap_eig(
                edge_index_, node_num[i], dtype=dtype, device=device
            )
            lap_eigvecs.append(lap_eigvec)

            start = end

        lap_eigvec_ = torch.cat(
            [F.pad(i, (0, max_n - i.size(1)), value=float("0")) for i in lap_eigvecs]
        )

        lap_dim = lap_eigvec_.size(-1)
        if self.lap_node_id_k > lap_dim:
            lap_eigvec_ = F.pad(
                lap_eigvec_,
                (0, self.lap_node_id_k - lap_dim),
                value=float("0"),
            )  # [sum(n_node), Dl]
        else:
            lap_eigvec_ = lap_eigvec_[:, : self.lap_node_id_k]  # [sum(n_node), Dl]

        if self.lap_eig_dropout is not None:
            lap_eigvec_ = self.lap_eig_dropout(lap_eigvec_[..., None, None]).view(
                lap_eigvec_.size()
            )

        node_mask = get_node_mask(node_num, device)

        lap_node_id = self.handle_eigvec(lap_eigvec_, node_mask)
        lap_index_embed = get_index_embed(lap_node_id, node_mask, padded_index)

        return self.lap_encoder(lap_index_embed)

    def forward(
        self,
        edge_index,
        edge_data,
        node_data,
        node_num,
        edge_num,
        padded_index,
        padding_mask,
        padded_node_mask,
        padded_edge_mask,
    ):
        padded_feature = self.process_batch(
            node_data,
            edge_data,
            edge_index,
            node_num,
            edge_num,
            padded_node_mask,
            padded_edge_mask,
        )

        if self.lap_node_id:
            padded_feature = padded_feature + self.get_eigvec_embed(
                edge_index,
                node_num,
                edge_num,
                padded_index,
                padded_feature.dtype,
                padded_feature.device,
            )

        if self.type_id:
            padded_feature = padded_feature + self.get_type_embed(padded_index)

        padded_feature = padded_feature.masked_fill(padding_mask.bool()[..., None], float("0"))

        return padded_feature  # [B, T, D]
