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

from graph_coder.data import get_index_and_mask, GraphCoderBatch


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


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
        padded_feature[padded_node_mask, :] = node_feature
        padded_feature[padded_edge_mask, :] = edge_feature

        return padded_feature

    def get_type_embed(self, padded_index):
        """
        :param padded_index: LongTensor([B, T, 2])
        :return: Tensor([B, T, D])
        """
        order = torch.eq(padded_index[..., 0], padded_index[..., 1]).long()  # [B, T]
        order_embed = self.order_encoder(order)
        return order_embed

    def forward(
        self,
        edge_index,
        edge_data,
        node_data,
        node_num,
        edge_num,
        lap_eigvec,
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
            padded_feature = padded_feature + self.lap_encoder(lap_eigvec)

        if self.type_id:
            padded_feature = padded_feature + self.get_type_embed(padded_index)

        padded_feature = padded_feature.masked_fill(padding_mask[..., None], float("0"))

        return padded_feature  # [B, T, D]
