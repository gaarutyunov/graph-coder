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

from typing import Generic, Optional, TypeVar

import torch
from fairseq import utils
from torch import nn
from torch.nn import LayerNorm

TGE = TypeVar("TGE", bound=nn.Module)


class TokenGTEncoderBase(nn.Module, Generic[TGE]):
    def __init__(
        self,
        graph_encoder: TGE,
        encoder_layers: int = 12,
        encoder_attention_heads: int = 8,
        encoder_embed_dim: int = 768,
        share_encoder_input_output_embed: Optional[bool] = None,
        remove_head: bool = False,
        layernorm_style: str = "prenorm",
        activation_fn: str = "gelu",
        return_attention: bool = False,
    ):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.num_attention_heads = encoder_attention_heads
        self.return_attention = return_attention

        if layernorm_style not in ["prenorm", "postnorm"]:
            raise NotImplementedError

        self.graph_encoder = graph_encoder

        self.share_input_output_embed = share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not remove_head
        self.masked_lm_pooler = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        self.lm_head_transform_weight = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(encoder_embed_dim)

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(
        self,
        edge_index,
        node_num,
        edge_num,
        padded_index,
        padding_mask,
        padded_feature,
    ):
        x = self.graph_encoder(
            edge_index,
            node_num,
            edge_num,
            padded_index,
            padding_mask,
            padded_feature
        )

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        return x

    def performer_finetune_setup(self):
        self.graph_encoder.performer_finetune_setup()
