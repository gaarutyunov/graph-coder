#  MIT License
#
#  Copyright (c) 2020 Phil Wang
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
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import math

import torch
from axial_positional_embedding import AxialPositionalEmbedding
from performer_pytorch.performer_pytorch import (
    cast_tuple,
    FixedPositionalEmbedding,
    default,
    Always,
    AbsolutePositionalEmbedding,
    Performer,
)
from torch import nn


class PerformerEncoder(nn.Module):
    """Encoder layer using Performer attention.

    This module is a modified :class:`performer_pytorch.performer_pytorch.PerformerLM`"""
    def __init__(
        self,
        *,
        max_seq_len,
        dim,
        depth,
        heads,
        dim_head=64,
        local_attn_heads=0,
        local_window_size=256,
        causal=False,
        ff_mult=4,
        nb_features=None,
        feature_redraw_interval=1000,
        reversible=False,
        ff_chunks=1,
        ff_glu=False,
        emb_dropout=0.0,
        ff_dropout=0.0,
        attn_dropout=0.0,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        use_scalenorm=False,
        use_rezero=False,
        cross_attend=False,
        no_projection=False,
        rotary_position_emb=True,
        axial_position_emb=False,
        axial_position_shape=None,
        auto_check_redraw=True,
        qkv_bias=False,
        attn_out_bias=False,
        shift_tokens=False,
    ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)

        if rotary_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif axial_position_emb:
            axial_position_shape = default(
                axial_position_shape, (math.ceil(max_seq_len / 64), 64)
            )
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(emb_dropout)

        self.performer = Performer(
            dim,
            depth,
            heads,
            dim_head,
            local_attn_heads,
            local_window_size,
            causal,
            ff_mult,
            nb_features,
            feature_redraw_interval,
            reversible,
            ff_chunks,
            generalized_attention,
            kernel_fn,
            use_scalenorm,
            use_rezero,
            ff_glu,
            ff_dropout,
            attn_dropout,
            cross_attend,
            no_projection,
            auto_check_redraw,
            qkv_bias,
            attn_out_bias,
            shift_tokens,
        )
        self.norm = nn.LayerNorm(dim)

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x: torch.Tensor, **kwargs):
        # token and positional embeddings
        x += self.pos_emb(x)

        x = self.dropout(x)

        # performer layers

        layer_pos_emb = self.layer_pos_emb(x)
        x = self.performer(x, pos_emb=layer_pos_emb, **kwargs)

        # norm

        return self.norm(x)
