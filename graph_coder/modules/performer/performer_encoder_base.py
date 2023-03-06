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
from typing import TypeVar, Generic

import torch
from axial_positional_embedding import AxialPositionalEmbedding
from performer_pytorch.performer_pytorch import (
    FixedPositionalEmbedding,
    default,
    Always,
    AbsolutePositionalEmbedding,
)
from torch import nn


TE = TypeVar("TE", bound=nn.Module)


class PerformerEncoderBase(nn.Module, Generic[TE]):
    """Encoder layer using Performer attention.

    This module is a modified :class:`performer_pytorch.performer_pytorch.PerformerLM`
    """

    def __init__(
        self,
        *,
        performer: TE,
        max_seq_len,
        dim,
        dim_head=64,
        emb_dropout=0.0,
        rotary_position_emb=True,
        axial_position_emb=False,
        axial_position_shape=None,
    ):
        super().__init__()

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

        self.performer = performer
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
