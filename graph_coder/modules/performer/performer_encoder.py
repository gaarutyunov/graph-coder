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

from performer_pytorch import Performer
from performer_pytorch.performer_pytorch import cast_tuple
from torch import nn

from .performer_encoder_base import PerformerEncoderBase


class PerformerEncoder(PerformerEncoderBase[Performer]):
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
        super().__init__(
            performer=Performer(
                dim,
                depth,
                heads,
                dim_head,
                cast_tuple(local_attn_heads),
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
            ),
            max_seq_len=max_seq_len,
            dim=dim,
            dim_head=dim_head,
            emb_dropout=emb_dropout,
            rotary_position_emb=rotary_position_emb,
            axial_position_emb=axial_position_emb,
            axial_position_shape=axial_position_shape,
        )
