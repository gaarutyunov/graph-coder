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
from torch import nn


class PerformerPipe(Performer):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        local_attn_heads=0,
        local_window_size=256,
        causal=False,
        ff_mult=4,
        nb_features=None,
        feature_redraw_interval=1000,
        reversible=False,
        ff_chunks=1,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        use_scalenorm=False,
        use_rezero=False,
        ff_glu=False,
        ff_dropout=0.0,
        attn_dropout=0.0,
        cross_attend=False,
        no_projection=False,
        auto_check_redraw=True,
        qkv_bias=True,
        attn_out_bias=True,
        shift_tokens=False,
    ):
        super().__init__(
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
        self.net.layers
        self.net.args_route

    def to_layers(self):
        layers = []

        return layers
