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
from typing import Dict, List, Tuple

from performer_pytorch import Performer
from performer_pytorch.reversible import SequentialSequence
from torch import nn

from graph_coder.pipe import Layers, PassThroughLayer, PipeModule


class PerformerPipe(PipeModule, Performer):
    net: SequentialSequence

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

    def performer_redraw(self, **kwargs):
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()

        return kwargs

    def to_layers(self) -> Layers:
        layers = [self.performer_redraw]

        args_route: List[Dict[str, Tuple[bool, ...]]] = [{}] * len(self.net.layers)

        for arg_name, arg_layers in self.net.args_route.items():
            for i, r in enumerate(arg_layers):
                args_route[i][arg_name] = r

        for (f, g), args_map in zip(self.net.layers, args_route):
            layers.extend(
                [
                    PassThroughLayer(
                        f,
                        "x",
                        ["x"] + [k for k, v in args_map.items() if v[0]],
                        callback=lambda res, **kwargs: kwargs["x"] + res,
                        args_mode="kwargs",
                    ),
                    PassThroughLayer(
                        g,
                        "x",
                        ["x"] + [k for k, v in args_map.items() if v[1]],
                        callback=lambda res, **kwargs: kwargs["x"] + res,
                        args_mode="kwargs",
                    ),
                ]
            )

        return layers
