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
from typing import Optional

from torch import nn

from graph_coder.pipe import Layers, PassThroughLayer, PipeModule

from .tokengt_encoder_base import TokenGTEncoderBase
from .tokengt_graph_encoder_pipe import TokenGTGraphEncoderPipe


class TokenGTEncoderPipe(TokenGTEncoderBase[PipeModule], PipeModule):
    def __init__(
        self,
        embedding: nn.Module,
        encoder_layers: int = 12,
        encoder_attention_heads: int = 8,
        encoder_embed_dim: int = 768,
        encoder_ffn_embed_dim: int = 768,
        act_dropout: float = 0.1,
        share_encoder_input_output_embed: Optional[bool] = None,
        remove_head: bool = False,
        lap_node_id: bool = False,
        lap_node_id_k: int = 8,
        lap_node_id_sign_flip: bool = False,
        lap_node_id_eig_dropout: float = 0.0,
        type_id: bool = False,
        stochastic_depth: bool = False,
        performer: bool = False,
        performer_finetune: bool = False,
        performer_nb_features: Optional[int] = None,
        performer_feature_redraw_interval: int = 1000,
        performer_generalized_attention: bool = False,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        encoder_normalize_before: bool = False,
        layernorm_style: str = "prenorm",
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        return_attention: bool = False,
        causal: bool = False,
        last_state_only: bool = True,
    ):
        super().__init__(
            TokenGTGraphEncoderPipe(
                # < for tokenization
                embedding=embedding,
                lap_node_id=lap_node_id,
                lap_node_id_k=lap_node_id_k,
                lap_node_id_sign_flip=lap_node_id_sign_flip,
                lap_node_id_eig_dropout=lap_node_id_eig_dropout,
                type_id=type_id,
                # >
                # < performer
                performer=performer,
                performer_finetune=performer_finetune,
                performer_nb_features=performer_nb_features,
                performer_feature_redraw_interval=performer_feature_redraw_interval,
                performer_generalized_attention=performer_generalized_attention,
                # >
                stochastic_depth=stochastic_depth,
                causal=causal,
                num_encoder_layers=encoder_layers,
                embedding_dim=encoder_embed_dim,
                ffn_embedding_dim=encoder_ffn_embed_dim,
                num_attention_heads=encoder_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=act_dropout,
                encoder_normalize_before=encoder_normalize_before,
                layernorm_style=layernorm_style,
                apply_graphormer_init=apply_graphormer_init,
                activation_fn=activation_fn,
                return_attention=return_attention,
                last_state_only=last_state_only,
            ),
            encoder_layers,
            encoder_attention_heads,
            encoder_embed_dim,
            share_encoder_input_output_embed,
            remove_head,
            layernorm_style,
            activation_fn,
            return_attention,
        )

    def to_layers(self) -> Layers:
        layers = [
            *self.graph_encoder.to_layers(),
            # args: *batch_args, *, x
            PassThroughLayer(self.lm_head_transform_weight, -1, -1),
            PassThroughLayer(self.activation_fn, -1, -1),
            PassThroughLayer(self.layer_norm, -1, -1),
        ]

        return layers
