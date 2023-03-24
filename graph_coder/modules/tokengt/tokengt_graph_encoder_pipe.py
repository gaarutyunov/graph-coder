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
from typing import List

from torch.nn import Identity

from graph_coder.pipe import Layers, PassThroughLayer, PipeLayer, PipeModule
from ...data import GraphCoderBatch

from .tokengt_graph_encoder import TokenGTGraphEncoder
from .tokengt_graph_encoder_layer_pipe import TokenGTGraphEncoderLayerPipe


class TokenGTGraphEncoderPipe(TokenGTGraphEncoder, PipeModule):
    layers: List[TokenGTGraphEncoderLayerPipe]

    def build_tokengt_graph_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        encoder_layers,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        drop_path,
        performer,
        performer_nb_features,
        performer_generalized_attention,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        layernorm_style,
        return_attention,
        causal: bool = False,
    ):
        return TokenGTGraphEncoderLayerPipe(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            encoder_layers=encoder_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            drop_path=drop_path,
            performer=performer,
            performer_nb_features=performer_nb_features,
            performer_generalized_attention=performer_generalized_attention,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            layernorm_style=layernorm_style,
            return_attention=return_attention,
            causal=causal,
        )

    def performer_redraw(self, *args):
        if self.performer and self.performer_auto_check_redraw:
            self.performer_proj_updater.redraw_projections()

        return args

    def to_layers(self) -> Layers:
        layers: Layers = [
            PipeLayer(self.performer_redraw),
            PassThroughLayer(
                self.graph_feature,
                [
                    GraphCoderBatch.get_arg_idx(name)
                    for name in [
                        "edge_index",
                        "edge_data",
                        "node_data",
                        "node_num",
                        "edge_num",
                        "padded_index",
                        "padding_mask",
                        "padded_node_mask",
                        "padded_edge_mask",
                    ]
                ],
            ),
        ]
        # args: *batch_args, *, x

        if self.quant_noise is not None:
            layers.append(PassThroughLayer(self.quant_noise, -1, -1))

        if self.emb_layer_norm is not None:
            layers.append(PassThroughLayer(self.emb_layer_norm, -1, -1))

        layers.append(
            PassThroughLayer(
                self.dropout_module,
                -1,
                -1,
                lambda res, *args: res.transpose(0, 1),
            )
        )
        for layer in self.layers:
            layers.extend(layer.to_layers())
        # args: *batch_args, *, x

        layers.append(
            PassThroughLayer(Identity(), -1, -1, lambda res, *args: res.transpose(0, 1))
        )

        return layers
