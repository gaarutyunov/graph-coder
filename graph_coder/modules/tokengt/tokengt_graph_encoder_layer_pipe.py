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
from torch.nn import Identity

from graph_coder.pipe import (
    CloneLayer,
    Layers,
    PassThroughLayer,
    PipeModule,
    RemoveArgsLayer,
)
from .tokengt_graph_encoder_layer import TokenGTGraphEncoderLayer


class TokenGTGraphEncoderLayerPipe(TokenGTGraphEncoderLayer, PipeModule):
    def to_layers(self) -> Layers:
        if self.layernorm_style == "prenorm":
            layers: Layers = [
                PassThroughLayer(CloneLayer(), -2),
                # args: *batch_args, *, x, padding_mask, residual
                PassThroughLayer(self.self_attn_layer_norm, -3, -3),
                PassThroughLayer(
                    self.self_attn,
                    out=-3,
                    args_getter=lambda *args: (
                        args[-3],
                        args[-3],
                        args[-3],
                        None,
                        args[-2],
                        self.return_attention,
                        None,
                        False,
                        self.return_attention,
                    ),
                ),
                PassThroughLayer(self.dropout_module, -3, -3),
                PassThroughLayer(self.drop_path1, -3, -3),
                PassThroughLayer(
                    Identity(),
                    -3,
                    -3,
                    callback=lambda res, *args: res + args[-1],
                ),
                PassThroughLayer(Identity(), -1, -3),
                PassThroughLayer(self.final_layer_norm, -3, -3),
                PassThroughLayer(self.feedforward, -3, -3),
                PassThroughLayer(self.drop_path2, -3, -3),
                PassThroughLayer(
                    Identity(),
                    -3,
                    -3,
                    callback=lambda res, *args: res + args[-1],
                ),
                RemoveArgsLayer(-1),
            ]
        elif self.layernorm_style == "postnorm":
            layers = [
                PassThroughLayer(CloneLayer(), -2),
                # args: *batch_args, *, x, padding_mask, residual
                PassThroughLayer(
                    self.self_attn,
                    out=-3,
                    args_getter=lambda *args: (
                        args[-3],
                        args[-3],
                        args[-3],
                        None,
                        args[-2],
                        self.return_attention,
                        None,
                        False,
                        self.return_attention,
                    ),
                ),
                PassThroughLayer(self.dropout_module, -3, -3),
                PassThroughLayer(
                    Identity(),
                    -3,
                    -3,
                    callback=lambda res, *args: res + args[-1],
                ),
                PassThroughLayer(self.self_attn_layer_norm, -3, -3),
                PassThroughLayer(Identity(), -1, -3),
                PassThroughLayer(self.feedforward, -3, -3),
                PassThroughLayer(
                    Identity(),
                    -3,
                    -3,
                    callback=lambda res, *args: res + args[-1],
                ),
                PassThroughLayer(self.final_layer_norm, -3, -3),
            ]
        else:
            raise NotImplementedError

        return layers
