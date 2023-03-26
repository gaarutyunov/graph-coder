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

from graph_coder.pipe import Layers, PassThroughLayer, PipeModule, RemoveArgsLayer
from .tokengt_graph_encoder_layer import TokenGTGraphEncoderLayer
from graph_coder.data import get_arg_idx


class TokenGTGraphEncoderLayerPipe(TokenGTGraphEncoderLayer, PipeModule):
    def to_layers(self) -> Layers:
        if self.layernorm_style == "prenorm":
            layers: Layers = [
                PassThroughLayer(Identity(), -1),
                # args: *batch_args, *, x, residual
                PassThroughLayer(self.self_attn_layer_norm, -2, -2),
                PassThroughLayer(
                    self.self_attn,
                    out=-2,
                    args_getter=lambda *args: (
                        args[-2],
                        args[-2],
                        args[-2],
                        None,
                        args[get_arg_idx("padding_mask")],
                        self.return_attention,
                        None,
                        False,
                        self.return_attention,
                    ),
                ),
                PassThroughLayer(self.dropout_module, -2, -2),
                PassThroughLayer(self.drop_path1, -2, -2),
                PassThroughLayer(
                    Identity(),
                    -2,
                    -2,
                    callback=lambda res, *args: res + args[-1],
                ),
                PassThroughLayer(Identity(), -1, -2),
                PassThroughLayer(self.final_layer_norm, -2, -2),
                PassThroughLayer(self.feedforward, -2, -2),
                PassThroughLayer(self.drop_path2, -2, -2),
                PassThroughLayer(
                    Identity(),
                    -2,
                    -2,
                    callback=lambda res, *args: res + args[-1],
                ),
                RemoveArgsLayer(-1),
            ]
        elif self.layernorm_style == "postnorm":
            layers = [
                PassThroughLayer(Identity(), -1),
                # args: *batch_args, *, x, residual
                PassThroughLayer(
                    self.self_attn,
                    out=-2,
                    args_getter=lambda *args: (
                        args[-2],
                        args[-2],
                        args[-2],
                        None,
                        args[get_arg_idx("padding_mask")],
                        self.return_attention,
                        None,
                        False,
                        self.return_attention,
                    ),
                ),
                PassThroughLayer(self.dropout_module, -2, -2),
                PassThroughLayer(
                    Identity(),
                    -2,
                    -2,
                    callback=lambda res, *args: res + args[-1],
                ),
                PassThroughLayer(self.self_attn_layer_norm, -2, -2),
                PassThroughLayer(Identity(), -1, -2),
                PassThroughLayer(self.feedforward, -2, -2),
                PassThroughLayer(
                    Identity(),
                    -2,
                    -2,
                    callback=lambda res, *args: res + args[-1],
                ),
                PassThroughLayer(self.final_layer_norm, -2, -2),
            ]
        else:
            raise NotImplementedError

        return layers
