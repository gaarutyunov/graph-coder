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

from graph_coder.modules.pass_through import PassThroughLayer
from .tokengt_graph_encoder_layer import TokenGTGraphEncoderLayer
from graph_coder.pipe import PipeModule, Layers


class TokenGTGraphEncoderLayerPipe(TokenGTGraphEncoderLayer, PipeModule):
    def to_layers(self) -> Layers:
        if self.layernorm_style == "prenorm":
            layers: Layers = [
                PassThroughLayer(Identity(), "residual", ["x"]),
                PassThroughLayer(self.self_attn_layer_norm, "x", ["x"]),
                PassThroughLayer(
                    self.self_attn,
                    "x",
                    args_getter=lambda **kwargs: {
                        "query": kwargs["x"],
                        "key": kwargs["x"],
                        "value": kwargs["x"],
                        "key_padding_mask": kwargs["padding_mask"],
                        "need_weights": self.return_attention,
                        "need_head_weights": self.return_attention,
                    },
                ),
                PassThroughLayer(self.dropout_module, "x", ["x"]),
                PassThroughLayer(self.drop_path1, "x", ["x"]),
                PassThroughLayer(
                    Identity(),
                    "x",
                    ["x"],
                    callback=lambda res, **kwargs: res + kwargs["residual"],
                ),
                PassThroughLayer(Identity(), "residual", ["x"]),
                PassThroughLayer(self.final_layer_norm, "x", ["x"]),
                PassThroughLayer(self.feedforward, "x", ["x"]),
                PassThroughLayer(self.drop_path2, "x", ["x"]),
                PassThroughLayer(
                    Identity(),
                    "x",
                    ["x"],
                    callback=lambda res, **kwargs: res + kwargs["residual"],
                ),
            ]
        elif self.layernorm_style == "postnorm":
            layers = [
                PassThroughLayer(Identity(), "residual", ["x"]),
                PassThroughLayer(
                    self.self_attn,
                    "x",
                    args_getter=lambda **kwargs: {
                        "query": kwargs["x"],
                        "key": kwargs["x"],
                        "value": kwargs["x"],
                        "key_padding_mask": kwargs["padding_mask"],
                        "need_weights": self.return_attention,
                        "need_head_weights": self.return_attention,
                    },
                ),
                PassThroughLayer(self.dropout_module, "x", ["x"]),
                PassThroughLayer(
                    Identity(),
                    "x",
                    ["x"],
                    callback=lambda res, **kwargs: res + kwargs["residual"],
                ),
                PassThroughLayer(self.self_attn_layer_norm, "x", ["x"]),
                PassThroughLayer(Identity(), "residual", ["x"]),
                PassThroughLayer(self.feedforward, "x", ["x"]),
                PassThroughLayer(
                    Identity(),
                    "x",
                    ["x"],
                    callback=lambda res, **kwargs: res + kwargs["residual"],
                ),
                PassThroughLayer(self.final_layer_norm, "x", ["x"]),
            ]
        else:
            raise NotImplementedError

        return layers
