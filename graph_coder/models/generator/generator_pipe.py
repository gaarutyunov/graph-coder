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
import torch
from torch import nn

from graph_coder.data import GraphCoderBatch
from graph_coder.pipe import Layers, PassThroughLayer, pipe_wrap, PipeModule

from .generator_base import GraphCoderGeneratorBase
from .layers import LmLayer
from .layers_pipe import CodeLayerPipe, GraphLayerPipe, TextLayerPipe


class GraphCoderGeneratorPipe(GraphCoderGeneratorBase[PipeModule], PipeModule):
    def __init__(
        self,
        embedding: nn.Module,
        text_encoder: PipeModule,
        code_encoder: PipeModule,
        graph_encoder: PipeModule,
        decoder: PipeModule,
        hidden_size: int,
        vocab_size: int,
        eos_token_id: int = 0,
        max_length: int = 64,
    ) -> None:
        super().__init__(
            layers=[
                TextLayerPipe(embedding, text_encoder, eos_token_id),
                GraphLayerPipe(graph_encoder),
                CodeLayerPipe(embedding, code_encoder, eos_token_id),
            ],
            decoder=decoder,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            max_length=max_length,
            lm_layer=LmLayer(vocab_size, max_length, hidden_size),  # type: ignore[arg-type]
        )

    def has_source(self, *args):
        return GraphCoderBatch.from_tuple(args).has_source

    def has_docstring(self, *args):
        return GraphCoderBatch.from_tuple(args).has_docstring

    def has_graph(self, *args):
        return GraphCoderBatch.from_tuple(args).has_graph

    @pipe_wrap
    def to_layers(self) -> Layers:
        layers = []

        for layer in self.layers:
            layers.extend(layer.to_layers())

        layers.extend(
            [
                # args: *batch_args, tgt, memory
                *self.decoder.to_layers(),
                # args: *batch_args, tgt
                PassThroughLayer(
                    self.dense,
                    -1,
                    -1,
                    callback=lambda res, *args: torch.tanh(res).contiguous(),
                ),
                # args: *batch_args, tgt (hidden_states)
                LmLayer(self.vocab_size, self.max_length, self.hidden_size),
                # torch.Tensor (logits)
            ]
        )

        return layers
