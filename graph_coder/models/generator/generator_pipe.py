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
from graph_coder.pipe import (
    ConditionalLayer,
    Layers,
    PassThroughLayer,
    pipe_wrap,
    PipeModule,
    RemoveArgsLayer,
)

from .generator_base import GraphCoderGeneratorBase
from .layers_pipe import CodeLayerPipe, GraphLayerPipe, TextLayerPipe


class GraphCoderGeneratorPipe(GraphCoderGeneratorBase[PipeModule], PipeModule):
    def __init__(
        self,
        embedding: nn.Module,
        text_encoder: PipeModule,
        code_encoder: PipeModule,
        graph_encoder: PipeModule,
        criterion: nn.Module,
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
        )
        self.criterion = criterion

    def get_states(self, *args):
        batch = GraphCoderBatch.from_tuple(args)

        hidden_states = args[-1]
        largs = list(args)
        if batch.has_docstring:
            largs.append(hidden_states[:, : batch.docstring_size])

        if batch.has_graph:
            if batch.has_docstring:
                start = batch.docstring_size
            else:
                start = 0
            if batch.has_source:
                end = batch.source_size
            else:
                end = -1
            largs.append(hidden_states[:, start:-end])

        if batch.has_source:
            largs.append(hidden_states[:, -batch.source_size :])

        return tuple(largs)

    def has_source(self, *args):
        return GraphCoderBatch.from_tuple(args).has_source

    def has_docstring(self, *args):
        return GraphCoderBatch.from_tuple(args).has_docstring

    def has_graph(self, *args):
        return GraphCoderBatch.from_tuple(args).has_graph

    def calc_loss(self, *args):
        batch = GraphCoderBatch.from_tuple(args)

        lm_logits = []
        target_ids = []

        if batch.has_docstring:
            target_ids.append(batch.docstring[batch.docstring_attn_mask.bool()])
            lm_logits.append(args[-3][batch.docstring_attn_mask.bool()])
        if batch.has_graph:
            target_ids.extend(
                [
                    batch.node_data[batch.node_data_attn_mask.bool()],
                    batch.edge_data[batch.edge_data_attn_mask.bool()],
                ]
            )
            lm_logits.append(
                args[-1][batch.padded_node_mask][batch.node_data_attn_mask.bool()]
            )
            lm_logits.append(
                args[-1][batch.padded_edge_mask][batch.edge_data_attn_mask.bool()]
            )
        if batch.has_source:
            target_ids.append(batch.source[batch.source_attn_mask.bool()])
            lm_logits.append(args[-2][batch.source_attn_mask.bool()])

        target_ids_ = torch.cat(target_ids)
        lm_logits_ = torch.cat(lm_logits, dim=0)

        shift_logits = lm_logits_[:-1, :].contiguous()
        shift_labels = target_ids_[1:].contiguous().long()

        return self.criterion(shift_logits, shift_labels)

    @pipe_wrap
    def to_layers(self) -> Layers:
        layers = []

        for layer in self.layers:
            layers.extend(layer.to_layers())

        layers.extend(
            [
                # args: *batch_args, tgt, memory
                *self.decoder.to_layers(),
                RemoveArgsLayer(-1),
                # args: *batch_args, tgt
                PassThroughLayer(
                    self.dense,
                    -1,
                    -1,
                    callback=lambda res, *args: torch.tanh(res).contiguous(),
                ),
                # args: *batch_args, tgt (hidden_states)
                self.get_states,
                # args: *batch_args, hidden_states, text_states?, graph_states, code_states
                ConditionalLayer(
                    PassThroughLayer(self.lm_head, -3, -3),
                    self.has_docstring,
                ),
                # args: *batch_args, hidden_states, docstring_result?, graph_states, code_states
                ConditionalLayer(
                    PassThroughLayer(
                        self.lm_graph_head,
                        -2,
                        callback=lambda res, *args: res.view(
                            res.size(0), -1, self.hidden_size
                        ),
                    ),
                    self.has_graph,
                ),
                # args: *batch_args, hidden_states, docstring_result?, graph_states, code_states, graph_result
                ConditionalLayer(
                    PassThroughLayer(self.lm_head, -2, -2),
                    self.has_source,
                ),
                # args: *batch_args, hidden_states, docstring_result?, graph_states, source_result, graph_result
                ConditionalLayer(
                    PassThroughLayer(
                        self.lm_head,
                        -1,
                        -1,
                        callback=lambda res, *args: res.view(
                            res.size(0),
                            args[-3].size(1),
                            -1,
                            self.vocab_size,
                        ),
                    ),
                    self.has_graph,
                ),
                # args: *batch_args, hidden_states, docstring_result?, graph_states, source_result, graph_result
                RemoveArgsLayer(-3),
                # args: *batch_args, hidden_states, docstring_result?, source_result, graph_result
                self.calc_loss
                # torch.Tensor (loss)
            ]
        )

        return layers
