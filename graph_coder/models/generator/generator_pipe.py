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
from graph_coder.pipe import ConditionalLayer, Layers, PassThroughLayer, PipeModule

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

    def get_states(self, **kwargs):
        batch = GraphCoderBatch.from_dict(kwargs)

        hidden_states = kwargs["hidden_states"]
        if batch.has_docstring:
            kwargs["text_states"] = hidden_states[:, : batch.docstring_size]

        if batch.has_graph:
            if batch.has_docstring:
                start = batch.docstring_size + 1
            else:
                start = 0
            if batch.has_source:
                end = batch.source_size + 1
            else:
                end = 0
            kwargs["graph_states"] = hidden_states[:, start : -end - 1]

        if batch.has_source:
            kwargs["code_states"] = hidden_states[:, -batch.source_size - 1 : -1]

        return kwargs

    def has_source(self, **kwargs):
        return GraphCoderBatch.from_dict(kwargs).has_source

    def has_docstring(self, **kwargs):
        return GraphCoderBatch.from_dict(kwargs).has_docstring

    def has_graph(self, **kwargs):
        return GraphCoderBatch.from_dict(kwargs).has_graph

    def tgt_and_memory(self, **kwargs):
        kwargs["tgt"] = torch.cat(kwargs["tgt_"], dim=1)
        kwargs["memory"] = torch.cat(kwargs["x_"], dim=1)

        return kwargs

    def add_accums(self, **kwargs):
        kwargs["tgt_"] = []
        kwargs["x_"] = []
        kwargs["result_"] = {}

        return kwargs

    def calc_logits(self, **kwargs):
        batch = GraphCoderBatch.from_dict(kwargs)
        result = kwargs["result_"]

        lm_logits = []
        target_ids = []

        if batch.has_docstring:
            target_ids.append(batch.docstring[batch.docstring_attn_mask.bool()])
            lm_logits.append(result["docstring"][batch.docstring_attn_mask.bool()])
            # add eos token
            device = batch.docstring.device
            target_ids.append(torch.tensor([self.eos_token_id], device=device))
            lm_logits.append(
                torch.tensor([self.eos_token_id], device=device).repeat(
                    1, self.vocab_size
                )
            )
        if batch.has_graph:
            target_ids.extend(
                [
                    batch.node_data[batch.node_data_attn_mask.bool()],
                    batch.edge_data[batch.edge_data_attn_mask.bool()],
                ]
            )
            lm_logits.append(
                result["graph"][result["padded_node_mask"].bool()][
                    batch.node_data_attn_mask.bool()
                ]
            )
            lm_logits.append(
                result["graph"][result["padded_edge_mask"].bool()][
                    batch.edge_data_attn_mask.bool()
                ]
            )
            # add eos token
            device = batch.node_data.device
            target_ids.append(torch.tensor([self.eos_token_id], device=device))
            lm_logits.append(
                torch.tensor([self.eos_token_id], device=device).repeat(
                    1, self.vocab_size
                )
            )
        if batch.has_source:
            target_ids.append(batch.source[batch.source_attn_mask.bool()])
            lm_logits.append(result["source"][batch.source_attn_mask.bool()])
            # add eos token
            device = batch.source.device
            target_ids.append(torch.tensor([self.eos_token_id], device=device))
            lm_logits.append(
                torch.tensor([self.eos_token_id], device=device).repeat(
                    1, self.vocab_size
                )
            )

        target_ids_ = torch.cat(target_ids)
        lm_logits_ = torch.cat(lm_logits, dim=0)

        shift_logits = lm_logits_[:-1, :].contiguous()
        shift_labels = target_ids_[1:].contiguous().long()

        return self.criterion(shift_logits, shift_labels)

    def to_layers(self) -> Layers:
        layers = [
            self.add_accums,
        ]

        for layer in self.layers:
            layers.extend(layer.to_layers())

        layers.extend(
            [
                self.tgt_and_memory,
                *self.decoder.to_layers(),
                PassThroughLayer(
                    self.dense,
                    "hidden_states",
                    ["tgt"],
                    callback=lambda res, **kwargs: torch.tanh(res).contiguous(),
                ),
                self.get_states,
                ConditionalLayer(
                    PassThroughLayer(self.lm_head, "docstring", ["text_states"]),
                    self.has_docstring,
                ),
                ConditionalLayer(
                    PassThroughLayer(
                        self.lm_graph_head,
                        "graph",
                        ["graph_states"],
                        callback=lambda res, **kwargs: res.view(
                            res.size(0), -1, self.hidden_size
                        ),
                    ),
                    self.has_graph,
                ),
                ConditionalLayer(
                    PassThroughLayer(self.lm_head, "source", ["code_states"]),
                    self.has_source,
                ),
                ConditionalLayer(
                    PassThroughLayer(
                        self.lm_head,
                        "graph",
                        ["graph"],
                        callback=lambda res, **kwargs: res.view(
                            res.size(0),
                            kwargs["graph_states"].size(1),
                            -1,
                            self.vocab_size,
                        ),
                    ),
                    self.has_graph,
                ),
                self.calc_logits
            ]
        )

        return layers
