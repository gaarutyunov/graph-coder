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
from torch.nn import Identity

from graph_coder.data import GraphCoderBatch
from graph_coder.pipe import ConditionalLayer, Layers, PassThroughLayer, PipeModule

from .layers import CodeLayer, GraphLayer, TextLayer


def append_kwarg(key: str, other: str):
    def inner(**kwargs):
        kwargs[key].append(kwargs[other])

        return kwargs

    return inner


class TextLayerPipe(TextLayer[PipeModule], PipeModule):
    def add_eos(self, **kwargs):
        batch = GraphCoderBatch.from_dict(kwargs)

        eos = torch.empty(
            (batch.batch_size, 1),
            device=batch.docstring.device,
            dtype=batch.docstring.dtype,
        ).fill_(self.eos_token_id)

        kwargs["text"] = torch.cat([batch.docstring, eos], dim=1)

        return kwargs

    def condition(self, **kwargs):
        return GraphCoderBatch.from_dict(kwargs).has_docstring

    def to_layers(self) -> Layers:
        return [
            ConditionalLayer(self.add_eos, self.condition),
            ConditionalLayer(
                PassThroughLayer(self.embedding, "emb", ["text"]), self.condition
            ),
            ConditionalLayer(append_kwarg("tgt_", "emb"), self.condition),
            ConditionalLayer(
                PassThroughLayer(Identity(), "x", ["emb"]), self.condition
            ),
            *[
                ConditionalLayer(layer, self.condition)
                for layer in self.text_encoder.to_layers()
            ],
            ConditionalLayer(append_kwarg("x_", "x"), self.condition),
        ]


class GraphLayerPipe(GraphLayer[PipeModule], PipeModule):
    def condition(self, **kwargs):
        return GraphCoderBatch.from_dict(kwargs).has_graph

    def add_target(self, **kwargs):
        batch = GraphCoderBatch.from_dict(kwargs)
        (
            _,
            padded_feature,
            padding_mask,
            kwargs["result_"]["padded_node_mask"],
            kwargs["result_"]["padded_edge_mask"],
        ) = self.graph_encoder.graph_encoder.graph_feature.process_batch(  # type: ignore[union-attr]
            batch
        )  # type: ignore[operator]
        padded_feature = padded_feature.masked_fill(padding_mask[..., None], float("0"))
        kwargs["tgt_"].append(padded_feature)

        return kwargs

    def to_layers(self) -> Layers:
        return [
            *[
                ConditionalLayer(layer, self.condition)
                for layer in self.graph_encoder.to_layers()
            ],
            ConditionalLayer(append_kwarg("x_", "x"), self.condition),
            ConditionalLayer(self.add_target, self.condition),
        ]


class CodeLayerPipe(CodeLayer[PipeModule], PipeModule):
    def add_eos(self, **kwargs):
        batch = GraphCoderBatch.from_dict(kwargs)

        eos = torch.empty(
            (batch.batch_size, 1),
            device=batch.source.device,
            dtype=batch.source.dtype,
        ).fill_(self.eos_token_id)

        kwargs["code"] = torch.cat([eos, batch.source, torch.clone(eos)], dim=1)

        return kwargs

    def condition(self, **kwargs):
        return GraphCoderBatch.from_dict(kwargs).has_source

    def to_layers(self) -> Layers:
        return [
            ConditionalLayer(self.add_eos, self.condition),
            ConditionalLayer(
                PassThroughLayer(self.embedding, "emb", ["code"]), self.condition
            ),
            ConditionalLayer(append_kwarg("tgt_", "emb"), self.condition),
            ConditionalLayer(
                PassThroughLayer(Identity(), "x", ["emb"]), self.condition
            ),
            *[
                ConditionalLayer(layer, self.condition)
                for layer in self.code_encoder.to_layers()
            ],
            ConditionalLayer(append_kwarg("x_", "x"), self.condition),
        ]
