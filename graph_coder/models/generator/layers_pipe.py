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
import typing

import torch
from torch.nn import Identity

from graph_coder.data import GraphCoderBatch, get_arg_idx
from graph_coder.pipe import (
    ConditionalLayer,
    Layers,
    PassThroughLayer,
    PipeModule,
    RemoveArgsLayer,
)

from .layers import CodeLayer, GraphLayer, TextLayer


def cat_arg(key: int, other: typing.Union[int, torch.Tensor]):
    def inner(*args):
        largs = list(args)
        obj = other if isinstance(other, torch.Tensor) else largs[other]
        if len(largs) <= key:
            largs.append(obj)
        else:
            largs[key] = torch.cat([largs[key], obj], dim=1)

        return tuple(largs)

    return inner


class TextLayerPipe(TextLayer[PipeModule], PipeModule):
    def condition(self, *args):
        return GraphCoderBatch.from_tuple(args).has_docstring

    def to_layers(self) -> Layers:
        return [
            ConditionalLayer(
                PassThroughLayer(
                    self.embedding, get_arg_idx("docstring")
                ),
                self.condition,
            ),
            # args: *batch_args, tgt
            ConditionalLayer(
                PassThroughLayer(Identity(), -1),
                self.condition,
            ),
            # args: *batch_args, tgt, x
            *[
                ConditionalLayer(layer, self.condition)
                for layer in self.text_encoder.to_layers()
            ],
            # args: *batch_args, tgt, x (memory)
        ]


class GraphLayerPipe(GraphLayer[PipeModule], PipeModule):
    def condition(self, *args):
        return GraphCoderBatch.from_tuple(args).has_graph

    def condition_only_graph(self, *args):
        batch = GraphCoderBatch.from_tuple(args)
        return batch.has_graph and not batch.has_docstring

    def condition_docstring(self, *args):
        batch = GraphCoderBatch.from_tuple(args)
        return batch.has_graph and batch.has_docstring

    def cat_target_and_memory(self, *args):
        batch = GraphCoderBatch.from_tuple(args)
        padded_feature = self.graph_encoder.graph_encoder.graph_feature.process_batch(  # type: ignore[union-attr]
            batch.padded_feature,
        )  # type: ignore[operator]
        padded_feature = padded_feature.masked_fill(
            batch.padding_mask.bool()[..., None], float("0")
        )

        memory = torch.cat([args[-2], args[-1]], dim=1)
        tgt = torch.cat([args[-3], padded_feature], dim=1)

        return *batch.to_tuple(), tgt, memory

    def add_target_and_memory(self, *args):
        batch = GraphCoderBatch.from_tuple(args)
        padded_feature = self.graph_encoder.graph_encoder.graph_feature.process_batch(  # type: ignore[union-attr]
            batch.padded_feature
        )  # type: ignore[operator]
        padded_feature = padded_feature.masked_fill(
            batch.padding_mask.bool()[..., None], float("0")
        )

        return *batch.to_tuple(), padded_feature, args[-1]

    def to_layers(self) -> Layers:
        return [
            *[
                ConditionalLayer(layer, self.condition)
                for layer in self.graph_encoder.to_layers()
            ],
            # args: *batch_args, tgt?, memory?, x
            ConditionalLayer(self.add_target_and_memory, self.condition_only_graph),
            # args: *batch_args, tgt, memory
            ConditionalLayer(self.cat_target_and_memory, self.condition_docstring),
            # args: *batch_args, tgt, memory
        ]


class CodeLayerPipe(CodeLayer[PipeModule], PipeModule):
    def condition(self, *args):
        return GraphCoderBatch.from_tuple(args).has_source

    def to_layers(self) -> Layers:
        return [
            ConditionalLayer(
                PassThroughLayer(self.embedding, get_arg_idx("source")),
                self.condition,
            ),
            # args: *batch_args, *, tgt, memory, emb
            ConditionalLayer(cat_arg(-3, -1), self.condition),
            # args: *batch_args, *, tgt, memory, emb (x)
            *[
                ConditionalLayer(layer, self.condition)
                for layer in self.code_encoder.to_layers()
            ],
            # args: *batch_args, *, tgt, memory, x
            ConditionalLayer(cat_arg(-2, -1), self.condition),
            RemoveArgsLayer(-1),
            # args: *batch_args, *, tgt, memory
        ]
