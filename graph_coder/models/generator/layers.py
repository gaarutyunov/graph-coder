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
from torch import nn

from graph_coder.data import GraphCoderBatch
from graph_coder.pipe import Kwargs

TE = typing.TypeVar("TE", bound=nn.Module)


class TextLayer(nn.Module, typing.Generic[TE]):
    def __init__(self, embedding: nn.Module, text_encoder: TE, eos_token_id: int):
        super().__init__()
        self.embedding = embedding
        self.text_encoder = text_encoder
        self.eos_token_id = eos_token_id

    def forward(
        self,
        **kwargs: Kwargs,
    ) -> typing.Dict[str, Kwargs]:
        batch = GraphCoderBatch.from_dict(kwargs)

        if not batch.has_docstring:
            return kwargs

        eos = torch.empty(
            (batch.batch_size, 1),
            device=batch.docstring.device,
            dtype=batch.docstring.dtype,
        ).fill_(self.eos_token_id)
        text = torch.cat([batch.docstring, eos], dim=1)

        emb = self.embedding(text)
        if "tgt" in kwargs:
            kwargs["tgt"] = torch.cat([kwargs["tgt"], emb], dim=1)
        else:
            kwargs["tgt"] = emb

        text_encoded = self.text_encoder(emb)
        if "memory" in kwargs:
            kwargs["memory"] = torch.cat([kwargs["memory"], text_encoded], dim=1)
        else:
            kwargs["memory"] = text_encoded

        return kwargs


class CodeLayer(nn.Module, typing.Generic[TE]):
    def __init__(self, embedding: nn.Module, code_encoder: TE, eos_token_id: int):
        super().__init__()
        self.embedding = embedding
        self.code_encoder = code_encoder
        self.eos_token_id = eos_token_id

    def forward(
        self,
        **kwargs: Kwargs,
    ) -> typing.Dict[str, Kwargs]:
        batch = GraphCoderBatch.from_dict(kwargs)

        if not batch.has_source:
            return kwargs

        eos = torch.empty(
            (batch.batch_size, 1),
            device=batch.source.device,
            dtype=batch.source.dtype,
        ).fill_(self.eos_token_id)
        text = torch.cat([eos, batch.source, torch.clone(eos)], dim=1)

        emb = self.embedding(text)
        if "tgt" in kwargs:
            kwargs["tgt"] = torch.cat([kwargs["tgt"], emb], dim=1)
        else:
            kwargs["tgt"] = emb

        source_code_encoded = self.code_encoder(emb)
        if "memory" in kwargs:
            kwargs["memory"] = torch.cat([kwargs["memory"], source_code_encoded], dim=1)
        else:
            kwargs["memory"] = source_code_encoded

        return kwargs


class GraphLayer(nn.Module, typing.Generic[TE]):
    def __init__(self, graph_encoder: TE):
        super().__init__()
        self.graph_encoder = graph_encoder

    def forward(
        self,
        **kwargs: Kwargs,
    ) -> typing.Dict[str, Kwargs]:
        batch = GraphCoderBatch.from_dict(kwargs)

        if not batch.has_graph:
            return kwargs

        x_ = self.graph_encoder(
            kwargs["edge_index"],
            kwargs["edge_data"],
            kwargs["node_data"],
            kwargs["node_num"],
            kwargs["edge_num"],
            kwargs["lap_eigvec"],
        )

        if "memory" in kwargs:
            kwargs["memory"] = torch.cat([kwargs["memory"], x_], dim=1)
        else:
            kwargs["memory"] = x_

        (
            _,
            padded_feature,
            padding_mask,
            kwargs["padded_node_mask"],
            kwargs["padded_edge_mask"],
        ) = self.graph_encoder.graph_encoder.graph_feature.process_batch(  # type: ignore[union-attr]
            kwargs["node_data"],
            kwargs["edge_data"],
            kwargs["edge_index"],
            kwargs["node_num"],
            kwargs["edge_num"],
        )  # type: ignore[operator]
        padded_feature = padded_feature.masked_fill(padding_mask[..., None], float("0"))
        if "tgt" in kwargs:
            kwargs["tgt"] = torch.cat([kwargs["tgt"], padded_feature], dim=1)
        else:
            kwargs["tgt"] = padded_feature

        return kwargs
