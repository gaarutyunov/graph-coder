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

TE = typing.TypeVar("TE", bound=nn.Module)


Kwargs = typing.Union[
    torch.Tensor, typing.List[torch.Tensor], typing.Dict[str, torch.Tensor]
]


class TextLayer(nn.Module, typing.Generic[TE]):
    def __init__(self, embedding: nn.Module, text_encoder: TE, eos_token_id: int):
        super().__init__()
        self.embedding = embedding
        self.text_encoder = text_encoder
        self.eos_token_id = eos_token_id

    def forward(
        self,
        **kwargs: Kwargs,
    ) -> typing.Dict[str, torch.Tensor]:
        batch = GraphCoderBatch.from_dict(kwargs)

        if not batch.has_docstring:
            return kwargs

        x, tgt = kwargs["x_"], kwargs["tgt_"]

        eos = torch.empty(
            (batch.batch_size, 1),
            device=batch.docstring.device,
            dtype=batch.docstring.dtype,
        ).fill_(self.eos_token_id)
        text = torch.cat([batch.docstring, eos], dim=1)

        emb = self.embedding(text)
        tgt.append(emb)

        text_encoded = self.text_encoder(emb)
        x.append(text_encoded)

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
    ) -> typing.Dict[str, torch.Tensor]:
        batch = GraphCoderBatch.from_dict(kwargs)

        if not batch.has_source:
            return kwargs

        x, tgt = kwargs["x_"], kwargs["tgt_"]

        eos = torch.empty(
            (batch.batch_size, 1),
            device=batch.source.device,
            dtype=batch.source.dtype,
        ).fill_(self.eos_token_id)
        text = torch.cat([eos, batch.source, torch.clone(eos)], dim=1)

        emb = self.embedding(text)
        tgt.append(emb)

        source_code_encoded = self.code_encoder(emb)
        x.append(source_code_encoded)

        return kwargs


class GraphLayer(nn.Module, typing.Generic[TE]):
    def __init__(self, graph_encoder: TE):
        super().__init__()
        self.graph_encoder = graph_encoder

    def forward(
        self,
        **kwargs: Kwargs,
    ) -> typing.Dict[str, torch.Tensor]:
        batch = GraphCoderBatch.from_dict(kwargs)

        if not batch.has_graph:
            return kwargs

        x, tgt, result = kwargs["x_"], kwargs["tgt_"], kwargs["result_"]

        x_ = self.graph_encoder(**kwargs)

        x.append(x_)

        (
            _,
            padded_feature,
            padding_mask,
            result["padded_node_mask"],
            result["padded_edge_mask"],
        ) = self.graph_encoder.graph_encoder.graph_feature.process_batch(  # type: ignore[union-attr]
            batch
        )  # type: ignore[operator]
        padded_feature = padded_feature.masked_fill(padding_mask[..., None], float("0"))
        tgt.append(padded_feature)

        return kwargs
