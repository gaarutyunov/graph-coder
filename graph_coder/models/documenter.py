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
from typing import Dict

import torch
from torch import nn

from .base import GraphCoderBase
from graph_coder.data import GraphCoderBatch


class GraphCoderDocumenter(GraphCoderBase):
    """Graph-coder model for documentation generation"""

    def __init__(
        self,
        embedding: nn.Module,
        text_encoder: nn.Module,
        code_encoder: nn.Module,
        graph_encoder: nn.Module,
        decoder: nn.Module,
        hidden_size: int,
        vocab_size: int,
        eos_token_id: int = 0,
        max_length: int = 64,
    ) -> None:
        super().__init__(embedding, text_encoder, code_encoder, graph_encoder, decoder)
        self.hidden_size = hidden_size
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_graph_head = nn.Linear(
            hidden_size, hidden_size * max_length, bias=False
        )

    def forward(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:  # TODO: fix eos like generator
        batch = GraphCoderBatch.from_dict(kwargs)

        x = []
        tgt = []
        result = {}

        if batch.has_source:
            # add eos token
            eos = torch.empty(
                (batch.batch_size, 1),
                device=batch.source.device,
                dtype=batch.source.dtype,
            ).fill_(self.eos_token_id)
            text = torch.cat([batch.source, eos], dim=1)

            emb = self.embedding(text)
            source_encoded = self.code_encoder(emb)
            x.append(source_encoded)
            tgt.append(emb)

        if batch.has_graph:
            graph_encoded = self.graph_encoder(**kwargs)
            x.append(graph_encoded)
            (
                _,
                padded_feature,
                padding_mask,
                result["padded_node_mask"],
                result["padded_edge_mask"],
            ) = self.graph_encoder.graph_encoder.graph_feature.process_batch(  # type: ignore[union-attr]
                batch
            )  # type: ignore[operator]
            padded_feature = padded_feature.masked_fill(
                padding_mask[..., None], float("0")
            )
            tgt.append(padded_feature)

        if batch.has_docstring:
            # add eos token
            eos = torch.empty(
                (batch.batch_size, 1),
                device=batch.docstring.device,
                dtype=batch.docstring.dtype,
            ).fill_(self.eos_token_id)
            text = torch.cat([eos, batch.docstring, torch.clone(eos)], dim=1)

            emb = self.embedding(text)
            docstring_encoded = self.text_encoder(emb)
            x.append(docstring_encoded)
            tgt.append(emb)
        else:  # we need to add eos anyway at the end of sequence
            # add eos token
            eos = torch.empty(
                (batch.batch_size, 1),
                device=batch.source.device,
                dtype=batch.source.dtype,
            ).fill_(self.eos_token_id)

            emb = self.embedding(eos)
            docstring_encoded = self.text_encoder(emb)
            x.append(docstring_encoded)
            tgt.append(emb)

        x_, tgt_ = torch.cat(x, dim=1), torch.cat(tgt, dim=1)

        out = self.decoder(tgt_, x_)
        hidden_states = torch.tanh(self.dense(out)).contiguous()

        if batch.has_source:
            result["source"] = self.lm_head(hidden_states[:, : batch.source_size])

        if batch.has_graph:
            if batch.has_source:
                start = batch.source_size + 1
            else:
                start = 0
            if batch.has_docstring:
                end = batch.docstring_size + 1
            else:
                end = 0
            graph_states = hidden_states[:, start : -end - 1]
            graph = self.lm_graph_head(graph_states)
            graph = graph.view(graph.size(0), -1, self.hidden_size)
            graph = self.lm_head(graph)
            result["graph"] = graph.view(
                graph.size(0), graph_states.size(1), -1, self.vocab_size
            )

        if batch.has_docstring:
            result["docstring"] = self.lm_head(
                hidden_states[:, -batch.docstring_size - 1 : -1]
            )

        return result
