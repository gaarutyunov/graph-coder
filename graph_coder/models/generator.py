#  Copyright 2023 German Arutyunov
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import torch
from torch import nn
from typing import Dict

from graph_coder.data import GraphCoderBatch


class GraphCoderGenerator(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        encoder: nn.Module,
        graph_encoder: nn.Module,
        decoder: nn.Module,
        hidden_size: int,
        vocab_size: int,
        eos_token_id: int = 0,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.eos_token_id = eos_token_id
        self.encoder = encoder
        self.graph_encoder = graph_encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_graph_head = nn.Linear(hidden_size, vocab_size * max_length, bias=False)

    def forward(self, batch: GraphCoderBatch) -> Dict[str, torch.Tensor]:
        x = []
        tgt = []

        if batch.has_docstring:
            emb = self.embedding(batch.docstring)
            docstring_encoded = self.encoder(emb)
            x.append(docstring_encoded)
            tgt.append(emb)

        if batch.has_graph:
            graph_encoded = self.graph_encoder(batch)
            device = graph_encoded.device
            if len(x) != 0:
                eos = torch.tensor(
                    [self.eos_token_id], device=device, dtype=torch.float
                ).repeat(graph_encoded.size(0), 1, graph_encoded.size(-1))
                x.append(eos)
                tgt.append(eos)
            x.append(graph_encoded)
            tgt.append(
                self.embedding(
                    torch.cat(
                        [
                            batch.node_data,
                            batch.edge_data,
                        ],
                        dim=1,
                    )
                ).sum(-2)
            )

        if batch.has_source:
            emb = self.embedding(batch.source)
            source_code_encoded = self.encoder(emb)
            device = source_code_encoded.device
            if len(x) != 0:
                eos = torch.tensor(
                    [self.eos_token_id], device=device, dtype=torch.float
                ).repeat(source_code_encoded.size(0), 1, source_code_encoded.size(-1))
                x.append(eos)
                tgt.append(eos)
            x.append(source_code_encoded)
            tgt.append(emb)

        x, tgt = torch.cat(x, dim=1), torch.cat(tgt, dim=1)

        out = self.decoder(tgt, x)
        hidden_states = torch.tanh(self.dense(out)).contiguous()

        result = {}

        if batch.has_docstring:
            result["docstring"] = self.lm_head(hidden_states[:, : batch.docstring_size])

        if batch.has_graph:
            if batch.has_docstring:
                start = batch.docstring_size + 1
            else:
                start = 0
            if batch.has_source:
                end = batch.source_size + 1
            else:
                end = 0
            result["graph"] = self.lm_graph_head(hidden_states[:, start:-end])

        if batch.has_source:
            result["source"] = self.lm_head(hidden_states[:, -batch.source_size :])

        return result
