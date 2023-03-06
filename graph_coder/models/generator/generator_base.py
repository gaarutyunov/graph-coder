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
from typing import Dict, Generic, List, TypeVar, Union

import torch
from torch import nn

from graph_coder.data import GraphCoderBatch

TL = TypeVar("TL", bound=nn.Module)


class GraphCoderGeneratorBase(nn.Module, Generic[TL]):
    """Graph-coder model for code generation"""

    def __init__(
        self,
        layers: List[TL],
        decoder: TL,
        hidden_size: int,
        vocab_size: int,
        eos_token_id: int = 0,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.decoder = decoder
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_graph_head = nn.Linear(
            hidden_size, hidden_size * max_length, bias=False
        )

    def forward(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        new_kwargs: Dict[
            str, Union[List[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]
        ] = {
            **kwargs,
            "x_": [],
            "tgt_": [],
            "result_": {},
        }

        for layer in self.layers:
            new_kwargs = layer(**new_kwargs)

        x, tgt, result = (
            torch.cat(new_kwargs["x_"], dim=1),
            torch.cat(new_kwargs["tgt_"], dim=1),
            new_kwargs["result_"],
        )

        out = self.decoder(tgt, x)
        hidden_states = torch.tanh(self.dense(out)).contiguous()

        batch = GraphCoderBatch.from_dict(kwargs)

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
            graph_states = hidden_states[:, start : -end - 1]
            graph = self.lm_graph_head(graph_states)
            graph = graph.view(graph.size(0), -1, self.hidden_size)
            graph = self.lm_head(graph)
            result["graph"] = graph.view(
                graph.size(0), graph_states.size(1), -1, self.vocab_size
            )

        if batch.has_source:
            result["source"] = self.lm_head(
                hidden_states[:, -batch.source_size - 1 : -1]
            )

        return result
