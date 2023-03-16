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
from typing import Any, Dict, Generic, List, Tuple, TypeVar, Union

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
        max_seq_length: int = 512,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.max_seq_length = max_seq_length
        self.decoder = decoder
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_graph_head = nn.Linear(
            hidden_size, hidden_size * max_length, bias=False
        )

    def forward(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        for layer in self.layers:
            kwargs = layer(**kwargs)

        result: Dict[str, Any] = {
            "padded_node_mask": kwargs["padded_node_mask"],
            "padded_edge_mask": kwargs["padded_edge_mask"],
        }

        out = self.decoder(kwargs["tgt"], kwargs["memory"])
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

    def generate(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        kwargs["source"], kwargs["source_attn_mask"] = self._predict_code_token(
            **kwargs
        )

        for i in range(self.max_seq_length - 1):
            new_token, attn_mask = self._predict_code_token(**kwargs)

            kwargs["source"] = torch.cat([kwargs["source"], new_token], dim=1)
            kwargs["source_attn_mask"] = torch.cat(
                [kwargs["source_attn_mask"], attn_mask], dim=1
            )

        return {
            "source": kwargs["source"],
            "docstring": kwargs["docstring"],
        }

    def _predict_code_token(
        self, **kwargs: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            kwargs = layer(**kwargs)

        out = self.decoder(kwargs["tgt"], kwargs["memory"])
        hidden_states = torch.tanh(self.dense(out)).contiguous()
        logits = self.lm_head(hidden_states[:, -1:])
        token = logits.argmax(dim=-1)[0]
        attn_mask = torch.ones_like(token, device=token.device)

        return token, attn_mask
