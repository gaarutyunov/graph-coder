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
from typing import Any, Dict, Generic, List, Tuple, TypeVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers import top_k_top_p_filtering

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
        self.lm_text = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_code = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_graph = nn.Linear(hidden_size, hidden_size * max_length, bias=False)
        self.lm_graph_vocab = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        for layer in self.layers:
            kwargs = layer(**kwargs)

        result: Dict[str, Any] = {}

        out = self.decoder(kwargs["tgt"], kwargs["memory"])
        hidden_states = torch.tanh(self.dense(out)).contiguous()

        batch = GraphCoderBatch.from_dict(kwargs)

        if batch.has_docstring:
            result["docstring"] = self.lm_text(hidden_states[:, : batch.docstring_size])

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
            graph = self.lm_graph(graph_states)
            graph = graph.view(graph.size(0), -1, self.hidden_size)
            graph = self.lm_graph_vocab(graph)
            result["graph"] = graph.view(
                graph.size(0), -1, self.max_length, self.vocab_size
            )

        if batch.has_source:
            result["source"] = self.lm_code(
                hidden_states[:, -batch.source_size - 1 : -1]
            )

        return result

    def generate(
        self,
        num_samples: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch = GraphCoderBatch.from_dict(kwargs)
        new_token, attn_mask = self._predict_code_token(
            num_samples, temperature, top_k, top_p, repetition_penalty, **kwargs
        )

        def append_token(t, m):
            kwargs["source"] = torch.cat([kwargs["source"], t], dim=1)
            kwargs["source_attn_mask"] = torch.cat(
                [kwargs["source_attn_mask"], m], dim=1
            )

        if not batch.has_source:
            kwargs["source"], kwargs["source_attn_mask"] = new_token, attn_mask
        else:
            append_token(new_token, attn_mask)

        for i in range(self.max_seq_length - 1):
            new_token, attn_mask = self._predict_code_token(
                num_samples, temperature, top_k, top_p, repetition_penalty, **kwargs
            )

            append_token(new_token, attn_mask)

        return {
            "source": kwargs["source"],
            "docstring": kwargs["docstring"],
        }

    def _predict_code_token(
        self,
        num_samples: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        **kwargs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            kwargs = layer(**kwargs)

        out = self.decoder(kwargs["tgt"], kwargs["memory"])
        hidden_states = torch.tanh(self.dense(out)).contiguous()
        logits = self.lm_code(hidden_states[0, -1, :]) / (
            temperature if temperature > 0 else 1.0
        )

        # reptition penalty from CTRL (https://arxiv.org/abs/1909.05858)
        if "source" in kwargs and repetition_penalty != 1.0:
            idx = set(kwargs["source"].view(-1).tolist())
            logits[list(idx)] /= repetition_penalty

        filtered_logits = top_k_top_p_filtering(
            logits.unsqueeze(0), top_k=top_k, top_p=top_p
        )
        if temperature == 0:
            next_token = torch.topk(filtered_logits, num_samples)[1]
        else:
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=num_samples,
                replacement=True,
            )

        next_token = next_token[0].unsqueeze(1)

        attn_mask = torch.ones_like(next_token, device=next_token.device)

        return next_token, attn_mask
