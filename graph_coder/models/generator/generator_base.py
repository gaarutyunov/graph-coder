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
from typing import Dict, Generic, List, Tuple, TypeVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers import top_k_top_p_filtering

from graph_coder.data import GraphCoderBatch

TL = TypeVar("TL", bound=nn.Module)


def append_token(
    batch: GraphCoderBatch, t: torch.Tensor, m: torch.Tensor
) -> Dict[str, torch.Tensor]:
    batch.source_ = {
        "input_ids": torch.cat([batch.source, t], dim=1),
        "attention_mask": torch.cat([batch.source_attn_mask, m], dim=1),
    }

    return batch.to_dict()


class GraphCoderGeneratorBase(nn.Module, Generic[TL]):
    """Graph-coder model for code generation"""

    def __init__(
        self,
        layers: List[TL],
        lm_layer: TL,
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
        self.lm_layer = lm_layer

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            kwargs = layer(**kwargs)

        out = self.decoder(kwargs["tgt"], kwargs["memory"])
        hidden_states = torch.tanh(self.dense(out)).contiguous()

        batch = GraphCoderBatch.from_dict(kwargs)
        args = (*batch.to_tuple(), hidden_states)

        return self.lm_layer(*args)

    def generate(
        self,
        num_samples: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        batch = GraphCoderBatch.from_dict(kwargs)

        new_token, attn_mask = self._predict_code_token(
            num_samples, temperature, top_k, top_p, repetition_penalty, **kwargs
        )

        kwargs = append_token(batch, new_token, attn_mask)

        for i in range(self.max_seq_length - 1):
            new_token, attn_mask = self._predict_code_token(
                num_samples, temperature, top_k, top_p, repetition_penalty, **kwargs
            )

            kwargs = append_token(batch, new_token, attn_mask)

        return kwargs["source"]

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
        logits = self.lm_layer.code(hidden_states[0, -1, :]) / (  # type: ignore[operator]
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
