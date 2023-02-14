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
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, embedding: nn.Module, ff: nn.Module) -> None:
        super().__init__()
        self.embedding = embedding
        self.ff = ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.Tensor [sum(num_tokens), D]
        :returns: torch.Tensor [sum(num_tokens), T]
        """
        x = self.embedding(x)
        x = self.ff(x.transpose(-1, -2))
        return x
