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
import abc
from typing import TypeVar, Generic

from torch import nn

from graph_coder.data import GraphCoderBatch

T = TypeVar("T")


class GraphCoderBase(nn.Module, Generic[T], abc.ABC):
    """Base class for graph-coder models"""

    def __init__(
        self,
        embedding: nn.Module,
        encoder: nn.Module,
        graph_encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.graph_encoder = graph_encoder
        self.decoder = decoder

    @abc.abstractmethod
    def forward(self, batch: GraphCoderBatch) -> T:
        # TODO: replace custom type with tensors for deepspeed offload compatibility
        pass
