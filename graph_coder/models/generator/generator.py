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
from torch import nn

from .generator_base import GraphCoderGeneratorBase
from .layers import TextLayer, GraphLayer, CodeLayer


class GraphCoderGenerator(GraphCoderGeneratorBase[nn.Module]):
    """Graph-coder model for code generation"""

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
        super().__init__(
            layers=[
                TextLayer(embedding, text_encoder, eos_token_id),
                GraphLayer(graph_encoder),
                CodeLayer(embedding, code_encoder, eos_token_id),
            ],
            decoder=decoder,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            max_length=max_length,
        )
        self.embedding = embedding
        self.text_encoder = text_encoder
        self.code_encoder = code_encoder
        self.graph_encoder = graph_encoder
