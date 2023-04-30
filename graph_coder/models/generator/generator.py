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
from typing import List, Optional

from torch import nn

from .generator_base import GraphCoderGeneratorBase
from .layers import CodeLayer, GraphLayer, LmLayer, TextLayer


class GraphCoderGenerator(GraphCoderGeneratorBase[nn.Module]):
    """Graph-coder model for code generation"""

    def __init__(
        self,
        embedding: nn.Module,
        decoder: nn.Module,
        hidden_size: int,
        vocab_size: int,
        eos_token_id: int = 0,
        max_length: int = 64,
        max_seq_length: int = 512,
        shift: bool = True,
        layers: Optional[List[nn.Module]] = None,
        text_encoder: Optional[nn.Module] = None,
        code_encoder: Optional[nn.Module] = None,
        graph_encoder: Optional[nn.Module] = None,
    ) -> None:
        if layers is None:
            assert (
                text_encoder is not None
            ), "text_encoder must be provided if layers is None"
            assert (
                code_encoder is not None
            ), "code_encoder must be provided if layers is None"
            assert (
                graph_encoder is not None
            ), "graph_encoder must be provided if layers is None"

            layers = [
                TextLayer(embedding, text_encoder, eos_token_id),
                GraphLayer(graph_encoder),
                CodeLayer(embedding, code_encoder, eos_token_id),
            ]

        lm_kwargs = {
            "has_docstring": False,
            "has_graph": False,
            "has_source": False,
        }

        for layer in layers:
            if isinstance(layer, TextLayer):
                lm_kwargs["has_docstring"] = True
            elif isinstance(layer, GraphLayer):
                lm_kwargs["has_graph"] = True
            elif isinstance(layer, CodeLayer):
                lm_kwargs["has_source"] = True
            else:
                raise ValueError(f"Unknown layer type: {layer.__class__.__name__}")

        if layers is None:
            layers = [
                TextLayer(embedding, text_encoder, eos_token_id),
                GraphLayer(graph_encoder),
                CodeLayer(embedding, code_encoder, eos_token_id),
            ]

        super().__init__(
            layers=layers,
            decoder=decoder,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            max_length=max_length,
            max_seq_length=max_seq_length,
            lm_layer=LmLayer(
                vocab_size, max_length, hidden_size, shift=shift, **lm_kwargs
            ),
        )
        self.embedding = embedding
        self.text_encoder = text_encoder
        self.code_encoder = code_encoder
        self.graph_encoder = graph_encoder
