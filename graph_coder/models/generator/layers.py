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
import pickle
import typing

import torch
from torch import nn

from graph_coder.data import GraphCoderBatch
from graph_coder.pipe import Kwarg

TE = typing.TypeVar("TE", bound=nn.Module)


class TextLayer(nn.Module, typing.Generic[TE]):
    def __init__(self, embedding: nn.Module, text_encoder: TE, eos_token_id: int):
        super().__init__()
        self.embedding = embedding
        self.text_encoder = text_encoder
        self.eos_token_id = eos_token_id

    def forward(
        self,
        **kwargs: Kwarg,
    ) -> typing.Dict[str, Kwarg]:
        batch = GraphCoderBatch.from_dict(kwargs)

        if not batch.has_docstring:
            return kwargs

        emb = self.embedding(kwargs["docstring"])
        if "tgt" in kwargs:
            kwargs["tgt"] = torch.cat([kwargs["tgt"], emb], dim=1)
        else:
            kwargs["tgt"] = emb

        text_encoded = self.text_encoder(emb)
        if "memory" in kwargs:
            kwargs["memory"] = torch.cat([kwargs["memory"], text_encoded], dim=1)
        else:
            kwargs["memory"] = text_encoded

        return kwargs


class CodeLayer(nn.Module, typing.Generic[TE]):
    def __init__(self, embedding: nn.Module, code_encoder: TE, eos_token_id: int):
        super().__init__()
        self.embedding = embedding
        self.code_encoder = code_encoder
        self.eos_token_id = eos_token_id

    def forward(
        self,
        **kwargs: Kwarg,
    ) -> typing.Dict[str, Kwarg]:
        batch = GraphCoderBatch.from_dict(kwargs)

        if not batch.has_source:
            return kwargs

        emb = self.embedding(kwargs["source"])
        if "tgt" in kwargs:
            kwargs["tgt"] = torch.cat([kwargs["tgt"], emb], dim=1)
        else:
            kwargs["tgt"] = emb

        source_code_encoded = self.code_encoder(emb)
        if "memory" in kwargs:
            kwargs["memory"] = torch.cat([kwargs["memory"], source_code_encoded], dim=1)
        else:
            kwargs["memory"] = source_code_encoded

        return kwargs


class GraphLayer(nn.Module, typing.Generic[TE]):
    def __init__(self, graph_encoder: TE):
        super().__init__()
        self.graph_encoder = graph_encoder

    def forward(
        self,
        **kwargs: Kwarg,
    ) -> typing.Dict[str, Kwarg]:
        batch = GraphCoderBatch.from_dict(kwargs)

        if not batch.has_graph:
            return kwargs

        x_ = self.graph_encoder(
            batch.edge_index,
            batch.edge_data,
            batch.node_data,
            batch.node_num,
            batch.edge_num,
            batch.padded_index,
            batch.padding_mask,
            batch.padded_node_mask,
            batch.padded_edge_mask,
        )

        if "memory" in kwargs:
            kwargs["memory"] = torch.cat([kwargs["memory"], x_], dim=1)
        else:
            kwargs["memory"] = x_

        padded_feature = self.graph_encoder.graph_encoder.graph_feature.process_batch(  # type: ignore[union-attr]
            batch.node_data,
            batch.edge_data,
            batch.edge_index,
            batch.node_num,
            batch.edge_num,
            batch.padded_node_mask,
            batch.padded_edge_mask,
        )  # type: ignore[operator]
        padded_feature = padded_feature.masked_fill(
            batch.padding_mask.bool()[..., None], float("0")
        )
        if "tgt" in kwargs:
            kwargs["tgt"] = torch.cat([kwargs["tgt"], padded_feature], dim=1)
        else:
            kwargs["tgt"] = padded_feature

        return kwargs


class LmLayer(nn.Module):
    """Layer that calculates logits"""

    def __init__(
        self, vocab_size: int, max_length: int, hidden_size: int, shift: bool = True
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.shift = shift
        self.text = nn.Linear(hidden_size, vocab_size, bias=False)
        self.graph_inner = nn.Linear(hidden_size, hidden_size * max_length, bias=False)
        self.graph_outer = nn.Linear(hidden_size, vocab_size, bias=False)
        self.code = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, *args: torch.Tensor):
        batch = GraphCoderBatch.from_tuple(args)
        hidden_states = args[-1]
        lm_logits = []
        docstring, graph, source = None, None, None

        if batch.has_docstring:
            docstring = self.text(hidden_states[:, : batch.docstring_size])
            lm_logits.append(docstring[batch.docstring_attn_mask.bool()])

        if batch.has_graph:
            if batch.has_docstring:
                start = batch.docstring_size
            else:
                start = 0
            if batch.has_source:
                end = batch.source_size
            else:
                end = 1

            graph_states = hidden_states[:, start:-end]
            graph = self.graph_inner(graph_states)
            graph = graph.view(graph.size(0), -1, self.hidden_size)
            graph = self.graph_outer(graph)
            graph = graph.view(graph.size(0), -1, self.max_length, self.vocab_size)
            try:
                lm_logits.extend(
                    [
                        graph[batch.padded_node_mask.bool()][
                            batch.node_data_attn_mask.bool()
                        ],
                        graph[batch.padded_edge_mask.bool()][
                            batch.edge_data_attn_mask.bool()
                        ],
                    ]
                )
            except Exception as e:
                fmt = f"""\
                idx: {batch.idx.tolist()}
                hidden_states.shape: {hidden_states.shape}
                docstring_states.shape: {docstring.shape if docstring is not None else 0}
                graph_states.shape: {graph.shape}
                padded_node_mask.shape: {batch.padded_node_mask.shape}
                padded_edge_mask.shape: {batch.padded_edge_mask.shape}
                node_attn.shape: {batch.node_data_attn_mask.shape}
                edge_attn.shape: {batch.edge_data_attn_mask.shape}
                node_data.shape: {batch.node_data.shape}
                edge_data.shape: {batch.edge_data.shape}
                batch.source_size: {batch.source_size}"""
                with open("batch.pkl", mode="wb") as f:
                    pickle.dump(batch, f)
                raise Exception(fmt) from e

        if batch.has_source:
            source = self.code(hidden_states[:, -batch.source_size - 1 : -1])
            lm_logits.append(source[batch.source_attn_mask.bool()])

        logits = torch.cat(lm_logits, dim=0)

        if self.shift:
            logits = logits[:-1, :].contiguous()

        return logits
