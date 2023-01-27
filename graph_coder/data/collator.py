#  Copyright 2023 German Arutyunov
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.functional import F
from transformers import PreTrainedTokenizerFast
from typing import List, Dict

from .batch import GraphCoderBatch
from .ast import AstExample
from .algos import lap_eig
from catalyst.utils import get_device


def pad(
    batch: List[str],
    num: List[int],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 64,
    device: torch.device = get_device(),
) -> Dict[str, torch.Tensor]:
    """Pad a batch of strings restoring the original packs."""
    encoding = (
        tokenizer(
            batch,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        .to(device)
        .convert_to_tensors()
    )
    inputs_ids = []
    attention_mask = []

    start = 0

    for n in num:
        end = start + n
        inputs_ids.append(encoding["input_ids"][start:end])
        attention_mask.append(encoding["attention_mask"][start:end])
        start = end

    return {
        "input_ids": pad_sequence(
            inputs_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        ),
        "attention_mask": pad_sequence(
            attention_mask, batch_first=True, padding_value=False
        ),
    }


@torch.no_grad()
def collate_ast(
    batch: List[AstExample],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 64,
    max_seq_length: int = 8192,
    device: torch.device = get_device(),
) -> GraphCoderBatch:
    """Collate a batch of examples into a batch of tensors."""
    idx = []
    edge_index = []
    edge_data = []
    node_data = []
    node_num = []
    edge_num = []
    sources = []
    docstrings = []
    lap_eigvals = []
    lap_eigvecs = []

    for data in batch:
        sources.append(data.source)
        docstrings.append(data.docstring)

        graph_data = data.graph
        edge_index_ = (
            torch.tensor(graph_data.edge_index, dtype=torch.long, device=device)
            .t()
            .contiguous()
        )
        lap_eigval, lap_eigvec = lap_eig(edge_index_, len(graph_data.x))
        lap_eigval = lap_eigval[None, :].expand_as(lap_eigvec)
        lap_eigvals.append(lap_eigval)
        lap_eigvecs.append(lap_eigvec)

        idx.append(graph_data.idx)
        edge_index.append(edge_index_)

        node_data.extend(graph_data.x)
        edge_data.extend(graph_data.edge_attr)

        node_num.append(len(graph_data.x))
        edge_num.append(len(graph_data.edge_attr))

    max_n = max(node_num)

    pad_ = partial(pad, max_length=max_length, device=device)

    return GraphCoderBatch(
        idx=torch.tensor(idx, dtype=torch.long, device=device),
        edge_index=torch.cat(edge_index, dim=1),
        node_num=torch.tensor(node_num, dtype=torch.long, device=device),
        edge_num=torch.tensor(edge_num, dtype=torch.long, device=device),
        lap_eigval=torch.cat(
            [F.pad(i, (0, max_n - i.size(1)), value=float("0")) for i in lap_eigvals]
        ),
        lap_eigvec=torch.cat(
            [F.pad(i, (0, max_n - i.size(1)), value=float("0")) for i in lap_eigvecs]
        ),
        source_=tokenizer(
            sources,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=max_seq_length,
        )
        .to(device)
        .convert_to_tensors(),
        docstring_=tokenizer(
            docstrings,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=max_seq_length,
        )
        .to(device)
        .convert_to_tensors(),
        edge_data_=pad_(edge_data, edge_num, tokenizer),
        node_data_=pad_(node_data, node_num, tokenizer),
    )
