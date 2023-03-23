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
from typing import Dict, List, Tuple

import torch
from torch.functional import F
from transformers import PreTrainedTokenizerFast

from .algos import lap_eig
from .ast import AstExample

from .batch import GraphCoderBatch


def pad(
    batch: List[str],
    num: List[List[int]],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 64,
) -> List[Dict[str, torch.Tensor]]:
    """Pad a batch of strings restoring the original packs."""
    encoding = (
        tokenizer(
            batch,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
    ).data
    inputs_ids = []
    attention_mask = []

    start = 0

    for g in num:
        inputs_ids_ = []
        attention_mask_ = []
        for n in g:
            end = start + n
            inputs_ids_.append(encoding["input_ids"][start:end])
            attention_mask_.append(encoding["attention_mask"][start:end])
            start = end
        inputs_ids.append(inputs_ids_)
        attention_mask.append(attention_mask_)

    return [
        {
            "input_ids": torch.cat(inputs_ids),
            "attention_mask": torch.cat(attention_mask),
        }
        for inputs_ids, attention_mask in zip(inputs_ids, attention_mask)
    ]


def collate_ast(
    batch: List[AstExample],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 64,
    max_seq_length: int = 8192,
    dtype: torch.dtype = torch.float,
    use_dict: bool = True,
    num_samples: int = 1,
    lap_node_id_k: int = 8
) -> torch.Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
    """Collate a batch of examples into a batch of tensors."""
    if num_samples > 1:
        assert len(batch) == 1, "You can only repeat with batch_size = 1"
        batch = batch * num_samples
    idx = []
    edge_index = []
    edge_data = []
    node_data = []
    node_num = []
    edge_num = []
    sources = []
    docstrings = []
    lap_eigvecs = []

    for data in batch:
        sources.append(data.source)
        docstrings.append(data.docstring)

        graph_data = data.graph
        edge_index_ = (
            torch.tensor(graph_data.edge_index, dtype=torch.long).t().contiguous()
        )
        _, lap_eigvec = lap_eig(edge_index_, len(graph_data.x), dtype=dtype)
        lap_eigvecs.append(lap_eigvec)

        idx.append(graph_data.idx)
        edge_index.append(edge_index_)

        node_data.extend(graph_data.x)
        edge_data.extend(graph_data.edge_attr)

        node_num.append(len(graph_data.x))
        edge_num.append(len(graph_data.edge_attr))

    max_n = max(node_num)

    node_data_, edge_data_ = pad(
        batch=node_data + edge_data,
        num=[node_num, edge_num],
        max_length=max_length,
        tokenizer=tokenizer,
    )

    source_ = tokenizer(
        sources,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,
        max_length=max_seq_length,
    ).data

    docstring_ = tokenizer(
        docstrings,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,
        max_length=max_seq_length,
    ).data

    lap_eigvec_ = torch.cat(
        [F.pad(i, (0, max_n - i.size(1)), value=float("0")) for i in lap_eigvecs]
    )

    lap_dim = lap_eigvec_.size(-1)
    if lap_node_id_k > lap_dim:
        lap_eigvec_ = F.pad(
            lap_eigvec_,
            (0, lap_node_id_k - lap_dim),
            value=float("0"),
        )  # [sum(n_node), Dl]
    else:
        lap_eigvec_ = lap_eigvec_[:, : lap_node_id_k]  # [sum(n_node), Dl]

    res = GraphCoderBatch(
        idx=torch.tensor(idx, dtype=torch.long),
        edge_index=torch.cat(edge_index, dim=1),
        node_num=torch.tensor(node_num, dtype=torch.long),
        edge_num=torch.tensor(edge_num, dtype=torch.long),
        lap_eigvec=lap_eigvec_,
        source_={
            "input_ids": source_["input_ids"].type(torch.long),
            "attention_mask": source_["attention_mask"].type(torch.long),
        },
        docstring_={
            "input_ids": docstring_["input_ids"].type(torch.long),
            "attention_mask": docstring_["attention_mask"].type(torch.long),
        },
        node_data_=node_data_,
        edge_data_=edge_data_,
    )

    if use_dict:
        return res.to_dict()

    return res.to_tuple()
