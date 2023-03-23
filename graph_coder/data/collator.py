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


def get_index_and_mask(
    edge_index: torch.Tensor,
    node_num: torch.Tensor,
    edge_num: torch.Tensor,
):
    seq_len = [n + e for n, e in zip(node_num, edge_num)]
    b = len(seq_len)
    max_len = max(seq_len)
    max_n = max(node_num)
    device = edge_index.device

    token_pos = torch.arange(max_len, device=device)[None, :].expand(
        b, max_len
    )  # [B, T]

    seq_len_ = torch.tensor(seq_len, device=device, dtype=torch.long)[:, None]  # [B, 1]
    node_num = node_num[:, None]  # [B, 1]
    edge_num = edge_num[:, None]  # [B, 1]

    node_index = torch.arange(max_n, device=device, dtype=torch.long)[None, :].expand(
        b, max_n
    )  # [B, max_n]
    node_index = node_index[None, node_index < node_num].repeat(
        2, 1
    )  # [2, sum(node_num)]

    padded_node_mask = torch.less(token_pos, node_num)
    padded_edge_mask = torch.logical_and(
        torch.greater_equal(token_pos, node_num),
        torch.less(token_pos, node_num + edge_num),
    )

    padded_index = torch.zeros(
        b, max_len, 2, device=device, dtype=torch.long
    )  # [B, T, 2]
    padded_index[padded_node_mask, :] = node_index.t()
    padded_index[padded_edge_mask, :] = edge_index.t()

    padding_mask = torch.greater_equal(token_pos, seq_len_)  # [B, T]

    return (
        padded_index,
        padding_mask,
        padded_node_mask,
        padded_edge_mask,
    )


def get_index_embed(node_id, node_mask, padded_index):
    """
    :param node_id: Tensor([sum(node_num), D])
    :param node_mask: BoolTensor([B, max_n])
    :param padded_index: LongTensor([B, T, 2])
    :return: Tensor([B, T, 2D])
    """
    b, max_n = node_mask.size()
    max_len = padded_index.size(1)
    d = node_id.size(-1)

    padded_node_id = torch.zeros(
        b, max_n, d, device=node_id.device, dtype=node_id.dtype
    )  # [B, max_n, D]
    padded_node_id[node_mask] = node_id

    padded_node_id = padded_node_id[:, :, None, :].expand(b, max_n, 2, d)
    padded_index = padded_index[..., None].expand(b, max_len, 2, d)
    index_embed = padded_node_id.gather(1, padded_index)  # [B, T, 2, D]
    index_embed = index_embed.view(b, max_len, 2 * d)
    return index_embed


def get_node_mask(node_num):
    b = len(node_num)
    max_n = max(node_num)
    node_index = torch.arange(max_n, dtype=torch.long)[None, :].expand(
        b, max_n
    )  # [B, max_n]
    node_num = node_num[:, None]  # [B, 1]
    node_mask = torch.less(node_index, node_num)  # [B, max_n]
    return node_mask


def get_random_sign_flip(eigvec, node_mask):
    b, max_n = node_mask.size()
    d = eigvec.size(1)

    sign_flip = torch.rand(b, d, device=eigvec.device, dtype=eigvec.dtype)
    sign_flip[sign_flip >= 0.5] = 1.0
    sign_flip[sign_flip < 0.5] = -1.0
    sign_flip = sign_flip[:, None, :].expand(b, max_n, d)
    sign_flip = sign_flip[node_mask]
    return sign_flip


@torch.no_grad()
def collate_ast(
    batch: List[AstExample],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 64,
    max_seq_length: int = 8192,
    dtype: torch.dtype = torch.float,
    use_dict: bool = True,
    num_samples: int = 1,
    lap_node_id_k: int = 8,
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
        lap_eigvec_ = lap_eigvec_[:, :lap_node_id_k]  # [sum(n_node), Dl]

    node_num_, edge_num_ = torch.tensor(node_num, dtype=torch.long), torch.tensor(
        edge_num, dtype=torch.long
    )

    edge_index_ = torch.cat(edge_index, dim=1)

    padded_index, padding_mask, padded_node_mask, padded_edge_mask = get_index_and_mask(
        edge_index_, node_num_, edge_num_
    )

    res = GraphCoderBatch(
        idx=torch.tensor(idx, dtype=torch.long),
        edge_index=edge_index_,
        node_num=node_num_,
        edge_num=edge_num_,
        lap_eigvec=get_index_embed(
            node_id=lap_eigvec_,
            node_mask=get_node_mask(node_num_),
            padded_index=padded_index,
        ),
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
        padded_index=padded_index,
        padding_mask=padding_mask,
        padded_node_mask=padded_node_mask,
        padded_edge_mask=padded_edge_mask,
    )

    if use_dict:
        return res.to_dict()

    return res.to_tuple()
