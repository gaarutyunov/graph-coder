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
from transformers import PreTrainedTokenizerFast

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
    node_data: torch.Tensor,
    edge_data: torch.Tensor,
    edge_index: torch.Tensor,
    node_num: torch.Tensor,
    edge_num: torch.Tensor,
):
    seq_len = node_num + edge_num
    b = len(seq_len)
    max_len = max(seq_len)
    max_n = max(node_num)
    device = edge_index.device

    token_pos = torch.arange(max_len, device=device)[None, :].expand(
        b, max_len
    )  # [B, T]

    seq_len_ = seq_len[:, None]  # [B, 1]
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

    d = node_data.size(-1)

    padded_feature = torch.zeros(
        b, max_len, d, device=device, dtype=node_data.dtype
    )  # [B, T, D]
    padded_feature[padded_node_mask.bool(), :] = node_data
    padded_feature[padded_edge_mask.bool(), :] = edge_data

    return (
        padded_index,
        padding_mask.long(),
        padded_node_mask.long(),
        padded_edge_mask.long(),
        padded_feature,
    )


@torch.no_grad()
def collate_ast(
    batch: List[AstExample],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 64,
    max_seq_length: int = 8192,
    use_dict: bool = True,
    num_samples: int = 1,
    shift: bool = True,
) -> torch.Union[
    Dict[str, torch.Tensor], Tuple[Tuple[torch.Tensor, ...], torch.Tensor]
]:
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

    for data in batch:
        sources.append(data.source)
        docstrings.append(data.docstring)

        graph_data = data.graph
        edge_index_ = (
            torch.tensor(graph_data.edge_index, dtype=torch.long).t().contiguous()
        )

        idx.append(graph_data.idx)
        edge_index.append(edge_index_)

        node_data.extend(graph_data.x)
        edge_data.extend(graph_data.edge_attr)

        node_num.append(len(graph_data.x))
        edge_num.append(len(graph_data.edge_attr))

    node_data_, edge_data_ = pad(
        batch=node_data + edge_data,
        num=[node_num, edge_num],
        max_length=max_length,
        tokenizer=tokenizer,
    )

    eos = torch.empty(
        (len(batch), 1),
        dtype=torch.long,
    ).fill_(tokenizer.eos_token_id)
    eos_attn = torch.ones_like(eos)

    docstring_ = tokenizer(
        docstrings,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,
        max_length=max_seq_length,
    ).data
    docstring_ = {k: v.long() for k, v in docstring_.items()}

    if docstring_["input_ids"].size(-1) > 0:
        docstring_ = {
            "input_ids": torch.cat(
                [docstring_["input_ids"].long(), eos],
                dim=1,
            ),
            "attention_mask": torch.cat(
                [
                    docstring_["attention_mask"].long(),
                    eos_attn,
                ],
                dim=1,
            ),
        }

    source_ = tokenizer(
        sources,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,
        max_length=max_seq_length,
    ).data
    source_ = {k: v.long() for k, v in source_.items()}

    if source_["input_ids"].size(-1) > 0:
        source_ = {
            "input_ids": torch.cat(
                [eos, source_["input_ids"].long(), eos],
                dim=1,
            ),
            "attention_mask": torch.cat(
                [
                    eos_attn,
                    source_["attention_mask"].long(),
                    eos_attn,
                ],
                dim=1,
            ),
        }

    node_num_, edge_num_ = torch.tensor(node_num, dtype=torch.long), torch.tensor(
        edge_num, dtype=torch.long
    )

    edge_index_ = torch.cat(edge_index, dim=1)

    (
        padded_index,
        padding_mask,
        padded_node_mask,
        padded_edge_mask,
        padded_feature,
    ) = get_index_and_mask(
        node_data_["input_ids"],
        edge_data_["input_ids"],
        edge_index_,
        node_num_,
        edge_num_,
    )

    padded_feature_attn_mask = (padded_feature != tokenizer.pad_token_id).long()

    res = GraphCoderBatch(
        idx=torch.tensor(idx, dtype=torch.long),
        edge_index=edge_index_,
        node_num=node_num_,
        edge_num=edge_num_,
        source_=source_,
        docstring_=docstring_,
        padded_feature_={
            "input_ids": padded_feature,
            "attention_mask": padded_feature_attn_mask,
        },
        padded_index=padded_index,
        padding_mask=padding_mask,
    )

    labels = torch.cat(
        [
            res.docstring[res.docstring_attn_mask.bool()],
            res.padded_feature[res.padded_feature_attn_mask.bool()],
            res.source[res.source_attn_mask.bool()],
        ],
    )

    if shift:
        labels = labels[1:].contiguous()

    if use_dict:
        return {**res.to_dict(), "labels": labels}

    return res.to_tuple(), labels
