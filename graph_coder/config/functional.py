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
"""Functional utilities to use in config files"""
import pandas as pd
import torch
from torch._C._profiler import ProfilerActivity
from transformers import PreTrainedTokenizerFast, AutoTokenizer


def get_pretrained_tokenizer(
    name: str, pad_token_id: int = 1, eos_token_id: int = 0
) -> PreTrainedTokenizerFast:
    """Get a pretrained tokenizer with pad and eos tokens"""
    tokenizer = AutoTokenizer.from_pretrained(name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token_id

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = eos_token_id

    return tokenizer


def get_vocab_size(tokenizer: PreTrainedTokenizerFast) -> int:
    """Get the size of the vocabulary with added tokens"""
    return tokenizer._tokenizer.get_vocab_size(with_added_tokens=True)


def filter_has_docstring(index: pd.DataFrame) -> pd.DataFrame:
    """Filters out rows that don't have docstrings"""
    return index[index.has_docstring].dropna(axis=0)


def filter_is_processed(index: pd.DataFrame) -> pd.DataFrame:
    """Filters out rows that are not processed"""
    return index[index.processed].dropna(axis=0)


def filter_max_nodes(index: pd.DataFrame, max_nodes: int = 1000) -> pd.DataFrame:
    """Filters out rows that have more than `max_nodes` nodes"""
    return index[index.nodes <= max_nodes].dropna(axis=0)


def filter_max_tokens(index: pd.DataFrame, max_tokens: int = 1000) -> pd.DataFrame:
    """Filters out rows that have more than `max_tokens` nodes+edges"""
    return index[(index.nodes + index.edges) <= max_tokens].dropna(axis=0)


def get_device(name: str) -> torch.device:
    """Get device by name"""
    return torch.device(name)


def get_dtype(name: str) -> torch.dtype:
    """Get dtype by name"""
    return getattr(torch, name)


def get_activity(idx: int) -> ProfilerActivity:
    """Get profiler activity by index"""
    return ProfilerActivity(idx)
