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
import pathlib
import re
from typing import Any, Dict, List

import pandas as pd
import torch
from torch._C._profiler import ProfilerActivity
from transformers import AutoTokenizer, PreTrainedTokenizerFast


__num_re__ = re.compile(r"\d+")


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


def filter_max_size(index: pd.DataFrame, max_size: int = 1000) -> pd.DataFrame:
    """Filters out rows that have more than `max_size` size"""
    return index[index["size"] <= max_size].dropna(axis=0)


def filter_max_lines(index: pd.DataFrame, max_lines: int = 1000) -> pd.DataFrame:
    """Filters out rows that have more than `max_lines` lines (end_lineno-lineno)"""
    return index[(index.end_lineno - index.lineno) <= max_lines].dropna(axis=0)


def filter_unique_by_column(index: pd.DataFrame, column: str) -> pd.DataFrame:
    """Filters out rows that have the same value in `column`"""
    return index.drop_duplicates(subset=[column]).dropna(axis=0)


def get_device(name: str) -> torch.device:
    """Get device by name"""
    return torch.device(name)


def get_dtype(name: str) -> torch.dtype:
    """Get dtype by name"""
    return getattr(torch, name)


def get_activity(idx: int) -> ProfilerActivity:
    """Get profiler activity by index"""
    return ProfilerActivity(idx)


def get_keys(obj: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Get only specified keys from object"""
    return {k: v for k, v in obj.items() if k in keys}


def get_log_path(root: str) -> str:
    """Get log path with model version"""
    path = pathlib.Path(root)
    path.mkdir(exist_ok=True, parents=True)
    versions = [0]

    for version in path.iterdir():
        m = __num_re__.search(version.stem)
        if m is None:
            continue
        versions.append(int(m[0]))

    versions = sorted(versions)

    new_path = path / f"version{versions[-1]+1}"
    new_path.mkdir(exist_ok=True, parents=True)

    return str(new_path)
