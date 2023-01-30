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
from collections import OrderedDict
from typing import Iterable, Dict, Any
from catalyst import utils
from transformers import PreTrainedTokenizerFast, AutoTokenizer


def process_configs(configs: Iterable[str], ordered: bool = False) -> Dict[str, Any]:
    """Merges YAML configs and prepares env."""

    config: Dict[str, Any] = OrderedDict() if ordered else {}  # type: ignore[assignment]

    for config_path in configs:
        config_part = utils.load_config(config_path, ordered=ordered)
        config = utils.merge_dicts(config, config_part)

    return config


def get_pretrained_tokenizer(
    name: str, pad_token_id: int = 1, eos_token_id: int = 0
) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token_id

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = eos_token_id

    return tokenizer


def get_vocab_size(tokenizer: PreTrainedTokenizerFast) -> int:
    return tokenizer._tokenizer.get_vocab_size(with_added_tokens=True)