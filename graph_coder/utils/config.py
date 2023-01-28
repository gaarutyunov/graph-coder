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
from catalyst.contrib.scripts.run import (
    run_from_params,
)
from catalyst.registry import REGISTRY
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from graph_coder.config import ConfigBuilder
from typing import Optional


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


def _add_all_to_registry():
    """Add all graph-coder modules to registry."""
    import graph_coder.data
    import graph_coder.datasets
    import graph_coder.models
    import graph_coder.modules
    import graph_coder.runners
    import graph_coder.utils

    REGISTRY.add_from_module(graph_coder.data)
    REGISTRY.add_from_module(graph_coder.datasets)
    REGISTRY.add_from_module(graph_coder.models)
    REGISTRY.add_from_module(graph_coder.modules)
    REGISTRY.add_from_module(graph_coder.runners)
    REGISTRY.add_from_module(graph_coder.utils)


def run_model(
    root: str,
    name: Optional[str] = None,
    size: Optional[str] = None,
    arch: Optional[str] = None,
):
    """Run a model from a config directory with the specified name, size and arch.

    If root is a path to a file, it will be used as the config file."""
    _add_all_to_registry()
    experiment_params = ConfigBuilder(root, name, size, arch).load().build()

    run_from_params(experiment_params)
