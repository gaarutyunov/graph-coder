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
from pathlib import Path

from graph_coder.data import collate_ast
from graph_coder.datasets import AstDataset
from graph_coder.utils import partial, get_pretrained_tokenizer, filter_has_docstring


def test_filter_index():
    dataset = AstDataset(
        Path(__file__).parent / "./data",
        collate_fn=partial(
            collate_ast, tokenizer=get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
        ),
        batch_size=2,
        filter_index=filter_has_docstring,
    )

    assert len(dataset.index[dataset.index["path"] == "function_003.py"]) == 0
    assert len(dataset.index[dataset.index["path"] == "multi_class_001.py"]) == 0
