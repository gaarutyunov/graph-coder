#  Copyright 2023 German Arutyunov
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import math
from pathlib import Path

from graph_coder.datasets import AstDataset
from graph_coder.utils import get_pretrained_tokenizer


def test_loaders():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(tokenizer=tokenizer, root=Path(__file__).parent / "./data")

    train = dataset.loaders["train"]
    val = dataset.loaders["val"]
    test = dataset.loaders["test"]

    assert len(train) == math.ceil(len(dataset.index) * 0.6)
    assert len(val) == math.floor(len(dataset.index) * 0.2)
    assert len(test) == math.floor(len(dataset.index) * 0.2)
