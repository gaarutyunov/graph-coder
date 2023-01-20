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

from pathlib import Path

from graph_coder.datasets import AstDataset
from graph_coder.utils import get_pretrained_tokenizer


def test_collator():
    dataset = AstDataset(tokenizer=get_pretrained_tokenizer("EleutherAI/gpt-neox-20b"), root=Path(__file__).parent / "./data", batch_size=2)
    loader = dataset.loaders["train"]

    for batch in loader:
        assert batch.edge_index.size(0) == 2
        assert batch.edge_data.size(0) == 2 or batch.edge_data.size(0) == 1
        assert (
            batch.edge_data.size(1) == (batch.edge_data != -100).sum(dim=1).max().item()
        )
        assert batch.node_data.size(0) == 2 or batch.node_data.size(0) == 1
        assert (
            batch.node_data.size(1) == (batch.node_data != -100).sum(dim=1).max().item()
        )
