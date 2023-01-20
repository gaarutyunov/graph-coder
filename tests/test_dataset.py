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

import numpy as np

from graph_coder.datasets import AstDataset
from graph_coder.utils import get_pretrained_tokenizer


def test_dataset():
    dataset = AstDataset(tokenizer=get_pretrained_tokenizer("EleutherAI/gpt-neox-20b"), root=Path(__file__).parent / "./data")
    assert dataset is not None
    assert np.all(
        dataset.index["source"] != "function_002.py"
    ), "Small graph should be skipped"

    for i in range(len(dataset)):
        data = dataset[i]
        assert data is not None
        assert len(data.graph.edge_index[0]) == 2, "Edge should have 2 nodes"
        assert len(data.graph.edge_index) > 0, "Graph should have edges"

        if dataset.index.iloc[i]["source"] == "function_003.py":
            assert len(data.docstring) == 0, "Docstring should be empty"
