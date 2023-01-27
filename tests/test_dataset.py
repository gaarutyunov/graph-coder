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
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch

from graph_coder.data import collate_ast, pad
from graph_coder.datasets import AstDataset
from graph_coder.utils import get_pretrained_tokenizer, filter_has_docstring


def test_dataset():
    dataset = AstDataset(
        collate_fn=partial(
            collate_ast, tokenizer=get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
        ),
        root=Path(__file__).parent / "./data",
    )
    assert dataset is not None
    assert np.all(
        dataset.index["path"] != "function_002.py"
    ), "Small graph should be skipped"

    for i in range(len(dataset)):
        data = dataset[i]
        assert data is not None
        assert len(data.graph.edge_index[0]) == 2, "Edge should have 2 nodes"
        assert len(data.graph.edge_index) > 0, "Graph should have edges"

        if dataset.index.iloc[i]["path"] == "function_003.py":
            assert len(data.docstring) == 0, "Docstring should be empty"


def test_collator():
    dataset = AstDataset(
        collate_fn=partial(
            collate_ast, tokenizer=get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
        ),
        root=Path(__file__).parent / "./data",
        batch_size=2,
    )
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


def test_encoding():
    dataset = AstDataset(
        root=Path(__file__).parent / "data",
        introspect=True,
    )

    assert sorted(dataset.index["encoding"].unique().tolist()) == ['big5', 'utf-16', 'utf-8']


def test_error():
    dataset = AstDataset(
        collate_fn=partial(
            collate_ast, tokenizer=get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
        ),
        root=Path(__file__).parent / "./data",
        introspect=True,
    )
    with open(dataset.log_file, "r") as log:
        lines = log.readlines()
        assert len(lines) > 0
        assert lines[-2].endswith(
            "[WARN]   Refactoring error_001.py: bad input: type=0, value='', context=('', (2, 0))\n"
        )
        assert lines[-1].endswith(
            "[ERROR]  Parsing error_001.py: expected an indented block (<unknown>, line 1)\n"
        )


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


def test_loaders():
    dataset = AstDataset(
        collate_fn=partial(
            collate_ast, tokenizer=get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
        ),
        root=Path(__file__).parent / "./data",
    )

    train = dataset.loaders["train"]
    val = dataset.loaders["val"]
    test = dataset.loaders["test"]

    assert len(train) == math.ceil(len(dataset.index) * 0.6)
    assert len(val) == math.floor(len(dataset.index) * 0.2)
    assert len(test) == math.floor(len(dataset.index) * 0.2)


def test_pad():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")

    texts = [
        "This is a test",
        "This is another test",
        "This is a third test",
        "This is a fourth test",
        """This is not a test, but a fifth test. This is some huge text. It is needed to test the truncation.
        This is not a test, but a fifth test. This is some huge text. It is needed to test the truncation.
        This is not a test, but a fifth test. This is some huge text. It is needed to test the truncation.
        This is not a test, but a fifth test. This is some huge text. It is needed to test the truncation.
        This is not a test, but a fifth test. This is some huge text. It is needed to test the truncation.""",
    ]
    num = [2, 3]

    padded = pad(texts, num, tokenizer)

    assert padded["input_ids"].shape == (2, 3, 64)
    assert padded["attention_mask"].shape == (2, 3, 64)
    assert torch.all(padded["input_ids"][0, -1] == 1)
    assert padded["attention_mask"][0, -1].sum() == 0
