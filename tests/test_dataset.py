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
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from graph_coder.data import collate_ast, pad, GraphCoderBatch
from graph_coder.datasets import AstDataset
from graph_coder.config.functional import (
    get_pretrained_tokenizer,
    filter_has_docstring,
    filter_is_processed,
    get_dtype,
    filter_max_nodes,
)


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
        batch = GraphCoderBatch.from_dict(batch)
        assert batch.edge_index.size(0) == 2
        assert (
            batch.edge_data.size(1) == (batch.edge_data != -100).sum(dim=1).max().item()
        )
        assert (
            batch.node_data.size(1) == (batch.node_data != -100).sum(dim=1).max().item()
        )


def test_encoding():
    dataset = AstDataset(
        root=Path(__file__).parent / "data",
        introspect=True,
    )

    assert sorted(dataset.index["encoding"].unique().tolist()) == [
        "big5",
        "utf-16",
        "utf-8",
    ]


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
            "[ERROR]  Parsing error_001.py: expected an indented block (error_001.py, line 1)\n"
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


def test_filters_index():
    dataset = AstDataset(
        Path(__file__).parent / "./data",
        collate_fn=partial(
            collate_ast, tokenizer=get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
        ),
        batch_size=2,
        filter_index=[filter_has_docstring, partial(filter_max_nodes, max_nodes=13)],
    )

    assert len(dataset.index) == 3


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

    other_texts = [
        "This is a test",
        "This is a fourth test",
        """This is not a test, but a fifth test. This is some huge text. It is needed to test the truncation.
        This is not a test, but a fifth test. This is some huge text. It is needed to test the truncation.
        This is not a test, but a fifth test. This is some huge text. It is needed to test the truncation.
        This is not a test, but a fifth test. This is some huge text. It is needed to test the truncation.""",
    ]
    num = [2, 3]
    other_num = [1, 2]

    padded, other_padded = pad(texts + other_texts, [num, other_num], tokenizer)

    assert padded["input_ids"].shape == (5, 64)
    assert padded["attention_mask"].shape == (5, 64)
    assert torch.all(padded["input_ids"][-1] != 1)
    assert torch.all(padded["attention_mask"][-1] == 1)

    assert other_padded["input_ids"].shape == (3, 64)
    assert other_padded["attention_mask"].shape == (3, 64)
    assert torch.all(other_padded["input_ids"][-1] != 1)
    assert torch.all(other_padded["attention_mask"][-1] == 1)


def test_preprocess():
    dataset = AstDataset(
        root=Path(__file__).parent / "./data",
        introspect=True,
    )
    dataset.process()
    assert dataset.is_processed
    with patch.object(dataset, "_get_processed", wraps=dataset._get_processed) as mock:
        _ = dataset[0]
        mock.assert_called_with(0)
    with patch.object(dataset, "_process", wraps=dataset._process) as mock:
        dataset.process()
        mock.assert_not_called()
    dataset = AstDataset(
        root=Path(__file__).parent / "./data",
        filter_index=filter_is_processed,
    )
    assert dataset.index.shape[0] == len(dataset)
    summary = """Dataset summary for AstDataset:

- Number of graphs: 12
- Avg. number of nodes: 21
- Avg. number of edges: 26
- Number of documented graphs: 8
- Number of processed graphs: 12
"""
    io = StringIO()
    dataset._print_summary(io)
    assert io.getvalue() == summary


def test_dtype():
    dataset = AstDataset(
        collate_fn=partial(
            collate_ast,
            tokenizer=get_pretrained_tokenizer("EleutherAI/gpt-neox-20b"),
            dtype=get_dtype("half"),
        ),
        root=Path(__file__).parent / "./data",
    )

    for batch in dataset.loaders["train"]:
        batch = GraphCoderBatch.from_dict(batch)

        assert batch.node_data.dtype == torch.long
        assert batch.edge_data.dtype == torch.long
        assert batch.docstring.dtype == torch.long
        assert batch.source.dtype == torch.long
        assert batch.lap_eigval.dtype == torch.half
        assert batch.lap_eigvec.dtype == torch.half
