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
from functools import partial
from io import StringIO
from pathlib import Path

from graph_coder.config.functional import filter_unique_by_column
from graph_coder.datasets import FuncAstDataset, get


def test_dataset():
    dataset = FuncAstDataset(
        root=Path(__file__).parent / "func_data",
        index_file="index.jsonl",
        introspect=True,
    )

    assert dataset is not None
    assert len(dataset) == 9


def test_filter_unique():
    dataset = FuncAstDataset(
        root=Path(__file__).parent / "func_data",
        index_file="index.jsonl",
        introspect=True,
        filter_index=[partial(filter_unique_by_column, column="hash")],
    )

    assert dataset is not None
    assert len(dataset) == 7


def test_register():
    dataset = get("func_ast")
    assert dataset.__name__ == FuncAstDataset.__name__


def test_process():
    FuncAstDataset(
        root=Path(__file__).parent / "func_data",
        index_file="index.jsonl",
        introspect=True,
    ).process()

    dataset = FuncAstDataset(
        root=Path(__file__).parent / "func_data",
        index_file="index.jsonl",
    )

    summary = """\
Summary for FuncAstDataset:

- Number of graphs: 9
- Avg. number of nodes: 19
- Avg. number of edges: 22
- Number of documented graphs: 2
- Number of processed graphs: 9
- Dataset size: 11.7 kB

Splits:
- train: 6 batches
- valid: 2 batches
- infer: 1 batches
"""
    io = StringIO()
    dataset._print_summary(io)
    assert io.getvalue() == summary
