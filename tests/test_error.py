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

from graph_coder.data import collate_ast
from graph_coder.datasets import AstDataset
from graph_coder.utils import get_pretrained_tokenizer, partial


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
        assert len(lines) == 2
        assert lines[0].endswith(
            "[WARN]   Refactoring error_001.py: bad input: type=0, value='', context=('', (2, 0))\n"
        )
        assert lines[1].endswith(
            "[ERROR]  Parsing error_001.py: expected an indented block (<unknown>, line 1)\n"
        )
