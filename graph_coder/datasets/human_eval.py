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
import gzip
import json
import os
import typing
from typing import Optional, Union
from urllib.request import urlopen

import aiofiles
import pandas as pd
from black import format_str, Mode
from torch.utils.data import DataLoader

from graph_coder.data import HumanEvalExample
from graph_coder.utils import run_async
from .ast import FilterFn
from .func_ast import FuncAstDataset
from .registry import register

__DOWNLOAD_URL__ = (
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
)


@register("human_eval")
class HumanEvalDataset(FuncAstDataset[HumanEvalExample]):
    def __init__(
        self,
        root: typing.Union[os.PathLike, str],
        url: str = __DOWNLOAD_URL__,
        collate_fn: Optional[typing.Callable] = None,
        min_nodes: int = 1,
        index_file: str = "index.jsonl",
        random_seed: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        batch_size: int = 1,
        log_file: str = "log.txt",
        introspect: bool = False,
        preprocess: bool = False,
        in_memory: bool = False,
        print_summary: bool = False,
        filter_index: Optional[Union[typing.Iterable[FilterFn], FilterFn]] = None,
        processed_dir: Optional[typing.Union[os.PathLike, str]] = None,
    ) -> None:
        self.url = url
        super().__init__(
            root,
            collate_fn,
            min_nodes,
            index_file,
            random_seed,
            test_size,
            val_size,
            batch_size,
            log_file,
            introspect,
            preprocess,
            in_memory,
            print_summary,
            filter_index,
            processed_dir,
        )

    def __getitem__(self, index: int) -> HumanEvalExample:
        ex = super().__getitem__(index)

        if isinstance(ex, HumanEvalExample):
            return ex

        path = self.index.iloc[index]["path"]

        with open((self.root / path).parent / "meta.json", mode="r") as f:
            data = json.load(f)

        return HumanEvalExample(
            source=data["prompt"],
            graph=ex.graph,
            docstring=ex.docstring,
            canonical_solution=data["canonical_solution"],
            prompt=data["prompt"],
            test=data["test"],
            task_id=data["task_id"],
            entry_point=data["entry_point"],
        )

    def split(self):
        self._loaders = {
            "infer": DataLoader(
                self, collate_fn=self.collate_fn, batch_size=self.batch_size
            )
        }

    def download(self):
        if (self.root / "HumanEval.pkl").exists():
            return

        if not (self.root / "HumanEval.jsonl.gz").exists():
            with urlopen(self.url) as response:
                with open(self.root / "HumanEval.jsonl.gz", mode="wb") as f:
                    f.write(response.read())

        with gzip.GzipFile(filename=self.root / "HumanEval.jsonl.gz") as gz:
            df = pd.read_json(gz, lines=True)

        df.to_pickle(self.root / "HumanEval.pkl")

    def introspect(self):
        df = pd.read_pickle(self.root / "HumanEval.pkl")
        run_async(self._write_sources(df))
        super().introspect()

    async def _write_sources(self, df: pd.DataFrame):
        for idx, row in df.iterrows():
            task_root = self.root / f"{row['task_id']}"
            task_root.mkdir(exist_ok=True, parents=True)

            async with aiofiles.open(task_root / "source.py", mode="w") as f:
                source = row["prompt"] + "    pass\n"
                try:
                    source = self._refactor(source, task_root / "source.py")
                    source = format_str(source, mode=Mode())
                except:
                    pass
                await f.write(source)

            async with aiofiles.open(task_root / "meta.json", mode="w") as f:
                await f.write(row.to_json(orient="columns", indent=4))  # type: ignore[call-overload]
