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

import ast
import datetime
import json
import os
import typing
import aiofiles
import chardet
import networkx as nx
import pandas as pandas
import pandas as pd

from functools import lru_cache
from typing import Union, AsyncGenerator, Optional, Tuple, List, Dict

from pathlib import Path
from tqdm.auto import tqdm
from lib2to3.refactor import MultiprocessRefactoringTool, get_fixers_from_package

from graph_coder.data import AstExample
from graph_coder.datasets.base import BaseDataset
from graph_coder.utils import run_async
from graph_coder.ast import F


class AstDataset(BaseDataset):
    index: Optional[pandas.DataFrame]

    def __init__(
        self,
        root: typing.Union[os.PathLike, str],
        collate_fn: Optional[typing.Callable] = None,
        min_nodes: int = 10,
        index_file: str = "index.jsonl",
        random_seed: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        batch_size: int = 1,
        log_file: str = "log.txt",
        introspect: bool = False,
        filter_index: Optional[typing.Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> None:
        super().__init__(
            collate_fn,
            random_seed,
            test_size,
            val_size,
            batch_size,
        )
        self.root = Path(root).expanduser()
        self.log_file = self.root / log_file
        self.index_file = self.root / index_file
        if introspect and self.index_file.exists():
            os.remove(self.index_file)
            if self.log_file.exists():
                os.truncate(self.log_file, 0)
        self.min_nodes = min_nodes
        self.random_seed = random_seed
        self.refactoring_tool = MultiprocessRefactoringTool(
            get_fixers_from_package("lib2to3.fixes")
        )
        self.encodings = ["big5", "latin-1"]
        self.index: Optional[pandas.DataFrame] = None
        self.filter_index = filter_index
        self.introspect()
        self.split()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index: int) -> AstExample:
        assert (
            self.index is not None
        ), "Dataset is not introspected yet, call .introspect()"
        path, lineno, end_lineno, encoding = self.index.iloc[index][
            ["path", "lineno", "end_lineno", "encoding"]
        ]

        code = self._get_source(path, encoding)
        graph_source = "\n".join(code[lineno - 1 : end_lineno])
        node, graph = F.parse_graph(graph_source, path)

        return AstExample(
            source=graph_source,
            graph=F.graph_to_data(index, graph),
            docstring=F.get_docstring(node),
        )

    @lru_cache(maxsize=16)
    def _get_source(self, path: str, encoding: str = "utf-8") -> List[str]:
        with open(self.root / path, "r", encoding=encoding) as f:
            return self._refactor(f.read(), self.root / path).splitlines()

    def introspect(self):
        if not self.index_file.exists():
            self.index_file.touch()
            run_async(self._introspect())
        self.index = pandas.read_json(self.index_file, lines=True)
        if self.filter_index is not None:
            self.index = self.filter_index(self.index)

    async def _introspect(self):
        async for graph_meta in self._parse_root():
            await self._write_index(**graph_meta)

    async def _parse_root(self):
        for file in tqdm(
            self.root.rglob("**/*.py"),
            desc=f"Introspecting dataset in {self.root}",
            unit="files",
        ):
            async for graph_meta in self._parse_source(file):
                yield {"path": str(file.relative_to(self.root)), **graph_meta}

    async def _write_index(self, **kwargs) -> None:
        async with aiofiles.open(self.index_file, "a") as f:
            await f.write(json.dumps(kwargs) + "\n")

    async def _parse_source(
        self, file: Path
    ) -> AsyncGenerator[Dict[str, Union[nx.Graph, Optional[int]]], None]:
        source, encoding = await self._try_open(file)

        if source is None:
            return

        try:
            source = self._refactor(source, file.relative_to(self.root))
        except Exception as e:
            await self.log("WARN", f"Refactoring {file.relative_to(self.root)}: {e}")

        try:
            mod = ast.parse(source, filename=str(file))
        except Exception as e:
            await self.log("ERROR", f"Parsing {file.relative_to(self.root)}: {e}")
            return

        for node in mod.body:
            _, graph = F.node_to_graph(node)
            nodes = graph.number_of_nodes()

            if self.min_nodes > nodes:
                continue

            yield {
                "lineno": node.lineno,
                "end_lineno": node.end_lineno,
                "nodes": nodes,
                "edges": graph.number_of_edges(),
                "has_docstring": F.get_docstring(node) != "",
                "encoding": encoding,
            }

    def _refactor(self, source: str, name: Union[str, os.PathLike]) -> str:
        if not source.endswith("\n"):
            source += "\n"
        return str(self.refactoring_tool.refactor_string(source, str(name)))

    async def _try_open(self, file: Path) -> Tuple[Optional[str], Optional[str]]:
        lines = []
        detector = chardet.UniversalDetector()

        try:
            encoding: Optional[str] = "utf-8"
            async with aiofiles.open(file, "r", encoding=encoding) as f:
                source: Optional[str] = await f.read()
                return source, encoding
        except Exception as e:
            await self.log(
                "WARN",
                f"Opening {file.relative_to(self.root)} as utf-8: {e}",
            )
            pass

        async with aiofiles.open(file, "rb") as f:
            encoding = None

            for line in await f.readlines():
                lines.append(line)
                if not detector.done:
                    detector.feed(line)
                if detector.done and encoding is None:
                    detector.close()
                    encoding = detector.result["encoding"]

        if encoding is None:
            source, encoding = await self._try_encodings(file, lines)
        else:
            encoding = encoding.lower()
            source = b"".join(lines).decode(encoding)

        return source, encoding

    async def _try_encodings(
        self, file: Path, lines: List[bytes]
    ) -> Tuple[Optional[str], Optional[str]]:
        for encoding in self.encodings:
            try:
                return b"".join(lines).decode(encoding), encoding
            except Exception as e:
                await self.log("ERROR", f"Decoding {file.relative_to(self.root)}: {e}")

        return None, None

    async def log(self, level: str, msg: str):
        level = f"[{level}]"
        async with aiofiles.open(self.log_file, "a") as log:
            await log.write(
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {level:8s} {msg}\n"
            )
