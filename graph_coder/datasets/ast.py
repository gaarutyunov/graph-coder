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
import inspect
import json
import os
import typing
from hashlib import md5

import aiofiles
import chardet
import humanize
import networkx as nx
import pandas as pandas
import pandas as pd

from functools import lru_cache
from typing import Union, AsyncGenerator, Optional, Tuple, List, Dict

from pathlib import Path
from tqdm.auto import tqdm
from lib2to3.refactor import MultiprocessRefactoringTool, get_fixers_from_package

from graph_coder.data import AstExample
from .registry import register
from .base import BaseDataset
from graph_coder.logger import AsyncLogger
from graph_coder.utils import run_async
from graph_coder.ast import F
from black import format_str, Mode

FilterFn = typing.Callable[[pd.DataFrame], pd.DataFrame]


@register("ast")
class AstDataset(BaseDataset[AstExample]):
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
        preprocess: bool = False,
        in_memory: bool = False,
        print_summary: bool = False,
        filter_index: Optional[Union[typing.Iterable[FilterFn], FilterFn]] = None,
        processed_dir: Optional[typing.Union[os.PathLike, str]] = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.log_file = self.root / log_file
        super().__init__(
            AsyncLogger(self.log_file),
            collate_fn,
            random_seed,
            test_size,
            val_size,
            batch_size,
            in_memory,
            preprocess,
        )
        if processed_dir is not None:
            self._processed_dir = Path(processed_dir).expanduser()
        else:
            self._processed_dir = self.root / "__processed__"
        self._processed_dir.mkdir(exist_ok=True)
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
        self.filter_index = filter_index
        self.introspect()
        if self.preprocess:
            self.process()
        self.split()
        if print_summary:
            self._print_summary()
        if self.in_memory:
            self.load()

    @property
    def processed_dir(self) -> typing.Union[os.PathLike, str]:
        return self._processed_dir

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index: int) -> AstExample:
        item = self._get_item(index)
        if item is not None:
            return item
        assert (
            self.index is not None
        ), "Dataset is not introspected yet, call .introspect()"
        path, lineno, end_lineno, encoding = self.index.iloc[index][
            ["path", "lineno", "end_lineno", "encoding"]
        ]

        code = self._get_source(path, encoding)
        graph_source = "\n".join(code[lineno - 1 : end_lineno])
        graph_source = format_str(graph_source, mode=Mode())
        node, graph = F.parse_graph(graph_source, path)

        return AstExample(
            source=graph_source,
            graph=F.graph_to_data(index, graph),
            docstring=F.get_docstring(node),
        )

    @lru_cache(maxsize=16)
    def _get_source(self, path: str, encoding: str = "utf-8") -> List[str]:
        with open(self.root / path, "r", encoding=encoding) as f:
            source = f.read()
            try:
                source = self._refactor(source, path)
            except:
                pass
            return source.splitlines()

    def introspect(self):
        """Introspects the dataset by discovering and indexing all the graphs in the source files.

        Checks whether the files are already processed.

        Finally, applies filter provided by the `filter_index` property."""
        if not self.index_file.exists():
            self.index_file.touch()
            run_async(self._introspect())
        self.index = pandas.read_json(self.index_file, lines=True)
        self.index["processed"] = self.index.index.map(self.is_item_processed)
        self.index["size"] = self.index.index.map(self.item_size)
        if self.filter_index is not None:
            if inspect.isfunction(self.filter_index):
                self.index = self.filter_index(self.index)
            else:
                for fn in self.filter_index:
                    self.index = fn(self.index)

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
        source, encoding, mod = await self._parse_ast(file)
        if mod is None:
            return

        for node in mod.body:
            res = self._parse_ast_node(node)

            yield {
                **res,
                "encoding": encoding,
            }

    async def _parse_ast(
        self, file: Path
    ) -> Tuple[Optional[str], Optional[str], Optional[ast.Module]]:
        source, encoding = await self._try_open(file)

        if source is None:
            return None, None, None

        try:
            source = self._refactor(source, file.relative_to(self.root))
        except Exception as e:
            await self._logger.warn(f"Refactoring {file.relative_to(self.root)}: {e}")

        try:
            mod = ast.parse(source, filename=str(file))
        except Exception as e:
            await self._logger.error(f"Parsing {file.relative_to(self.root)}: {e}")
            return None, None, None

        return source, encoding, mod

    def _hash(self, source: str, encoding: str) -> str:
        return md5(bytes(source, encoding=encoding)).hexdigest()

    def _parse_ast_node(self, node: ast.AST) -> Optional[Dict[str, typing.Any]]:
        _, graph = F.node_to_graph(node)
        nodes = graph.number_of_nodes()

        if self.min_nodes > nodes:
            return None

        hash_ = self._hash(ast.unparse(node), "utf-8")

        return {
            "lineno": node.lineno,
            "end_lineno": node.end_lineno,
            "nodes": nodes,
            "edges": graph.number_of_edges(),
            "has_docstring": F.get_docstring(node) != "",
            "hash": hash_,
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
            await self._logger.warn(
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
                await self._logger.error(f"Decoding {file.relative_to(self.root)}: {e}")

        return None, None

    def summary(self):
        """Prints a summary of the dataset."""
        self._print_summary()

    def _print_summary(self, out: Optional[typing.TextIO] = None):
        assert self.index is not None, "Run .introspect() first"
        print(f"Summary for {self.__class__.__name__}:\n", file=out)
        print(f"- Number of graphs: {len(self.index):,}", file=out)
        print(f"- Avg. number of nodes: {self.index['nodes'].mean():.0f}", file=out)
        print(f"- Avg. number of edges: {self.index['edges'].mean():.0f}", file=out)
        print(
            f"- Number of documented graphs: {self.index['has_docstring'].sum():,}",
            file=out,
        )
        print(
            f"- Number of processed graphs: {self.index['processed'].sum():,}", file=out
        )
        print(
            f"- Dataset size: {humanize.naturalsize(self.index['size'].sum())}",
            file=out,
        )
        print("\nSplits:", file=out)
        for split, loader in self.loaders.items():
            print(f"- {split}: {len(loader):,} batches", file=out)
