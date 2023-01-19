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

import ast
import asyncio
import datetime
import json
import os
import typing
import aiofiles
import networkx as nx
import pandas as pandas
import torch
import transformers

from functools import lru_cache, partial
from typing import Union, AsyncGenerator, Optional, Tuple, List, Dict
from astmonkey import transformers as ast_transformers
from pathlib import Path
from tqdm.auto import tqdm
from lib2to3.refactor import MultiprocessRefactoringTool, get_fixers_from_package

from graph_coder.data import AstData
from graph_coder.data.collator import collate_ast
from graph_coder.data import AstExample
from graph_coder.datasets.base import BaseDataset
from graph_coder.utils.graph_node_visitor import GraphNodeVisitor


def graph_to_data(idx: int, graph: nx.Graph) -> AstData:
    x = []
    edge_index = []
    edge_attr = []
    graph: nx.Graph = nx.convert_node_labels_to_integers(graph, label_attribute="label")

    for node, label in graph.nodes(data="label"):
        x.append(label)

    for u, v, label in graph.edges(data="label"):
        edge_index.append((u, v))
        edge_attr.append(label)

    return AstData(
        idx=idx,
        x=x,
        edge_index=torch.LongTensor(edge_index).t().contiguous(),
        edge_attr=edge_attr,
    )


def get_docstring(node: ast.AST) -> str:
    try:
        doc = ast.get_docstring(node)
        return doc if doc is not None else ""
    except TypeError:
        return ""


def node_to_graph(node: ast.AST) -> Tuple[ast.AST, nx.Graph]:
    node = ast_transformers.ParentChildNodeTransformer().visit(node)
    visitor = GraphNodeVisitor()
    visitor.visit(node)

    return node, visitor.graph


def parse_graph(code: str) -> Tuple[ast.AST, nx.Graph]:
    mod = ast.parse(code)
    node = mod.body[0]

    return node_to_graph(node)


class AstDataset(BaseDataset):
    index: Optional[pandas.DataFrame]

    def __init__(
        self,
        root: typing.Union[os.PathLike, str],
        tokenizer: transformers.PreTrainedTokenizerFast,
        min_nodes: int = 10,
        max_length: int = 64,
        index_file: str = "index.jsonl",
        random_seed: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        collate_fn: typing.Callable = None,
        batch_size: int = 1,
        log_file: str = "log.txt",
        introspect: bool = False,
    ) -> None:
        super().__init__(
            collate_fn
            or partial(collate_ast, tokenizer=tokenizer, max_length=max_length),
            random_seed,
            test_size,
            val_size,
            batch_size,
        )
        self.tokenizer = tokenizer
        self.root = Path(root).expanduser()
        self.log_file = self.root / log_file
        if not self.log_file.exists():
            self.log_file.touch()
        else:
            os.truncate(self.log_file, 0)
        self.index_file = self.root / index_file
        if introspect and self.index_file.exists():
            os.remove(self.index_file)
        self.min_nodes = min_nodes
        self.random_seed = random_seed
        self.max_length = max_length
        self._loaders = {}
        self.refactoring_tool = MultiprocessRefactoringTool(
            get_fixers_from_package("lib2to3.fixes")
        )
        self.introspect()
        self.split()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index: int) -> AstExample:
        source, lineno, end_lineno = self.index.iloc[index][
            ["source", "lineno", "end_lineno"]
        ]

        code = self._get_source(source)
        graph_source = "".join(code[lineno - 1 : end_lineno])
        node, graph = parse_graph(graph_source)

        return AstExample(
            source=graph_source,
            graph=graph_to_data(index, graph),
            docstring=get_docstring(node),
        )

    @lru_cache(maxsize=16)
    def _get_source(self, source: str) -> List[str]:
        with open(self.root / source, "r") as f:
            return f.readlines()

    def introspect(self):
        if not self.index_file.exists():
            self.index_file.touch()
            asyncio.run(self._introspect())
        self.index = pandas.read_json(self.index_file, lines=True)

    async def _introspect(self):
        async for graph in tqdm(
            self._parse_root(), desc=f"Introspecting dataset in {self.root}"
        ):
            await self._write_index(**graph)

    async def _parse_root(self):
        for file in self.root.rglob("**/*.py"):
            async for graph in self._parse_source(file):
                yield {
                    "source": str(file.relative_to(self.root)),
                    "lineno": graph["lineno"],
                    "end_lineno": graph["end_lineno"],
                    "nodes": graph["nodes"],
                    "edges": graph["edges"],
                }

    async def _write_index(self, **kwargs) -> None:
        async with aiofiles.open(self.index_file, "a") as f:
            await f.write(json.dumps(kwargs) + "\n")

    async def _parse_source(
        self, file: Path
    ) -> AsyncGenerator[Dict[str, Union[nx.Graph, int]], None]:
        async with aiofiles.open(file, "r") as f:
            source = await f.read()
            try:
                source = self.refactoring_tool.refactor_string(source, str(file))
            except Exception as e:
                await self.log(
                    "ERROR", f"Refactoring {file.relative_to(self.root)}: {e}"
                )
                return

            try:
                mod = ast.parse(str(source))
            except Exception as e:
                await self.log("ERROR", f"Parsing {file.relative_to(self.root)}: {e}")
                return

            for node in mod.body:
                _, graph = node_to_graph(node)

                nodes = graph.number_of_nodes()

                if self.min_nodes > nodes:
                    continue

                yield {
                    "graph": graph,
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno,
                    "nodes": nodes,
                    "edges": graph.number_of_edges(),
                }

    async def log(self, level: str, msg: str):
        level = f"[{level}]"
        async with aiofiles.open(self.log_file, "a") as log:
            await log.write(
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {level:8s} {msg}\n"
            )
