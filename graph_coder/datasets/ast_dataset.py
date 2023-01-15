import ast
import asyncio
import json
import os
import typing
import aiofiles
import networkx as nx
import pandas as pandas
import pydot
import torch
import transformers

from functools import lru_cache, partial
from typing import Union, AsyncGenerator, Optional
from astmonkey import transformers as ast_transformers
from pathlib import Path
from transformers import TensorType
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data
from tqdm.auto import tqdm

from graph_coder.data.collator import collate
from graph_coder.datasets.ast_example import AstExample
from graph_coder.datasets.base import BaseDataset
from graph_coder.utils.graph_node_visitor import GraphNodeVisitor


class AstDataset(Dataset, BaseDataset):
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
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.root = Path(root).expanduser()
        self.index_file = self.root / index_file
        self.min_nodes = min_nodes
        self.random_seed = random_seed
        self.collate_fn = collate_fn or partial(collate, pad_token_id=self.tokenizer.pad_token_id)
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.max_length = max_length
        self._loaders = {}
        self.introspect()
        self.split()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index: int) -> AstExample:
        source, lineno, end_lineno = self.index.iloc[index][
            ["source", "lineno", "end_lineno"]
        ]

        code = self._get_source(source)
        graph_source = "".join(code[lineno - 1: end_lineno])
        node, graph = self._parse_graph(graph_source)
        docstring = self._get_docstring(node)
        data = self._graph_to_data(graph)
        data.idx = index

        return AstExample(
            source=self.tokenizer(graph_source).convert_to_tensors(TensorType.PYTORCH)["input_ids"],
            graph=data,
            docstring=self.tokenizer(docstring).convert_to_tensors(TensorType.PYTORCH)["input_ids"],
        )

    @property
    def loaders(self) -> dict[str, DataLoader]:
        return self._loaders

    @lru_cache(maxsize=16)
    def _get_source(self, source: str) -> list[str]:
        with open(self.root / source, "r") as f:
            return f.readlines()

    def introspect(self):
        if not self.index_file.exists():
            self.index_file.touch()
            asyncio.run(self._introspect())
        self.index = pandas.read_json(self.index_file, lines=True)

    def split(self):
        train_size = 1.0 - self.test_size - self.val_size
        datasets = random_split(
            self,
            [train_size, self.val_size, self.test_size],
            generator=torch.Generator().manual_seed(self.random_seed)
            if self.random_seed
            else None,
        )

        self._loaders = dict(
            zip(
                ["train", "val", "test"],
                [
                    DataLoader(
                        dataset, collate_fn=self.collate_fn, batch_size=self.batch_size
                    )
                    for dataset in datasets
                ],
            )
        )

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
    ) -> AsyncGenerator[dict[str, Union[pydot.Dot, int]], None]:
        async with aiofiles.open(file, "r") as f:
            source = await f.read()
            mod = ast.parse(source)

            for node in mod.body:
                _, graph = self._node_to_graph(node)

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

    def _parse_graph(self, code: str) -> tuple[ast.AST, nx.Graph]:
        mod = ast.parse(code)
        node = mod.body[0]

        return self._node_to_graph(node)

    def _node_to_graph(self, node: ast.AST) -> tuple[ast.AST, nx.Graph]:
        node = ast_transformers.ParentChildNodeTransformer().visit(node)
        visitor = GraphNodeVisitor()
        visitor.visit(node)

        return node, visitor.graph

    def _get_docstring(self, node: ast.AST) -> str:
        try:
            doc = ast.get_docstring(node)
            return doc if doc is not None else ""
        except TypeError:
            return ""

    def _graph_to_data(self, graph: nx.Graph) -> Data:
        x = []
        edge_index = []
        edge_attr = []
        graph: nx.Graph = nx.convert_node_labels_to_integers(
            graph, label_attribute="label"
        )

        for node, label in graph.nodes(data="label"):
            x.append(label)

        for u, v, label in graph.edges(data="label"):
            edge_index.append((u, v))
            edge_attr.append(label)

        return Data(
            x=self.tokenizer(
                x, max_length=self.max_length, padding="max_length", truncation=True
            ).convert_to_tensors(TensorType.PYTORCH)["input_ids"],
            edge_index=torch.LongTensor(edge_index).t().contiguous(),
            edge_attr=self.tokenizer(
                edge_attr,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            ).convert_to_tensors(TensorType.PYTORCH)["input_ids"],
        )
