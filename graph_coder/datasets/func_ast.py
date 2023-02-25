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
import ast
from pathlib import Path
from typing import AsyncGenerator, Dict, Union, Optional

import networkx as nx

from .registry import register
from .ast import AstDataset


@register("func_ast")
class FuncAstDataset(AstDataset):
    async def _parse_source(
        self, file: Path
    ) -> AsyncGenerator[Dict[str, Union[nx.Graph, Optional[int]]], None]:
        source, encoding, mod = await self._parse_ast(file)

        if mod is None:
            return

        for node in mod.body:
            if isinstance(node, ast.ClassDef):
                for res in self._iter_node(node):
                    if res is None:
                        continue
                    yield {
                        **res,
                        "encoding": encoding,
                    }
                continue

            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            res = self._parse_ast_node(node)
            if res is None:
                continue

            yield {
                **res,
                "encoding": encoding,
            }

    def _iter_node(self, node: ast.AST):
        for subnode in node.body:
            if not isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            yield self._parse_ast_node(subnode)
