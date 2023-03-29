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
from pathlib import Path

import networkx as nx

from graph_coder.ast.functional import parse_graph
from graph_coder.data import AstExample
from graph_coder.datasets import FuncAstDataset
from graph_coder.visualize.ast import export_code, export_graph, export_tokens


def test_visualize_ast():
    dataset = FuncAstDataset(
        root=Path(__file__).parent / "func_data",
    )

    func = dataset.index[dataset.index.path.str.endswith("multi_function_001.py")].iloc[
        [1]
    ]
    idx = func.index[0]

    item: AstExample = dataset[idx]

    _, g = parse_graph(item.source, Path(func.iloc[0]["path"]).stem, True)
    _, gg = parse_graph(item.source, Path(func.iloc[0]["path"]).stem, False)
    g.remove_edges_from(nx.selfloop_edges(g))
    gg.remove_edges_from(nx.selfloop_edges(gg))

    source = Path(__file__).parent / "logs" / "source.tiff"
    graph = Path(__file__).parent / "logs" / "graph.tiff"
    tokens = Path(__file__).parent / "logs" / "tokens.tiff"

    with open(source, mode="wb") as f:
        export_code(item.source, f)

    with open(graph, mode="wb") as f:
        export_graph(g, f)

    with open(tokens, mode="wb") as f:
        export_tokens(gg, f)

    assert source.stat().st_size > 100_000
    assert graph.stat().st_size > 100_000
    assert tokens.stat().st_size > 100_000
