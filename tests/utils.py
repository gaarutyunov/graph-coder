import pathlib
import networkx as nx

__snapshots_dir__ = pathlib.Path(__file__).parent / "snapshots"


def get_graph_from_snapshot(idx: int) -> nx.Graph:
    return nx.read_gpickle(__snapshots_dir__ / f"{idx}.bin")


def assert_equals_snapshot(graph: nx.Graph, idx: int):
    assert nx.utils.graphs_equal(graph, get_graph_from_snapshot(idx))
