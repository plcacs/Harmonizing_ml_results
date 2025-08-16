from dbt.compilation import Graph, Linker
from dbt.graph.cli import parse_difference
from dbt.graph.queue import GraphQueue
from dbt.graph.selector import NodeSelector
from queue import Empty
from typing import List, Tuple

def _mock_manifest(nodes: str) -> mock.MagicMock:
    ...

class TestLinker:

    def test_linker_add_node(self, linker: Linker) -> None:
        ...

    def test_linker_write_graph(self, linker: Linker) -> None:
        ...

    def assert_would_join(self, queue: GraphQueue) -> None:
        ...

    def _get_graph_queue(self, manifest: mock.MagicMock, linker: Linker, include: List[str] = None, exclude: List[str] = None) -> GraphQueue:
        ...

    def test_linker_add_dependency(self, linker: Linker) -> None:
        ...

    def test_linker_add_disjoint_dependencies(self, linker: Linker) -> None:
        ...

    def test_linker_dependencies_limited_to_some_nodes(self, linker: Linker) -> None:
        ...

    def test__find_cycles__cycles(self, linker: Linker) -> None:
        ...

    def test__find_cycles__no_cycles(self, linker: Linker) -> None:
        ...
