import os
import tempfile
from queue import Empty
from unittest import mock
from typing import List, Optional, Tuple
import pytest
from dbt.compilation import Graph, Linker
from dbt.graph.cli import parse_difference
from dbt.graph.queue import GraphQueue
from dbt.graph.selector import NodeSelector
from unittest.mock import MagicMock


def _mock_manifest(nodes: str) -> MagicMock:
    config: MagicMock = mock.MagicMock(enabled=True)
    manifest = mock.MagicMock(nodes={n: mock.MagicMock(unique_id=n,
                                                         package_name='pkg',
                                                         name=n,
                                                         empty=False,
                                                         config=config,
                                                         fqn=['pkg', n],
                                                         is_versioned=False)
                                       for n in nodes})
    manifest.expect.side_effect = lambda n: mock.MagicMock(unique_id=n)
    return manifest


class TestLinker:

    @pytest.fixture
    def linker(self) -> Linker:
        return Linker()

    def test_linker_add_node(self, linker: Linker) -> None:
        expected_nodes: List[str] = ['A', 'B', 'C']
        for node in expected_nodes:
            linker.add_node(node)
        actual_nodes = linker.nodes()
        for node in expected_nodes:
            assert node in actual_nodes
        assert len(actual_nodes) == len(expected_nodes)

    def test_linker_write_graph(self, linker: Linker) -> None:
        expected_nodes: List[str] = ['A', 'B', 'C']
        for node in expected_nodes:
            linker.add_node(node)
        manifest: MagicMock = _mock_manifest('ABC')
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:
            linker.write_graph(fname, manifest)
            assert os.path.exists(fname)
        finally:
            os.unlink(fname)

    def assert_would_join(self, queue: GraphQueue) -> None:
        """test join() without timeout risk"""
        assert queue.inner.unfinished_tasks == 0

    def _get_graph_queue(self, manifest: MagicMock, linker: Linker,
                         include: Optional[List[str]] = None,
                         exclude: Optional[List[str]] = None) -> GraphQueue:
        graph = Graph(linker.graph)
        selector = NodeSelector(graph, manifest)
        spec = parse_difference(include, exclude)
        return selector.get_graph_queue(spec)

    def test_linker_add_dependency(self, linker: Linker) -> None:
        actual_deps: List[Tuple[str, str]] = [('A', 'B'), ('A', 'C'), ('B', 'C')]
        for l, r in actual_deps:
            linker.dependency(l, r)
        queue: GraphQueue = self._get_graph_queue(_mock_manifest('ABC'), linker)
        got = queue.get(block=False)
        assert got.unique_id == 'C'
        with pytest.raises(Empty):
            queue.get(block=False)
        assert not queue.empty()
        queue.mark_done('C')
        assert not queue.empty()
        got = queue.get(block=False)
        assert got.unique_id == 'B'
        with pytest.raises(Empty):
            queue.get(block=False)
        assert not queue.empty()
        queue.mark_done('B')
        assert not queue.empty()
        got = queue.get(block=False)
        assert got.unique_id == 'A'
        with pytest.raises(Empty):
            queue.get(block=False)
        assert queue.empty()
        queue.mark_done('A')
        self.assert_would_join(queue)
        assert queue.empty()

    def test_linker_add_disjoint_dependencies(self, linker: Linker) -> None:
        actual_deps: List[Tuple[str, str]] = [('A', 'B')]
        additional_node: str = 'Z'
        for l, r in actual_deps:
            linker.dependency(l, r)
        linker.add_node(additional_node)
        queue: GraphQueue = self._get_graph_queue(_mock_manifest('ABCZ'), linker)
        first = queue.get(block=False)
        assert first.unique_id == 'B'
        assert not queue.empty()
        queue.mark_done('B')
        assert not queue.empty()
        second = queue.get(block=False)
        assert second.unique_id in {'A', 'Z'}
        assert not queue.empty()
        queue.mark_done(second.unique_id)
        assert not queue.empty()
        third = queue.get(block=False)
        assert third.unique_id in {'A', 'Z'}
        with pytest.raises(Empty):
            queue.get(block=False)
        assert second.unique_id != third.unique_id
        assert queue.empty()
        queue.mark_done(third.unique_id)
        self.assert_would_join(queue)
        assert queue.empty()

    def test_linker_dependencies_limited_to_some_nodes(self, linker: Linker) -> None:
        actual_deps: List[Tuple[str, str]] = [('A', 'B'), ('B', 'C'), ('C', 'D')]
        for l, r in actual_deps:
            linker.dependency(l, r)
        queue: GraphQueue = self._get_graph_queue(_mock_manifest('ABCD'), linker, ['B'])
        got = queue.get(block=False)
        assert got.unique_id == 'B'
        assert queue.empty()
        queue.mark_done('B')
        self.assert_would_join(queue)
        queue_2: GraphQueue = self._get_graph_queue(_mock_manifest('ABCD'), linker, ['A', 'B'])
        got = queue_2.get(block=False)
        assert got.unique_id == 'B'
        assert not queue_2.empty()
        with pytest.raises(Empty):
            queue_2.get(block=False)
        queue_2.mark_done('B')
        assert not queue_2.empty()
        got = queue_2.get(block=False)
        assert got.unique_id == 'A'
        assert queue_2.empty()
        with pytest.raises(Empty):
            queue_2.get(block=False)
        assert queue_2.empty()
        queue_2.mark_done('A')
        self.assert_would_join(queue_2)

    def test__find_cycles__cycles(self, linker: Linker) -> None:
        actual_deps: List[Tuple[str, str]] = [('A', 'B'), ('B', 'C'), ('C', 'A')]
        for l, r in actual_deps:
            linker.dependency(l, r)
        assert linker.find_cycles() is not None

    def test__find_cycles__no_cycles(self, linker: Linker) -> None:
        actual_deps: List[Tuple[str, str]] = [('A', 'B'), ('B', 'C'), ('C', 'D')]
        for l, r in actual_deps:
            linker.dependency(l, r)
        assert linker.find_cycles() is None
