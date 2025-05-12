import threading
from queue import PriorityQueue
from typing import Dict, Generator, List, Optional, Set, Tuple
import networkx as nx
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import Exposure, GraphMemberNode, Metric, SourceDefinition
from dbt.node_types import NodeType
from .graph import UniqueId


class GraphQueue:
    """A fancy queue that is backed by the dependency graph.
    Note: this will mutate input!

    This queue is thread-safe for `mark_done` calls, though you must ensure
    that separate threads do not call `.empty()` or `__len__()` and `.get()` at
    the same time, as there is an unlocked race!
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        manifest: Manifest,
        selected: Set[str],
        preserve_edges: bool = True,
    ) -> None:
        self.graph: nx.DiGraph = graph if preserve_edges else nx.classes.function.create_empty_copy(graph)
        self.manifest: Manifest = manifest
        self._selected: Set[str] = selected
        self.inner: PriorityQueue = PriorityQueue()
        self.in_progress: Set[str] = set()
        self.queued: Set[str] = set()
        self.lock: threading.Lock = threading.Lock()
        self._scores: Dict[str, int] = self._get_scores(self.graph)
        self._find_new_additions(list(self.graph.nodes()))
        self.some_task_done: threading.Condition = threading.Condition(self.lock)

    def get_selected_nodes(self) -> Set[str]:
        return self._selected.copy()

    def _include_in_cost(self, node_id: str) -> bool:
        node: GraphMemberNode = self.manifest.expect(node_id)
        if node.resource_type != NodeType.Model:
            return False
        assert not isinstance(node, (SourceDefinition, Exposure, Metric))
        if node.is_ephemeral:
            return False
        return True

    @staticmethod
    def _grouped_topological_sort(graph: nx.DiGraph) -> Generator[List[str], None, None]:
        """Topological sort of given graph that groups ties.

        Adapted from `nx.topological_sort`, this function returns a topo sort of a graph however
        instead of arbitrarily ordering ties in the sort order, ties are grouped together in
        lists.

        Args:
            graph: The graph to be sorted.

        Returns:
            A generator that yields lists of nodes, one list per graph depth level.
        """
        indegree_map: Dict[str, int] = {v: d for v, d in graph.in_degree() if d > 0}
        zero_indegree: List[str] = [v for v, d in graph.in_degree() if d == 0]
        while zero_indegree:
            yield zero_indegree
            new_zero_indegree: List[str] = []
            for v in zero_indegree:
                for _, child in graph.edges(v):
                    indegree_map[child] -= 1
                    if not indegree_map[child]:
                        new_zero_indegree.append(child)
            zero_indegree = new_zero_indegree

    def _get_scores(self, graph: nx.DiGraph) -> Dict[str, int]:
        """Scoring nodes for processing order.

        Scores are calculated by the graph depth level. Lowest score (0) should be processed first.

        Args:
            graph: The graph to be scored.

        Returns:
            A dictionary consisting of `node name`:`score` pairs.
        """
        subgraphs: Generator[nx.DiGraph, None, None] = (
            graph.subgraph(x) for x in nx.connected_components(nx.Graph(graph))
        )
        scores: Dict[str, int] = {}
        for subgraph in subgraphs:
            grouped_nodes: Generator[List[str], None, None] = self._grouped_topological_sort(subgraph)
            for level, group in enumerate(grouped_nodes):
                for node in group:
                    scores[node] = level
        return scores

    def get(
        self, block: bool = True, timeout: Optional[float] = None
    ) -> GraphMemberNode:
        """Get a node off the inner priority queue. By default, this blocks.

        This takes the lock, but only for part of it.

        :param block: If True, block until the inner queue has data
        :param timeout: If set, block for timeout seconds waiting for data.
        :return: The node as present in the manifest.

        See `queue.PriorityQueue` for more information on `get()` behavior and
        exceptions.
        """
        _, node_id: Tuple[int, str] = self.inner.get(block=block, timeout=timeout)
        with self.lock:
            self._mark_in_progress(node_id)
        return self.manifest.expect(node_id)

    def __len__(self) -> int:
        """The length of the queue is the number of tasks left for the queue to
        give out, regardless of where they are. Incomplete tasks are not part
        of the length.

        This takes the lock.
        """
        with self.lock:
            return len(self.graph) - len(self.in_progress)

    def empty(self) -> bool:
        """The graph queue is 'empty' if it all remaining nodes in the graph
        are in progress.

        This takes the lock.
        """
        return len(self) == 0

    def _already_known(self, node: str) -> bool:
        """Decide if a node is already known (either handed out as a task, or
        in the queue.

        Callers must hold the lock.

        :param str node: The node ID to check
        :returns bool: If the node is in progress/queued.
        """
        return node in self.in_progress or node in self.queued

    def _find_new_additions(self, candidates: List[str]) -> None:
        """Find any nodes in the graph that need to be added to the internal
        queue and add them.
        """
        for node in candidates:
            if self.graph.in_degree(node) == 0 and (not self._already_known(node)):
                self.inner.put((self._scores[node], node))
                self.queued.add(node)

    def mark_done(self, node_id: str) -> None:
        """Given a node's unique ID, mark it as done.

        This method takes the lock.

        :param str node_id: The node ID to mark as complete.
        """
        with self.lock:
            self.in_progress.remove(node_id)
            successors: List[str] = list(self.graph.successors(node_id))
            self.graph.remove_node(node_id)
            self._find_new_additions(successors)
            self.inner.task_done()
            self.some_task_done.notify_all()

    def _mark_in_progress(self, node_id: str) -> None:
        """Mark the node as 'in progress'.

        Callers must hold the lock.

        :param str node_id: The node ID to mark as in progress.
        """
        self.queued.remove(node_id)
        self.in_progress.add(node_id)

    def join(self) -> None:
        """Join the queue. Blocks until all tasks are marked as done.

        Make sure not to call this before the queue reports that it is empty.
        """
        self.inner.join()

    def wait_until_something_was_done(self) -> int:
        """Block until a task is done, then return the number of unfinished
        tasks.
        """
        with self.lock:
            self.some_task_done.wait()
            return self.inner.unfinished_tasks
