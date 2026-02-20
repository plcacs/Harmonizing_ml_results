import dataclasses
import json
import os
import pickle
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import networkx as nx  # type: ignore
import sqlparse

import dbt.tracking
from dbt.adapters.factory import get_adapter
from dbt.clients import jinja
from dbt.context.providers import (
    generate_runtime_model_context,
    generate_runtime_unit_test_context,
)
from dbt.contracts.graph.manifest import Manifest, UniqueID
from dbt.contracts.graph.nodes import (
    GenericTestNode,
    GraphMemberNode,
    InjectedCTE,
    ManifestNode,
    ManifestSQLNode,
    ModelNode,
    SeedNode,
    UnitTestDefinition,
    UnitTestNode,
)
from dbt.events.types import FoundStats, WritingInjectedSQLForNode
from dbt.exceptions import (
    DbtInternalError,
    DbtRuntimeError,
    ForeignKeyConstraintToSyntaxError,
    GraphDependencyNotFoundError,
    ParsingError,
)
from dbt.flags import get_flags
from dbt.graph import Graph
from dbt.node_types import ModelLanguage, NodeType
from dbt_common.clients.system import make_directory
from dbt_common.contracts.constraints import ConstraintType
from dbt_common.events.contextvars import get_node_info
from dbt_common.events.format import pluralize
from dbt_common.events.functions import fire_event
from dbt_common.events.types import Note
from dbt_common.invocation import get_invocation_id

graph_file_name = "graph.gpickle"


def print_compile_stats(stats: Dict[NodeType, int]) -> None:
    # create tracking event for resource_counts
    if dbt.tracking.active_user is not None:
        resource_counts = {k.pluralize(): v for k, v in stats.items()}
        dbt.tracking.track_resource_counts(resource_counts)

    # do not include resource types that are not actually defined in the project
    stat_line = ", ".join(
        [pluralize(ct, t).replace("_", " ") for t, ct in stats.items() if ct != 0]
    )
    fire_event(FoundStats(stat_line=stat_line))


def _node_enabled(node: ManifestNode) -> bool:
    # Disabled models are already excluded from the manifest
    if node.resource_type == NodeType.Test and not node.config.enabled:
        return False
    else:
        return True


def _generate_stats(manifest: Manifest) -> Dict[NodeType, int]:
    stats: Dict[NodeType, int] = defaultdict(int)
    for node in manifest.nodes.values():
        if _node_enabled(node):
            stats[node.resource_type] += 1

    # Disabled nodes don't appear in the following collections, so we don't check.
    stats[NodeType.Source] += len(manifest.sources)
    stats[NodeType.Exposure] += len(manifest.exposures)
    stats[NodeType.Metric] += len(manifest.metrics)
    stats[NodeType.Macro] += len(manifest.macros)
    stats[NodeType.Group] += len(manifest.groups)
    stats[NodeType.SemanticModel] += len(manifest.semantic_models)
    stats[NodeType.SavedQuery] += len(manifest.saved_queries)
    stats[NodeType.Unit] += len(manifest.unit_tests)

    # TODO: should we be counting dimensions + entities?

    return stats


def _add_prepended_cte(prepended_ctes: List[InjectedCTE], new_cte: InjectedCTE) -> None:
    for cte in prepended_ctes:
        if cte.id == new_cte.id and new_cte.sql:
            cte.sql = new_cte.sql
            return
    if new_cte.sql:
        prepended_ctes.append(new_cte)


def _extend_prepended_ctes(prepended_ctes: List[InjectedCTE], new_prepended_ctes: List[InjectedCTE]) -> None:
    for new_cte in new_prepended_ctes:
        _add_prepended_cte(prepended_ctes, new_cte)


def _get_tests_for_node(manifest: Manifest, unique_id: UniqueID) -> List[UniqueID]:
    """Get a list of tests that depend on the node with the
    provided unique id"""

    tests = []
    if unique_id in manifest.child_map:
        for child_unique_id in manifest.child_map[unique_id]:
            if child_unique_id.startswith("test."):
                tests.append(child_unique_id)

    return tests


@dataclasses.dataclass
class SeenDetails:
    node_id: UniqueID
    visits: int = 0
    ancestors: Set[UniqueID] = dataclasses.field(default_factory=set)
    awaits_tests: Set[Tuple[UniqueID, Tuple[UniqueID, ...]]] = dataclasses.field(
        default_factory=set
    )


class Linker:
    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        if data is None:
            data = {}
        self.graph: nx.DiGraph = nx.DiGraph(**data)

    def edges(self) -> Any:
        return self.graph.edges()

    def nodes(self) -> Any:
        return self.graph.nodes()

    def find_cycles(self) -> Optional[str]:
        try:
            cycle = nx.find_cycle(self.graph)
        except nx.NetworkXNoCycle:
            return None
        else:
            # cycles is a List[Tuple[str, ...]]
            return " --> ".join(c[0] for c in cycle)

    def dependency(self, node1: str, node2: str) -> None:
        "indicate that node1 depends on node2"
        self.graph.add_node(node1)
        self.graph.add_node(node2)
        self.graph.add_edge(node2, node1)

    def add_node(self, node: str) -> None:
        self.graph.add_node(node)

    def write_graph(self, outfile: str, manifest: Manifest) -> None:
        """Write the graph to a gpickle file. Before doing so, serialize and
        include all nodes in their corresponding graph entries.
        """
        out_graph = self.graph.copy()
        for node_id in self.graph:
            data = manifest.expect(node_id).to_dict(omit_none=True)
            out_graph.add_node(node_id, **data)
        with open(outfile, "wb") as outfh:
            pickle.dump(out_graph, outfh, protocol=pickle.HIGHEST_PROTOCOL)

    def link_node(self, node: GraphMemberNode, manifest: Manifest) -> None:
        self.add_node(node.unique_id)

        for dependency in node.depends_on_nodes:
            if dependency in manifest.nodes:
                self.dependency(node.unique_id, (manifest.nodes[dependency].unique_id))
            elif dependency in manifest.sources:
                self.dependency(node.unique_id, (manifest.sources[dependency].unique_id))
            elif dependency in manifest.metrics:
                self.dependency(node.unique_id, (manifest.metrics[dependency].unique_id))
            elif dependency in manifest.semantic_models:
                self.dependency(node.unique_id, (manifest.semantic_models[dependency].unique_id))
            else:
                raise GraphDependencyNotFoundError(node, dependency)

    def link_graph(self, manifest: Manifest) -> None:
        for source in manifest.sources.values():
            self.add_node(source.unique_id)
        for node in manifest.nodes.values():
            self.link_node(node, manifest)
        for semantic_model in manifest.semantic_models.values():
            self.link_node(semantic_model, manifest)
        for exposure in manifest.exposures.values():
            self.link_node(exposure, manifest)
        for metric in manifest.metrics.values():
            self.link_node(metric, manifest)
        for unit_test in manifest.unit_tests.values():
            self.link_node(unit_test, manifest)
        for saved_query in manifest.saved_queries.values():
            self.link_node(saved_query, manifest)

        cycle = self.find_cycles()

        if cycle:
            raise RuntimeError("Found a cycle: {}".format(cycle))

    def add_test_edges(self, manifest: Manifest) -> None:
        if not get_flags().USE_FAST_TEST_EDGES:
            self.add_test_edges_1(manifest)
        else:
            self.add_test_edges_2(manifest)

    def add_test_edges_1(self, manifest: Manifest) -> None:
        """This method adds additional edges to the DAG. For a given non-test
        executable node, add an edge from an upstream test to the given node if
        the set of nodes the test depends on is a subset of the upstream nodes
        for the given node."""

        # HISTORICAL NOTE: To understand the motivation behind this function,
        # consider a node A with tests and a node B which depends (either directly
        # or indirectly) on A. It would be nice if B were not executed until
        # all of the tests on A are finished. After all, we don't want to
        # propagate bad data. We can enforce that behavior by adding new
        # dependencies (edges) from tests to nodes that should wait on them.
        #
        # This function implements a rough approximation of the behavior just
        # described. In fact, for tests that only depend on a single node, it
        # always works.
        #
        # Things get trickier for tests that depend on multiple nodes. In that
        # case, if we are not careful, we will introduce cycles. That seems to
        # be the reason this function adds dependencies from a downstream node to
        # an upstream test if and only if the downstream node is already a
        # descendant of all the nodes the upstream test depends on. By following
        # that rule, it never makes the node dependent on new upstream nodes other
        # than the tests themselves, and no cycles will be created.
        #
        # One drawback (Drawback 1) of the approach taken in this function is
        # that it could still allow a downstream node to proceed before all
        # testing is done on its ancestors, if it happens to have ancestors that
        # are not also ancestors of a test with multiple dependencies.
        #
        # Another drawback (Drawback 2) is that the approach below adds far more
        # edges than are strictly needed. After all, if we have A -> B -> C,
        # there is no need to add a new edge A -> C. But this function often does.
        #
        # Drawback 2 is resolved in the new add_test_edges_2() implementation
        # below, which is also typically much faster. Drawback 1 has been left in
        # place in order to conservatively retain existing behavior, and so that
        # the new implementation can be verified against this existing
        # implementation by ensuring both resulting graphs have the same transitive
        # reduction.

        # MOTIVATING IDEA: Given a graph...
        #
        # model1 --> model2 --> model3
        #   |             |
        #   |            \/
        #  \/          test 2
        # test1
        #
        # ...produce the following...
        #
        # model1 --> model2 --> model3
        #   |       /\    |      /\ /\
        #   |       |    \/      |  |
        #  \/       |  test2 ----|  |
        # test1 ----|---------------|

        for node_id in self.graph:
            # If node is executable (in manifest.nodes) and does _not_
            # represent a test, continue.
            if (
                node_id in manifest.nodes
                and manifest.nodes[node_id].resource_type != NodeType.Test
            ):
                # Get *everything* upstream of the node
                all_upstream_nodes = nx.traversal.bfs_tree(self.graph, node_id, reverse=True)
                # Get the set of upstream nodes not including the current node.
                upstream_nodes = set([n for n in all_upstream_nodes if n != node_id])

                # Get all tests that depend on any upstream nodes.
                upstream_tests = []
                for upstream_node in upstream_nodes:
                    # This gets tests with unique_ids starting with "test."
                    upstream_tests += _get_tests_for_node(manifest, upstream_node)

                for upstream_test in upstream_tests:
                    # Get the set of all nodes that the test depends on
                    # including the upstream_node itself. This is necessary
                    # because tests can depend on multiple nodes (ex:
                    # relationship tests). Test nodes do not distinguish
                    # between what node the test is "testing" and what
                    # node(s) it depends on.
                    test_depends_on = set(manifest.nodes[upstream_test].depends_on_nodes)

                    # If the set of nodes that an upstream test depends on
                    # is a subset of all upstream nodes of the current node,
                    # add an edge from the upstream test to the current node.
                    if test_depends_on.issubset(upstream_nodes):
                        self.graph.add_edge(upstream_test, node_id, edge_type="parent_test")

    def add_test_edges_2(self, manifest: Manifest) -> None:
        graph = self.graph
        new_edges = self._get_test_edges_2(graph, manifest)
        for e in new_edges:
            graph.add_edge(e[0], e[1], edge_type="parent_test")

    @staticmethod
    def _get_test_edges_2(
        graph: nx.DiGraph, manifest: Manifest
    ) -> Iterable[Tuple[UniqueID, UniqueID]]:
        # This function enforces the same execution behavior as add_test_edges,
        # but executes far more quickly and adds far fewer edges. See the
        # HISTORICAL NOTE above.
        #
        # The idea is to first scan for "single-tested" nodes (which have tests
        # that depend only upon on that node) and "multi-tested" nodes (which
        # have tests that depend on multiple nodes). Single-tested nodes are
        # handled quickly and easily.
        #
        # The less common but more complex case of multi-tested nodes is handled
        # by a specialized function.

        new_edges: List[Tuple[UniqueID, UniqueID]] = []

        source_nodes: List[UniqueID] = []
        executable_nodes: Set[UniqueID] = set()
        multi_tested_nodes: Set[UniqueID] = set()
        # Dictionary mapping nodes with single-dep tests to a list of those tests.
        single_tested_nodes: Dict[UniqueID, List[UniqueID]] = defaultdict(list)
        for node_id in graph.nodes:
            manifest_node = manifest.nodes.get(node_id, None)
            if manifest_node is None:
                continue

            if next(graph.predecessors(node_id), None) is None:
                source_nodes.append(node_id)

            if manifest_node.resource_type != NodeType.Test:
                executable_nodes.add(node_id)
            else:
                test_deps = manifest_node.depends_on_nodes
                if len(test_deps) == 1:
                    single_tested_nodes[test_deps[0]].append(node_id)
                elif len(test_deps) > 1:
                    multi_tested_nodes.update(manifest_node.depends_on_nodes)

        # Now that we have all the necessary information conveniently organized,
        # add new edges for single-tested nodes.
        for node_id, test_ids in single_tested_nodes.items():
            succs = [s for s in graph.successors(node_id) if s in executable_nodes]
            for succ_id in succs:
                for test_id in test_ids:
                    new_edges.append((test_id, succ_id))

        # Get the edges for multi-tested nodes separately, if needed.
        if len(multi_tested_nodes) > 0:
            multi_test_edges = Linker._get_multi_test_edges(
                graph, manifest, source_nodes, executable_nodes, multi_tested_nodes
            )
            new_edges += multi_test_edges

        return new_edges

    @staticmethod
    def _get_multi_test_edges(
        graph: nx.DiGraph,
        manifest: Manifest,
        source_nodes: Iterable[UniqueID],
        executable_nodes: Set[UniqueID],
        multi_tested_nodes: Set[UniqueID],
    ) -> List[Tuple[UniqueID, UniqueID]]:
        # Works through the graph in a breadth-first style, processing nodes from
        # a ready queue which initially consists of nodes with no ancestors,
        # and adding more nodes to the ready queue after all their ancestors
        # have been processed. All the while, the relevant details of all nodes
        # "seen" by the search so far are maintained in a SeenDetails record,
        # including the ancestor set which tests it is "awaiting" (i.e. tests of
        # its ancestors). The processing step adds test edges when every dependency
        # of an awaited test is an ancestor of a node that is being processed.
        # Downstream nodes are then exempted from awaiting the test.
        #
        # Memory consumption is potentially O(n^2) with n the number of nodes in
        # the graph, since the average number of ancestors and tests being awaited
        # for each of the n nodes could itself be O(n) but we only track ancestors
        # that are multi-tested, which should keep things closer to O(n) in real-
        # world scenarios.

        new_edges: List[Tuple[UniqueID, UniqueID]] = []
        ready: deque = deque(source_nodes)
        details: Dict[UniqueID, SeenDetails] = {node_id: SeenDetails(node_id) for node_id in source_nodes}

        while len(ready) > 0:
            curr_details: SeenDetails = details[ready.pop()]
            test_ids = _get_tests_for_node(manifest, curr_details.node_id)
            new_awaits_for_succs = curr_details.awaits_tests.copy()
            for test_id in test_ids:
                deps: List[UniqueID] = sorted(manifest.nodes[test_id].depends_on_nodes)
                if len(deps) > 1:
                    # Tests with only one dep were already handled.
                    new_awaits_for_succs.add((test_id, tuple(deps)))

            for succ_id in [
                s for s in graph.successors(curr_details.node_id) if s in executable_nodes
            ]:
                suc_details = details.get(succ_id, None)
                if suc_details is None:
                    suc_details = SeenDetails(succ_id)
                    details[succ_id] = suc_details
                suc_details.visits +=