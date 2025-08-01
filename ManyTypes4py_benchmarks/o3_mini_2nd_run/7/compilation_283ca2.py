#!/usr/bin/env python3
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import dataclasses
import json
import os
import pickle
from collections import defaultdict, deque

import networkx as nx
import sqlparse
import dbt.tracking
from dbt.adapters.factory import get_adapter
from dbt.clients import jinja
from dbt.context.providers import generate_runtime_model_context, generate_runtime_unit_test_context
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

graph_file_name: str = 'graph.gpickle'


def print_compile_stats(stats: Dict[Any, int]) -> None:
    if dbt.tracking.active_user is not None:
        resource_counts: Dict[str, int] = {k.pluralize(): v for k, v in stats.items()}
        dbt.tracking.track_resource_counts(resource_counts)
    stat_line: str = ', '.join([pluralize(ct, t).replace('_', ' ') for t, ct in stats.items() if ct != 0])
    fire_event(FoundStats(stat_line=stat_line))


def _node_enabled(node: GraphMemberNode) -> bool:
    if node.resource_type == NodeType.Test and (not node.config.enabled):
        return False
    else:
        return True


def _generate_stats(manifest: Manifest) -> Dict[Any, int]:
    stats: Dict[Any, int] = defaultdict(int)
    for node in manifest.nodes.values():
        if _node_enabled(node):
            stats[node.resource_type] += 1
    stats[NodeType.Source] += len(manifest.sources)
    stats[NodeType.Exposure] += len(manifest.exposures)
    stats[NodeType.Metric] += len(manifest.metrics)
    stats[NodeType.Macro] += len(manifest.macros)
    stats[NodeType.Group] += len(manifest.groups)
    stats[NodeType.SemanticModel] += len(manifest.semantic_models)
    stats[NodeType.SavedQuery] += len(manifest.saved_queries)
    stats[NodeType.Unit] += len(manifest.unit_tests)
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


def _get_tests_for_node(manifest: Manifest, unique_id: UniqueID) -> List[str]:
    """Get a list of tests that depend on the node with the
    provided unique id"""
    tests: List[str] = []
    if unique_id in manifest.child_map:
        for child_unique_id in manifest.child_map[unique_id]:
            if child_unique_id.startswith('test.'):
                tests.append(child_unique_id)
    return tests


@dataclasses.dataclass
class SeenDetails:
    node_id: str
    visits: int = 0
    ancestors: Set[str] = dataclasses.field(default_factory=set)
    awaits_tests: Set[Tuple[str, Tuple[str, ...]]] = dataclasses.field(default_factory=set)


class Linker:
    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        if data is None:
            data = {}
        self.graph: nx.DiGraph = nx.DiGraph(**data)

    def edges(self) -> Iterable[Tuple[Any, Any]]:
        return self.graph.edges()

    def nodes(self) -> Iterable[Any]:
        return self.graph.nodes()

    def find_cycles(self) -> Optional[str]:
        try:
            cycle = nx.find_cycle(self.graph)
        except nx.NetworkXNoCycle:
            return None
        else:
            return ' --> '.join((c[0] for c in cycle))

    def dependency(self, node1: str, node2: str) -> None:
        """indicate that node1 depends on node2"""
        self.graph.add_node(node1)
        self.graph.add_node(node2)
        self.graph.add_edge(node2, node1)

    def add_node(self, node: str) -> None:
        self.graph.add_node(node)

    def write_graph(self, outfile: str, manifest: Manifest) -> None:
        """Write the graph to a gpickle file. Before doing so, serialize and
        include all nodes in their corresponding graph entries.
        """
        out_graph: nx.DiGraph = self.graph.copy()
        for node_id in self.graph:
            data: Dict[str, Any] = manifest.expect(node_id).to_dict(omit_none=True)
            out_graph.add_node(node_id, **data)
        with open(outfile, 'wb') as outfh:
            pickle.dump(out_graph, outfh, protocol=pickle.HIGHEST_PROTOCOL)

    def link_node(self, node: GraphMemberNode, manifest: Manifest) -> None:
        self.add_node(node.unique_id)
        for dependency_id in node.depends_on_nodes:
            if dependency_id in manifest.nodes:
                self.dependency(node.unique_id, manifest.nodes[dependency_id].unique_id)
            elif dependency_id in manifest.sources:
                self.dependency(node.unique_id, manifest.sources[dependency_id].unique_id)
            elif dependency_id in manifest.metrics:
                self.dependency(node.unique_id, manifest.metrics[dependency_id].unique_id)
            elif dependency_id in manifest.semantic_models:
                self.dependency(node.unique_id, manifest.semantic_models[dependency_id].unique_id)
            else:
                raise GraphDependencyNotFoundError(node, dependency_id)

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
            raise RuntimeError('Found a cycle: {}'.format(cycle))

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
        for node_id in self.graph:
            if node_id in manifest.nodes and manifest.nodes[node_id].resource_type != NodeType.Test:
                all_upstream_nodes = nx.traversal.bfs_tree(self.graph, node_id, reverse=True)
                upstream_nodes: Set[str] = set(n for n in all_upstream_nodes if n != node_id)
                upstream_tests: List[str] = []
                for upstream_node in upstream_nodes:
                    upstream_tests += _get_tests_for_node(manifest, upstream_node)
                for upstream_test in upstream_tests:
                    test_depends_on: Set[str] = set(manifest.nodes[upstream_test].depends_on_nodes)
                    if test_depends_on.issubset(upstream_nodes):
                        self.graph.add_edge(upstream_test, node_id, edge_type='parent_test')

    def add_test_edges_2(self, manifest: Manifest) -> None:
        new_edges = self._get_test_edges_2(self.graph, manifest)
        for e in new_edges:
            self.graph.add_edge(e[0], e[1], edge_type='parent_test')

    @staticmethod
    def _get_test_edges_2(graph: nx.DiGraph, manifest: Manifest) -> List[Tuple[str, str]]:
        new_edges: List[Tuple[str, str]] = []
        source_nodes: List[str] = []
        executable_nodes: Set[str] = set()
        multi_tested_nodes: Set[str] = set()
        single_tested_nodes: Dict[str, List[str]] = defaultdict(list)
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
        for node_id, test_ids in single_tested_nodes.items():
            succs = [s for s in graph.successors(node_id) if s in executable_nodes]
            for succ_id in succs:
                for test_id in test_ids:
                    new_edges.append((test_id, succ_id))
        if len(multi_tested_nodes) > 0:
            multi_test_edges = Linker._get_multi_test_edges(graph, manifest, source_nodes, executable_nodes, multi_tested_nodes)
            new_edges += multi_test_edges
        return new_edges

    @staticmethod
    def _get_multi_test_edges(
        graph: nx.DiGraph,
        manifest: Manifest,
        source_nodes: List[str],
        executable_nodes: Set[str],
        multi_tested_nodes: Set[str],
    ) -> List[Tuple[str, str]]:
        new_edges: List[Tuple[str, str]] = []
        ready: deque[str] = deque(source_nodes)
        details: Dict[str, SeenDetails] = {node_id: SeenDetails(node_id) for node_id in source_nodes}
        while len(ready) > 0:
            curr_node_id = ready.pop()
            curr_details: SeenDetails = details[curr_node_id]
            test_ids: List[str] = _get_tests_for_node(manifest, curr_details.node_id)
            new_awaits_for_succs: Set[Tuple[str, Tuple[str, ...]]] = curr_details.awaits_tests.copy()
            for test_id in test_ids:
                deps = sorted(manifest.nodes[test_id].depends_on_nodes)
                if len(deps) > 1:
                    new_awaits_for_succs.add((test_id, tuple(deps)))
            for succ_id in [s for s in graph.successors(curr_details.node_id) if s in executable_nodes]:
                suc_details: Optional[SeenDetails] = details.get(succ_id, None)
                if suc_details is None:
                    suc_details = SeenDetails(succ_id)
                    details[succ_id] = suc_details
                suc_details.visits += 1
                suc_details.awaits_tests.update(new_awaits_for_succs)
                suc_details.ancestors.update(curr_details.ancestors)
                if curr_details.node_id in multi_tested_nodes:
                    suc_details.ancestors.add(curr_details.node_id)
                if suc_details.visits == graph.in_degree(succ_id):
                    if len(suc_details.awaits_tests) > 0:
                        removes: Set[Tuple[str, Tuple[str, ...]]] = set()
                        for awt in suc_details.awaits_tests:
                            if not any((True for a in awt[1] if a not in suc_details.ancestors)):
                                removes.add(awt)
                                new_edges.append((awt[0], succ_id))
                        suc_details.awaits_tests.difference_update(removes)
                    ready.appendleft(succ_id)
            del details[curr_details.node_id]
        return new_edges

    def get_graph(self, manifest: Manifest) -> Graph:
        self.link_graph(manifest)
        return Graph(self.graph)

    def get_graph_summary(self, manifest: Manifest) -> Dict[int, Dict[str, Any]]:
        """Create a smaller summary of the graph, suitable for basic diagnostics
        and performance tuning. The summary includes only the edge structure,
        node types, and node names. Each of the n nodes is assigned an integer
        index 0, 1, 2,..., n-1 for compactness"""
        graph_nodes: Dict[int, Dict[str, Any]] = dict()
        index_dict: Dict[str, int] = dict()
        for node_index, node_name in enumerate(self.graph):
            index_dict[node_name] = node_index
            data: Dict[str, Any] = manifest.expect(node_name).to_dict(omit_none=True)
            graph_nodes[node_index] = {'name': node_name, 'type': data['resource_type']}
        for node_index, node in graph_nodes.items():
            successors = [index_dict[n] for n in self.graph.successors(node['name'])]
            if successors:
                node['succ'] = successors
        return graph_nodes


class Compiler:
    def __init__(self, config: Any) -> None:
        self.config: Any = config

    def initialize(self) -> None:
        make_directory(self.config.project_target_path)

    def _create_node_context(
        self, node: GraphMemberNode, manifest: Manifest, extra_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if isinstance(node, UnitTestNode):
            context: Dict[str, Any] = generate_runtime_unit_test_context(node, self.config, manifest)
        else:
            context = generate_runtime_model_context(node, self.config, manifest)
        context.update(extra_context)
        if isinstance(node, GenericTestNode):
            jinja.add_rendered_test_kwargs(context, node)
        return context

    def add_ephemeral_prefix(self, name: str) -> str:
        adapter = get_adapter(self.config)
        relation_cls = adapter.Relation
        return relation_cls.add_ephemeral_prefix(name)

    def _recursively_prepend_ctes(
        self, model: ModelNode, manifest: Manifest, extra_context: Dict[str, Any]
    ) -> Tuple[ModelNode, List[InjectedCTE]]:
        """This method is called by the 'compile_node' method. Starting
        from the node that it is passed in, it will recursively call
        itself using the 'extra_ctes'.  The 'ephemeral' models do
        not produce SQL that is executed directly, instead they
        are rolled up into the models that refer to them by
        inserting CTEs into the SQL.
        """
        if model.compiled_code is None:
            raise DbtRuntimeError('Cannot inject ctes into an uncompiled node', model)
        if not getattr(self.config.args, 'inject_ephemeral_ctes', True):
            return model, []
        if model.extra_ctes_injected:
            return model, model.extra_ctes
        if len(model.extra_ctes) == 0:
            if not isinstance(model, SeedNode):
                model.extra_ctes_injected = True
            return model, []
        prepended_ctes: List[InjectedCTE] = []
        for cte in model.extra_ctes:
            if cte.id not in manifest.nodes:
                raise DbtInternalError(f'During compilation, found a cte reference that could not be resolved: {cte.id}')
            cte_model = manifest.nodes[cte.id]
            assert not isinstance(cte_model, SeedNode)
            if not cte_model.is_ephemeral_model:
                raise DbtInternalError(f'{cte.id} is not ephemeral')
            if cte_model.compiled is True and cte_model.extra_ctes_injected is True:
                new_prepended_ctes = cte_model.extra_ctes
            else:
                cte_model = self._compile_code(cte_model, manifest, extra_context)  # type: ignore
                cte_model, new_prepended_ctes = self._recursively_prepend_ctes(cte_model, manifest, extra_context)  # type: ignore
                self._write_node(cte_model)
            _extend_prepended_ctes(prepended_ctes, new_prepended_ctes)
            new_cte_name: str = self.add_ephemeral_prefix(cte_model.identifier)  # type: ignore
            rendered_sql: str = cte_model._pre_injected_sql or cte_model.compiled_code  # type: ignore
            sql: str = f' {new_cte_name} as (\n{rendered_sql}\n)'
            _add_prepended_cte(prepended_ctes, InjectedCTE(id=cte.id, sql=sql))
        if not model.extra_ctes_injected:
            injected_sql = inject_ctes_into_sql(model.compiled_code, prepended_ctes)
            model.extra_ctes_injected = True
            model._pre_injected_sql = model.compiled_code
            model.compiled_code = injected_sql
            model.extra_ctes = prepended_ctes
        return model, model.extra_ctes

    def _compile_code(
        self, node: GraphMemberNode, manifest: Manifest, extra_context: Optional[Dict[str, Any]] = None
    ) -> GraphMemberNode:
        if extra_context is None:
            extra_context = {}
        if node.language == ModelLanguage.python:
            context: Dict[str, Any] = self._create_node_context(node, manifest, extra_context)
            postfix: str = jinja.get_rendered('{{ py_script_postfix(model) }}', context, node)
            node.compiled_code = f'{node.raw_code}\n\n{postfix}'
        else:
            context = self._create_node_context(node, manifest, extra_context)
            node.compiled_code = jinja.get_rendered(node.raw_code, context, node)
        node.compiled = True
        if node.resource_type == NodeType.Test and node.relation_name is None and node.is_relational:
            adapter = get_adapter(self.config)
            relation_cls = adapter.Relation
            relation_name: str = str(relation_cls.create_from(self.config, node))
            node.relation_name = relation_name
        if isinstance(node, ModelNode):
            for constraint in node.all_constraints:
                if constraint.type == ConstraintType.foreign_key and constraint.to:
                    constraint.to = self._compile_relation_for_foreign_key_constraint_to(manifest, node, constraint.to)
        return node

    def _compile_relation_for_foreign_key_constraint_to(
        self, manifest: Manifest, node: ModelNode, to_expression: str
    ) -> str:
        try:
            foreign_key_node = manifest.find_node_from_ref_or_source(to_expression)
        except ParsingError:
            raise ForeignKeyConstraintToSyntaxError(node, to_expression)
        if not foreign_key_node:
            raise GraphDependencyNotFoundError(node, to_expression)
        adapter = get_adapter(self.config)
        relation_name: str = str(adapter.Relation.create_from(self.config, foreign_key_node))
        return relation_name

    def compile(
        self, manifest: Manifest, write: bool = True, add_test_edges: bool = False
    ) -> Graph:
        self.initialize()
        linker = Linker()
        linker.link_graph(manifest)
        summaries: Dict[str, Any] = dict()
        summaries['_invocation_id'] = get_invocation_id()
        summaries['linked'] = linker.get_graph_summary(manifest)
        if add_test_edges:
            manifest.build_parent_and_child_maps()
            linker.add_test_edges(manifest)
            summaries['with_test_edges'] = linker.get_graph_summary(manifest)
        with open(os.path.join(self.config.project_target_path, 'graph_summary.json'), 'w') as out_stream:
            try:
                out_stream.write(json.dumps(summaries))
            except Exception as e:
                fire_event(Note(msg=f'An error was encountered writing the graph summary information: {e}'))
        stats: Dict[Any, int] = _generate_stats(manifest)
        if write:
            self.write_graph_file(linker, manifest)
        if self.config.args.which != 'list':
            stats = _generate_stats(manifest)
            print_compile_stats(stats)
        return Graph(linker.graph)

    def write_graph_file(self, linker: Linker, manifest: Manifest) -> None:
        filename: str = graph_file_name
        graph_path: str = os.path.join(self.config.project_target_path, filename)
        flags = get_flags()
        if flags.WRITE_JSON:
            linker.write_graph(graph_path, manifest)

    def _write_node(self, node: GraphMemberNode, split_suffix: Optional[str] = None) -> GraphMemberNode:
        if not node.extra_ctes_injected or node.resource_type in (NodeType.Snapshot, NodeType.Seed):
            return node
        fire_event(WritingInjectedSQLForNode(node_info=get_node_info()))
        if node.compiled_code:
            node.compiled_path = node.get_target_write_path(self.config.target_path, 'compiled', split_suffix)
            node.write_node(self.config.project_root, node.compiled_path, node.compiled_code)
        return node

    def compile_node(
        self,
        node: GraphMemberNode,
        manifest: Manifest,
        extra_context: Optional[Dict[str, Any]] = None,
        write: bool = True,
        split_suffix: Optional[str] = None,
    ) -> GraphMemberNode:
        """This is the main entry point into this code. It's called by
        CompileRunner.compile, GenericRPCRunner.compile, and
        RunTask.get_hook_sql. It calls '_compile_code' to render
        the node's raw_code into compiled_code, and then calls the
        recursive method to "prepend" the ctes.
        """
        if isinstance(node, UnitTestDefinition):
            return node  # type: ignore
        from sqlparse.lexer import Lexer

        if hasattr(Lexer, 'get_default_instance'):
            Lexer.get_default_instance()
        node = self._compile_code(node, manifest, extra_context)
        node, _ = self._recursively_prepend_ctes(node, manifest, extra_context if extra_context is not None else {})
        if write:
            self._write_node(node, split_suffix=split_suffix)
        return node


def inject_ctes_into_sql(sql: str, ctes: List[InjectedCTE]) -> str:
    """
    `ctes` is a list of InjectedCTEs like:

        [
            InjectedCTE(
                id="cte_id_1",
                sql="__dbt__cte__ephemeral as (select * from table)",
            ),
            InjectedCTE(
                id="cte_id_2",
                sql="__dbt__cte__events as (select id, type from events)",
            ),
        ]

    Given `sql` like:

      "with internal_cte as (select * from sessions)
       select * from internal_cte"

    This will spit out:

      "with __dbt__cte__ephemeral as (select * from table),
            __dbt__cte__events as (select id, type from events),
            internal_cte as (select * from sessions)
       select * from internal_cte"

    (Whitespace enhanced for readability.)
    """
    if len(ctes) == 0:
        return sql
    parsed_stmts = sqlparse.parse(sql)
    parsed = parsed_stmts[0]
    with_stmt = None
    for token in parsed.tokens:
        if token.is_keyword and token.normalized == 'WITH':
            with_stmt = token
        elif token.is_keyword and token.normalized == 'RECURSIVE' and (with_stmt is not None):
            with_stmt = token
            break
        elif not token.is_whitespace and with_stmt is not None:
            break
    if with_stmt is None:
        first_token = parsed.token_first()
        with_token = sqlparse.sql.Token(sqlparse.tokens.Keyword, 'with')
        parsed.insert_before(first_token, with_token)
        injected_ctes = ', '.join((c.sql for c in ctes)) + ' '
        injected_ctes_token = sqlparse.sql.Token(sqlparse.tokens.Keyword, injected_ctes)
        parsed.insert_after(with_token, injected_ctes_token)
    else:
        injected_ctes = ', '.join((c.sql for c in ctes))
        injected_ctes_token = sqlparse.sql.Token(sqlparse.tokens.Keyword, injected_ctes)
        parsed.insert_after(with_stmt, injected_ctes_token)
        comma_token = sqlparse.sql.Token(sqlparse.tokens.Punctuation, ', ')
        parsed.insert_after(injected_ctes_token, comma_token)
    return str(parsed)