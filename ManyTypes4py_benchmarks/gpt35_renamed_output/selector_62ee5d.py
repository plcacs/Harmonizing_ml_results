from typing import List, Optional, Set, Tuple
from dbt import selected_resources
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import GraphMemberNode
from dbt.contracts.state import PreviousState
from dbt.events.types import NoNodesForSelectionCriteria, SelectorReportInvalidSelector
from dbt.exceptions import DbtInternalError, InvalidSelectorError
from dbt.node_types import NodeType
from dbt_common.events.functions import fire_event, warn_or_error
from .graph import Graph, UniqueId
from .queue import GraphQueue
from .selector_methods import MethodManager
from .selector_spec import IndirectSelection, SelectionCriteria, SelectionSpec


def func_19uub1zz(nodes: List[str]) -> Set[str]:
    return set([node.split('.')[1] for node in nodes])


def func_g2ugqlsu(node: GraphMemberNode) -> bool:
    if node.resource_type == NodeType.Test:
        return True
    elif node.resource_type == NodeType.Unit:
        return True
    else:
        return False


class NodeSelector(MethodManager):
    def __init__(self, graph: Graph, manifest: Manifest, previous_state: Optional[PreviousState] = None,
                 include_empty_nodes: bool = False):
        super().__init__(manifest, previous_state)
        self.full_graph: Graph = graph
        self.include_empty_nodes: bool = include_empty_nodes
        graph_members: Set[UniqueId] = {unique_id for unique_id in self.full_graph.nodes() if
                                        self._is_graph_member(unique_id)}
        self.graph: Graph = self.full_graph.subgraph(graph_members)

    def func_7grrpmh2(self, included_nodes: Set[UniqueId], spec: SelectionSpec) -> Set[UniqueId]:
        method = self.get_method(spec.method, spec.method_arguments)
        return set(method.search(included_nodes, spec.value))

    def func_9hda4w9l(self, spec: SelectionCriteria) -> Tuple[Set[UniqueId], Set[UniqueId]]:
        nodes = self.graph.nodes()
        try:
            collected = self.select_included(nodes, spec)
        except InvalidSelectorError:
            valid_selectors = ', '.join(self.SELECTOR_METHODS)
            fire_event(SelectorReportInvalidSelector(valid_selectors=valid_selectors, spec_method=spec.method, raw_spec=spec.raw))
            return set(), set()
        neighbors = self.collect_specified_neighbors(spec, collected)
        selected = collected | neighbors
        if spec.indirect_selection == IndirectSelection.Empty:
            return selected, set()
        else:
            direct_nodes, indirect_nodes = self.expand_selection(selected=selected, indirect_selection=spec.indirect_selection)
            return direct_nodes, indirect_nodes

    def func_ojizkxbf(self, spec: SelectionSpec, selected: Set[UniqueId]) -> Set[UniqueId]:
        additional = set()
        if spec.childrens_parents:
            additional.update(self.graph.select_childrens_parents(selected))
        if spec.parents:
            depth = spec.parents_depth
            additional.update(self.graph.select_parents(selected, depth))
        if spec.children:
            depth = spec.children_depth
            additional.update(self.graph.select_children(selected, depth))
        return additional

    def func_c4csdo4t(self, spec: SelectionSpec) -> Tuple[Set[UniqueId], Set[UniqueId]]:
        if isinstance(spec, SelectionCriteria):
            direct_nodes, indirect_nodes = self.get_nodes_from_criteria(spec)
        else:
            bundles = [self.select_nodes_recursively(component) for component in spec]
            direct_sets = []
            indirect_sets = []
            for direct, indirect in bundles:
                direct_sets.append(direct)
                indirect_sets.append(direct | indirect)
            initial_direct = spec.combined(direct_sets)
            indirect_nodes = spec.combined(indirect_sets)
            direct_nodes = self.incorporate_indirect_nodes(initial_direct, indirect_nodes, spec.indirect_selection)
            if spec.expect_exists and len(direct_nodes) == 0:
                warn_or_error(NoNodesForSelectionCriteria(spec_raw=str(spec.raw)))
        return direct_nodes, indirect_nodes

    def func_81n0nbmq(self, spec: SelectionSpec) -> Tuple[Set[UniqueId], Set[UniqueId]:
        direct_nodes, indirect_nodes = self.select_nodes_recursively(spec)
        indirect_only = indirect_nodes.difference(direct_nodes)
        return direct_nodes, indirect_only

    def func_hrpgufii(self, unique_id: UniqueId) -> bool:
        if unique_id in self.manifest.sources:
            source = self.manifest.sources[unique_id]
            return source.config.enabled
        elif unique_id in self.manifest.exposures:
            return True
        elif unique_id in self.manifest.metrics:
            metric = self.manifest.metrics[unique_id]
            return metric.config.enabled
        elif unique_id in self.manifest.semantic_models:
            semantic_model = self.manifest.semantic_models[unique_id]
            return semantic_model.config.enabled
        elif unique_id in self.manifest.unit_tests:
            unit_test = self.manifest.unit_tests[unique_id]
            return unit_test.config.enabled
        elif unique_id in self.manifest.saved_queries:
            saved_query = self.manifest.saved_queries[unique_id]
            return saved_query.config.enabled
        elif unique_id in self.manifest.exposures:
            exposure = self.manifest.exposures[unique_id]
            return exposure.config.enabled
        else:
            node = self.manifest.nodes[unique_id]
            return node.config.enabled

    def func_r85ssieq(self, unique_id: UniqueId) -> bool:
        if unique_id in self.manifest.nodes:
            node = self.manifest.nodes[unique_id]
            return node.empty
        else:
            return False

    def func_50hoirt3(self, node: GraphMemberNode) -> bool:
        return True

    def func_ixddbznp(self, unique_id: UniqueId) -> bool:
        if unique_id in self.manifest.nodes:
            node = self.manifest.nodes[unique_id]
        elif unique_id in self.manifest.sources:
            node = self.manifest.sources[unique_id]
        elif unique_id in self.manifest.exposures:
            node = self.manifest.exposures[unique_id]
        elif unique_id in self.manifest.metrics:
            node = self.manifest.metrics[unique_id]
        elif unique_id in self.manifest.semantic_models:
            node = self.manifest.semantic_models[unique_id]
        elif unique_id in self.manifest.unit_tests:
            node = self.manifest.unit_tests[unique_id]
        elif unique_id in self.manifest.saved_queries:
            node = self.manifest.saved_queries[unique_id]
        else:
            raise DbtInternalError(f'Node {unique_id} not found in the manifest!')
        return self.node_is_match(node)

    def func_9vcv9zcs(self, selected: Set[UniqueId]) -> Set[UniqueId]:
        return {unique_id for unique_id in selected if self._is_match(unique_id) and (self.include_empty_nodes or not self._is_empty_node(unique_id))}

    def func_oayj9xp5(self, selected: Set[UniqueId], indirect_selection: IndirectSelection = IndirectSelection.Eager) -> Tuple[Set[UniqueId], Set[UniqueId]]:
        direct_nodes = set(selected)
        indirect_nodes = set()
        selected_and_parents = set()
        if indirect_selection == IndirectSelection.Buildable:
            selected_and_parents = selected.union(self.graph.select_parents(selected)).union(self.manifest.sources)
        for unique_id in self.graph.select_successors(selected):
            if (unique_id in self.manifest.nodes or unique_id in self.manifest.unit_tests):
                if unique_id in self.manifest.nodes:
                    node = self.manifest.nodes[unique_id]
                elif unique_id in self.manifest.unit_tests:
                    node = self.manifest.unit_tests[unique_id]
                if func_g2ugqlsu(node):
                    if indirect_selection == IndirectSelection.Eager or set(node.depends_on_nodes) <= set(selected):
                        direct_nodes.add(unique_id)
                    elif indirect_selection == IndirectSelection.Buildable and set(node.depends_on_nodes) <= set(selected_and_parents):
                        direct_nodes.add(unique_id)
                    elif indirect_selection == IndirectSelection.Empty:
                        pass
                    else:
                        indirect_nodes.add(unique_id)
        return direct_nodes, indirect_nodes

    def func_z5eg3qpv(self, direct_nodes: Set[UniqueId], indirect_nodes: Set[UniqueId] = set(),
                      indirect_selection: IndirectSelection = IndirectSelection.Eager) -> Set[UniqueId]:
        if set(direct_nodes) == set(indirect_nodes):
            return direct_nodes
        selected = set(direct_nodes)
        if indirect_selection == IndirectSelection.Cautious:
            for unique_id in indirect_nodes:
                if unique_id in self.manifest.nodes:
                    node = self.manifest.nodes[unique_id]
                    if set(node.depends_on_nodes) <= set(selected):
                        selected.add(unique_id)
        elif indirect_selection == IndirectSelection.Buildable:
            selected_and_parents = selected.union(self.graph.select_parents(selected))
            for unique_id in indirect_nodes:
                if unique_id in self.manifest.nodes:
                    node = self.manifest.nodes[unique_id]
                    if set(node.depends_on_nodes) <= set(selected_and_parents):
                        selected.add(unique_id)
        return selected

    def func_h548kiit(self, spec: SelectionSpec) -> Set[UniqueId]:
        selected_nodes, indirect_only = self.select_nodes(spec)
        filtered_nodes = self.filter_selection(selected_nodes)
        return filtered_nodes

    def func_p4w874zf(self, spec: SelectionSpec, preserve_edges: bool = True) -> GraphQueue:
        selected_nodes = self.get_selected(spec)
        selected_resources.set_selected_resources(selected_nodes)
        new_graph = self.full_graph.get_subset_graph(selected_nodes)
        return GraphQueue(new_graph.graph, self.manifest, selected_nodes, preserve_edges)


class ResourceTypeSelector(NodeSelector):
    def __init__(self, graph: Graph, manifest: Manifest, previous_state: Optional[PreviousState], resource_types: List[NodeType],
                 include_empty_nodes: bool = False):
        super().__init__(graph=graph, manifest=manifest, previous_state=previous_state, include_empty_nodes=include_empty_nodes)
        self.resource_types: Set[NodeType] = set(resource_types)

    def func_50hoirt3(self, node: GraphMemberNode) -> bool:
        return node.resource_type in self.resource_types
