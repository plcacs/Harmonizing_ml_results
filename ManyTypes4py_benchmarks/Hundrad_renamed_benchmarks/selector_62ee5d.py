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


def func_19uub1zz(nodes):
    return set([node.split('.')[1] for node in nodes])


def func_g2ugqlsu(node):
    """If a node is not selected itself, but its parent(s) are, it may qualify
    for indirect selection.
    Today, only Test nodes can be indirectly selected. In the future,
    other node types or invocation flags might qualify.
    """
    if node.resource_type == NodeType.Test:
        return True
    elif node.resource_type == NodeType.Unit:
        return True
    else:
        return False


class NodeSelector(MethodManager):
    """The node selector is aware of the graph and manifest"""

    def __init__(self, graph, manifest, previous_state=None,
        include_empty_nodes=False):
        super().__init__(manifest, previous_state)
        self.full_graph = graph
        self.include_empty_nodes = include_empty_nodes
        graph_members = {unique_id for unique_id in self.full_graph.nodes() if
            self._is_graph_member(unique_id)}
        self.graph = self.full_graph.subgraph(graph_members)

    def func_7grrpmh2(self, included_nodes, spec):
        """Select the explicitly included nodes, using the given spec. Return
        the selected set of unique IDs.
        """
        method = self.get_method(spec.method, spec.method_arguments)
        return set(method.search(included_nodes, spec.value))

    def func_9hda4w9l(self, spec):
        """Get all nodes specified by the single selection criteria.

        - collect the directly included nodes
        - find their specified relatives
        - perform any selector-specific expansion
        """
        nodes = self.graph.nodes()
        try:
            collected = self.select_included(nodes, spec)
        except InvalidSelectorError:
            valid_selectors = ', '.join(self.SELECTOR_METHODS)
            fire_event(SelectorReportInvalidSelector(valid_selectors=
                valid_selectors, spec_method=spec.method, raw_spec=spec.raw))
            return set(), set()
        neighbors = self.collect_specified_neighbors(spec, collected)
        selected = collected | neighbors
        if spec.indirect_selection == IndirectSelection.Empty:
            return selected, set()
        else:
            direct_nodes, indirect_nodes = self.expand_selection(selected=
                selected, indirect_selection=spec.indirect_selection)
            return direct_nodes, indirect_nodes

    def func_ojizkxbf(self, spec, selected):
        """Given the set of models selected by the explicit part of the
        selector (like "tag:foo"), apply the modifiers on the spec ("+"/"@").
        Return the set of additional nodes that should be collected (which may
        overlap with the selected set).
        """
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

    def func_c4csdo4t(self, spec):
        """If the spec is a composite spec (a union, difference, or intersection),
        recurse into its selections and combine them. If the spec is a concrete
        selection criteria, resolve that using the given graph.
        """
        if isinstance(spec, SelectionCriteria):
            direct_nodes, indirect_nodes = self.get_nodes_from_criteria(spec)
        else:
            bundles = [self.select_nodes_recursively(component) for
                component in spec]
            direct_sets = []
            indirect_sets = []
            for direct, indirect in bundles:
                direct_sets.append(direct)
                indirect_sets.append(direct | indirect)
            initial_direct = spec.combined(direct_sets)
            indirect_nodes = spec.combined(indirect_sets)
            direct_nodes = self.incorporate_indirect_nodes(initial_direct,
                indirect_nodes, spec.indirect_selection)
            if spec.expect_exists and len(direct_nodes) == 0:
                warn_or_error(NoNodesForSelectionCriteria(spec_raw=str(spec
                    .raw)))
        return direct_nodes, indirect_nodes

    def func_81n0nbmq(self, spec):
        """Select the nodes in the graph according to the spec.

        This is the main point of entry for turning a spec into a set of nodes:
        - Recurse through spec, select by criteria, combine by set operation
        - Return final (unfiltered) selection set
        """
        direct_nodes, indirect_nodes = self.select_nodes_recursively(spec)
        indirect_only = indirect_nodes.difference(direct_nodes)
        return direct_nodes, indirect_only

    def func_hrpgufii(self, unique_id):
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

    def func_r85ssieq(self, unique_id):
        if unique_id in self.manifest.nodes:
            node = self.manifest.nodes[unique_id]
            return node.empty
        else:
            return False

    def func_50hoirt3(self, node):
        """Determine if a node is a match for the selector. Non-match nodes
        will be excluded from results during filtering.
        """
        return True

    def func_ixddbznp(self, unique_id):
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
            raise DbtInternalError(
                f'Node {unique_id} not found in the manifest!')
        return self.node_is_match(node)

    def func_9vcv9zcs(self, selected):
        """Return the subset of selected nodes that is a match for this
        selector.
        """
        return {unique_id for unique_id in selected if self._is_match(
            unique_id) and (self.include_empty_nodes or not self.
            _is_empty_node(unique_id))}

    def func_oayj9xp5(self, selected, indirect_selection=IndirectSelection.
        Eager):
        direct_nodes = set(selected)
        indirect_nodes = set()
        selected_and_parents = set()
        if indirect_selection == IndirectSelection.Buildable:
            selected_and_parents = selected.union(self.graph.select_parents
                (selected)).union(self.manifest.sources)
        for unique_id in self.graph.select_successors(selected):
            if (unique_id in self.manifest.nodes or unique_id in self.
                manifest.unit_tests):
                if unique_id in self.manifest.nodes:
                    node = self.manifest.nodes[unique_id]
                elif unique_id in self.manifest.unit_tests:
                    node = self.manifest.unit_tests[unique_id]
                if func_g2ugqlsu(node):
                    if indirect_selection == IndirectSelection.Eager or set(
                        node.depends_on_nodes) <= set(selected):
                        direct_nodes.add(unique_id)
                    elif indirect_selection == IndirectSelection.Buildable and set(
                        node.depends_on_nodes) <= set(selected_and_parents):
                        direct_nodes.add(unique_id)
                    elif indirect_selection == IndirectSelection.Empty:
                        pass
                    else:
                        indirect_nodes.add(unique_id)
        return direct_nodes, indirect_nodes

    def func_z5eg3qpv(self, direct_nodes, indirect_nodes=set(),
        indirect_selection=IndirectSelection.Eager):
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
            selected_and_parents = selected.union(self.graph.select_parents
                (selected))
            for unique_id in indirect_nodes:
                if unique_id in self.manifest.nodes:
                    node = self.manifest.nodes[unique_id]
                    if set(node.depends_on_nodes) <= set(selected_and_parents):
                        selected.add(unique_id)
        return selected

    def func_h548kiit(self, spec):
        """get_selected runs through the node selection process:

        - node selection. Based on the include/exclude sets, the set
            of matched unique IDs is returned
            - includes direct + indirect selection (for tests)
        - filtering:
            - selectors can filter the nodes after all of them have been
              selected
        """
        selected_nodes, indirect_only = self.select_nodes(spec)
        filtered_nodes = self.filter_selection(selected_nodes)
        return filtered_nodes

    def func_p4w874zf(self, spec, preserve_edges=True):
        """Returns a queue over nodes in the graph that tracks progress of
        dependencies.
        """
        selected_nodes = self.get_selected(spec)
        selected_resources.set_selected_resources(selected_nodes)
        new_graph = self.full_graph.get_subset_graph(selected_nodes)
        return GraphQueue(new_graph.graph, self.manifest, selected_nodes,
            preserve_edges)


class ResourceTypeSelector(NodeSelector):

    def __init__(self, graph, manifest, previous_state, resource_types,
        include_empty_nodes=False):
        super().__init__(graph=graph, manifest=manifest, previous_state=
            previous_state, include_empty_nodes=include_empty_nodes)
        self.resource_types = set(resource_types)

    def func_50hoirt3(self, node):
        return node.resource_type in self.resource_types
