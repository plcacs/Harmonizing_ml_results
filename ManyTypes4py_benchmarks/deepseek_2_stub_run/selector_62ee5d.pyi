```python
from typing import Any, Set, Tuple
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import GraphMemberNode
from dbt.contracts.state import PreviousState
from .graph import Graph
from .queue import GraphQueue
from .selector_methods import MethodManager
from .selector_spec import IndirectSelection, SelectionCriteria, SelectionSpec

def get_package_names(nodes: Any) -> Set[str]: ...

def can_select_indirectly(node: Any) -> bool: ...

class NodeSelector(MethodManager):
    full_graph: Graph
    include_empty_nodes: bool
    graph: Any

    def __init__(
        self,
        graph: Graph,
        manifest: Manifest,
        previous_state: PreviousState = ...,
        include_empty_nodes: bool = ...
    ) -> None: ...

    def select_included(self, included_nodes: Any, spec: SelectionCriteria) -> Set[Any]: ...

    def get_nodes_from_criteria(self, spec: SelectionCriteria) -> Tuple[Set[Any], Set[Any]]: ...

    def collect_specified_neighbors(self, spec: SelectionCriteria, selected: Set[Any]) -> Set[Any]: ...

    def select_nodes_recursively(self, spec: SelectionSpec) -> Tuple[Set[Any], Set[Any]]: ...

    def select_nodes(self, spec: SelectionSpec) -> Tuple[Set[Any], Set[Any]]: ...

    def _is_graph_member(self, unique_id: Any) -> bool: ...

    def _is_empty_node(self, unique_id: Any) -> bool: ...

    def node_is_match(self, node: GraphMemberNode) -> bool: ...

    def _is_match(self, unique_id: Any) -> bool: ...

    def filter_selection(self, selected: Set[Any]) -> Set[Any]: ...

    def expand_selection(
        self,
        selected: Set[Any],
        indirect_selection: IndirectSelection = ...
    ) -> Tuple[Set[Any], Set[Any]]: ...

    def incorporate_indirect_nodes(
        self,
        direct_nodes: Set[Any],
        indirect_nodes: Set[Any] = ...,
        indirect_selection: IndirectSelection = ...
    ) -> Set[Any]: ...

    def get_selected(self, spec: SelectionSpec) -> Set[Any]: ...

    def get_graph_queue(self, spec: SelectionSpec, preserve_edges: bool = ...) -> GraphQueue: ...

class ResourceTypeSelector(NodeSelector):
    resource_types: Set[Any]

    def __init__(
        self,
        graph: Graph,
        manifest: Manifest,
        previous_state: PreviousState,
        resource_types: Any,
        include_empty_nodes: bool = ...
    ) -> None: ...

    def node_is_match(self, node: GraphMemberNode) -> bool: ...
```