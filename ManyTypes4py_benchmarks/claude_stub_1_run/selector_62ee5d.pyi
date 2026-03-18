```pyi
from typing import Any, List, Optional, Set, Tuple
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import GraphMemberNode
from dbt.contracts.state import PreviousState
from dbt.node_types import NodeType
from .graph import Graph, UniqueId
from .queue import GraphQueue
from .selector_methods import MethodManager
from .selector_spec import IndirectSelection, SelectionCriteria, SelectionSpec

def get_package_names(nodes: Any) -> Set[Any]: ...
def can_select_indirectly(node: Any) -> bool: ...

class NodeSelector(MethodManager):
    full_graph: Graph
    include_empty_nodes: bool
    graph: Graph
    
    def __init__(
        self,
        graph: Graph,
        manifest: Manifest,
        previous_state: Optional[PreviousState] = None,
        include_empty_nodes: bool = False,
    ) -> None: ...
    
    def select_included(self, included_nodes: Any, spec: SelectionSpec) -> Set[UniqueId]: ...
    
    def get_nodes_from_criteria(self, spec: SelectionCriteria) -> Tuple[Set[UniqueId], Set[UniqueId]]: ...
    
    def collect_specified_neighbors(self, spec: SelectionSpec, selected: Set[UniqueId]) -> Set[UniqueId]: ...
    
    def select_nodes_recursively(self, spec: SelectionSpec) -> Tuple[Set[UniqueId], Set[UniqueId]]: ...
    
    def select_nodes(self, spec: SelectionSpec) -> Tuple[Set[UniqueId], Set[UniqueId]]: ...
    
    def _is_graph_member(self, unique_id: UniqueId) -> bool: ...
    
    def _is_empty_node(self, unique_id: UniqueId) -> bool: ...
    
    def node_is_match(self, node: GraphMemberNode) -> bool: ...
    
    def _is_match(self, unique_id: UniqueId) -> bool: ...
    
    def filter_selection(self, selected: Set[UniqueId]) -> Set[UniqueId]: ...
    
    def expand_selection(
        self,
        selected: Set[UniqueId],
        indirect_selection: IndirectSelection = IndirectSelection.Eager,
    ) -> Tuple[Set[UniqueId], Set[UniqueId]]: ...
    
    def incorporate_indirect_nodes(
        self,
        direct_nodes: Set[UniqueId],
        indirect_nodes: Set[UniqueId] = ...,
        indirect_selection: IndirectSelection = IndirectSelection.Eager,
    ) -> Set[UniqueId]: ...
    
    def get_selected(self, spec: SelectionSpec) -> Set[UniqueId]: ...
    
    def get_graph_queue(self, spec: SelectionSpec, preserve_edges: bool = True) -> GraphQueue: ...

class ResourceTypeSelector(NodeSelector):
    resource_types: Set[NodeType]
    
    def __init__(
        self,
        graph: Graph,
        manifest: Manifest,
        previous_state: PreviousState,
        resource_types: Any,
        include_empty_nodes: bool = False,
    ) -> None: ...
    
    def node_is_match(self, node: GraphMemberNode) -> bool: ...
```