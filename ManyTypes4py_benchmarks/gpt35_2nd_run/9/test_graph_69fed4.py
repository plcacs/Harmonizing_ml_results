from dbt.compilation import Linker
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import ModelNode
from dbt.graph.graph import Graph
from tests.unit.utils.manifest import make_model
from typing import Set, Dict

class TestGraph:

    def extra_parent_model(self) -> ModelNode:
    def non_shared_child_of_extra(self, extra_parent_model: ModelNode) -> ModelNode:
    def model_with_two_direct_parents(self, extra_parent_model: ModelNode, ephemeral_model: ModelNode) -> ModelNode:
    def local_manifest_extensions(self, manifest: Manifest, model_with_two_direct_parents: ModelNode, non_shared_child_of_extra: ModelNode, extra_parent_model: ModelNode) -> None:
    def graph(self, manifest: Manifest, local_manifest_extensions: None) -> Graph:
    def test_nodes(self, graph: Graph, manifest: Manifest) -> None:
    def test_descendantcs(self, graph: Graph, manifest: Manifest) -> None:
    def test_ancestors(self, graph: Graph, manifest: Manifest) -> None:
    def test_exclude_edge_type(self) -> None:
    def test_select_childrens_parents(self, graph: Graph, model_with_two_direct_parents: ModelNode, extra_parent_model: ModelNode, ephemeral_model: ModelNode) -> None:
    def test_select_children(self, graph: Graph, ephemeral_model: ModelNode, extra_parent_model: ModelNode) -> None:
    def test_select_parents(self, graph: Graph, non_shared_child_of_extra: ModelNode, table_model: ModelNode) -> None:
