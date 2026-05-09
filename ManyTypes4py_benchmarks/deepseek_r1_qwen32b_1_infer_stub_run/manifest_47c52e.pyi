from __future__ import annotations
import enum
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from multiprocessing.synchronize import Lock
from typing import Any, Callable, ClassVar, DefaultDict, Dict, Generic, List, Mapping, MutableMapping, Optional, Set, Tuple, TypeVar, Union
from typing_extensions import Protocol
import dbt_common.exceptions
import dbt_common.utils
from dbt import deprecations, tracking
from dbt.adapters.exceptions import DuplicateMacroInPackageError, DuplicateMaterializationNameError
from dbt.adapters.factory import get_adapter_package_names
from dbt.artifacts.resources import BaseResource, DeferRelation, NodeVersion, RefArgs
from dbt.artifacts.resources.v1.config import NodeConfig
from dbt.artifacts.schemas.manifest import ManifestMetadata, UniqueID, WritableManifest
from dbt.clients.jinja_static import statically_parse_ref_or_source
from dbt.contracts.files import AnySourceFile, FileHash, FixtureSourceFile, SchemaSourceFile, SourceFile
from dbt.contracts.graph.nodes import RESOURCE_CLASS_TO_NODE_CLASS, BaseNode, Documentation, Exposure, GenericTestNode, GraphMemberNode, Group, Macro, ManifestNode, Metric, ModelNode, SavedQuery, SeedNode, SemanticModel, SingularTestNode, SnapshotNode, SourceDefinition, UnitTestDefinition, UnitTestFileFixture, UnpatchedSourceDefinition
from dbt.contracts.graph.unparsed import SourcePatch, UnparsedVersion
from dbt.contracts.util import SourceKey
from dbt.events.types import ArtifactWritten, UnpinnedRefNewVersionAvailable
from dbt.exceptions import AmbiguousResourceNameRefError, CompilationError, DuplicateResourceNameError
from dbt.flags import get_flags
from dbt.mp_context import get_mp_context
from dbt.node_types import REFABLE_NODE_TYPES, VERSIONED_NODE_TYPES, AccessType, NodeType
from dbt_common.dataclass_schema import dbtClassMixin
from dbt_common.events.contextvars import get_node_info
from dbt_common.events.functions import fire_event
from dbt_common.helper_types import PathSet

PackageName = str
DocName = str
RefName = str

def find_unique_id_for_package(storage: DefaultDict[str, Dict[str, UniqueID]], key: str, package: Optional[str]) -> Optional[UniqueID]:
    ...

@dataclass
class ParsingInfo:
    static_analysis_parsed_path_count: int
    static_analysis_path_count: int

class Locality(enum.IntEnum):
    Core = 1
    Imported = 2
    Root = 3

M = TypeVar('M', bound='MacroCandidate')

@dataclass
class MacroCandidate:
    locality: Locality
    macro: Macro

    def __eq__(self, other: M) -> bool:
        ...

    def __lt__(self, other: M) -> bool:
        ...

@dataclass
class MaterializationCandidate(MacroCandidate):
    specificity: int

    @classmethod
    def from_macro(cls, candidate: MacroCandidate, specificity: int) -> MaterializationCandidate:
        ...

    def __eq__(self, other: MaterializationCandidate) -> bool:
        ...

    def __lt__(self, other: MaterializationCandidate) -> bool:
        ...

class CandidateList(List[M]):
    def last_candidate(self, valid_localities: Optional[List[Locality]] = None) -> Optional[M]:
        ...

    def last(self) -> Optional[Macro]:
        ...

D = TypeVar('D')

class Searchable(Protocol):
    @property
    def search_name(self) -> str:
        ...

T = TypeVar('T', bound=GraphMemberNode)

class Disabled(Generic[D]):
    pass

MaybeMetricNode = Optional[Union[Metric, Disabled[Metric]]]
MaybeSavedQueryNode = Optional[Union[SavedQuery, Disabled[SavedQuery]]]
MaybeDocumentation = Optional[Documentation]
MaybeParsedSource = Optional[Union[SourceDefinition, Disabled[SourceDefinition]]]
MaybeNonSource = Optional[Union[ManifestNode, Disabled[ManifestNode]]]

class DocLookup(dbtClassMixin):
    def __init__(self, manifest: Manifest) -> None:
        ...

    def get_unique_id(self, key: str, package: Optional[str]) -> Optional[UniqueID]:
        ...

    def find(self, key: str, package: Optional[str], manifest: Manifest) -> Optional[Documentation]:
        ...

    def add_doc(self, doc: Documentation) -> None:
        ...

    def populate(self, manifest: Manifest) -> None:
        ...

    def perform_lookup(self, unique_id: UniqueID, manifest: Manifest) -> Documentation:
        ...

class SourceLookup(dbtClassMixin):
    def __init__(self, manifest: Manifest) -> None:
        ...

    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]:
        ...

    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[SourceDefinition]:
        ...

    def add_source(self, source: SourceDefinition) -> None:
        ...

    def populate(self, manifest: Manifest) -> None:
        ...

    def perform_lookup(self, unique_id: UniqueID, manifest: Manifest) -> SourceDefinition:
        ...

class RefableLookup(dbtClassMixin):
    _lookup_types: Set[NodeType]
    _versioned_types: Set[NodeType]

    def __init__(self, manifest: Manifest) -> None:
        ...

    def get_unique_id(self, key: str, package: Optional[str], version: Optional[str], node: Optional[BaseNode] = None) -> Optional[UniqueID]:
        ...

    def find(self, key: str, package: Optional[str], version: Optional[str], manifest: Manifest, source_node: Optional[BaseNode] = None) -> Optional[ModelNode]:
        ...

    def add_node(self, node: ManifestNode) -> None:
        ...

    def populate(self, manifest: Manifest) -> None:
        ...

    def perform_lookup(self, unique_id: UniqueID, manifest: Manifest) -> ModelNode:
        ...

    def _find_unique_ids_for_package(self, key: str, package: Optional[str]) -> List[UniqueID]:
        ...

class MetricLookup(dbtClassMixin):
    def __init__(self, manifest: Manifest) -> None:
        ...

    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]:
        ...

    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[Metric]:
        ...

    def add_metric(self, metric: Metric) -> None:
        ...

    def populate(self, manifest: Manifest) -> None:
        ...

    def perform_lookup(self, unique_id: UniqueID, manifest: Manifest) -> Metric:
        ...

class SavedQueryLookup(dbtClassMixin):
    def __init__(self, manifest: Manifest) -> None:
        ...

    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]:
        ...

    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[SavedQuery]:
        ...

    def add_saved_query(self, saved_query: SavedQuery) -> None:
        ...

    def populate(self, manifest: Manifest) -> None:
        ...

    def perform_lookup(self, unique_id: UniqueID, manifest: Manifest) -> SavedQuery:
        ...

class SemanticModelByMeasureLookup(dbtClassMixin):
    def __init__(self, manifest: Manifest) -> None:
        ...

    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]:
        ...

    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[SemanticModel]:
        ...

    def add(self, semantic_model: SemanticModel) -> None:
        ...

    def populate(self, manifest: Manifest) -> None:
        ...

    def perform_lookup(self, unique_id: UniqueID, manifest: Manifest) -> SemanticModel:
        ...

class DisabledLookup(dbtClassMixin):
    def __init__(self, manifest: Manifest) -> None:
        ...

    def populate(self, manifest: Manifest) -> None:
        ...

    def add_node(self, node: GraphMemberNode) -> None:
        ...

    def find(self, search_name: str, package: Optional[str], version: Optional[str] = None, resource_types: Optional[List[NodeType]] = None) -> Optional[List[GraphMemberNode]]:
        ...

class AnalysisLookup(RefableLookup):
    _lookup_types: Set[NodeType]

class SingularTestLookup(dbtClassMixin):
    def __init__(self, manifest: Manifest) -> None:
        ...

    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]:
        ...

    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[SingularTestNode]:
        ...

    def add_singular_test(self, source: SingularTestNode) -> None:
        ...

    def populate(self, manifest: Manifest) -> None:
        ...

    def perform_lookup(self, unique_id: UniqueID, manifest: Manifest) -> SingularTestNode:
        ...

def _packages_to_search(current_project: str, node_package: str, target_package: Optional[str] = None) -> List[str]:
    ...

def _sort_values(dct: Dict[str, List[UniqueID]]) -> Dict[str, List[UniqueID]]:
    ...

def build_node_edges(nodes: List[ManifestNode]) -> Tuple[Dict[UniqueID, List[UniqueID]], Dict[UniqueID, List[UniqueID]]]:
    ...

def build_macro_edges(nodes: List[ManifestNode]) -> Dict[UniqueID, List[UniqueID]]:
    ...

def _deepcopy(value: Any) -> Any:
    ...

class MacroMethods:
    def __init__(self) -> None:
        ...

    def find_macro_candidate_by_name(self, name: str, root_project_name: str, package: Optional[str] = None) -> Optional[MacroCandidate]:
        ...

    def find_macro_by_name(self, name: str, root_project_name: str, package: Optional[str] = None) -> Optional[Macro]:
        ...

    def find_generate_macro_by_name(self, component: str, root_project_name: str, imported_package: Optional[str] = None) -> Optional[Macro]:
        ...

    def _find_macros_by_name(self, name: str, root_project_name: str, filter: Optional[Callable[[MacroCandidate], bool]] = None) -> CandidateList:
        ...

    def get_macros_by_name(self) -> Dict[str, List[Macro]]:
        ...

    @staticmethod
    def _build_macros_by_name(macros: Dict[UniqueID, Macro]) -> Dict[str, List[Macro]]:
        ...

    def get_macros_by_package(self) -> Dict[str, Dict[str, Macro]]:
        ...

    @staticmethod
    def _build_macros_by_package(macros: Dict[UniqueID, Macro]) -> Dict[str, Dict[str, Macro]]:
        ...

class Manifest(MacroMethods, dbtClassMixin):
    nodes: Dict[UniqueID, ModelNode]
    sources: Dict[UniqueID, SourceDefinition]
    macros: Dict[UniqueID, Macro]
    docs: Dict[UniqueID, Documentation]
    exposures: Dict[UniqueID, Exposure]
    metrics: Dict[UniqueID, Metric]
    groups: Dict[UniqueID, Group]
    selectors: Dict[str, Any]
    files: Dict[UniqueID, SourceFile]
    metadata: ManifestMetadata
    flat_graph: Dict[str, Dict[UniqueID, Dict[str, Any]]]
    state_check: ManifestStateCheck
    source_patches: Dict[str, SourcePatch]
    disabled: Dict[str, List[GraphMemberNode]]
    env_vars: Dict[str, str]
    semantic_models: Dict[UniqueID, SemanticModel]
    unit_tests: Dict[UniqueID, UnitTestDefinition]
    saved_queries: Dict[UniqueID, SavedQuery]
    fixtures: Dict[UniqueID, UnitTestFileFixture]
    _doc_lookup: Optional[DocLookup]
    _source_lookup: Optional[SourceLookup]
    _ref_lookup: Optional[RefableLookup]
    _metric_lookup: Optional[MetricLookup]
    _saved_query_lookup: Optional[SavedQueryLookup]
    _semantic_model_by_measure_lookup: Optional[SemanticModelByMeasureLookup]
    _disabled_lookup: Optional[DisabledLookup]
    _analysis_lookup: Optional[AnalysisLookup]
    _singular_test_lookup: Optional[SingularTestLookup]
    _parsing_info: ParsingInfo
    _lock: Lock
    _macros_by_name: Optional[Dict[str, List[Macro]]]
    _macros_by_package: Optional[Dict[str, Dict[str, Macro]]]

    def __pre_serialize__(self, context: Optional[Any] = None) -> Manifest:
        ...

    @classmethod
    def __post_deserialize__(cls, obj: Manifest) -> Manifest:
        ...

    def build_flat_graph(self) -> None:
        ...

    def build_disabled_by_file_id(self) -> Dict[str, GraphMemberNode]:
        ...

    def _get_parent_adapter_types(self, adapter_type: str) -> List[str]:
        ...

    def _materialization_candidates_for(self, project_name: str, materialization_name: str, adapter_type: str, specificity: int) -> CandidateList:
        ...

    def find_materialization_macro_by_name(self, project_name: str, materialization_name: str, adapter_type: str) -> Optional[Macro]:
        ...

    def get_resource_fqns(self) -> Dict[str, Set[Tuple[str, ...]]]:
        ...

    def get_used_schemas(self, resource_types: Optional[List[NodeType]] = None) -> frozenset[Tuple[Optional[str], Optional[str]]]:
        ...

    def get_used_databases(self) -> frozenset[Optional[str]]:
        ...

    def deepcopy(self) -> Manifest:
        ...

    def build_parent_and_child_maps(self) -> None:
        ...

    def build_macro_child_map(self) -> Dict[UniqueID, List[UniqueID]]:
        ...

    def build_group_map(self) -> None:
        ...

    def fill_tracking_metadata(self) -> None:
        ...

    @classmethod
    def from_writable_manifest(cls, writable_manifest: WritableManifest) -> Manifest:
        ...

    @staticmethod
    def _map_nodes_to_map_resources(nodes_map: Dict[UniqueID, ManifestNode]) -> Dict[UniqueID, BaseResource]:
        ...

    @staticmethod
    def _map_list_nodes_to_map_list_resources(nodes_map: Dict[UniqueID, List[ManifestNode]]) -> Dict[UniqueID, List[BaseResource]]:
        ...

    @classmethod
    def _map_resources_to_map_nodes(cls, resources_map: Dict[UniqueID, BaseResource]) -> Dict[UniqueID, ManifestNode]:
        ...

    @classmethod
    def _map_list_resources_to_map_list_nodes(cls, resources_map: Dict[UniqueID, List[BaseResource]]) -> Dict[UniqueID, List[ManifestNode]]:
        ...

    def writable_manifest(self) -> WritableManifest:
        ...

    def write(self, path: str) -> None:
        ...

    def expect(self, unique_id: UniqueID) -> GraphMemberNode:
        ...

    @property
    def doc_lookup(self) -> DocLookup:
        ...

    def rebuild_doc_lookup(self) -> None:
        ...

    @property
    def source_lookup(self) -> SourceLookup:
        ...

    def rebuild_source_lookup(self) -> None:
        ...

    @property
    def ref_lookup(self) -> RefableLookup:
        ...

    @property
    def metric_lookup(self) -> MetricLookup:
        ...

    @property
    def saved_query_lookup(self) -> SavedQueryLookup:
        ...

    @property
    def semantic_model_by_measure_lookup(self) -> SemanticModelByMeasureLookup:
        ...

    def rebuild_ref_lookup(self) -> None:
        ...

    @property
    def disabled_lookup(self) -> DisabledLookup:
        ...

    def rebuild_disabled_lookup(self) -> None:
        ...

    @property
    def analysis_lookup(self) -> AnalysisLookup:
        ...

    @property
    def singular_test_lookup(self) -> SingularTestLookup:
        ...

    @property
    def external_node_unique_ids(self) -> List[UniqueID]:
        ...

    def resolve_ref(self, source_node: BaseNode, target_model_name: str, target_model_package: Optional[str], target_model_version: Optional[str], current_project: str, node_package: str) -> Optional[Union[ModelNode, Disabled[ModelNode]]]:
        ...

    def resolve_source(self, target_source_name: str, target_table_name: str, current_project: str, node_package: str) -> Optional[Union[SourceDefinition, Disabled[SourceDefinition]]]:
        ...

    def resolve_metric(self, target_metric_name: str, target_metric_package: Optional[str], current_project: str, node_package: str) -> Optional[Union[Metric, Disabled[Metric]]]:
        ...

    def resolve_saved_query(self, target_saved_query_name: str, target_saved_query_package: Optional[str], current_project: str, node_package: str) -> Optional[Union[SavedQuery, Disabled[SavedQuery]]]:
        ...

    def resolve_semantic_model_for_measure(self, target_measure_name: str, current_project: str, node_package: str, target_package: Optional[str] = None) -> Optional[SemanticModel]:
        ...

    def resolve_doc(self, name: str, package: Optional[str], current_project: str, node_package: str) -> Optional[Documentation]:
        ...

    def is_invalid_private_ref(self, node: BaseNode, target_model: ModelNode, dependencies: Optional[Dict[str, Any]] = None) -> bool:
        ...

    def is_invalid_protected_ref(self, node: BaseNode, target_model: ModelNode, dependencies: Optional[Dict[str, Any]] = None) -> bool:
        ...

    def merge_from_artifact(self, other: Manifest) -> None:
        ...

    def add_macro(self, source_file: SourceFile, macro: Macro) -> None:
        ...

    def has_file(self, source_file: SourceFile) -> bool:
        ...

    def add_source(self, source_file: SourceFile, source: SourceDefinition) -> None:
        ...

    def add_node_nofile(self, node: ManifestNode) -> None:
        ...

    def add_node(self, source_file: SourceFile, node: ManifestNode, test_from: Optional[str] = None) -> None:
        ...

    def add_exposure(self, source_file: SourceFile, exposure: Exposure) -> None:
        ...

    def add_metric(self, source_file: SourceFile, metric: Metric, generated_from: Optional[str] = None) -> None:
        ...

    def add_group(self, source_file: SourceFile, group: Group) -> None:
        ...

    def add_disabled_nofile(self, node: GraphMemberNode) -> None:
        ...

    def add_disabled(self, source_file: SourceFile, node: GraphMemberNode, test_from: Optional[str] = None) -> None:
        ...

    def add_doc(self, source_file: SourceFile, doc: Documentation) -> None:
        ...

    def add_semantic_model(self, source_file: SourceFile, semantic_model: SemanticModel) -> None:
        ...

    def add_unit_test(self, source_file: SourceFile, unit_test: UnitTestDefinition) -> None:
        ...

    def add_fixture(self, source_file: SourceFile, fixture: UnitTestFileFixture) -> None:
        ...

    def add_saved_query(self, source_file: SourceFile, saved_query: SavedQuery) -> None:
        ...

    def find_node_from_ref_or_source(self, expression: str) -> Optional[BaseNode]:
        ...

    def __reduce_ex__(self, protocol: int) -> Tuple[Callable, Tuple[Any, ...]]:
        ...

    def _microbatch_macro_is_core(self, project_name: str) -> bool:
        ...

    def use_microbatch_batches(self, project_name: str) -> bool:
        ...

class MacroManifest(MacroMethods):
    def __init__(self, macros: Dict[UniqueID, Macro]) -> None:
        ...

AnyManifest = Union[Manifest, MacroManifest]

def _check_duplicates(value: GraphMemberNode, src: Dict[UniqueID, GraphMemberNode]) -> None:
    ...

K_T = TypeVar('K_T')
V_T = TypeVar('V_T')

def _expect_value(key: K_T, src: Dict[K_T, V_T], old_file: str, name: str) -> V_T:
    ...