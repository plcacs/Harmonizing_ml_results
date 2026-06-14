from __future__ import annotations

import enum
from dataclasses import dataclass
from itertools import chain
from multiprocessing.synchronize import Lock
from typing import Any, Callable, Dict, Generic, Iterable, List, Mapping, MutableMapping, Optional, Set, Tuple, Type, TypeVar, Union
from typing_extensions import Protocol

import dbt_common.exceptions
import dbt_common.utils
from dbt.adapters.exceptions import DuplicateMacroInPackageError, DuplicateMaterializationNameError
from dbt.adapters.factory import get_adapter_package_names, get_adapter_type_names
from dbt.artifacts.resources import BaseResource, DeferRelation, RefArgs
from dbt.artifacts.resources.v1.config import NodeConfig
from dbt.artifacts.schemas.manifest import ManifestMetadata, UniqueID, WritableManifest
from dbt.clients.jinja_static import statically_parse_ref_or_source
from dbt.contracts.files import AnySourceFile, FileHash, FixtureSourceFile, SchemaSourceFile, SourceFile
from dbt.contracts.graph.nodes import (
    RESOURCE_CLASS_TO_NODE_CLASS,
    BaseNode,
    Documentation,
    Exposure,
    GenericTestNode,
    GraphMemberNode,
    Group,
    Macro,
    ManifestNode,
    Metric,
    ModelNode,
    SavedQuery,
    SemanticModel,
    SingularTestNode,
    SnapshotNode,
    SourceDefinition,
    UnitTestDefinition,
    UnitTestFileFixture,
)
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

def find_unique_id_for_package(
    storage: Mapping[str, Mapping[str, UniqueID]],
    key: str,
    package: Optional[str],
) -> Optional[UniqueID]: ...

class DocLookup(dbtClassMixin):
    storage: Dict[str, Dict[str, UniqueID]]

    def __init__(self, manifest: Manifest) -> None: ...
    def get_unique_id(self, key: str, package: Optional[str]) -> Optional[UniqueID]: ...
    def find(self, key: str, package: Optional[str], manifest: Manifest) -> Optional[Documentation]: ...
    def add_doc(self, doc: Documentation) -> None: ...
    def populate(self, manifest: Manifest) -> None: ...
    def perform_lookup(self, unique_id: UniqueID, manifest: Manifest) -> Documentation: ...

class SourceLookup(dbtClassMixin):
    storage: Dict[str, Dict[str, UniqueID]]

    def __init__(self, manifest: Manifest) -> None: ...
    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]: ...
    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[SourceDefinition]: ...
    def add_source(self, source: SourceDefinition) -> None: ...
    def populate(self, manifest: Manifest) -> None: ...
    def perform_lookup(self, unique_id: UniqueID, manifest: Manifest) -> SourceDefinition: ...

class RefableLookup(dbtClassMixin):
    _lookup_types: Set[NodeType]
    _versioned_types: Set[NodeType]
    storage: Dict[str, Dict[str, UniqueID]]

    def __init__(self, manifest: Manifest) -> None: ...
    def get_unique_id(
        self,
        key: str,
        package: Optional[str],
        version: Optional[str],
        node: Optional[ManifestNode] = ...,
    ) -> Optional[UniqueID]: ...
    def find(
        self,
        key: str,
        package: Optional[str],
        version: Optional[str],
        manifest: Manifest,
        source_node: Optional[ManifestNode] = ...,
    ) -> Optional[ManifestNode]: ...
    def add_node(self, node: ManifestNode) -> None: ...
    def populate(self, manifest: Manifest) -> None: ...
    def perform_lookup(self, unique_id: UniqueID, manifest: Manifest) -> ManifestNode: ...
    def _find_unique_ids_for_package(self, key: str, package: Optional[str]) -> List[UniqueID]: ...

class MetricLookup(dbtClassMixin):
    storage: Dict[str, Dict[str, UniqueID]]

    def __init__(self, manifest: Manifest) -> None: ...
    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]: ...
    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[Metric]: ...
    def add_metric(self, metric: Metric) -> None: ...
    def populate(self, manifest: Manifest) -> None: ...
    def perform_lookup(self, unique_id: UniqueID, manifest: Manifest) -> Metric: ...

class SavedQueryLookup(dbtClassMixin):
    storage: Dict[str, Dict[str, UniqueID]]

    def __init__(self, manifest: Manifest) -> None: ...
    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]: ...
    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[SavedQuery]: ...
    def add_saved_query(self, saved_query: SavedQuery) -> None: ...
    def populate(self, manifest: Manifest) -> None: ...
    def perform_lookup(self, unique_id: UniqueID, manifest: Manifest) -> SavedQuery: ...

class SemanticModelByMeasureLookup(dbtClassMixin):
    storage: "DefaultDict[str, Dict[str, UniqueID]]"

    def __init__(self, manifest: Manifest) -> None: ...
    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]: ...
    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[SemanticModel]: ...
    def add(self, semantic_model: SemanticModel) -> None: ...
    def populate(self, manifest: Manifest) -> None: ...
    def perform_lookup(self, unique_id: UniqueID, manifest: Manifest) -> SemanticModel: ...

class DisabledLookup(dbtClassMixin):
    storage: Dict[str, Dict[str, List[GraphMemberNode]]]

    def __init__(self, manifest: Manifest) -> None: ...
    def populate(self, manifest: Manifest) -> None: ...
    def add_node(self, node: GraphMemberNode) -> None: ...
    def find(
        self,
        search_name: str,
        package: Optional[str],
        version: Optional[str] = ...,
        resource_types: Optional[Set[NodeType]] = ...,
    ) -> Optional[List[GraphMemberNode]]: ...

class AnalysisLookup(RefableLookup): ...

class SingularTestLookup(dbtClassMixin):
    storage: Dict[str, Dict[str, UniqueID]]

    def __init__(self, manifest: Manifest) -> None: ...
    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]: ...
    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[SingularTestNode]: ...
    def add_singular_test(self, source: SingularTestNode) -> None: ...
    def populate(self, manifest: Manifest) -> None: ...
    def perform_lookup(self, unique_id: UniqueID, manifest: Manifest) -> SingularTestNode: ...

def _packages_to_search(
    current_project: str,
    node_package: str,
    target_package: Optional[str] = ...,
) -> List[Optional[str]]: ...

def _sort_values(dct: Mapping[UniqueID, List[UniqueID]]) -> Dict[UniqueID, List[UniqueID]]: ...
def build_node_edges(
    nodes: Iterable[GraphMemberNode],
) -> Tuple[Dict[UniqueID, List[UniqueID]], Dict[UniqueID, List[UniqueID]]]: ...
def build_macro_edges(nodes: Iterable[GraphMemberNode]) -> Dict[UniqueID, List[UniqueID]]: ...
def _deepcopy(value: dbtClassMixin) -> dbtClassMixin: ...

class Locality(enum.IntEnum):
    Core: int
    Imported: int
    Root: int

@dataclass
class MacroCandidate:
    locality: Locality
    macro: Macro

    def __eq__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...

@dataclass
class MaterializationCandidate(MacroCandidate):
    specificity: int

    @classmethod
    def from_macro(cls: Type["MaterializationCandidate"], candidate: MacroCandidate, specificity: int) -> "MaterializationCandidate": ...
    def __eq__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...

M = TypeVar("M", bound=MacroCandidate)

class CandidateList(List[M]):
    def last_candidate(self, valid_localities: Optional[Set[Locality]] = ...) -> Optional[M]: ...
    def last(self) -> Optional[Macro]: ...

def _get_locality(macro: Macro, root_project_name: str, internal_packages: Set[str]) -> Locality: ...

class Searchable(Protocol):
    @property
    def search_name(self) -> str: ...

D = TypeVar("D")

@dataclass
class Disabled(Generic[D]):
    value: D

MaybeMetricNode = Optional[Union[Metric, Disabled[Metric]]]
MaybeSavedQueryNode = Optional[Union[SavedQuery, Disabled[SavedQuery]]]
MaybeDocumentation = Optional[Documentation]
MaybeParsedSource = Optional[Union[SourceDefinition, Disabled[SourceDefinition]]]
MaybeNonSource = Optional[Union[ManifestNode, Disabled[ManifestNode]]]
T = TypeVar("T", bound=GraphMemberNode)

class MacroMethods:
    macros: Dict[UniqueID, Macro]
    metadata: ManifestMetadata
    _macros_by_name: Optional[Dict[str, List[Macro]]]
    _macros_by_package: Optional[Dict[str, Dict[str, Macro]]]

    def __init__(self) -> None: ...
    def find_macro_candidate_by_name(
        self,
        name: str,
        root_project_name: str,
        package: Optional[str],
    ) -> Optional[MacroCandidate]: ...
    def find_macro_by_name(self, name: str, root_project_name: str, package: Optional[str]) -> Optional[Macro]: ...
    def find_generate_macro_by_name(
        self,
        component: str,
        root_project_name: str,
        imported_package: Optional[str] = ...,
    ) -> Optional[Macro]: ...
    def _find_macros_by_name(
        self,
        name: str,
        root_project_name: str,
        filter: Optional[Callable[[MacroCandidate], bool]] = ...,
    ) -> CandidateList[MacroCandidate]: ...
    def get_macros_by_name(self) -> Dict[str, List[Macro]]: ...
    @staticmethod
    def _build_macros_by_name(macros: Mapping[UniqueID, Macro]) -> Dict[str, List[Macro]]: ...
    def get_macros_by_package(self) -> Dict[str, Dict[str, Macro]]: ...
    @staticmethod
    def _build_macros_by_package(macros: Mapping[UniqueID, Macro]) -> Dict[str, Dict[str, Macro]]: ...

@dataclass
class ParsingInfo:
    static_analysis_parsed_path_count: int
    static_analysis_path_count: int

@dataclass
class ManifestStateCheck(dbtClassMixin):
    vars_hash: FileHash
    project_env_vars_hash: FileHash
    profile_env_vars_hash: FileHash
    profile_hash: FileHash
    project_hashes: Dict[str, FileHash]

NodeClassT = TypeVar("NodeClassT", bound="BaseNode")
ResourceClassT = TypeVar("ResourceClassT", bound="BaseResource")

@dataclass
class Manifest(MacroMethods, dbtClassMixin):
    nodes: Dict[UniqueID, ManifestNode]
    sources: Dict[UniqueID, SourceDefinition]
    macros: Dict[UniqueID, Macro]
    docs: Dict[UniqueID, Documentation]
    exposures: Dict[UniqueID, Exposure]
    metrics: Dict[UniqueID, Metric]
    groups: Dict[UniqueID, Group]
    selectors: Dict[str, Any]
    files: Dict[str, SourceFile]
    metadata: ManifestMetadata
    flat_graph: Dict[str, Dict[str, Any]]
    state_check: ManifestStateCheck
    source_patches: Dict[SourceKey, SourcePatch]
    disabled: Dict[UniqueID, List[GraphMemberNode]]
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
    child_map: Dict[UniqueID, List[UniqueID]]
    parent_map: Dict[UniqueID, List[UniqueID]]
    group_map: Dict[str, List[UniqueID]]

    def __pre_serialize__(self, context: Optional[Mapping[str, Any]] = ...) -> "Manifest": ...
    @classmethod
    def __post_deserialize__(cls: Type["Manifest"], obj: "Manifest") -> "Manifest": ...
    def build_flat_graph(self) -> None: ...
    def build_disabled_by_file_id(self) -> Dict[str, GraphMemberNode]: ...
    def _get_parent_adapter_types(self, adapter_type: Optional[str]) -> List[str]: ...
    def _materialization_candidates_for(
        self,
        project_name: str,
        materialization_name: str,
        adapter_type: Optional[str],
        specificity: int,
    ) -> CandidateList[MaterializationCandidate]: ...
    def find_materialization_macro_by_name(
        self,
        project_name: str,
        materialization_name: str,
        adapter_type: Optional[str],
    ) -> Optional[Macro]: ...
    def get_resource_fqns(self) -> Dict[str, Set[Tuple[str, ...]]]: ...
    def get_used_schemas(self, resource_types: Optional[Set[NodeType]] = ...) -> Set[Tuple[Optional[str], Optional[str]]]: ...
    def get_used_databases(self) -> Set[Optional[str]]: ...
    def deepcopy(self) -> "Manifest": ...
    def build_parent_and_child_maps(self) -> None: ...
    def build_macro_child_map(self) -> Dict[UniqueID, List[UniqueID]]: ...
    def build_group_map(self) -> None: ...
    def fill_tracking_metadata(self) -> None: ...
    @classmethod
    def from_writable_manifest(cls: Type["Manifest"], writable_manifest: WritableManifest) -> "Manifest": ...
    def _map_nodes_to_map_resources(self, nodes_map: Mapping[UniqueID, BaseNode]) -> Dict[UniqueID, BaseResource]: ...
    def _map_list_nodes_to_map_list_resources(self, nodes_map: Mapping[UniqueID, List[BaseNode]]) -> Dict[UniqueID, List[BaseResource]]: ...
    @classmethod
    def _map_resources_to_map_nodes(cls: Type["Manifest"], resources_map: Mapping[UniqueID, BaseResource]) -> Dict[UniqueID, BaseNode]: ...
    @classmethod
    def _map_list_resources_to_map_list_nodes(
        cls: Type["Manifest"],
        resources_map: Optional[Mapping[UniqueID, List[BaseResource]]],
    ) -> Dict[UniqueID, List[BaseNode]]: ...
    def writable_manifest(self) -> WritableManifest: ...
    def write(self, path: str) -> None: ...
    def expect(
        self,
        unique_id: UniqueID,
    ) -> Union[ManifestNode, SourceDefinition, Exposure, Metric, SemanticModel, UnitTestDefinition, SavedQuery]: ...
    @property
    def doc_lookup(self) -> DocLookup: ...
    def rebuild_doc_lookup(self) -> None: ...
    @property
    def source_lookup(self) -> SourceLookup: ...
    def rebuild_source_lookup(self) -> None: ...
    @property
    def ref_lookup(self) -> RefableLookup: ...
    @property
    def metric_lookup(self) -> MetricLookup: ...
    @property
    def saved_query_lookup(self) -> SavedQueryLookup: ...
    @property
    def semantic_model_by_measure_lookup(self) -> SemanticModelByMeasureLookup: ...
    def rebuild_ref_lookup(self) -> None: ...
    @property
    def disabled_lookup(self) -> DisabledLookup: ...
    def rebuild_disabled_lookup(self) -> None: ...
    @property
    def analysis_lookup(self) -> AnalysisLookup: ...
    @property
    def singular_test_lookup(self) -> SingularTestLookup: ...
    @property
    def external_node_unique_ids(self) -> List[UniqueID]: ...
    def resolve_ref(
        self,
        source_node: Optional[ManifestNode],
        target_model_name: str,
        target_model_package: Optional[str],
        target_model_version: Optional[str],
        current_project: str,
        node_package: str,
    ) -> MaybeNonSource: ...
    def resolve_source(
        self,
        target_source_name: str,
        target_table_name: str,
        current_project: str,
        node_package: str,
    ) -> MaybeParsedSource: ...
    def resolve_metric(
        self,
        target_metric_name: str,
        target_metric_package: Optional[str],
        current_project: str,
        node_package: str,
    ) -> MaybeMetricNode: ...
    def resolve_saved_query(
        self,
        target_saved_query_name: str,
        target_saved_query_package: Optional[str],
        current_project: str,
        node_package: str,
    ) -> MaybeSavedQueryNode: ...
    def resolve_semantic_model_for_measure(
        self,
        target_measure_name: str,
        current_project: str,
        node_package: str,
        target_package: Optional[str] = ...,
    ) -> Optional[SemanticModel]: ...
    def resolve_doc(
        self,
        name: str,
        package: Optional[str],
        current_project: str,
        node_package: str,
    ) -> Optional[Documentation]: ...
    def is_invalid_private_ref(
        self,
        node: ManifestNode,
        target_model: ManifestNode,
        dependencies: Optional[Mapping[str, Any]],
    ) -> bool: ...
    def is_invalid_protected_ref(
        self,
        node: ManifestNode,
        target_model: ManifestNode,
        dependencies: Optional[Mapping[str, Any]],
    ) -> bool: ...
    def merge_from_artifact(self, other: "Manifest") -> None: ...
    def add_macro(self, source_file: AnySourceFile, macro: Macro) -> None: ...
    def has_file(self, source_file: SourceFile) -> bool: ...
    def add_source(self, source_file: AnySourceFile, source: SourceDefinition) -> None: ...
    def add_node_nofile(self, node: ManifestNode) -> None: ...
    def add_node(self, source_file: AnySourceFile, node: ManifestNode, test_from: Optional[UniqueID] = ...) -> None: ...
    def add_exposure(self, source_file: AnySourceFile, exposure: Exposure) -> None: ...
    def add_metric(self, source_file: AnySourceFile, metric: Metric, generated_from: Optional[str] = ...) -> None: ...
    def add_group(self, source_file: AnySourceFile, group: Group) -> None: ...
    def add_disabled_nofile(self, node: GraphMemberNode) -> None: ...
    def add_disabled(self, source_file: AnySourceFile, node: GraphMemberNode, test_from: Optional[UniqueID] = ...) -> None: ...
    def add_doc(self, source_file: AnySourceFile, doc: Documentation) -> None: ...
    def add_semantic_model(self, source_file: AnySourceFile, semantic_model: SemanticModel) -> None: ...
    def add_unit_test(self, source_file: AnySourceFile, unit_test: UnitTestDefinition) -> None: ...
    def add_fixture(self, source_file: AnySourceFile, fixture: UnitTestFileFixture) -> None: ...
    def add_saved_query(self, source_file: AnySourceFile, saved_query: SavedQuery) -> None: ...
    def find_node_from_ref_or_source(self, expression: str) -> Optional[Union[ManifestNode, SourceDefinition]]: ...
    def __reduce_ex__(self, protocol: int) -> Tuple[type, Tuple[Any, ...]]: ...
    def _microbatch_macro_is_core(self, project_name: str) -> bool: ...
    def use_microbatch_batches(self, project_name: str) -> bool: ...

class MacroManifest(MacroMethods):
    macros: Dict[UniqueID, Macro]
    metadata: ManifestMetadata
    flat_graph: Dict[str, Any]
    _macros_by_name: Optional[Dict[str, List[Macro]]]
    _macros_by_package: Optional[Dict[str, Dict[str, Macro]]]

    def __init__(self, macros: Dict[UniqueID, Macro]) -> None: ...

AnyManifest = Union[Manifest, MacroManifest]

def _check_duplicates(value: BaseNode, src: Mapping[UniqueID, BaseNode]) -> None: ...

K_T = TypeVar("K_T")
V_T = TypeVar("V_T")

def _expect_value(key: K_T, src: Mapping[K_T, V_T], old_file: AnySourceFile, name: str) -> V_T: ...