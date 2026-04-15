import enum
from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import chain
from multiprocessing.synchronize import Lock
from typing import (
    Any,
    Callable,
    ClassVar,
    DefaultDict,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import Protocol as ProtocolExt
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
    SeedNode,
    SemanticModel,
    SingularTestNode,
    SnapshotNode,
    SourceDefinition,
    UnitTestDefinition,
    UnitTestFileFixture,
    UnpatchedSourceDefinition,
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

PackageName: Type[str] = ...
DocName: Type[str] = ...
RefName: Type[str] = ...

def find_unique_id_for_package(
    storage: Dict[str, Dict[Optional[str], UniqueID]],
    key: str,
    package: Optional[str],
) -> Optional[UniqueID]: ...

class DocLookup(dbtClassMixin):
    storage: Dict[str, Dict[str, UniqueID]]
    
    def __init__(self, manifest: "Manifest") -> None: ...
    def get_unique_id(self, key: str, package: Optional[str]) -> Optional[UniqueID]: ...
    def find(self, key: str, package: Optional[str], manifest: "Manifest") -> Optional[Documentation]: ...
    def add_doc(self, doc: Documentation) -> None: ...
    def populate(self, manifest: "Manifest") -> None: ...
    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> Documentation: ...

class SourceLookup(dbtClassMixin):
    storage: Dict[str, Dict[str, UniqueID]]
    
    def __init__(self, manifest: "Manifest") -> None: ...
    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]: ...
    def find(self, search_name: str, package: Optional[str], manifest: "Manifest") -> Optional[SourceDefinition]: ...
    def add_source(self, source: SourceDefinition) -> None: ...
    def populate(self, manifest: "Manifest") -> None: ...
    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> SourceDefinition: ...

class RefableLookup(dbtClassMixin):
    _lookup_types: ClassVar[Set[NodeType]]
    _versioned_types: ClassVar[Set[NodeType]]
    storage: Dict[str, Dict[str, UniqueID]]
    
    def __init__(self, manifest: "Manifest") -> None: ...
    def get_unique_id(
        self,
        key: str,
        package: Optional[str],
        version: Optional[NodeVersion],
        node: Optional[ManifestNode] = None,
    ) -> Optional[UniqueID]: ...
    def find(
        self,
        key: str,
        package: Optional[str],
        version: Optional[NodeVersion],
        manifest: "Manifest",
        source_node: Optional[ManifestNode] = None,
    ) -> Optional[ManifestNode]: ...
    def add_node(self, node: ManifestNode) -> None: ...
    def populate(self, manifest: "Manifest") -> None: ...
    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> ManifestNode: ...
    def _find_unique_ids_for_package(self, key: str, package: Optional[str]) -> List[UniqueID]: ...

class MetricLookup(dbtClassMixin):
    storage: Dict[str, Dict[str, UniqueID]]
    
    def __init__(self, manifest: "Manifest") -> None: ...
    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]: ...
    def find(self, search_name: str, package: Optional[str], manifest: "Manifest") -> Optional[Metric]: ...
    def add_metric(self, metric: Metric) -> None: ...
    def populate(self, manifest: "Manifest") -> None: ...
    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> Metric: ...

class SavedQueryLookup(dbtClassMixin):
    storage: Dict[str, Dict[str, UniqueID]]
    
    def __init__(self, manifest: "Manifest") -> None: ...
    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]: ...
    def find(self, search_name: str, package: Optional[str], manifest: "Manifest") -> Optional[SavedQuery]: ...
    def add_saved_query(self, saved_query: SavedQuery) -> None: ...
    def populate(self, manifest: "Manifest") -> None: ...
    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> SavedQuery: ...

class SemanticModelByMeasureLookup(dbtClassMixin):
    storage: DefaultDict[str, Dict[str, UniqueID]]
    
    def __init__(self, manifest: "Manifest") -> None: ...
    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]: ...
    def find(self, search_name: str, package: Optional[str], manifest: "Manifest") -> Optional[SemanticModel]: ...
    def add(self, semantic_model: SemanticModel) -> None: ...
    def populate(self, manifest: "Manifest") -> None: ...
    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> SemanticModel: ...

class DisabledLookup(dbtClassMixin):
    storage: Dict[str, Dict[str, List[ManifestNode]]]
    
    def __init__(self, manifest: "Manifest") -> None: ...
    def populate(self, manifest: "Manifest") -> None: ...
    def add_node(self, node: ManifestNode) -> None: ...
    def find(
        self,
        search_name: str,
        package: Optional[str],
        version: Optional[NodeVersion] = None,
        resource_types: Optional[Set[NodeType]] = None,
    ) -> Optional[List[ManifestNode]]: ...

class AnalysisLookup(RefableLookup):
    _lookup_types: ClassVar[Set[NodeType]] = ...
    _versioned_types: ClassVar[Set[NodeType]] = ...

class SingularTestLookup(dbtClassMixin):
    storage: Dict[str, Dict[str, UniqueID]]
    
    def __init__(self, manifest: "Manifest") -> None: ...
    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[UniqueID]: ...
    def find(self, search_name: str, package: Optional[str], manifest: "Manifest") -> Optional[SingularTestNode]: ...
    def add_singular_test(self, source: SingularTestNode) -> None: ...
    def populate(self, manifest: "Manifest") -> None: ...
    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> SingularTestNode: ...

def _packages_to_search(
    current_project: str,
    node_package: str,
    target_package: Optional[str] = None,
) -> List[Optional[str]]: ...

def _sort_values(dct: Dict[Any, List[Any]]) -> Dict[Any, List[Any]]: ...

def build_node_edges(
    nodes: List[ManifestNode],
) -> Tuple[Dict[UniqueID, List[UniqueID]], Dict[UniqueID, List[UniqueID]]]: ...

def build_macro_edges(nodes: List[ManifestNode]) -> Dict[UniqueID, List[UniqueID]]: ...

def _deepcopy(value: dbtClassMixin) -> dbtClassMixin: ...

class Locality(enum.IntEnum):
    Core = 1
    Imported = 2
    Root = 3

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
    def from_macro(cls, candidate: MacroCandidate, specificity: int) -> "MaterializationCandidate": ...
    def __eq__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...

M = TypeVar("M", bound=MacroCandidate)

class CandidateList(List[M]):
    def last_candidate(self, valid_localities: Optional[Set[Locality]] = None) -> Optional[M]: ...
    def last(self) -> Optional[Macro]: ...

def _get_locality(macro: Macro, root_project_name: str, internal_packages: Set[str]) -> Locality: ...

class Searchable(ProtocolExt):
    @property
    def search_name(self) -> str: ...

D = TypeVar("D")

@dataclass
class Disabled(Generic[D]):
    pass

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
    def find_macro_by_name(
        self,
        name: str,
        root_project_name: str,
        package: Optional[str],
    ) -> Optional[Macro]: ...
    def find_generate_macro_by_name(
        self,
        component: str,
        root_project_name: str,
        imported_package: Optional[str] = None,
    ) -> Optional[Macro]: ...
    def _find_macros_by_name(
        self,
        name: str,
        root_project_name: str,
        filter: Optional[Callable[[MacroCandidate], bool]] = None,
    ) -> CandidateList[MacroCandidate]: ...
    def get_macros_by_name(self) -> Dict[str, List[Macro]]: ...
    @staticmethod
    def _build_macros_by_name(macros: Dict[UniqueID, Macro]) -> Dict[str, List[Macro]]: ...
    def get_macros_by_package(self) -> Dict[str, Dict[str, Macro]]: ...
    @staticmethod
    def _build_macros_by_package(macros: Dict[UniqueID, Macro]) -> Dict[str, Dict[str, Macro]]: ...

@dataclass
class ParsingInfo:
    static_analysis_parsed_path_count: int = ...
    static_analysis_path_count: int = ...

@dataclass
class ManifestStateCheck(dbtClassMixin):
    vars_hash: FileHash = ...
    project_env_vars_hash: FileHash = ...
    profile_env_vars_hash: FileHash = ...
    profile_hash: FileHash = ...
    project_hashes: Dict[str, FileHash] = ...

NodeClassT = TypeVar("NodeClassT", bound=BaseNode)
ResourceClassT = TypeVar("ResourceClassT", bound=BaseResource)

@dataclass
class Manifest(MacroMethods, dbtClassMixin):
    nodes: Dict[UniqueID, ManifestNode]
    sources: Dict[UniqueID, SourceDefinition]
    macros: Dict[UniqueID, Macro]
    docs: Dict[UniqueID, Documentation]
    exposures: Dict[UniqueID, Exposure]
    metrics: Dict[UniqueID, Metric]
    groups: Dict[UniqueID, Group]
    selectors: Dict[UniqueID, Any]
    files: Dict[str, AnySourceFile]
    metadata: ManifestMetadata
    flat_graph: Dict[str, Dict[UniqueID, Dict[str, Any]]]
    state_check: ManifestStateCheck
    source_patches: Dict[SourceKey, SourcePatch]
    disabled: Dict[UniqueID, List[ManifestNode]]
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
    
    def __pre_serialize__(self, context: Optional[Any] = None) -> "Manifest": ...
    @classmethod
    def __post_deserialize__(cls, obj: "Manifest") -> "Manifest": ...
    def build_flat_graph(self) -> None: ...
    def build_disabled_by_file_id(self) -> Dict[str, ManifestNode]: ...
    def _get_parent_adapter_types(self, adapter_type: str) -> List[str]: ...
    def _materialization_candidates_for(
        self,
        project_name: str,
        materialization_name: str,
        adapter_type: str,
        specificity: int,
    ) -> CandidateList[MaterializationCandidate]: ...
    def find_materialization_macro_by_name(
        self,
        project_name: str,
        materialization_name: str,
        adapter_type: str,
    ) -> Optional[Macro]: ...
    def get_resource_fqns(self) -> Dict[str, Set[Tuple[str, ...]]]: ...
    def get_used_schemas(self, resource_types: Optional[Set[NodeType]] = None) -> frozenset[Tuple[Optional[str], Optional[str]]]: ...
    def get_used_databases(self) -> frozenset[Optional[str]]: ...
    def deepcopy(self) -> "Manifest": ...
    def build_parent_and_child_maps(self) -> None: ...
    def build_macro_child_map(self) -> Dict[UniqueID, List[UniqueID]]: ...
    def build_group_map(self) -> None: ...
    def fill_tracking_metadata(self) -> None: ...
    @classmethod
    def from_writable_manifest(cls, writable_manifest: WritableManifest) -> "Manifest": ...
    def _map_nodes_to_map_resources(self, nodes_map: Dict[UniqueID, ManifestNode]) -> Dict[UniqueID, BaseResource]: ...
    def _map_list_nodes_to_map_list_resources(
        self,
        nodes_map: Dict[UniqueID, List[ManifestNode]],
    ) -> Dict[UniqueID, List[BaseResource]]: ...
    @classmethod
    def _map_resources_to_map_nodes(
        cls,
        resources_map: Dict[UniqueID, BaseResource],
    ) -> Dict[UniqueID, ManifestNode]: ...
    @classmethod
    def _map_list_resources_to_map_list_nodes(
        cls,
        resources_map: Dict[UniqueID, List[BaseResource]],
    ) -> Dict[UniqueID, List[ManifestNode]]: ...
    def writable_manifest(self) -> WritableManifest: ...
    def write(self, path: str) -> None: ...
    def expect(self, unique_id: UniqueID) -> ManifestNode: ...
    @property
    def doc_lookup(self) -> DocLookup: ...
    def rebuild_doc_lookup(self) -> None: ...
    @property
    def source_lookup(self) -> SourceLookup: ...
    def rebuild_source_lookup(self) -> None: ...
    @property
    def ref_lookup(self) -> RefableLookup: ...
    @property
    def metric_lookup(self