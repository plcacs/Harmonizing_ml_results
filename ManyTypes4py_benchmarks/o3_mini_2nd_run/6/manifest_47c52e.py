from __future__ import annotations
import enum
from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import chain
from multiprocessing.synchronize import Lock
from typing import Any, Callable, ClassVar, DefaultDict, Dict, Generic, List, Mapping, MutableMapping, Optional, Set, Tuple, TypeVar, Union, FrozenSet
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

def find_unique_id_for_package(storage: Mapping[str, Mapping[str, str]], key: str, package: Optional[str]) -> Optional[str]:
    if key not in storage:
        return None
    pkg_dct = storage[key]
    if package is None:
        if not pkg_dct:
            return None
        else:
            return next(iter(pkg_dct.values()))
    elif package in pkg_dct:
        return pkg_dct[package]
    else:
        return None

class DocLookup(dbtClassMixin):
    def __init__(self, manifest: Manifest) -> None:
        self.storage: Dict[str, Dict[str, str]] = {}
        self.populate(manifest)

    def get_unique_id(self, key: str, package: Optional[str]) -> Optional[str]:
        return find_unique_id_for_package(self.storage, key, package)

    def find(self, key: str, package: Optional[str], manifest: Manifest) -> Optional[Documentation]:
        unique_id = self.get_unique_id(key, package)
        if unique_id is not None:
            return self.perform_lookup(unique_id, manifest)
        return None

    def add_doc(self, doc: Documentation) -> None:
        if doc.name not in self.storage:
            self.storage[doc.name] = {}
        self.storage[doc.name][doc.package_name] = doc.unique_id

    def populate(self, manifest: Manifest) -> None:
        for doc in manifest.docs.values():
            self.add_doc(doc)

    def perform_lookup(self, unique_id: str, manifest: Manifest) -> Documentation:
        if unique_id not in manifest.docs:
            raise dbt_common.exceptions.DbtInternalError(f'Doc {unique_id} found in cache but not found in manifest')
        return manifest.docs[unique_id]

class SourceLookup(dbtClassMixin):
    def __init__(self, manifest: Manifest) -> None:
        self.storage: Dict[str, Dict[str, str]] = {}
        self.populate(manifest)

    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[str]:
        return find_unique_id_for_package(self.storage, search_name, package)

    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[SourceDefinition]:
        unique_id = self.get_unique_id(search_name, package)
        if unique_id is not None:
            return self.perform_lookup(unique_id, manifest)
        return None

    def add_source(self, source: SourceDefinition) -> None:
        if source.search_name not in self.storage:
            self.storage[source.search_name] = {}
        self.storage[source.search_name][source.package_name] = source.unique_id

    def populate(self, manifest: Manifest) -> None:
        for source in manifest.sources.values():
            if hasattr(source, 'source_name'):
                self.add_source(source)

    def perform_lookup(self, unique_id: str, manifest: Manifest) -> SourceDefinition:
        if unique_id not in manifest.sources:
            raise dbt_common.exceptions.DbtInternalError(f'Source {unique_id} found in cache but not found in manifest')
        return manifest.sources[unique_id]

class RefableLookup(dbtClassMixin):
    _lookup_types: Set[str] = set(REFABLE_NODE_TYPES)
    _versioned_types: Set[str] = set(VERSIONED_NODE_TYPES)

    def __init__(self, manifest: Manifest) -> None:
        self.storage: Dict[str, Dict[str, str]] = {}
        self.populate(manifest)

    def get_unique_id(self, key: str, package: Optional[str], version: Optional[str], node: Optional[GraphMemberNode] = None) -> Optional[str]:
        if version:
            key = f'{key}.v{version}'
        unique_ids = self._find_unique_ids_for_package(key, package)
        if len(unique_ids) > 1:
            raise AmbiguousResourceNameRefError(key, unique_ids, node)
        else:
            return unique_ids[0] if unique_ids else None

    def find(self, key: str, package: Optional[str], version: Optional[str], manifest: Manifest, source_node: Optional[GraphMemberNode] = None) -> Optional[ManifestNode]:
        unique_id = self.get_unique_id(key, package, version, source_node)
        if unique_id is not None:
            node = self.perform_lookup(unique_id, manifest)
            if isinstance(node, ModelNode) and version is None and node.is_versioned and get_node_info():
                max_version = max([UnparsedVersion(v.version) for v in manifest.nodes.values() if isinstance(v, ModelNode) and v.name == node.name and (v.version is not None)])
                assert node.latest_version is not None
                if max_version > UnparsedVersion(node.latest_version):
                    fire_event(UnpinnedRefNewVersionAvailable(node_info=get_node_info(), ref_node_name=node.name, ref_node_package=node.package_name, ref_node_version=str(node.version), ref_max_version=str(max_version.v)))
            return node
        return None

    def add_node(self, node: ManifestNode) -> None:
        if node.resource_type in self._lookup_types:
            if node.name not in self.storage:
                self.storage[node.name] = {}
            if node.is_versioned:
                if node.search_name not in self.storage:
                    self.storage[node.search_name] = {}
                self.storage[node.search_name][node.package_name] = node.unique_id
                if node.is_latest_version:
                    self.storage[node.name][node.package_name] = node.unique_id
            else:
                self.storage[node.name][node.package_name] = node.unique_id

    def populate(self, manifest: Manifest) -> None:
        for node in manifest.nodes.values():
            self.add_node(node)

    def perform_lookup(self, unique_id: str, manifest: Manifest) -> ManifestNode:
        if unique_id in manifest.nodes:
            node = manifest.nodes[unique_id]
        else:
            raise dbt_common.exceptions.DbtInternalError(f'Node {unique_id} found in cache but not found in manifest')
        return node

    def _find_unique_ids_for_package(self, key: str, package: Optional[str]) -> List[str]:
        if key not in self.storage:
            return []
        pkg_dct = self.storage[key]
        if package is None:
            if not pkg_dct:
                return []
            else:
                return list(pkg_dct.values())
        elif package in pkg_dct:
            return [pkg_dct[package]]
        else:
            return []

class MetricLookup(dbtClassMixin):
    def __init__(self, manifest: Manifest) -> None:
        self.storage: Dict[str, Dict[str, str]] = {}
        self.populate(manifest)

    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[str]:
        return find_unique_id_for_package(self.storage, search_name, package)

    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[Metric]:
        unique_id = self.get_unique_id(search_name, package)
        if unique_id is not None:
            return self.perform_lookup(unique_id, manifest)
        return None

    def add_metric(self, metric: Metric) -> None:
        if metric.search_name not in self.storage:
            self.storage[metric.search_name] = {}
        self.storage[metric.search_name][metric.package_name] = metric.unique_id

    def populate(self, manifest: Manifest) -> None:
        for metric in manifest.metrics.values():
            if hasattr(metric, 'name'):
                self.add_metric(metric)

    def perform_lookup(self, unique_id: str, manifest: Manifest) -> Metric:
        if unique_id not in manifest.metrics:
            raise dbt_common.exceptions.DbtInternalError(f'Metric {unique_id} found in cache but not found in manifest')
        return manifest.metrics[unique_id]

class SavedQueryLookup(dbtClassMixin):
    """Lookup utility for finding SavedQuery nodes"""
    def __init__(self, manifest: Manifest) -> None:
        self.storage: Dict[str, Dict[str, str]] = {}
        self.populate(manifest)

    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[str]:
        return find_unique_id_for_package(self.storage, search_name, package)

    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[SavedQuery]:
        unique_id = self.get_unique_id(search_name, package)
        if unique_id is not None:
            return self.perform_lookup(unique_id, manifest)
        return None

    def add_saved_query(self, saved_query: SavedQuery) -> None:
        if saved_query.search_name not in self.storage:
            self.storage[saved_query.search_name] = {}
        self.storage[saved_query.search_name][saved_query.package_name] = saved_query.unique_id

    def populate(self, manifest: Manifest) -> None:
        for saved_query in manifest.saved_queries.values():
            if hasattr(saved_query, 'name'):
                self.add_saved_query(saved_query)

    def perform_lookup(self, unique_id: str, manifest: Manifest) -> SavedQuery:
        if unique_id not in manifest.saved_queries:
            raise dbt_common.exceptions.DbtInternalError(f'SavedQUery {unique_id} found in cache but not found in manifest')
        return manifest.saved_queries[unique_id]

class SemanticModelByMeasureLookup(dbtClassMixin):
    """Lookup utility for finding SemanticModel by measure

    This is possible because measure names are supposed to be unique across
    the semantic models in a manifest.
    """
    def __init__(self, manifest: Manifest) -> None:
        self.storage: DefaultDict[str, Dict[str, str]] = defaultdict(dict)
        self.populate(manifest)

    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[str]:
        return find_unique_id_for_package(self.storage, search_name, package)

    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[SemanticModel]:
        """Tries to find a SemanticModel based on a measure name"""
        unique_id = self.get_unique_id(search_name, package)
        if unique_id is not None:
            return self.perform_lookup(unique_id, manifest)
        return None

    def add(self, semantic_model: SemanticModel) -> None:
        """Sets all measures for a SemanticModel as paths to the SemanticModel's `unique_id`"""
        for measure in semantic_model.measures:
            self.storage[measure.name][semantic_model.package_name] = semantic_model.unique_id

    def populate(self, manifest: Manifest) -> None:
        """Populate storage with all the measure + package paths to the Manifest's SemanticModels"""
        for semantic_model in manifest.semantic_models.values():
            self.add(semantic_model=semantic_model)
        for disabled in manifest.disabled.values():
            for node in disabled:
                if isinstance(node, SemanticModel):
                    self.add(semantic_model=node)

    def perform_lookup(self, unique_id: str, manifest: Manifest) -> SemanticModel:
        """Tries to get a SemanticModel from the Manifest"""
        enabled_semantic_model = manifest.semantic_models.get(unique_id)
        disabled_semantic_model = manifest.disabled.get(unique_id)
        if isinstance(enabled_semantic_model, SemanticModel):
            return enabled_semantic_model
        elif disabled_semantic_model is not None and isinstance(disabled_semantic_model[0], SemanticModel):
            return disabled_semantic_model[0]
        else:
            raise dbt_common.exceptions.DbtInternalError(f'Semantic model `{unique_id}` found in cache but not found in manifest')

class DisabledLookup(dbtClassMixin):
    def __init__(self, manifest: Manifest) -> None:
        self.storage: Dict[str, Dict[str, List[ManifestNode]]] = {}
        self.populate(manifest)

    def populate(self, manifest: Manifest) -> None:
        for node in list(chain.from_iterable(manifest.disabled.values())):
            self.add_node(node)

    def add_node(self, node: ManifestNode) -> None:
        if node.search_name not in self.storage:
            self.storage[node.search_name] = {}
        if node.package_name not in self.storage[node.search_name]:
            self.storage[node.search_name][node.package_name] = []
        self.storage[node.search_name][node.package_name].append(node)

    def find(self, search_name: str, package: Optional[str], version: Optional[str] = None, resource_types: Optional[Set[str]] = None) -> Optional[List[ManifestNode]]:
        if version:
            search_name = f'{search_name}.v{version}'
        if search_name not in self.storage:
            return None
        pkg_dct = self.storage[search_name]
        nodes: List[ManifestNode] = []
        if package is None:
            if not pkg_dct:
                return None
            else:
                nodes = next(iter(pkg_dct.values()))
        elif package in pkg_dct:
            nodes = pkg_dct[package]
        else:
            return None
        if resource_types is None:
            return nodes
        else:
            new_nodes: List[ManifestNode] = []
            for node in nodes:
                if node.resource_type in resource_types:
                    new_nodes.append(node)
            if not new_nodes:
                return None
            else:
                return new_nodes

class AnalysisLookup(RefableLookup):
    _lookup_types: Set[str] = set([NodeType.Analysis])
    _versioned_types: Set[str] = set()

class SingularTestLookup(dbtClassMixin):
    def __init__(self, manifest: Manifest) -> None:
        self.storage: Dict[str, Dict[str, str]] = {}
        self.populate(manifest)

    def get_unique_id(self, search_name: str, package: Optional[str]) -> Optional[str]:
        return find_unique_id_for_package(self.storage, search_name, package)

    def find(self, search_name: str, package: Optional[str], manifest: Manifest) -> Optional[SingularTestNode]:
        unique_id = self.get_unique_id(search_name, package)
        if unique_id is not None:
            return self.perform_lookup(unique_id, manifest)
        return None

    def add_singular_test(self, source: SingularTestNode) -> None:
        if source.search_name not in self.storage:
            self.storage[source.search_name] = {}
        self.storage[source.search_name][source.package_name] = source.unique_id

    def populate(self, manifest: Manifest) -> None:
        for node in manifest.nodes.values():
            if isinstance(node, SingularTestNode):
                self.add_singular_test(node)

    def perform_lookup(self, unique_id: str, manifest: Manifest) -> SingularTestNode:
        if unique_id not in manifest.nodes:
            raise dbt_common.exceptions.DbtInternalError(f'Singular test {unique_id} found in cache but not found in manifest')
        node = manifest.nodes[unique_id]
        assert isinstance(node, SingularTestNode)
        return node

def _packages_to_search(current_project: str, node_package: str, target_package: Optional[str] = None) -> List[Optional[str]]:
    if target_package is not None:
        return [target_package]
    elif current_project == node_package:
        return [current_project, None]
    else:
        return [current_project, node_package, None]

def _sort_values(dct: Mapping[Any, List[Any]]) -> Dict[Any, List[Any]]:
    """Given a dictionary, sort each value. This makes output deterministic,
    which helps for tests.
    """
    return {k: sorted(v) for k, v in dct.items()}

def build_node_edges(nodes: List[ManifestNode]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build the forward and backward edges on the given list of ManifestNodes
    and return them as two separate dictionaries, each mapping unique IDs to
    lists of edges.
    """
    backward_edges: Dict[str, List[str]] = {}
    forward_edges: Dict[str, List[str]] = {n.unique_id: [] for n in nodes}
    for node in nodes:
        backward_edges[node.unique_id] = node.depends_on_nodes[:]  # type: ignore
        for unique_id in backward_edges[node.unique_id]:
            if unique_id in forward_edges.keys():
                forward_edges[unique_id].append(node.unique_id)
    return (_sort_values(forward_edges), _sort_values(backward_edges))

def build_macro_edges(nodes: List[BaseNode]) -> Dict[str, List[str]]:
    forward_edges: Dict[str, List[str]] = {n.unique_id: [] for n in nodes if n.unique_id.startswith('macro') or n.depends_on_macros}
    for node in nodes:
        for unique_id in node.depends_on_macros:
            if unique_id in forward_edges.keys():
                forward_edges[unique_id].append(node.unique_id)
    return _sort_values(forward_edges)

T_co = TypeVar('T_co', covariant=True)
def _deepcopy(value: T_co) -> T_co:
    return value.from_dict(value.to_dict(omit_none=True))

class Locality(enum.IntEnum):
    Core = 1
    Imported = 2
    Root = 3

@dataclass
class MacroCandidate:
    locality: Locality
    macro: Macro

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MacroCandidate):
            return NotImplemented
        return self.locality == other.locality

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, MacroCandidate):
            return NotImplemented
        if self.locality < other.locality:
            return True
        if self.locality > other.locality:
            return False
        return False

@dataclass
class MaterializationCandidate(MacroCandidate):
    specificity: int

    @classmethod
    def from_macro(cls, candidate: MacroCandidate, specificity: int) -> MaterializationCandidate:
        return cls(locality=candidate.locality, macro=candidate.macro, specificity=specificity)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MaterializationCandidate):
            return NotImplemented
        equal = self.specificity == other.specificity and self.locality == other.locality
        if equal:
            raise DuplicateMaterializationNameError(self.macro, other)
        return equal

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, MaterializationCandidate):
            return NotImplemented
        if self.specificity > other.specificity:
            return True
        if self.specificity < other.specificity:
            return False
        if self.locality < other.locality:
            return True
        if self.locality > other.locality:
            return False
        return False

M = TypeVar('M', bound=MacroCandidate)

class CandidateList(List[M]):
    def last_candidate(self, valid_localities: Optional[Set[Locality]] = None) -> Optional[M]:
        """
        Obtain the last (highest precedence) MacroCandidate from the CandidateList of any locality in valid_localities.
        If valid_localities is not specified, return the last MacroCandidate of any locality.
        """
        if not self:
            return None
        self.sort()
        if valid_localities is None:
            return self[-1]
        for candidate in reversed(self):
            if candidate.locality in valid_localities:
                return candidate
        return None

    def last(self) -> Optional[Macro]:
        last_candidate = self.last_candidate()
        return last_candidate.macro if last_candidate is not None else None

class Searchable(Protocol):
    @property
    def search_name(self) -> str:
        raise NotImplementedError('search_name not implemented')

D = TypeVar('D')

@dataclass
class Disabled(Generic[D]):
    pass

MaybeMetricNode = Optional[Union[Metric, Disabled[Metric]]]
MaybeSavedQueryNode = Optional[Union[SavedQuery, Disabled[SavedQuery]]]
MaybeDocumentation = Optional[Documentation]
MaybeParsedSource = Optional[Union[SourceDefinition, Disabled[SourceDefinition]]]
MaybeNonSource = Optional[Union[ManifestNode, Disabled[ManifestNode]]]
T = TypeVar('T', bound=GraphMemberNode)

class MacroMethods:
    def __init__(self) -> None:
        self.macros: Dict[str, Macro] = {}
        self.metadata: ManifestMetadata = ManifestMetadata()
        self._macros_by_name: Optional[Dict[str, List[Macro]]] = None
        self._macros_by_package: Optional[Dict[str, Dict[str, Macro]]] = None

    def find_macro_candidate_by_name(self, name: str, root_project_name: str, package: Optional[str]) -> Optional[MacroCandidate]:
        """Find a MacroCandidate in the graph by its name and package name, or None for
        any package. The root project name is used to determine priority:
         - locally defined macros come first
         - then imported macros
         - then macros defined in the root project
        """
        filter_func: Optional[Callable[[MacroCandidate], bool]] = None
        if package is not None:
            def filter_func(candidate: MacroCandidate) -> bool:
                return package == candidate.macro.package_name
        candidates: CandidateList[MacroCandidate] = self._find_macros_by_name(name=name, root_project_name=root_project_name, filter=filter_func)
        return candidates.last_candidate()

    def find_macro_by_name(self, name: str, root_project_name: str, package: Optional[str]) -> Optional[Macro]:
        macro_candidate = self.find_macro_candidate_by_name(name=name, root_project_name=root_project_name, package=package)
        return macro_candidate.macro if macro_candidate else None

    def find_generate_macro_by_name(self, component: str, root_project_name: str, imported_package: Optional[str] = None) -> Optional[Macro]:
        """
        The default `generate_X_name` macros are similar to regular ones, but only
        includes imported packages when searching for a package.
        - if package is not provided:
            - if there is a `generate_{component}_name` macro in the root
              project, return it
            - return the `generate_{component}_name` macro from the 'dbt'
              internal project
        - if package is provided
            - return the `generate_{component}_name` macro from the imported
              package, if one exists
        """
        def filter_func(candidate: MacroCandidate) -> bool:
            if imported_package:
                return candidate.locality == Locality.Imported and imported_package == candidate.macro.package_name
            else:
                return candidate.locality != Locality.Imported
        candidates: CandidateList[MacroCandidate] = self._find_macros_by_name(name=f'generate_{component}_name', root_project_name=root_project_name, filter=filter_func)
        return candidates.last()

    def _find_macros_by_name(self, name: str, root_project_name: str, filter: Optional[Callable[[MacroCandidate], bool]] = None) -> CandidateList[MacroCandidate]:
        """Find macros by their name."""
        candidates: CandidateList[MacroCandidate] = CandidateList()
        macros_by_name = self.get_macros_by_name()
        if name not in macros_by_name:
            return candidates
        packages: Set[str] = set(get_adapter_package_names(self.metadata.adapter_type))
        for macro in macros_by_name[name]:
            candidate = MacroCandidate(locality=_get_locality(macro, root_project_name, packages), macro=macro)
            if filter is None or filter(candidate):
                candidates.append(candidate)
        return candidates

    def get_macros_by_name(self) -> Dict[str, List[Macro]]:
        if self._macros_by_name is None:
            self._macros_by_name = self._build_macros_by_name(self.macros)
        return self._macros_by_name

    @staticmethod
    def _build_macros_by_name(macros: Dict[str, Macro]) -> Dict[str, List[Macro]]:
        macros_by_name: Dict[str, List[Macro]] = {}
        for macro in macros.values():
            if macro.name not in macros_by_name:
                macros_by_name[macro.name] = []
            macros_by_name[macro.name].append(macro)
        return macros_by_name

    def get_macros_by_package(self) -> Dict[str, Dict[str, Macro]]:
        if self._macros_by_package is None:
            self._macros_by_package = self._build_macros_by_package(self.macros)
        return self._macros_by_package

    @staticmethod
    def _build_macros_by_package(macros: Dict[str, Macro]) -> Dict[str, Dict[str, Macro]]:
        macros_by_package: Dict[str, Dict[str, Macro]] = {}
        for macro in macros.values():
            if macro.package_name not in macros_by_package:
                macros_by_package[macro.package_name] = {}
            macros_by_name = macros_by_package[macro.package_name]
            macros_by_name[macro.name] = macro
        return macros_by_package

def _get_locality(macro: Macro, root_project_name: str, internal_packages: Set[str]) -> Locality:
    if macro.package_name == root_project_name:
        return Locality.Root
    elif macro.package_name in internal_packages:
        return Locality.Core
    else:
        return Locality.Imported

@dataclass
class ParsingInfo:
    static_analysis_parsed_path_count: int = 0
    static_analysis_path_count: int = 0

@dataclass
class ManifestStateCheck(dbtClassMixin):
    vars_hash: FileHash = field(default_factory=FileHash.empty)
    project_env_vars_hash: FileHash = field(default_factory=FileHash.empty)
    profile_env_vars_hash: FileHash = field(default_factory=FileHash.empty)
    profile_hash: FileHash = field(default_factory=FileHash.empty)
    project_hashes: Dict[Any, Any] = field(default_factory=dict)

NodeClassT = TypeVar('NodeClassT', bound='BaseNode')
ResourceClassT = TypeVar('ResourceClassT', bound='BaseResource')

@dataclass
class Manifest(MacroMethods, dbtClassMixin):
    """The manifest for the full graph, after parsing and during compilation."""
    nodes: Dict[str, ManifestNode] = field(default_factory=dict)
    sources: Dict[str, SourceDefinition] = field(default_factory=dict)
    macros: Dict[str, Macro] = field(default_factory=dict)
    docs: Dict[str, Documentation] = field(default_factory=dict)
    exposures: Dict[str, Exposure] = field(default_factory=dict)
    metrics: Dict[str, Metric] = field(default_factory=dict)
    groups: Dict[str, Group] = field(default_factory=dict)
    selectors: Dict[str, Any] = field(default_factory=dict)
    files: Dict[str, AnySourceFile] = field(default_factory=dict)
    metadata: ManifestMetadata = field(default_factory=ManifestMetadata)
    flat_graph: Dict[str, Any] = field(default_factory=dict)
    state_check: ManifestStateCheck = field(default_factory=ManifestStateCheck)
    source_patches: Dict[str, SourcePatch] = field(default_factory=dict)
    disabled: Dict[str, List[ManifestNode]] = field(default_factory=dict)
    env_vars: Dict[str, Any] = field(default_factory=dict)
    semantic_models: Dict[str, SemanticModel] = field(default_factory=dict)
    unit_tests: Dict[str, UnitTestDefinition] = field(default_factory=dict)
    saved_queries: Dict[str, SavedQuery] = field(default_factory=dict)
    fixtures: Dict[str, Any] = field(default_factory=dict)
    _doc_lookup: Optional[DocLookup] = field(default=None, metadata={'serialize': lambda x: None, 'deserialize': lambda x: None})
    _source_lookup: Optional[SourceLookup] = field(default=None, metadata={'serialize': lambda x: None, 'deserialize': lambda x: None})
    _ref_lookup: Optional[RefableLookup] = field(default=None, metadata={'serialize': lambda x: None, 'deserialize': lambda x: None})
    _metric_lookup: Optional[MetricLookup] = field(default=None, metadata={'serialize': lambda x: None, 'deserialize': lambda x: None})
    _saved_query_lookup: Optional[SavedQueryLookup] = field(default=None, metadata={'serialize': lambda x: None, 'deserialize': lambda x: None})
    _semantic_model_by_measure_lookup: Optional[SemanticModelByMeasureLookup] = field(default=None, metadata={'serialize': lambda x: None, 'deserialize': lambda x: None})
    _disabled_lookup: Optional[DisabledLookup] = field(default=None, metadata={'serialize': lambda x: None, 'deserialize': lambda x: None})
    _analysis_lookup: Optional[AnalysisLookup] = field(default=None, metadata={'serialize': lambda x: None, 'deserialize': lambda x: None})
    _singular_test_lookup: Optional[SingularTestLookup] = field(default=None, metadata={'serialize': lambda x: None, 'deserialize': lambda x: None})
    _parsing_info: ParsingInfo = field(default_factory=ParsingInfo, metadata={'serialize': lambda x: None, 'deserialize': lambda x: None})
    _lock: Lock = field(default_factory=get_mp_context().Lock, metadata={'serialize': lambda x: None, 'deserialize': lambda x: None})
    _macros_by_name: Optional[Dict[str, List[Macro]]] = field(default=None, metadata={'serialize': lambda x: None, 'deserialize': lambda x: None})
    _macros_by_package: Optional[Dict[str, Dict[str, Macro]]] = field(default=None, metadata={'serialize': lambda x: None, 'deserialize': lambda x: None})

    def __pre_serialize__(self, context: Optional[Any] = None) -> Manifest:
        self.source_patches = {}
        return self

    @classmethod
    def __post_deserialize__(cls, obj: Manifest) -> Manifest:
        obj._lock = get_mp_context().Lock()
        return obj

    def build_flat_graph(self) -> None:
        """This attribute is used in context.common by each node, so we want to
        only build it once and avoid any concurrency issues around it.
        Make sure you don't call this until you're done with building your
        manifest!
        """
        self.flat_graph = {
            'exposures': {k: v.to_dict(omit_none=False) for k, v in self.exposures.items()},
            'groups': {k: v.to_dict(omit_none=False) for k, v in self.groups.items()},
            'metrics': {k: v.to_dict(omit_none=False) for k, v in self.metrics.items()},
            'nodes': {k: v.to_dict(omit_none=False) for k, v in self.nodes.items()},
            'sources': {k: v.to_dict(omit_none=False) for k, v in self.sources.items()},
            'semantic_models': {k: v.to_dict(omit_none=False) for k, v in self.semantic_models.items()},
            'saved_queries': {k: v.to_dict(omit_none=False) for k, v in self.saved_queries.items()}
        }

    def build_disabled_by_file_id(self) -> Dict[str, ManifestNode]:
        disabled_by_file_id: Dict[str, ManifestNode] = {}
        for node_list in self.disabled.values():
            for node in node_list:
                disabled_by_file_id[node.file_id] = node
        return disabled_by_file_id

    def _get_parent_adapter_types(self, adapter_type: str) -> List[str]:
        from dbt.adapters.factory import get_adapter_type_names
        return get_adapter_type_names(adapter_type) + ['default']

    def _materialization_candidates_for(self, project_name: str, materialization_name: str, adapter_type: str, specificity: int) -> CandidateList[MaterializationCandidate]:
        full_name = dbt_common.utils.get_materialization_macro_name(materialization_name=materialization_name, adapter_type=adapter_type, with_prefix=False)
        return CandidateList([MaterializationCandidate.from_macro(m, specificity) for m in self._find_macros_by_name(full_name, project_name)])

    def find_materialization_macro_by_name(self, project_name: str, materialization_name: str, adapter_type: str) -> Optional[Macro]:
        candidates = CandidateList(chain.from_iterable(
            (self._materialization_candidates_for(
                project_name=project_name,
                materialization_name=materialization_name,
                adapter_type=atype,
                specificity=specificity
            ) for specificity, atype in enumerate(self._get_parent_adapter_types(adapter_type)))
        ))
        core_candidates = [candidate for candidate in candidates if candidate.locality == Locality.Core]
        materialization_candidate = candidates.last_candidate()
        if materialization_candidate is not None and materialization_candidate.locality == Locality.Imported and core_candidates:
            if get_flags().require_explicit_package_overrides_for_builtin_materializations is False:
                deprecations.warn('package-materialization-override', package_name=materialization_candidate.macro.package_name, materialization_name=materialization_name)
            else:
                materialization_candidate = candidates.last_candidate(valid_localities={Locality.Core, Locality.Root})
        return materialization_candidate.macro if materialization_candidate else None

    def get_resource_fqns(self) -> Dict[str, Set[Tuple[str, ...]]]:
        resource_fqns: Dict[str, Set[Tuple[str, ...]]] = {}
        all_resources = chain(self.exposures.values(), self.nodes.values(), self.sources.values(), self.metrics.values(), self.semantic_models.values(), self.saved_queries.values(), self.unit_tests.values())
        for resource in all_resources:
            resource_type_plural = resource.resource_type.pluralize()
            if resource_type_plural not in resource_fqns:
                resource_fqns[resource_type_plural] = set()
            resource_fqns[resource_type_plural].add(tuple(resource.fqn))
        return resource_fqns

    def get_used_schemas(self, resource_types: Optional[Set[str]] = None) -> FrozenSet[Tuple[str, str]]:
        return frozenset({(node.database, node.schema) for node in chain(self.nodes.values(), self.sources.values()) if (not resource_types or node.resource_type in resource_types)})

    def get_used_databases(self) -> FrozenSet[str]:
        return frozenset((x.database for x in chain(self.nodes.values(), self.sources.values())))

    def deepcopy(self) -> Manifest:
        copy = Manifest(
            nodes={k: _deepcopy(v) for k, v in self.nodes.items()},
            sources={k: _deepcopy(v) for k, v in self.sources.items()},
            macros={k: _deepcopy(v) for k, v in self.macros.items()},
            docs={k: _deepcopy(v) for k, v in self.docs.items()},
            exposures={k: _deepcopy(v) for k, v in self.exposures.items()},
            metrics={k: _deepcopy(v) for k, v in self.metrics.items()},
            groups={k: _deepcopy(v) for k, v in self.groups.items()},
            selectors={k: _deepcopy(v) for k, v in self.selectors.items()},
            metadata=self.metadata,
            disabled={k: _deepcopy(v) for k, v in self.disabled.items()},
            files={k: _deepcopy(v) for k, v in self.files.items()},
            state_check=_deepcopy(self.state_check),
            semantic_models={k: _deepcopy(v) for k, v in self.semantic_models.items()},
            unit_tests={k: _deepcopy(v) for k, v in self.unit_tests.items()},
            saved_queries={k: _deepcopy(v) for k, v in self.saved_queries.items()}
        )
        copy.build_flat_graph()
        return copy

    def build_parent_and_child_maps(self) -> None:
        edge_members = list(chain(self.nodes.values(), self.sources.values(), self.exposures.values(), self.metrics.values(), self.semantic_models.values(), self.saved_queries.values(), self.unit_tests.values()))
        forward_edges, backward_edges = build_node_edges(edge_members)
        self.child_map = forward_edges
        self.parent_map = backward_edges

    def build_macro_child_map(self) -> Dict[str, List[str]]:
        edge_members = list(chain(self.nodes.values(), self.macros.values()))
        forward_edges = build_macro_edges(edge_members)
        return forward_edges

    def build_group_map(self) -> None:
        groupable_nodes = list(chain(self.nodes.values(), self.saved_queries.values(), self.semantic_models.values(), self.metrics.values()))
        group_map: Dict[str, List[str]] = {group.name: [] for group in self.groups.values()}
        for node in groupable_nodes:
            if node.group is not None:
                if node.group in group_map:
                    group_map[node.group].append(node.unique_id)
        self.group_map = group_map

    def fill_tracking_metadata(self) -> None:
        self.metadata.user_id = tracking.active_user.id if tracking.active_user else None
        self.metadata.send_anonymous_usage_stats = get_flags().SEND_ANONYMOUS_USAGE_STATS

    @classmethod
    def from_writable_manifest(cls, writable_manifest: WritableManifest) -> Manifest:
        manifest = Manifest(
            nodes=cls._map_resources_to_map_nodes(writable_manifest.nodes),
            disabled=cls._map_list_resources_to_map_list_nodes(writable_manifest.disabled),
            unit_tests=cls._map_resources_to_map_nodes(writable_manifest.unit_tests),
            sources=cls._map_resources_to_map_nodes(writable_manifest.sources),
            macros=cls._map_resources_to_map_nodes(writable_manifest.macros),
            docs=cls._map_resources_to_map_nodes(writable_manifest.docs),
            exposures=cls._map_resources_to_map_nodes(writable_manifest.exposures),
            metrics=cls._map_resources_to_map_nodes(writable_manifest.metrics),
            groups=cls._map_resources_to_map_nodes(writable_manifest.groups),
            semantic_models=cls._map_resources_to_map_nodes(writable_manifest.semantic_models),
            saved_queries=cls._map_resources_to_map_nodes(writable_manifest.saved_queries),
            selectors={selector_id: selector for selector_id, selector in writable_manifest.selectors.items()}
        )
        return manifest

    def _map_nodes_to_map_resources(cls, nodes_map: Mapping[str, Any]) -> Dict[str, Any]:
        return {node_id: node.to_resource() for node_id, node in nodes_map.items()}

    def _map_list_nodes_to_map_list_resources(cls, nodes_map: Optional[Mapping[str, List[Any]]]) -> Dict[str, List[Any]]:
        return {node_id: [node.to_resource() for node in node_list] for node_id, node_list in nodes_map.items()}  # type: ignore

    @classmethod
    def _map_resources_to_map_nodes(cls, resources_map: Mapping[str, Any]) -> Dict[str, ManifestNode]:
        return {node_id: RESOURCE_CLASS_TO_NODE_CLASS[type(resource)].from_resource(resource) for node_id, resource in resources_map.items()}

    @classmethod
    def _map_list_resources_to_map_list_nodes(cls, resources_map: Optional[Mapping[str, List[Any]]]) -> Dict[str, List[ManifestNode]]:
        if resources_map is None:
            return {}
        return {node_id: [RESOURCE_CLASS_TO_NODE_CLASS[type(resource)].from_resource(resource) for resource in resource_list] for node_id, resource_list in resources_map.items()}

    def writable_manifest(self) -> WritableManifest:
        self.build_parent_and_child_maps()
        self.build_group_map()
        self.fill_tracking_metadata()
        return WritableManifest(
            nodes=self._map_nodes_to_map_resources(self.nodes),
            sources=self._map_nodes_to_map_resources(self.sources),
            macros=self._map_nodes_to_map_resources(self.macros),
            docs=self._map_nodes_to_map_resources(self.docs),
            exposures=self._map_nodes_to_map_resources(self.exposures),
            metrics=self._map_nodes_to_map_resources(self.metrics),
            groups=self._map_nodes_to_map_resources(self.groups),
            selectors=self.selectors,
            metadata=self.metadata,
            disabled=self._map_list_nodes_to_map_list_resources(self.disabled),
            child_map=self.child_map,
            parent_map=self.parent_map,
            group_map=self.group_map,
            semantic_models=self._map_nodes_to_map_resources(self.semantic_models),
            unit_tests=self._map_nodes_to_map_resources(self.unit_tests),
            saved_queries=self._map_nodes_to_map_resources(self.saved_queries)
        )

    def write(self, path: str) -> None:
        writable = self.writable_manifest()
        writable.write(path)
        fire_event(ArtifactWritten(artifact_type=writable.__class__.__name__, artifact_path=path))

    def expect(self, unique_id: str) -> Union[ManifestNode, SourceDefinition, Exposure, Metric, SemanticModel, UnitTestDefinition, SavedQuery]:
        if unique_id in self.nodes:
            return self.nodes[unique_id]
        elif unique_id in self.sources:
            return self.sources[unique_id]
        elif unique_id in self.exposures:
            return self.exposures[unique_id]
        elif unique_id in self.metrics:
            return self.metrics[unique_id]
        elif unique_id in self.semantic_models:
            return self.semantic_models[unique_id]
        elif unique_id in self.unit_tests:
            return self.unit_tests[unique_id]
        elif unique_id in self.saved_queries:
            return self.saved_queries[unique_id]
        else:
            raise dbt_common.exceptions.DbtInternalError('Expected node {} not found in manifest'.format(unique_id))

    @property
    def doc_lookup(self) -> DocLookup:
        if self._doc_lookup is None:
            self._doc_lookup = DocLookup(self)
        return self._doc_lookup

    def rebuild_doc_lookup(self) -> None:
        self._doc_lookup = DocLookup(self)

    @property
    def source_lookup(self) -> SourceLookup:
        if self._source_lookup is None:
            self._source_lookup = SourceLookup(self)
        return self._source_lookup

    def rebuild_source_lookup(self) -> None:
        self._source_lookup = SourceLookup(self)

    @property
    def ref_lookup(self) -> RefableLookup:
        if self._ref_lookup is None:
            self._ref_lookup = RefableLookup(self)
        return self._ref_lookup

    @property
    def metric_lookup(self) -> MetricLookup:
        if self._metric_lookup is None:
            self._metric_lookup = MetricLookup(self)
        return self._metric_lookup

    @property
    def saved_query_lookup(self) -> SavedQueryLookup:
        """Retuns a SavedQueryLookup, instantiating it first if necessary."""
        if self._saved_query_lookup is None:
            self._saved_query_lookup = SavedQueryLookup(self)
        return self._saved_query_lookup

    @property
    def semantic_model_by_measure_lookup(self) -> SemanticModelByMeasureLookup:
        """Gets (and creates if necessary) the lookup utility for getting SemanticModels by measures"""
        if self._semantic_model_by_measure_lookup is None:
            self._semantic_model_by_measure_lookup = SemanticModelByMeasureLookup(self)
        return self._semantic_model_by_measure_lookup

    def rebuild_ref_lookup(self) -> None:
        self._ref_lookup = RefableLookup(self)

    @property
    def disabled_lookup(self) -> DisabledLookup:
        if self._disabled_lookup is None:
            self._disabled_lookup = DisabledLookup(self)
        return self._disabled_lookup

    def rebuild_disabled_lookup(self) -> None:
        self._disabled_lookup = DisabledLookup(self)

    @property
    def analysis_lookup(self) -> AnalysisLookup:
        if self._analysis_lookup is None:
            self._analysis_lookup = AnalysisLookup(self)
        return self._analysis_lookup

    @property
    def singular_test_lookup(self) -> SingularTestLookup:
        if self._singular_test_lookup is None:
            self._singular_test_lookup = SingularTestLookup(self)
        return self._singular_test_lookup

    @property
    def external_node_unique_ids(self) -> List[str]:
        return [node.unique_id for node in self.nodes.values() if node.is_external_node]

    def resolve_ref(self, source_node: GraphMemberNode, target_model_name: str, target_model_package: Optional[str], target_model_version: Optional[str], current_project: str, node_package: str) -> Optional[Union[ManifestNode, Disabled[ManifestNode]]]:
        node: Optional[ManifestNode] = None
        disabled: Optional[List[ManifestNode]] = None
        candidates = _packages_to_search(current_project, node_package, target_model_package)
        for pkg in candidates:
            node = self.ref_lookup.find(target_model_name, pkg, target_model_version, self, source_node)
            if node is not None and hasattr(node, 'config') and node.config.enabled:
                return node
            if disabled is None:
                disabled = self.disabled_lookup.find(target_model_name, pkg, version=target_model_version, resource_types=REFABLE_NODE_TYPES)
        if disabled:
            return Disabled(disabled[0])
        return None

    def resolve_source(self, target_source_name: str, target_table_name: str, current_project: str, node_package: str) -> Optional[Union[SourceDefinition, Disabled[SourceDefinition]]]:
        search_name = f'{target_source_name}.{target_table_name}'
        candidates = _packages_to_search(current_project, node_package)
        source: Optional[SourceDefinition] = None
        disabled: Optional[List[SourceDefinition]] = None
        for pkg in candidates:
            source = self.source_lookup.find(search_name, pkg, self)
            if source is not None and source.config.enabled:
                return source
            if disabled is None:
                disabled = self.disabled_lookup.find(f'{target_source_name}.{target_table_name}', pkg)
        if disabled:
            return Disabled(disabled[0])
        return None

    def resolve_metric(self, target_metric_name: str, target_metric_package: Optional[str], current_project: str, node_package: str) -> Optional[Union[Metric, Disabled[Metric]]]:
        metric: Optional[Metric] = None
        disabled: Optional[List[Metric]] = None
        candidates = _packages_to_search(current_project, node_package, target_metric_package)
        for pkg in candidates:
            metric = self.metric_lookup.find(target_metric_name, pkg, self)
            if metric is not None and metric.config.enabled:
                return metric
            if disabled is None:
                disabled = self.disabled_lookup.find(f'{target_metric_name}', pkg)
        if disabled:
            return Disabled(disabled[0])
        return None

    def resolve_saved_query(self, target_saved_query_name: str, target_saved_query_package: Optional[str], current_project: str, node_package: str) -> Optional[Union[SavedQuery, Disabled[SavedQuery]]]:
        """Tries to find the SavedQuery by name within the available project and packages.

        Will return the first enabled SavedQuery matching the name found while iterating over
        the scoped packages. If no enabled SavedQuery node match is found, returns the last
        disabled SavedQuery node. Otherwise it returns None.
        """
        disabled: Optional[List[SavedQuery]] = None
        candidates = _packages_to_search(current_project, node_package, target_saved_query_package)
        for pkg in candidates:
            saved_query = self.saved_query_lookup.find(target_saved_query_name, pkg, self)
            if saved_query is not None and saved_query.config.enabled:
                return saved_query
            if disabled is None:
                disabled = self.disabled_lookup.find(f'{target_saved_query_name}', pkg)
        if disabled:
            return Disabled(disabled[0])
        return None

    def resolve_semantic_model_for_measure(self, target_measure_name: str, current_project: str, node_package: str, target_package: Optional[str] = None) -> Optional[SemanticModel]:
        """Tries to find the SemanticModel that a measure belongs to"""
        candidates = _packages_to_search(current_project, node_package, target_package)
        for pkg in candidates:
            semantic_model = self.semantic_model_by_measure_lookup.find(target_measure_name, pkg, self)
            if semantic_model is not None:
                return semantic_model
        return None

    def resolve_doc(self, name: str, package: Optional[str], current_project: str, node_package: str) -> Optional[Documentation]:
        """Resolve the given documentation. This follows the same algorithm as
        resolve_ref except the is_enabled checks are unnecessary as docs are
        always enabled.
        """
        candidates = _packages_to_search(current_project, node_package, package)
        for pkg in candidates:
            result = self.doc_lookup.find(name, pkg, self)
            if result is not None:
                return result
        return None

    def is_invalid_private_ref(self, node: ManifestNode, target_model: ManifestNode, dependencies: Optional[Mapping[str, Any]]) -> bool:
        dependencies = dependencies or {}
        if not isinstance(target_model, ModelNode):
            return False
        is_private_ref = target_model.access == AccessType.Private and node.resource_type != NodeType.SqlOperation and (node.resource_type != NodeType.RPCCall)
        target_dependency = dependencies.get(target_model.package_name)
        restrict_package_access = target_dependency.restrict_access if target_dependency else False
        return is_private_ref and (not hasattr(node, 'group') or not node.group or node.group != target_model.group or (node.package_name != target_model.package_name and restrict_package_access))

    def is_invalid_protected_ref(self, node: ManifestNode, target_model: ManifestNode, dependencies: Optional[Mapping[str, Any]]) -> bool:
        dependencies = dependencies or {}
        if not isinstance(target_model, ModelNode):
            return False
        is_protected_ref = target_model.access == AccessType.Protected and node.resource_type != NodeType.SqlOperation and (node.resource_type != NodeType.RPCCall)
        target_dependency = dependencies.get(target_model.package_name)
        restrict_package_access = target_dependency.restrict_access if target_dependency else False
        return is_protected_ref and (node.package_name != target_model.package_name and restrict_package_access)

    def merge_from_artifact(self, other: Manifest) -> None:
        """Update this manifest by adding the 'defer_relation' attribute to all nodes
        with a counterpart in the stateful manifest used for deferral.

        Only non-ephemeral refable nodes are examined.
        """
        refables = set(REFABLE_NODE_TYPES)
        for unique_id, node in other.nodes.items():
            current = self.nodes.get(unique_id)
            if current and node.resource_type in refables and (not node.is_ephemeral):
                assert isinstance(node.config, NodeConfig)
                defer_relation = DeferRelation(database=node.database, schema=node.schema, alias=node.alias, relation_name=node.relation_name, resource_type=node.resource_type, name=node.name, description=node.description, compiled_code=node.compiled_code if not isinstance(node, SeedNode) else None, meta=node.meta, tags=node.tags, config=node.config)
                self.nodes[unique_id] = replace(current, defer_relation=defer_relation)
        self.build_flat_graph()

    def add_macro(self, source_file: SourceFile, macro: Macro) -> None:
        if macro.unique_id in self.macros:
            raise DuplicateMacroInPackageError(macro=macro, macro_mapping=self.macros)
        self.macros[macro.unique_id] = macro
        if self._macros_by_name is None:
            self._macros_by_name = self._build_macros_by_name(self.macros)
        if macro.name not in self._macros_by_name:
            self._macros_by_name[macro.name] = []
        self._macros_by_name[macro.name].append(macro)
        if self._macros_by_package is None:
            self._macros_by_package = self._build_macros_by_package(self.macros)
        if macro.package_name not in self._macros_by_package:
            self._macros_by_package[macro.package_name] = {}
        self._macros_by_package[macro.package_name][macro.name] = macro
        source_file.macros.append(macro.unique_id)

    def has_file(self, source_file: SourceFile) -> bool:
        key = source_file.file_id
        if key is None:
            return False
        if key not in self.files:
            return False
        my_checksum = self.files[key].checksum
        return my_checksum == source_file.checksum

    def add_source(self, source_file: SourceFile, source: SourceDefinition) -> None:
        _check_duplicates(source, self.sources)
        self.sources[source.unique_id] = source
        source_file.sources.append(source.unique_id)

    def add_node_nofile(self, node: ManifestNode) -> None:
        _check_duplicates(node, self.nodes)
        self.nodes[node.unique_id] = node

    def add_node(self, source_file: SourceFile, node: ManifestNode, test_from: Optional[str] = None) -> None:
        self.add_node_nofile(node)
        if isinstance(source_file, SchemaSourceFile):
            if isinstance(node, GenericTestNode):
                assert test_from
                source_file.add_test(node.unique_id, test_from)
            elif isinstance(node, Metric):
                source_file.metrics.append(node.unique_id)
            elif isinstance(node, Exposure):
                source_file.exposures.append(node.unique_id)
            elif isinstance(node, Group):
                source_file.groups.append(node.unique_id)
            elif isinstance(node, SnapshotNode):
                source_file.snapshots.append(node.unique_id)
        elif isinstance(source_file, FixtureSourceFile):
            pass
        else:
            source_file.nodes.append(node.unique_id)

    def add_exposure(self, source_file: SourceFile, exposure: Exposure) -> None:
        _check_duplicates(exposure, self.exposures)
        self.exposures[exposure.unique_id] = exposure
        source_file.exposures.append(exposure.unique_id)

    def add_metric(self, source_file: SourceFile, metric: Metric, generated_from: Optional[str] = None) -> None:
        _check_duplicates(metric, self.metrics)
        self.metrics[metric.unique_id] = metric
        if not generated_from:
            source_file.metrics.append(metric.unique_id)
        else:
            source_file.add_metrics_from_measures(generated_from, metric.unique_id)

    def add_group(self, source_file: SourceFile, group: Group) -> None:
        _check_duplicates(group, self.groups)
        self.groups[group.unique_id] = group
        source_file.groups.append(group.unique_id)

    def add_disabled_nofile(self, node: ManifestNode) -> None:
        if node.unique_id in self.disabled:
            self.disabled[node.unique_id].append(node)
        else:
            self.disabled[node.unique_id] = [node]

    def add_disabled(self, source_file: SourceFile, node: ManifestNode, test_from: Optional[str] = None) -> None:
        self.add_disabled_nofile(node)
        if isinstance(source_file, SchemaSourceFile):
            if isinstance(node, GenericTestNode):
                assert test_from
                source_file.add_test(node.unique_id, test_from)
            if isinstance(node, Metric):
                source_file.metrics.append(node.unique_id)
            if isinstance(node, SavedQuery):
                source_file.saved_queries.append(node.unique_id)
            if isinstance(node, SemanticModel):
                source_file.semantic_models.append(node.unique_id)
            if isinstance(node, Exposure):
                source_file.exposures.append(node.unique_id)
            if isinstance(node, UnitTestDefinition):
                source_file.unit_tests.append(node.unique_id)
        elif isinstance(source_file, FixtureSourceFile):
            pass
        else:
            source_file.nodes.append(node.unique_id)

    def add_doc(self, source_file: SourceFile, doc: Documentation) -> None:
        _check_duplicates(doc, self.docs)
        self.docs[doc.unique_id] = doc
        source_file.docs.append(doc.unique_id)

    def add_semantic_model(self, source_file: SourceFile, semantic_model: SemanticModel) -> None:
        _check_duplicates(semantic_model, self.semantic_models)
        self.semantic_models[semantic_model.unique_id] = semantic_model
        source_file.semantic_models.append(semantic_model.unique_id)

    def add_unit_test(self, source_file: SourceFile, unit_test: UnitTestDefinition) -> None:
        if unit_test.unique_id in self.unit_tests:
            raise DuplicateResourceNameError(unit_test, self.unit_tests[unit_test.unique_id])
        self.unit_tests[unit_test.unique_id] = unit_test
        source_file.unit_tests.append(unit_test.unique_id)

    def add_fixture(self, source_file: SourceFile, fixture: Any) -> None:
        if fixture.unique_id in self.fixtures:
            raise DuplicateResourceNameError(fixture, self.fixtures[fixture.unique_id])
        self.fixtures[fixture.unique_id] = fixture
        source_file.fixture = fixture.unique_id

    def add_saved_query(self, source_file: SourceFile, saved_query: SavedQuery) -> None:
        _check_duplicates(saved_query, self.saved_queries)
        self.saved_queries[saved_query.unique_id] = saved_query
        source_file.saved_queries.append(saved_query.unique_id)

    def find_node_from_ref_or_source(self, expression: str) -> Optional[Union[ManifestNode, SourceDefinition]]:
        ref_or_source = statically_parse_ref_or_source(expression)
        node: Optional[Union[ManifestNode, SourceDefinition]] = None
        if isinstance(ref_or_source, RefArgs):
            node = self.ref_lookup.find(ref_or_source.name, ref_or_source.package, ref_or_source.version, self)
        else:
            source_name, source_table_name = (ref_or_source[0], ref_or_source[1])
            node = self.source_lookup.find(f'{source_name}.{source_table_name}', None, self)
        return node

    def __reduce_ex__(self, protocol: int) -> Tuple[Any, ...]:
        args = (self.nodes, self.sources, self.macros, self.docs, self.exposures, self.metrics, self.groups, self.selectors, self.files, self.metadata, self.flat_graph, self.state_check, self.source_patches, self.disabled, self.env_vars, self.semantic_models, self.unit_tests, self.saved_queries, self._doc_lookup, self._source_lookup, self._ref_lookup, self._metric_lookup, self._semantic_model_by_measure_lookup, self._disabled_lookup, self._analysis_lookup, self._singular_test_lookup)
        return (self.__class__, args)

    def _microbatch_macro_is_core(self, project_name: str) -> bool:
        microbatch_is_core = False
        candidate = self.find_macro_candidate_by_name(name='get_incremental_microbatch_sql', root_project_name=project_name, package=None)
        if candidate is not None and candidate.locality == Locality.Core:
            microbatch_is_core = True
        return microbatch_is_core

    def use_microbatch_batches(self, project_name: str) -> bool:
        return get_flags().require_batched_execution_for_custom_microbatch_strategy or self._microbatch_macro_is_core(project_name=project_name)

class MacroManifest(MacroMethods):
    def __init__(self, macros: Dict[str, Macro]) -> None:
        self.macros = macros
        self.metadata = ManifestMetadata(user_id=tracking.active_user.id if tracking.active_user else None, send_anonymous_usage_stats=get_flags().SEND_ANONYMOUS_USAGE_STATS if tracking.active_user else None)
        self.flat_graph = {}
        self._macros_by_name = None
        self._macros_by_package = None

AnyManifest = Union[Manifest, MacroManifest]

def _check_duplicates(value: Any, src: Mapping[str, Any]) -> None:
    if value.unique_id in src:
        raise DuplicateResourceNameError(value, src[value.unique_id])

K_T = TypeVar('K_T')
V_T = TypeVar('V_T')

def _expect_value(key: K_T, src: Mapping[K_T, V_T], old_file: Any, name: str) -> V_T:
    if key not in src:
        raise CompilationError('Expected to find "{}" in cached "result.{}" based on cached file information: {}!'.format(key, name, old_file))
    return src[key]