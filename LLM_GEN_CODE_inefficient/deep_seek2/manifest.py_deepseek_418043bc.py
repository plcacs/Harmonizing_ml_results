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
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from typing_extensions import Protocol

import dbt_common.exceptions
import dbt_common.utils
from dbt import deprecations, tracking
from dbt.adapters.exceptions import (
    DuplicateMacroInPackageError,
    DuplicateMaterializationNameError,
)
from dbt.adapters.factory import get_adapter_package_names

# to preserve import paths
from dbt.artifacts.resources import BaseResource, DeferRelation, NodeVersion, RefArgs
from dbt.artifacts.resources.v1.config import NodeConfig
from dbt.artifacts.schemas.manifest import ManifestMetadata, UniqueID, WritableManifest
from dbt.clients.jinja_static import statically_parse_ref_or_source
from dbt.contracts.files import (
    AnySourceFile,
    FileHash,
    FixtureSourceFile,
    SchemaSourceFile,
    SourceFile,
)
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
from dbt.exceptions import (
    AmbiguousResourceNameRefError,
    CompilationError,
    DuplicateResourceNameError,
)
from dbt.flags import get_flags
from dbt.mp_context import get_mp_context
from dbt.node_types import (
    REFABLE_NODE_TYPES,
    VERSIONED_NODE_TYPES,
    AccessType,
    NodeType,
)
from dbt_common.dataclass_schema import dbtClassMixin
from dbt_common.events.contextvars import get_node_info
from dbt_common.events.functions import fire_event
from dbt_common.helper_types import PathSet

PackageName = str
DocName = str
RefName = str


def find_unique_id_for_package(storage: Dict[str, Dict[PackageName, UniqueID]], key: str, package: Optional[PackageName]) -> Optional[UniqueID]:
    if key not in storage:
        return None

    pkg_dct: Mapping[PackageName, UniqueID] = storage[key]

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
    def __init__(self, manifest: "Manifest") -> None:
        self.storage: Dict[str, Dict[PackageName, UniqueID]] = {}
        self.populate(manifest)

    def get_unique_id(self, key: str, package: Optional[PackageName]) -> Optional[UniqueID]:
        return find_unique_id_for_package(self.storage, key, package)

    def find(self, key: str, package: Optional[PackageName], manifest: "Manifest") -> Optional[Documentation]:
        unique_id = self.get_unique_id(key, package)
        if unique_id is not None:
            return self.perform_lookup(unique_id, manifest)
        return None

    def add_doc(self, doc: Documentation) -> None:
        if doc.name not in self.storage:
            self.storage[doc.name] = {}
        self.storage[doc.name][doc.package_name] = doc.unique_id

    def populate(self, manifest: "Manifest") -> None:
        for doc in manifest.docs.values():
            self.add_doc(doc)

    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> Documentation:
        if unique_id not in manifest.docs:
            raise dbt_common.exceptions.DbtInternalError(
                f"Doc {unique_id} found in cache but not found in manifest"
            )
        return manifest.docs[unique_id]


class SourceLookup(dbtClassMixin):
    def __init__(self, manifest: "Manifest") -> None:
        self.storage: Dict[str, Dict[PackageName, UniqueID]] = {}
        self.populate(manifest)

    def get_unique_id(self, search_name: str, package: Optional[PackageName]) -> Optional[UniqueID]:
        return find_unique_id_for_package(self.storage, search_name, package)

    def find(self, search_name: str, package: Optional[PackageName], manifest: "Manifest") -> Optional[SourceDefinition]:
        unique_id = self.get_unique_id(search_name, package)
        if unique_id is not None:
            return self.perform_lookup(unique_id, manifest)
        return None

    def add_source(self, source: SourceDefinition) -> None:
        if source.search_name not in self.storage:
            self.storage[source.search_name] = {}

        self.storage[source.search_name][source.package_name] = source.unique_id

    def populate(self, manifest: "Manifest") -> None:
        for source in manifest.sources.values():
            if hasattr(source, "source_name"):
                self.add_source(source)

    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> SourceDefinition:
        if unique_id not in manifest.sources:
            raise dbt_common.exceptions.DbtInternalError(
                f"Source {unique_id} found in cache but not found in manifest"
            )
        return manifest.sources[unique_id]


class RefableLookup(dbtClassMixin):
    # model, seed, snapshot
    _lookup_types: ClassVar[Set[NodeType]] = set(REFABLE_NODE_TYPES)
    _versioned_types: ClassVar[Set[NodeType]] = set(VERSIONED_NODE_TYPES)

    def __init__(self, manifest: "Manifest") -> None:
        self.storage: Dict[str, Dict[PackageName, UniqueID]] = {}
        self.populate(manifest)

    def get_unique_id(
        self,
        key: str,
        package: Optional[PackageName],
        version: Optional[NodeVersion],
        node: Optional[GraphMemberNode] = None,
    ) -> Optional[UniqueID]:
        if version:
            key = f"{key}.v{version}"

        unique_ids = self._find_unique_ids_for_package(key, package)
        if len(unique_ids) > 1:
            raise AmbiguousResourceNameRefError(key, unique_ids, node)
        else:
            return unique_ids[0] if unique_ids else None

    def find(
        self,
        key: str,
        package: Optional[PackageName],
        version: Optional[NodeVersion],
        manifest: "Manifest",
        source_node: Optional[GraphMemberNode] = None,
    ) -> Optional[ManifestNode]:
        unique_id = self.get_unique_id(key, package, version, source_node)
        if unique_id is not None:
            node = self.perform_lookup(unique_id, manifest)
            # If this is an unpinned ref (no 'version' arg was passed),
            # AND this is a versioned node,
            # AND this ref is being resolved at runtime -- get_node_info != {}
            # Only ModelNodes can be versioned.
            if (
                isinstance(node, ModelNode)
                and version is None
                and node.is_versioned
                and get_node_info()
            ):
                # Check to see if newer versions are available, and log an "FYI" if so
                max_version: UnparsedVersion = max(
                    [
                        UnparsedVersion(v.version)
                        for v in manifest.nodes.values()
                        if isinstance(v, ModelNode)
                        and v.name == node.name
                        and v.version is not None
                    ]
                )
                assert node.latest_version is not None  # for mypy, whenever i may find it
                if max_version > UnparsedVersion(node.latest_version):
                    fire_event(
                        UnpinnedRefNewVersionAvailable(
                            node_info=get_node_info(),
                            ref_node_name=node.name,
                            ref_node_package=node.package_name,
                            ref_node_version=str(node.version),
                            ref_max_version=str(max_version.v),
                        )
                    )

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
                if node.is_latest_version:  # type: ignore
                    self.storage[node.name][node.package_name] = node.unique_id
            else:
                self.storage[node.name][node.package_name] = node.unique_id

    def populate(self, manifest: "Manifest") -> None:
        for node in manifest.nodes.values():
            self.add_node(node)

    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> ManifestNode:
        if unique_id in manifest.nodes:
            node = manifest.nodes[unique_id]
        else:
            raise dbt_common.exceptions.DbtInternalError(
                f"Node {unique_id} found in cache but not found in manifest"
            )
        return node

    def _find_unique_ids_for_package(self, key: str, package: Optional[PackageName]) -> List[str]:
        if key not in self.storage:
            return []

        pkg_dct: Mapping[PackageName, UniqueID] = self.storage[key]

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
    def __init__(self, manifest: "Manifest") -> None:
        self.storage: Dict[str, Dict[PackageName, UniqueID]] = {}
        self.populate(manifest)

    def get_unique_id(self, search_name: str, package: Optional[PackageName]) -> Optional[UniqueID]:
        return find_unique_id_for_package(self.storage, search_name, package)

    def find(self, search_name: str, package: Optional[PackageName], manifest: "Manifest") -> Optional[Metric]:
        unique_id = self.get_unique_id(search_name, package)
        if unique_id is not None:
            return self.perform_lookup(unique_id, manifest)
        return None

    def add_metric(self, metric: Metric) -> None:
        if metric.search_name not in self.storage:
            self.storage[metric.search_name] = {}

        self.storage[metric.search_name][metric.package_name] = metric.unique_id

    def populate(self, manifest: "Manifest") -> None:
        for metric in manifest.metrics.values():
            if hasattr(metric, "name"):
                self.add_metric(metric)

    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> Metric:
        if unique_id not in manifest.metrics:
            raise dbt_common.exceptions.DbtInternalError(
                f"Metric {unique_id} found in cache but not found in manifest"
            )
        return manifest.metrics[unique_id]


class SavedQueryLookup(dbtClassMixin):
    """Lookup utility for finding SavedQuery nodes"""

    def __init__(self, manifest: "Manifest") -> None:
        self.storage: Dict[str, Dict[PackageName, UniqueID]] = {}
        self.populate(manifest)

    def get_unique_id(self, search_name: str, package: Optional[PackageName]) -> Optional[UniqueID]:
        return find_unique_id_for_package(self.storage, search_name, package)

    def find(self, search_name: str, package: Optional[PackageName], manifest: "Manifest") -> Optional[SavedQuery]:
        unique_id = self.get_unique_id(search_name, package)
        if unique_id is not None:
            return self.perform_lookup(unique_id, manifest)
        return None

    def add_saved_query(self, saved_query: SavedQuery) -> None:
        if saved_query.search_name not in self.storage:
            self.storage[saved_query.search_name] = {}

        self.storage[saved_query.search_name][saved_query.package_name] = saved_query.unique_id

    def populate(self, manifest: "Manifest") -> None:
        for saved_query in manifest.saved_queries.values():
            if hasattr(saved_query, "name"):
                self.add_saved_query(saved_query)

    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> SavedQuery:
        if unique_id not in manifest.saved_queries:
            raise dbt_common.exceptions.DbtInternalError(
                f"SavedQUery {unique_id} found in cache but not found in manifest"
            )
        return manifest.saved_queries[unique_id]


class SemanticModelByMeasureLookup(dbtClassMixin):
    """Lookup utility for finding SemanticModel by measure

    This is possible because measure names are supposed to be unique across
    the semantic models in a manifest.
    """

    def __init__(self, manifest: "Manifest") -> None:
        self.storage: DefaultDict[str, Dict[PackageName, UniqueID]] = defaultdict(dict)
        self.populate(manifest)

    def get_unique_id(self, search_name: str, package: Optional[PackageName]) -> Optional[UniqueID]:
        return find_unique_id_for_package(self.storage, search_name, package)

    def find(
        self, search_name: str, package: Optional[PackageName], manifest: "Manifest"
    ) -> Optional[SemanticModel]:
        """Tries to find a SemanticModel based on a measure name"""
        unique_id = self.get_unique_id(search_name, package)
        if unique_id is not None:
            return self.perform_lookup(unique_id, manifest)
        return None

    def add(self, semantic_model: SemanticModel) -> None:
        """Sets all measures for a SemanticModel as paths to the SemanticModel's `unique_id`"""
        for measure in semantic_model.measures:
            self.storage[measure.name][semantic_model.package_name] = semantic_model.unique_id

    def populate(self, manifest: "Manifest") -> None:
        """Populate storage with all the measure + package paths to the Manifest's SemanticModels"""
        for semantic_model in manifest.semantic_models.values():
            self.add(semantic_model=semantic_model)
        for disabled in manifest.disabled.values():
            for node in disabled:
                if isinstance(node, SemanticModel):
                    self.add(semantic_model=node)

    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> SemanticModel:
        """Tries to get a SemanticModel from the Manifest"""
        enabled_semantic_model: Optional[SemanticModel] = manifest.semantic_models.get(unique_id)
        disabled_semantic_model: Optional[List] = manifest.disabled.get(unique_id)

        if isinstance(enabled_semantic_model, SemanticModel):
            return enabled_semantic_model
        elif disabled_semantic_model is not None and isinstance(
            disabled_semantic_model[0], SemanticModel
        ):
            return disabled_semantic_model[0]
        else:
            raise dbt_common.exceptions.DbtInternalError(
                f"Semantic model `{unique_id}` found in cache but not found in manifest"
            )


# This handles both models/seeds/snapshots and sources/metrics/exposures/semantic_models
class DisabledLookup(dbtClassMixin):
    def __init__(self, manifest: "Manifest") -> None:
        self.storage: Dict[str, Dict[PackageName, List[Any]]] = {}
        self.populate(manifest)

    def populate(self, manifest: "Manifest") -> None:
        for node in list(chain.from_iterable(manifest.disabled.values())):
            self.add_node(node)

    def add_node(self, node: GraphMemberNode) -> None:
        if node.search_name not in self.storage:
            self.storage[node.search_name] = {}
        if node.package_name not in self.storage[node.search_name]:
            self.storage[node.search_name][node.package_name] = []
        self.storage[node.search_name][node.package_name].append(node)

    # This should return a list of disabled nodes. It's different from
    # the other Lookup functions in that it returns full nodes, not just unique_ids
    def find(
        self,
        search_name: str,
        package: Optional[PackageName],
        version: Optional[NodeVersion] = None,
        resource_types: Optional[List[NodeType]] = None,
    ) -> Optional[List[Any]]:
        if version:
            search_name = f"{search_name}.v{version}"

        if search_name not in self.storage:
            return None

        pkg_dct: Mapping[PackageName, List[Any]] = self.storage[search_name]

        nodes = []
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
            new_nodes = []
            for node in nodes:
                if node.resource_type in resource_types:
                    new_nodes.append(node)
            if not new_nodes:
                return None
            else:
                return new_nodes


class AnalysisLookup(RefableLookup):
    _lookup_types: ClassVar[Set[NodeType]] = set([NodeType.Analysis])
    _versioned_types: ClassVar[Set[NodeType]] = set()


class SingularTestLookup(dbtClassMixin):
    def __init__(self, manifest: "Manifest") -> None:
        self.storage: Dict[str, Dict[PackageName, UniqueID]] = {}
        self.populate(manifest)

    def get_unique_id(self, search_name: str, package: Optional[PackageName]) -> Optional[UniqueID]:
        return find_unique_id_for_package(self.storage, search_name, package)

    def find(
        self, search_name: str, package: Optional[PackageName], manifest: "Manifest"
    ) -> Optional[SingularTestNode]:
        unique_id = self.get_unique_id(search_name, package)
        if unique_id is not None:
            return self.perform_lookup(unique_id, manifest)
        return None

    def add_singular_test(self, source: SingularTestNode) -> None:
        if source.search_name not in self.storage:
            self.storage[source.search_name] = {}

        self.storage[source.search_name][source.package_name] = source.unique_id

    def populate(self, manifest: "Manifest") -> None:
        for node in manifest.nodes.values():
            if isinstance(node, SingularTestNode):
                self.add_singular_test(node)

    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> SingularTestNode:
        if unique_id not in manifest.nodes:
            raise dbt_common.exceptions.DbtInternalError(
                f"Singular test {unique_id} found in cache but not found in manifest"
            )
        node = manifest.nodes[unique_id]
        assert isinstance(node, SingularTestNode)
        return node


def _packages_to_search(
    current_project: str,
    node_package: str,
    target_package: Optional[str] = None,
) -> List[Optional[str]]:
    if target_package is not None:
        return [target_package]
    elif current_project == node_package:
        return [current_project, None]
    else:
        return [current_project, node_package, None]


def _sort_values(dct: Dict[Any, Any]) -> Dict[Any, Any]:
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
    # pre-populate the forward edge dict for simplicity
    forward_edges: Dict[str, List[str]] = {n.unique_id: [] for n in nodes}
    for node in nodes:
        backward_edges[node.unique_id] = node.depends_on_nodes[:]
        for unique_id in backward_edges[node.unique_id]:
            if unique_id in forward_edges.keys():
                forward_edges[unique_id].append(node.unique_id)
    return _sort_values(forward_edges), _sort_values(backward_edges)


# Build a map of children of macros and generic tests
def build_macro_edges(nodes: List[Any]) -> Dict[str, List[str]]:
    forward_edges: Dict[str, List[str]] = {
        n.unique_id: [] for n in nodes if n.unique_id.startswith("macro") or n.depends_on_macros
    }
    for node in nodes:
        for unique_id in node.depends_on_macros:
            if unique_id in forward_edges.keys():
                forward_edges[unique_id].append(node.unique_id)
    return _sort_values(forward_edges)


def _deepcopy(value: Any) -> Any:
    return value.from_dict(value.to_dict(omit_none=True))


class Locality(enum.IntEnum):
    Core = 1
    Imported = 2
    Root = 3


@dataclass
class MacroCandidate:
    locality: Locality
    macro: Macro

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MacroCandidate):
            return NotImplemented
        return self.locality == other.locality

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, MacroCandidate):
            return NotImplemented
        if self.locality < other.locality:
            return True
        if self.locality > other.locality:
            return False
        return False


@dataclass
class MaterializationCandidate(MacroCandidate):
    # specificity describes where in the inheritance chain this materialization candidate is
    # a specificity of 0 means a materialization defined by the current adapter
    # the highest the specificity describes a default materialization. the value itself depends on
    # how many adapters there are in the inheritance chain
    specificity: int

    @classmethod
    def from_macro(cls, candidate: MacroCandidate, specificity: int) -> "MaterializationCandidate":
        return cls(
            locality=candidate.locality,
            macro=candidate.macro,
            specificity=specificity,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MaterializationCandidate):
            return NotImplemented
        equal = self.specificity == other.specificity and self.locality == other.locality
        if equal:
            raise DuplicateMaterializationNameError(self.macro, other)

        return equal

    def __lt__(self, other: object) -> bool:
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


M = TypeVar("M", bound=MacroCandidate)


class CandidateList(List[M]):
    def last_candidate(
        self, valid_localities: Optional[List[Locality]] = None
    ) -> Optional[MacroCandidate]:
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


def _get_locality(macro: Macro, root_project_name: str, internal_packages: Set[str]) -> Locality:
    if macro.package_name == root_project_name:
        return Locality.Root
    elif macro.package_name in internal_packages:
        return Locality.Core
    else:
        return Locality.Imported


class Searchable(Protocol):
    resource_type: NodeType
    package_name: str

    @property
    def search_name(self) -> str:
        raise NotImplementedError("search_name not implemented")


D = TypeVar("D")


@dataclass
class Disabled(Generic[D]):
    target: D


MaybeMetricNode = Optional[Union[Metric, Disabled[Metric]]]


MaybeSavedQueryNode = Optional[Union[SavedQuery, Disabled[SavedQuery]]]


MaybeDocumentation = Optional[Documentation]


MaybeParsedSource = Optional[
    Union[
        SourceDefinition,
        Disabled[SourceDefinition],
    ]
]


MaybeNonSource = Optional[Union[ManifestNode, Disabled[ManifestNode]]]


T = TypeVar("T", bound=GraphMemberNode)


# This contains macro methods that are in both the Manifest
# and the MacroManifest
class MacroMethods:
    # Just to make mypy happy. There must be a better way.
    def __init__(self):
        self.macros = []
        self.metadata = {}
        self._macros_by_name = {}
        self._macros_by_package = {}

    def find_macro_candidate_by_name(
        self, name: str, root_project_name: str, package: Optional[str]
    ) -> Optional[MacroCandidate]:
        """Find a MacroCandidate in the graph by its name and package name, or None for
        any package. The root project name is used to determine priority:
         - locally defined macros come first
         - then imported macros
         - then macros defined in the root project
        """
        filter: Optional[Callable[[MacroCandidate], bool]] = None
        if package is not None:

            def filter(candidate: MacroCandidate) -> bool:
                return package == candidate.macro.package_name

        candidates: CandidateList = self._find_macros_by_name(
            name=name,
            root_project_name=root_project_name,
            filter=filter,
        )

        return candidates.last_candidate()

    def find_macro_by_name(
        self, name: str, root_project_name: str, package: Optional[str]
    ) -> Optional[Macro]:
        macro_candidate = self.find_macro_candidate_by_name(
            name=name, root_project_name=root_project_name, package=package
        )
        return macro_candidate.macro if macro_candidate else None

    def find_generate_macro_by_name(
        self, component: str, root_project_name: str, imported_package: Optional[str] = None
    ) -> Optional[Macro]:
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

        def filter(candidate: MacroCandidate) -> bool:
            if imported_package:
                return (
                    candidate.locality == Locality.Imported
                    and imported_package == candidate.macro.package_name
                )
            else:
                return candidate.locality != Locality.Imported

        candidates: CandidateList = self._find_macros_by_name(
            name=f"generate_{component}_name",
            root_project_name=root_project_name,
            filter=filter,
        )

        return candidates.last()

    def _find_macros_by_name(
        self,
        name: str,
        root_project_name: str,
        filter: Optional[Callable[[MacroCandidate], bool]] = None,
    ) -> CandidateList:
        """Find macros by their name."""
        candidates: CandidateList = CandidateList()

        macros_by_name = self.get_macros_by_name()
        if name not in macros_by_name:
            return candidates

        packages = set(get_adapter_package_names(self.metadata.adapter_type))
        for macro in macros_by_name[name]:
            candidate = MacroCandidate(
                locality=_get_locality(macro, root_project_name, packages),
                macro=macro,
            )
            if filter is None or filter(candidate):
                candidates.append(candidate)

        return candidates

    def get_macros_by_name(self) -> Dict[str, List[Macro]]:
        if self._macros_by_name is None:
            # The by-name mapping doesn't exist yet (perhaps because the manifest
            # was deserialized), so we build it.
            self._macros_by_name = self._build_macros_by_name(self.macros)

        return self._macros_by_name

    @staticmethod
    def _build_macros_by_name(macros: Mapping[str, Macro]) -> Dict[str, List[Macro]]:
        # Convert a macro dictionary keyed on unique id to a flattened version
        # keyed on macro name for faster lookup by name. Since macro names are
        # not necessarily unique, the dict value is a list.
        macros_by_name: Dict[str, List[Macro]] = {}
        for macro in macros.values():
            if macro.name not in macros_by_name:
                macros_by_name[macro.name] = []

            macros_by_name[macro.name].append(macro)

        return macros_by_name

    def get_macros_by_package(self) -> Dict[str, Dict[str, Macro]]:
        if self._macros_by_package is None:
            # The by-package mapping doesn't exist yet (perhaps because the manifest
            # was deserialized), so we build it.
            self._macros_by_package = self._build_macros_by_package(self.macros)

        return self._macros_by_package

    @staticmethod
    def _build_macros_by_package(macros: Mapping[str, Macro]) -> Dict[str, Dict[str, Macro]]:
        # Convert a macro dictionary keyed on unique id to a flattened version
        # keyed on package name for faster lookup by name.
        macros_by_package: Dict[str, Dict[str, Macro]] = {}
        for macro in macros.values():
            if macro.package_name not in macros_by_package:
                macros_by_package[macro.package_name] = {}
            macros_by_name = macros_by_package[macro.package_name]
            macros_by_name[macro.name] = macro

        return macros_by_package


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
    project_hashes: MutableMapping[str, FileHash] = field(default_factory=dict)


NodeClassT = TypeVar("NodeClassT", bound="BaseNode")
ResourceClassT = TypeVar("ResourceClassT", bound="BaseResource")


@dataclass
class Manifest(MacroMethods, dbtClassMixin):
    """The manifest for the full graph, after parsing and during compilation."""

    # These attributes are both positional and by keyword. If an attribute
    # is added it must all be added in the __reduce_ex__ method in the
    # args tuple in the right position.
    nodes: MutableMapping[str, ManifestNode] = field(default_factory=dict)
    sources: MutableMapping[str, SourceDefinition] = field(default_factory=dict)
    macros: MutableMapping[str, Macro] = field(default_factory=dict)
    docs: MutableMapping[str, Documentation] = field(default_factory=dict)
    exposures: MutableMapping[str, Exposure] = field(default_factory=dict)
    metrics: MutableMapping[str, Metric] = field(default_factory=dict)
    groups: MutableMapping[str, Group] = field(default_factory=dict)
    selectors: MutableMapping[str, Any] = field(default_factory=dict)
    files: MutableMapping[str, AnySourceFile] = field(default_factory=dict)
    metadata: ManifestMetadata = field(default_factory=ManifestMetadata)
    flat_graph: Dict[str, Any] = field(default_factory=dict)
    state_check: ManifestStateCheck = field(default_factory=ManifestStateCheck)
    source_patches: MutableMapping[SourceKey, SourcePatch] = field(default_factory=dict)
    disabled: MutableMapping[str, List[GraphMemberNode]] = field(default_factory=dict)
    env_vars: MutableMapping[str, str] = field(default_factory=dict)
    semantic_models: MutableMapping[str, SemanticModel] = field(default_factory=dict)
    unit_tests: MutableMapping[str, UnitTestDefinition] = field(default_factory=dict)
    saved_queries: MutableMapping[str, SavedQuery] = field(default_factory=dict)
    fixtures: MutableMapping[str, UnitTestFileFixture] = field(default_factory=dict)

    _doc_lookup: Optional[DocLookup] = field(
        default=None, metadata={"serialize": lambda x: None, "deserialize": lambda x: None}
    )
    _source_lookup: Optional[SourceLookup] = field(
        default=None, metadata={"serialize": lambda x: None, "deserialize": lambda x: None}
    )
    _ref_lookup: Optional[RefableLookup] = field(
        default=None, metadata={"serialize": lambda x: None, "deserialize": lambda x: None}
    )
    _metric_lookup: Optional[MetricLookup] = field(
        default=None, metadata={"serialize": lambda x: None, "deserialize": lambda x: None}
    )
    _saved_query_lookup: Optional[SavedQueryLookup] = field(
        default=None, metadata={"serialize": lambda x: None, "deserialize": lambda x: None}
    )
    _semantic_model_by_measure_lookup: Optional[SemanticModelByMeasureLookup] = field(
        default=None, metadata={"serialize": lambda x: None, "deserialize": lambda x: None}
    )
    _disabled_lookup: Optional[DisabledLookup] = field(
        default=None, metadata={"serialize": lambda x: None, "deserialize": lambda x: None}
    )
    _analysis_lookup: Optional[AnalysisLookup] = field(
        default=None, metadata={"serialize": lambda x: None, "deserialize": lambda x