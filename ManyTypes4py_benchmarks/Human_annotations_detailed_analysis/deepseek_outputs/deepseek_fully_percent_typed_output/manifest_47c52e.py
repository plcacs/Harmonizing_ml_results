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
    cast,
    Iterator,
    FrozenSet,
    Iterable,
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


def find_unique_id_for_package(
    storage: Dict[str, Dict[PackageName, UniqueID]], 
    key: str, 
    package: Optional[PackageName]
) -> Optional[UniqueID]:
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

    def find(
        self, key: str, package: Optional[PackageName], manifest: "Manifest"
    ) -> Optional[Documentation]:
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

    def find(
        self, search_name: str, package: Optional[PackageName], manifest: "Manifest"
    ) -> Optional[SourceDefinition]:
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
            if (
                isinstance(node, ModelNode)
                and version is None
                and node.is_versioned
                and get_node_info()
            ):
                max_version: UnparsedVersion = max(
                    [
                        UnparsedVersion(v.version)
                        for v in manifest.nodes.values()
                        if isinstance(v, ModelNode)
                        and v.name == node.name
                        and v.version is not None
                    ]
                )
                assert node.latest_version is not None
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
                if node.is_latest_version:
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

    def find(
        self, search_name: str, package: Optional[PackageName], manifest: "Manifest"
    ) -> Optional[Metric]:
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
    def __init__(self, manifest: "Manifest") -> None:
        self.storage: Dict[str, Dict[PackageName, UniqueID]] = {}
        self.populate(manifest)

    def get_unique_id(self, search_name: str, package: Optional[PackageName]) -> Optional[UniqueID]:
        return find_unique_id_for_package(self.storage, search_name, package)

    def find(
        self, search_name: str, package: Optional[PackageName], manifest: "Manifest"
    ) -> Optional[SavedQuery]:
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
                f"SavedQuery {unique_id} found in cache but not found in manifest"
            )
        return manifest.saved_queries[unique_id]


class SemanticModelByMeasureLookup(dbtClassMixin):
    def __init__(self, manifest: "Manifest") -> None:
        self.storage: DefaultDict[str, Dict[PackageName, UniqueID]] = defaultdict(dict)
        self.populate(manifest)

    def get_unique_id(self, search_name: str, package: Optional[PackageName]) -> Optional[UniqueID]:
        return find_unique_id_for_package(self.storage, search_name, package)

    def find(
        self, search_name: str, package: Optional[PackageName], manifest: "Manifest"
    ) -> Optional[SemanticModel]:
        unique_id = self.get_unique_id(search_name, package)
        if unique_id is not None:
            return self.perform_lookup(unique_id, manifest)
        return None

    def add(self, semantic_model: SemanticModel) -> None:
        for measure in semantic_model.measures:
            self.storage[measure.name][semantic_model.package_name] = semantic_model.unique_id

    def populate(self, manifest: "Manifest") -> None:
        for semantic_model in manifest.semantic_models.values():
            self.add(semantic_model=semantic_model)
        for disabled in manifest.disabled.values():
            for node in disabled:
                if isinstance(node, SemanticModel):
                    self.add(semantic_model=node)

    def perform_lookup(self, unique_id: UniqueID, manifest: "Manifest") -> SemanticModel:
        enabled_semantic_model: Optional[SemanticModel] = manifest.semantic_models.get(unique_id)
        disabled_semantic_model: Optional[List[SemanticModel]] = manifest.disabled.get(unique_id)

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


class DisabledLookup(dbtClassMixin):
    def __init__(self, manifest: "Manifest") -> None:
        self.storage: Dict[str, Dict[PackageName, List[GraphMemberNode]]] = {}
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

    def find(
        self,
        search_name: str,
        package: Optional[PackageName],
        version: Optional[NodeVersion] = None,
        resource_types: Optional[List[NodeType]] = None,
    ) -> Optional[List[GraphMemberNode]]:
        if version:
            search_name = f"{search_name}.v{version}"

        if search_name not in self.storage:
            return None

        pkg_dct: Mapping[PackageName, List[GraphMemberNode]] = self.storage[search_name]

        nodes: List[GraphMemberNode] = []
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
    _lookup_types: ClassVar[Set[NodeType]] = {NodeType.Analysis}
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

        self.storage[source.search_name][source.package_name] = source