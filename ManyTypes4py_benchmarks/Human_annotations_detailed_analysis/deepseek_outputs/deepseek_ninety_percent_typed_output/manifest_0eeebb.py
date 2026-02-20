import datetime
import json
import os
import pprint
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Type, Union

import msgpack
from jinja2.nodes import Call

import dbt.deprecations
import dbt.exceptions
import dbt.tracking
import dbt.utils
import dbt_common.utils
from dbt import plugins
from dbt.adapters.capability import Capability
from dbt.adapters.factory import (
    get_adapter,
    get_adapter_package_names,
    get_relation_class_by_name,
    register_adapter,
)
from dbt.artifacts.resources import FileHash, NodeRelation, NodeVersion
from dbt.artifacts.resources.types import BatchSize
from dbt.artifacts.schemas.base import Writable
from dbt.clients.jinja import MacroStack, get_rendered
from dbt.clients.jinja_static import statically_extract_macro_calls
from dbt.config import Project, RuntimeConfig
from dbt.constants import (
    MANIFEST_FILE_NAME,
    PARTIAL_PARSE_FILE_NAME,
    SEMANTIC_MANIFEST_FILE_NAME,
)
from dbt.context.configured import generate_macro_context
from dbt.context.docs import generate_runtime_docs_context
from dbt.context.macro_resolver import MacroResolver, TestMacroNamespace
from dbt.context.providers import ParseProvider, generate_runtime_macro_context
from dbt.context.query_header import generate_query_header_context
from dbt.contracts.files import ParseFileType, SchemaSourceFile
from dbt.contracts.graph.manifest import (
    Disabled,
    MacroManifest,
    Manifest,
    ManifestStateCheck,
    ParsingInfo,
)
from dbt.contracts.graph.nodes import (
    Exposure,
    GenericTestNode,
    Macro,
    ManifestNode,
    Metric,
    ModelNode,
    ResultNode,
    SavedQuery,
    SeedNode,
    SemanticManifestNode,
    SemanticModel,
    SourceDefinition,
)
from dbt.contracts.graph.semantic_manifest import SemanticManifest
from dbt.events.types import (
    ArtifactWritten,
    DeprecatedModel,
    DeprecatedReference,
    InvalidConcurrentBatchesConfig,
    InvalidDisabledTargetInTestNode,
    MicrobatchModelNoEventTimeInputs,
    NodeNotFoundOrDisabled,
    ParsedFileLoadFailed,
    ParsePerfInfoPath,
    PartialParsingError,
    PartialParsingErrorProcessingFile,
    PartialParsingNotEnabled,
    PartialParsingSkipParsing,
    SpacesInResourceNameDeprecation,
    StateCheckVarsHash,
    UnableToPartialParse,
    UpcomingReferenceDeprecation,
)
from dbt.exceptions import (
    AmbiguousAliasError,
    InvalidAccessTypeError,
    TargetNotFoundError,
    scrub_secrets,
)
from dbt.flags import get_flags
from dbt.mp_context import get_mp_context
from dbt.node_types import AccessType, NodeType
from dbt.parser.analysis import AnalysisParser
from dbt.parser.base import Parser
from dbt.parser.docs import DocumentationParser
from dbt.parser.fixtures import FixtureParser
from dbt.parser.generic_test import GenericTestParser
from dbt.parser.hooks import HookParser
from dbt.parser.macros import MacroParser
from dbt.parser.models import ModelParser
from dbt.parser.partial import PartialParsing, special_override_macros
from dbt.parser.read_files import (
    FileDiff,
    ReadFiles,
    ReadFilesFromDiff,
    ReadFilesFromFileSystem,
    load_source_file,
)
from dbt.parser.schemas import SchemaParser
from dbt.parser.search import FileBlock
from dbt.parser.seeds import SeedParser
from dbt.parser.singular_test import SingularTestParser
from dbt.parser.snapshots import SnapshotParser
from dbt.parser.sources import SourcePatcher
from dbt.parser.unit_tests import process_models_for_unit_test
from dbt.version import __version__
from dbt_common.clients.jinja import parse
from dbt_common.clients.system import make_directory, path_exists, read_json, write_file
from dbt_common.constants import SECRET_ENV_PREFIX
from dbt_common.dataclass_schema import StrEnum, dbtClassMixin
from dbt_common.events.base_types import EventLevel
from dbt_common.events.functions import fire_event, get_invocation_id, warn_or_error
from dbt_common.events.types import Note
from dbt_common.exceptions.base import DbtValidationError
from dbt_common.helper_types import PathSet
from dbt_semantic_interfaces.enum_extension import assert_values_exhausted
from dbt_semantic_interfaces.type_enums import MetricType

PERF_INFO_FILE_NAME = "perf_info.json"


def extended_mashumaro_encoder(data: Any) -> bytes:
    return msgpack.packb(data, default=extended_msgpack_encoder, use_bin_type=True)


def extended_msgpack_encoder(obj: Any) -> Any:
    if type(obj) is datetime.date:
        date_bytes = msgpack.ExtType(1, obj.isoformat().encode())
        return date_bytes
    elif type(obj) is datetime.datetime:
        datetime_bytes = msgpack.ExtType(2, obj.isoformat().encode())
        return datetime_bytes

    return obj


def extended_mashumuro_decoder(data: bytes) -> Any:
    return msgpack.unpackb(data, ext_hook=extended_msgpack_decoder, raw=False)


def extended_msgpack_decoder(code: int, data: bytes) -> Any:
    if code == 1:
        d = datetime.date.fromisoformat(data.decode())
        return d
    elif code == 2:
        dt = datetime.datetime.fromisoformat(data.decode())
        return dt
    else:
        return msgpack.ExtType(code, data)


def version_to_str(version: Optional[Union[str, int]]) -> str:
    if isinstance(version, int):
        return str(version)
    elif isinstance(version, str):
        return version

    return ""


class ReparseReason(StrEnum):
    version_mismatch = "01_version_mismatch"
    file_not_found = "02_file_not_found"
    vars_changed = "03_vars_changed"
    profile_changed = "04_profile_changed"
    deps_changed = "05_deps_changed"
    project_config_changed = "06_project_config_changed"
    load_file_failure = "07_load_file_failure"
    exception = "08_exception"
    proj_env_vars_changed = "09_project_env_vars_changed"
    prof_env_vars_changed = "10_profile_env_vars_changed"


# Part of saved performance info
@dataclass
class ParserInfo(dbtClassMixin):
    parser: str
    elapsed: float
    parsed_path_count: int = 0


# Part of saved performance info
@dataclass
class ProjectLoaderInfo(dbtClassMixin):
    project_name: str
    elapsed: float
    parsers: List[ParserInfo] = field(default_factory=list)
    parsed_path_count: int = 0


# Part of saved performance info
@dataclass
class ManifestLoaderInfo(dbtClassMixin, Writable):
    path_count: int = 0
    parsed_path_count: int = 0
    static_analysis_path_count: int = 0
    static_analysis_parsed_path_count: int = 0
    is_partial_parse_enabled: Optional[bool] = None
    is_static_analysis_enabled: Optional[bool] = None
    read_files_elapsed: Optional[float] = None
    load_macros_elapsed: Optional[float] = None
    parse_project_elapsed: Optional[float] = None
    patch_sources_elapsed: Optional[float] = None
    process_manifest_elapsed: Optional[float] = None
    load_all_elapsed: Optional[float] = None
    projects: List[ProjectLoaderInfo] = field(default_factory=list)
    _project_index: Dict[str, ProjectLoaderInfo] = field(default_factory=dict)

    def __post_serialize__(self, dct: Dict, context: Optional[Dict] = None) -> Dict:
        del dct["_project_index"]
        return dct


# The ManifestLoader loads the manifest. The standard way to use the
# ManifestLoader is using the 'get_full_manifest' class method, but
# many tests use abbreviated processes.
class ManifestLoader:
    def __init__(
        self,
        root_project: RuntimeConfig,
        all_projects: Mapping[str, RuntimeConfig],
        macro_hook: Optional[Callable[[Manifest], Any]] = None,
        file_diff: Optional[FileDiff] = None,
    ) -> None:
        self.root_project: RuntimeConfig = root_project
        self.all_projects: Mapping[str, RuntimeConfig] = all_projects
        self.file_diff = file_diff
        self.manifest: Manifest = Manifest()
        self.new_manifest = self.manifest
        self.manifest.metadata = root_project.get_metadata()
        self.macro_resolver: Optional[MacroResolver] = None  # built after macros are loaded
        self.started_at: float = time.time()
        # This is a MacroQueryStringSetter callable, which is called
        # later after we set the MacroManifest in the adapter. It sets
        # up the query headers.
        self.macro_hook: Callable[[Manifest], Any]
        if macro_hook is None:
            self.macro_hook = lambda m: None
        else:
            self.macro_hook = macro_hook

        self._perf_info: ManifestLoaderInfo = self.build_perf_info()

        # State check determines whether the saved_manifest and the current
        # manifest match well enough to do partial parsing
        self.manifest.state_check: ManifestStateCheck = self.build_manifest_state_check()
        # We need to know if we're actually partially parsing. It could
        # have been enabled, but not happening because of some issue.
        self.partially_parsing: bool = False
        self.partial_parser: Optional[PartialParsing] = None
        self.skip_parsing: bool = False

        # This is a saved manifest from a previous run that's used for partial parsing
        self.saved_manifest: Optional[Manifest] = self.read_manifest_for_partial_parse()

    # This is the method that builds a complete manifest. We sometimes
    # use an abbreviated process in tests.
    @classmethod
    def get_full_manifest(
        cls,
        config: RuntimeConfig,
        *,
        file_diff: Optional[FileDiff] = None,
        reset: bool = False,
        write_perf_info: bool = False,
    ) -> Manifest:
        adapter = get_adapter(config)  # type: ignore
        # reset is set in a TaskManager load_manifest call, since
        # the config and adapter may be persistent.
        if reset:
            config.clear_dependencies()
            adapter.clear_macro_resolver()
        macro_hook = adapter.connections.set_query_header

        flags = get_flags()
        if not flags.PARTIAL_PARSE_FILE_DIFF:
            file_diff = FileDiff.from_dict(
                {
                    "deleted": [],
                    "changed": [],
                    "added": [],
                }
            )
        # Hack to test file_diffs
        elif os.environ.get("DBT_PP_FILE_DIFF_TEST"):
            file_diff_path = "file_diff.json"
            if path_exists(file_diff_path):
                file_diff_dct = read_json(file_diff_path)
                file_diff = FileDiff.from_dict(file_diff_dct)

        # Start performance counting
        start_load_all = time.perf_counter()

        projects = config.load_dependencies()
        loader = cls(
            config,
            projects,
            macro_hook=macro_hook,
            file_diff=file_diff,
        )

        manifest = loader.load()

        _check_manifest(manifest, config)
        manifest.build_flat_graph()

        # This needs to happen after loading from a partial parse,
        # so that the adapter has the query headers from the macro_hook.
        loader.save_macros_to_adapter(adapter)

        # Save performance info
        loader._perf_info.load_all_elapsed = time.perf_counter() - start_load_all
        loader.track_project_load()

        if write_perf_info:
            loader.write_perf_info(config.project_target_path)

        return manifest

    # This is where the main action happens
    def load(self) -> Manifest:
        start_read_files = time.perf_counter()

        # This updates the "files" dictionary in self.manifest, and creates
        # the partial_parser_files dictionary (see read_files.py),
        # which is a dictionary of projects to a dictionary
        # of parsers to lists of file strings. The file strings are
        # used to get the SourceFiles from the manifest files.
        saved_files = self.saved_manifest.files if self.saved_manifest else {}
        file_reader: Optional[ReadFiles] = None
        if self.file_diff:
            # We're getting files from a file diff
            file_reader = ReadFilesFromDiff(
                all_projects=self.all_projects,
                files=self.manifest.files,
                saved_files=saved_files,
                root_project_name=self.root_project.project_name,
                file_diff=self.file_diff,
            )
        else:
            # We're getting files from the file system
            file_reader = ReadFilesFromFileSystem(
                all_projects=self.all_projects,
                files=self.manifest.files,
                saved_files=saved_files,
            )

        # Set the files in the manifest and save the project_parser_files
        file_reader.read_files()
        self.manifest.files = file_reader.files
        project_parser_files = orig_project_parser_files = file_reader.project_parser_files
        self._perf_info.path_count = len(self.manifest.files)
        self._perf_info.read_files_elapsed = time.perf_counter() - start_read_files

        self.skip_parsing = False
        project_parser_files = self.safe_update_project_parser_files_partially(
            project_parser_files
        )

        if self.manifest._parsing_info is None:
            self.manifest._parsing_info = ParsingInfo()

        if self.skip_parsing:
            fire_event(PartialParsingSkipParsing())
        else:
            # Load Macros and tests
            # We need to parse the macros first, so they're resolvable when
            # the other files are loaded.  Also need to parse tests, specifically
            # generic tests
            start_load_macros = time.perf_counter()
            self.load_and_parse_macros(project_parser_files)

            # If we're partially parsing check that certain macros have not been changed
            if self.partially_parsing and self.skip_partial_parsing_because_of_macros():
                fire_event(
                    UnableToPartialParse(
                        reason="change detected to override macro. Starting full parse."
                    )
                )

                # Get new Manifest with original file records and move over the macros
                self.manifest = self.new_manifest  # contains newly read files
                project_parser_files = orig_project_parser_files
                self.partially_parsing = False
                self.load_and_parse_macros(project_parser_files)

            self._perf_info.load_macros_elapsed = time.perf_counter() - start_load_macros

            # Now that the macros are parsed, parse the rest of the files.
            # This is currently done on a per project basis.
            start_parse_projects = time.perf_counter()

            # Load the rest of the files except for schema yaml files
            parser_types: List[Type[Parser]] = [
                ModelParser,
                SnapshotParser,
                AnalysisParser,
                SingularTestParser,
                SeedParser,
                DocumentationParser,
                HookParser,
                FixtureParser,
            ]
            for project in self.all_projects.values():
                if project.project_name not in project_parser_files:
                    continue
                self.parse_project(
                    project, project_parser_files[project.project_name], parser_types
                )

            # Now that we've loaded most of the nodes (except for schema tests, sources, metrics)
            # load up the Lookup objects to resolve them by name, so the SourceFiles store
            # the unique_id instead of the name. Sources are loaded from yaml files, so
            # aren't in place yet
            self.manifest.rebuild_ref_lookup()
            self.manifest.rebuild_doc_lookup()
            self.manifest.rebuild_disabled_lookup()

            # Load yaml files
            parser_types = [SchemaParser]  # type: ignore
            for project in self.all_projects.values():
                if project.project_name not in project_parser_files:
                    continue
                self.parse_project(
                    project, project_parser_files[project.project_name], parser_types
                )

            self.cleanup_disabled()

            self._perf_info.parse_project_elapsed = time.perf_counter() - start_parse_projects

            # patch_sources converts the UnparsedSourceDefinitions in the
            # Manifest.sources to SourceDefinition via 'patch_source'
            # in SourcePatcher
            start_patch = time.perf_counter()
            patcher = SourcePatcher(self.root_project, self.manifest)
            patcher.construct_sources()
            self.manifest.sources = patcher.sources
            self._perf_info.patch_sources_elapsed = time.perf_counter() - start_patch

            # We need to rebuild disabled in order to include disabled sources
            self.manifest.rebuild_disabled_lookup()

            # copy the selectors from the root_project to the manifest
            self.manifest.selectors = self.root_project.manifest_selectors

            # inject any available external nodes
            self.manifest.build_parent_and_child_maps()
            external_nodes_modified = self.inject_external_nodes()
            if external_nodes_modified:
                self.manifest.rebuild_ref_lookup()

            # update the refs, sources, docs and metrics depends_on.nodes
            # These check the created_at time on the nodes to
            # determine whether they need processing.
            start_process = time.perf_counter