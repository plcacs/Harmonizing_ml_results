import datetime
import json
import os
import pprint
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Type, Union, cast
import msgpack
from jinja2.nodes import Call
import dbt.deprecations
import dbt.exceptions
import dbt.tracking
import dbt.utils
import dbt_common.utils
from dbt import plugins
from dbt.adapters.capability import Capability
from dbt.adapters.factory import get_adapter, get_adapter_package_names, get_relation_class_by_name, register_adapter
from dbt.artifacts.resources import FileHash, NodeRelation, NodeVersion
from dbt.artifacts.resources.types import BatchSize
from dbt.artifacts.schemas.base import Writable
from dbt.clients.jinja import MacroStack, get_rendered
from dbt.clients.jinja_static import statically_extract_macro_calls
from dbt.config import Project, RuntimeConfig
from dbt.constants import MANIFEST_FILE_NAME, PARTIAL_PARSE_FILE_NAME, SEMANTIC_MANIFEST_FILE_NAME
from dbt.context.configured import generate_macro_context
from dbt.context.docs import generate_runtime_docs_context
from dbt.context.macro_resolver import MacroResolver, TestMacroNamespace
from dbt.context.providers import ParseProvider, generate_runtime_macro_context
from dbt.context.query_header import generate_query_header_context
from dbt.contracts.files import ParseFileType, SchemaSourceFile
from dbt.contracts.graph.manifest import Disabled, MacroManifest, Manifest, ManifestStateCheck, ParsingInfo
from dbt.contracts.graph.nodes import Exposure, GenericTestNode, Macro, ManifestNode, Metric, ModelNode, ResultNode, SavedQuery, SeedNode, SemanticManifestNode, SemanticModel, SourceDefinition
from dbt.contracts.graph.semantic_manifest import SemanticManifest
from dbt.events.types import ArtifactWritten, DeprecatedModel, DeprecatedReference, InvalidConcurrentBatchesConfig, InvalidDisabledTargetInTestNode, MicrobatchModelNoEventTimeInputs, NodeNotFoundOrDisabled, ParsedFileLoadFailed, ParsePerfInfoPath, PartialParsingError, PartialParsingErrorProcessingFile, PartialParsingNotEnabled, PartialParsingSkipParsing, SpacesInResourceNameDeprecation, StateCheckVarsHash, UnableToPartialParse, UpcomingReferenceDeprecation
from dbt.exceptions import AmbiguousAliasError, InvalidAccessTypeError, TargetNotFoundError, scrub_secrets
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
from dbt.parser.read_files import FileDiff, ReadFiles, ReadFilesFromDiff, ReadFilesFromFileSystem, load_source_file
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

PERF_INFO_FILE_NAME = 'perf_info.json'

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

def version_to_str(version: Union[int, str, None]) -> str:
    if isinstance(version, int):
        return str(version)
    elif isinstance(version, str):
        return version
    return ''

class ReparseReason(StrEnum):
    version_mismatch = '01_version_mismatch'
    file_not_found = '02_file_not_found'
    vars_changed = '03_vars_changed'
    profile_changed = '04_profile_changed'
    deps_changed = '05_deps_changed'
    project_config_changed = '06_project_config_changed'
    load_file_failure = '07_load_file_failure'
    exception = '08_exception'
    proj_env_vars_changed = '09_project_env_vars_changed'
    prof_env_vars_changed = '10_profile_env_vars_changed'

@dataclass
class ParserInfo(dbtClassMixin):
    parsed_path_count: int = 0

@dataclass
class ProjectLoaderInfo(dbtClassMixin):
    parsers: List[ParserInfo] = field(default_factory=list)
    parsed_path_count: int = 0

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

    def __post_serialize__(self, dct: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        del dct['_project_index']
        return dct

class ManifestLoader:
    def __init__(self, root_project: RuntimeConfig, all_projects: Dict[str, Project], macro_hook: Optional[Callable[[Any], None] = None, file_diff: Optional[FileDiff] = None):
        self.root_project: RuntimeConfig = root_project
        self.all_projects: Dict[str, Project] = all_projects
        self.file_diff: Optional[FileDiff] = file_diff
        self.manifest: Manifest = Manifest()
        self.new_manifest: Manifest = self.manifest
        self.manifest.metadata = root_project.get_metadata()
        self.macro_resolver: Optional[MacroResolver] = None
        self.started_at: float = time.time()
        self.macro_hook: Callable[[Any], None] = macro_hook if macro_hook is not None else lambda m: None
        self._perf_info: ManifestLoaderInfo = self.build_perf_info()
        self.manifest.state_check: ManifestStateCheck = self.build_manifest_state_check()
        self.partially_parsing: bool = False
        self.partial_parser: Optional[PartialParsing] = None
        self.skip_parsing: bool = False
        self.saved_manifest: Optional[Manifest] = self.read_manifest_for_partial_parse()

    @classmethod
    def get_full_manifest(cls, config: RuntimeConfig, *, file_diff: Optional[FileDiff] = None, reset: bool = False, write_perf_info: bool = False) -> Manifest:
        adapter = get_adapter(config)
        if reset:
            config.clear_dependencies()
            adapter.clear_macro_resolver()
        macro_hook = adapter.connections.set_query_header
        flags = get_flags()
        if not flags.PARTIAL_PARSE_FILE_DIFF:
            file_diff = FileDiff.from_dict({'deleted': [], 'changed': [], 'added': []})
        elif os.environ.get('DBT_PP_FILE_DIFF_TEST'):
            file_diff_path = 'file_diff.json'
            if path_exists(file_diff_path):
                file_diff_dct = read_json(file_diff_path)
                file_diff = FileDiff.from_dict(file_diff_dct)
        start_load_all = time.perf_counter()
        projects = config.load_dependencies()
        loader = cls(config, projects, macro_hook=macro_hook, file_diff=file_diff)
        manifest = loader.load()
        _check_manifest(manifest, config)
        manifest.build_flat_graph()
        loader.save_macros_to_adapter(adapter)
        loader._perf_info.load_all_elapsed = time.perf_counter() - start_load_all
        loader.track_project_load()
        if write_perf_info:
            loader.write_perf_info(config.project_target_path)
        return manifest

    def load(self) -> Manifest:
        start_read_files = time.perf_counter()
        saved_files = self.saved_manifest.files if self.saved_manifest else {}
        file_reader: ReadFiles
        if self.file_diff:
            file_reader = ReadFilesFromDiff(all_projects=self.all_projects, files=self.manifest.files, saved_files=saved_files, root_project_name=self.root_project.project_name, file_diff=self.file_diff)
        else:
            file_reader = ReadFilesFromFileSystem(all_projects=self.all_projects, files=self.manifest.files, saved_files=saved_files)
        file_reader.read_files()
        self.manifest.files = file_reader.files
        project_parser_files = orig_project_parser_files = file_reader.project_parser_files
        self._perf_info.path_count = len(self.manifest.files)
        self._perf_info.read_files_elapsed = time.perf_counter() - start_read_files
        self.skip_parsing = False
        project_parser_files = self.safe_update_project_parser_files_partially(project_parser_files)
        if self.manifest._parsing_info is None:
            self.manifest._parsing_info = ParsingInfo()
        if self.skip_parsing:
            fire_event(PartialParsingSkipParsing())
        else:
            start_load_macros = time.perf_counter()
            self.load_and_parse_macros(project_parser_files)
            if self.partially_parsing and self.skip_partial_parsing_because_of_macros():
                fire_event(UnableToPartialParse(reason='change detected to override macro. Starting full parse.'))
                self.manifest = self.new_manifest
                project_parser_files = orig_project_parser_files
                self.partially_parsing = False
                self.load_and_parse_macros(project_parser_files)
            self._perf_info.load_macros_elapsed = time.perf_counter() - start_load_macros
            start_parse_projects = time.perf_counter()
            parser_types = [ModelParser, SnapshotParser, AnalysisParser, SingularTestParser, SeedParser, DocumentationParser, HookParser, FixtureParser]
            for project in self.all_projects.values():
                if project.project_name not in project_parser_files:
                    continue
                self.parse_project(project, project_parser_files[project.project_name], parser_types)
            self.manifest.rebuild_ref_lookup()
            self.manifest.rebuild_doc_lookup()
            self.manifest.rebuild_disabled_lookup()
            parser_types = [SchemaParser]
            for project in self.all_projects.values():
                if project.project_name not in project_parser_files:
                    continue
                self.parse_project(project, project_parser_files[project.project_name], parser_types)
            self.cleanup_disabled()
            self._perf_info.parse_project_elapsed = time.perf_counter() - start_parse_projects
            start_patch = time.perf_counter()
            patcher = SourcePatcher(self.root_project, self.manifest)
            patcher.construct_sources()
            self.manifest.sources = patcher.sources
            self._perf_info.patch_sources_elapsed = time.perf_counter() - start_patch
            self.manifest.rebuild_disabled_lookup()
            self.manifest.selectors = self.root_project.manifest_selectors
            self.manifest.build_parent_and_child_maps()
            external_nodes_modified = self.inject_external_nodes()
            if external_nodes_modified:
                self.manifest.rebuild_ref_lookup()
            start_process = time.perf_counter()
            self.process_sources(self.root_project.project_name)
            self.process_refs(self.root_project.project_name, self.root_project.dependencies)
            self.process_unit_tests(self.root_project.project_name)
            self.process_docs(self.root_project)
            self.process_metrics(self.root_project)
            self.process_saved_queries(self.root_project)
            self.process_model_inferred_primary_keys()
            self.check_valid_group_config()
            self.check_valid_access_property()
            self.check_valid_snapshot_config()
            self.check_valid_microbatch_config()
            semantic_manifest = SemanticManifest(self.manifest)
            if not semantic_manifest.validate():
                raise dbt.exceptions.ParsingError('Semantic Manifest validation failed.')
            self._perf_info.process_manifest_elapsed = time.perf_counter() - start_process
            self._perf_info.static_analysis_parsed_path_count = self.manifest._parsing_info.static_analysis_parsed_path_count
            self._perf_info.static_analysis_path_count = self.manifest._parsing_info.static_analysis_path_count
        external_nodes_modified = False
        if self.skip_parsing:
            self.manifest.build_parent_and_child_maps()
            external_nodes_modified = self.inject_external_nodes()
            if external_nodes_modified:
                self.manifest.rebuild_ref_lookup()
                self.process_refs(self.root_project.project_name, self.root_project.dependencies)
        if not self.skip_parsing or external_nodes_modified:
            self.write_manifest_for_partial_parse()
        self.check_for_model_deprecations()
        self.check_for_spaces_in_resource_names()
        self.check_for_microbatch_deprecations()
        self.check_forcing_batch_concurrency()
        self.check_microbatch_model_has_a_filtered_input()
        return self.manifest

    def safe_update_project_parser_files_partially(self, project_parser_files: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
        if self.saved_manifest is None:
            return project_parser_files
        self.partial_parser = PartialParsing(self.saved_manifest, self.manifest.files)
        self.skip_parsing = self.partial_parser.skip_parsing()
        if self.skip_parsing:
            self.manifest = self.saved_manifest
        else:
            self.saved_manifest.build_parent_and_child_maps()
            self.saved_manifest.build_group_map()
            try:
                project_parser_files = self.partial_parser.get_parsing_files()
                self.partially_parsing = True
                self.manifest = self.saved_manifest
            except Exception as exc:
                fire_event(UnableToPartialParse(reason='an error occurred. Switching to full reparse.'))
                tb_info = traceback.format_exc()
                tb_last_frame = traceback.extract_tb(exc.__traceback__)[-1]
                exc_info = {'traceback': tb_info, 'exception': tb_info.splitlines()[-1], 'code': tb_last_frame.line, 'location': f'line {tb_last_frame.lineno} in {tb_last_frame.name}'}
                parse_file_type = ''
                file_id = self.partial_parser.processing_file
                if file_id:
                    source_file = None
                    if file_id in self.saved_manifest.files:
                        source_file = self.saved_manifest.files[file_id]
                    elif file_id in self.manifest.files:
                        source_file = self.manifest.files[file_id]
                    if source_file:
                        parse_file_type = source_file.parse_file_type
                        fire_event(PartialParsingErrorProcessingFile(file=file_id))
                exc_info['parse_file_type'] = parse_file_type
                fire_event(PartialParsingError(exc_info=exc_info))
                if dbt.tracking.active_user is not None:
                    exc_info['full_reparse_reason'] = ReparseReason.exception
                    dbt.tracking.track_partial_parser(exc_info)
                if os.environ.get('DBT_PP