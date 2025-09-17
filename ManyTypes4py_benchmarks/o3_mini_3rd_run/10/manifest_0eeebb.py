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

PERF_INFO_FILE_NAME: str = 'perf_info.json'

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

def version_to_str(version: Union[int, str]) -> str:
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
    # You can add more specific fields if needed, for example:
    parser: Optional[str] = None
    elapsed: float = 0.0

@dataclass
class ProjectLoaderInfo(dbtClassMixin):
    project_name: str
    elapsed: float = 0.0
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

    def __post_serialize__(self, dct: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        del dct['_project_index']
        return dct

class ManifestLoader:
    def __init__(
        self,
        root_project: Project,
        all_projects: Mapping[str, Project],
        macro_hook: Optional[Callable[[Any], None]] = None,
        file_diff: Optional[FileDiff] = None,
    ) -> None:
        self.root_project: Project = root_project
        self.all_projects: Mapping[str, Project] = all_projects
        self.file_diff: Optional[FileDiff] = file_diff
        self.manifest: Manifest = Manifest()
        self.new_manifest: Manifest = self.manifest
        self.manifest.metadata = root_project.get_metadata()
        self.macro_resolver: Optional[MacroResolver] = None
        self.started_at: float = time.time()
        if macro_hook is None:
            self.macro_hook = lambda m: None
        else:
            self.macro_hook = macro_hook
        self._perf_info: ManifestLoaderInfo = self.build_perf_info()
        self.manifest.state_check = self.build_manifest_state_check()
        self.partially_parsing: bool = False
        self.partial_parser: Optional[PartialParsing] = None
        self.skip_parsing: bool = False
        self.saved_manifest: Optional[Manifest] = self.read_manifest_for_partial_parse()

    @classmethod
    def get_full_manifest(
        cls,
        config: RuntimeConfig,
        *,
        file_diff: Optional[FileDiff] = None,
        reset: bool = False,
        write_perf_info: bool = False,
    ) -> Manifest:
        adapter = get_adapter(config)
        if reset:
            config.clear_dependencies()
            adapter.clear_macro_resolver()
        macro_hook = adapter.connections.set_query_header
        flags = get_flags()
        if not flags.PARTIAL_PARSE_FILE_DIFF:
            file_diff = FileDiff.from_dict({'deleted': [], 'changed': [], 'added': []})
        elif os.environ.get('DBT_PP_FILE_DIFF_TEST'):
            file_diff_path: str = 'file_diff.json'
            if path_exists(file_diff_path):
                file_diff_dct: Dict[str, Any] = read_json(file_diff_path)
                file_diff = FileDiff.from_dict(file_diff_dct)
        start_load_all: float = time.perf_counter()
        projects: Mapping[str, Project] = config.load_dependencies()
        loader: ManifestLoader = cls(config, projects, macro_hook=macro_hook, file_diff=file_diff)
        manifest: Manifest = loader.load()
        _check_manifest(manifest, config)
        manifest.build_flat_graph()
        loader.save_macros_to_adapter(adapter)
        loader._perf_info.load_all_elapsed = time.perf_counter() - start_load_all
        loader.track_project_load()
        if write_perf_info:
            loader.write_perf_info(config.project_target_path)
        return manifest

    def load(self) -> Manifest:
        start_read_files: float = time.perf_counter()
        saved_files: Dict[str, Any] = self.saved_manifest.files if self.saved_manifest else {}
        file_reader: ReadFiles
        if self.file_diff:
            file_reader = ReadFilesFromDiff(
                all_projects=self.all_projects,
                files=self.manifest.files,
                saved_files=saved_files,
                root_project_name=self.root_project.project_name,
                file_diff=self.file_diff,
            )
        else:
            file_reader = ReadFilesFromFileSystem(
                all_projects=self.all_projects,
                files=self.manifest.files,
                saved_files=saved_files,
            )
        file_reader.read_files()
        self.manifest.files = file_reader.files
        project_parser_files: Dict[str, Dict[str, List[str]]] = file_reader.project_parser_files
        orig_project_parser_files: Dict[str, Dict[str, List[str]]] = project_parser_files
        self._perf_info.path_count = len(self.manifest.files)
        self._perf_info.read_files_elapsed = time.perf_counter() - start_read_files
        self.skip_parsing = False
        project_parser_files = self.safe_update_project_parser_files_partially(project_parser_files)
        if self.manifest._parsing_info is None:
            self.manifest._parsing_info = ParsingInfo()
        if self.skip_parsing:
            fire_event(PartialParsingSkipParsing())
        else:
            start_load_macros: float = time.perf_counter()
            self.load_and_parse_macros(project_parser_files)
            if self.partially_parsing and self.skip_partial_parsing_because_of_macros():
                fire_event(UnableToPartialParse(reason='change detected to override macro. Starting full parse.'))
                self.manifest = self.new_manifest
                project_parser_files = orig_project_parser_files
                self.partially_parsing = False
                self.load_and_parse_macros(project_parser_files)
            self._perf_info.load_macros_elapsed = time.perf_counter() - start_load_macros
            start_parse_projects: float = time.perf_counter()
            parser_types: List[Type[Parser]] = [ModelParser, SnapshotParser, AnalysisParser, SingularTestParser, SeedParser, DocumentationParser, HookParser, FixtureParser]
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
            start_patch: float = time.perf_counter()
            patcher = SourcePatcher(self.root_project, self.manifest)
            patcher.construct_sources()
            self.manifest.sources = patcher.sources
            self._perf_info.patch_sources_elapsed = time.perf_counter() - start_patch
            self.manifest.rebuild_disabled_lookup()
            self.manifest.selectors = self.root_project.manifest_selectors
            self.manifest.build_parent_and_child_maps()
            external_nodes_modified: bool = self.inject_external_nodes()
            if external_nodes_modified:
                self.manifest.rebuild_ref_lookup()
            start_process: float = time.perf_counter()
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
            self.check_forcing_batch_concurrency()
            self.check_microbatch_model_has_a_filtered_input()
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

    def safe_update_project_parser_files_partially(
        self, project_parser_files: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, Dict[str, List[str]]]:
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
                tb_info: str = traceback.format_exc()
                tb_last_frame = traceback.extract_tb(exc.__traceback__)[-1]
                exc_info: Dict[str, Any] = {
                    'traceback': tb_info,
                    'exception': tb_info.splitlines()[-1],
                    'code': tb_last_frame.line,
                    'location': f'line {tb_last_frame.lineno} in {tb_last_frame.name}',
                }
                parse_file_type: str = ''
                file_id: Optional[str] = self.partial_parser.processing_file
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
                if os.environ.get('DBT_PP_TEST'):
                    raise exc
        return project_parser_files

    def check_for_model_deprecations(self) -> None:
        self.manifest.build_parent_and_child_maps()  # type: ignore
        for node in self.manifest.nodes.values():
            if isinstance(node, ModelNode) and node.deprecation_date:
                if node.is_past_deprecation_date:
                    warn_or_error(
                        DeprecatedModel(
                            model_name=node.name,
                            model_version=version_to_str(node.version),
                            deprecation_date=node.deprecation_date.isoformat(),
                        )
                    )
                child_nodes = self.manifest.child_map[node.unique_id]
                for child_unique_id in child_nodes:
                    child_node = self.manifest.nodes.get(child_unique_id)
                    if not isinstance(child_node, ModelNode):
                        continue
                    if node.is_past_deprecation_date:
                        event_cls = DeprecatedReference
                    else:
                        event_cls = UpcomingReferenceDeprecation
                    warn_or_error(
                        event_cls(
                            model_name=child_node.name,
                            ref_model_package=node.package_name,
                            ref_model_name=node.name,
                            ref_model_version=version_to_str(node.version),
                            ref_model_latest_version=str(node.latest_version),
                            ref_model_deprecation_date=node.deprecation_date.isoformat(),
                        )
                    )

    def check_for_spaces_in_resource_names(self) -> None:
        improper_resource_names: int = 0
        level: EventLevel = EventLevel.ERROR if self.root_project.args.REQUIRE_RESOURCE_NAMES_WITHOUT_SPACES else EventLevel.WARN
        flags = get_flags()
        for node in self.manifest.nodes.values():
            if ' ' in node.name:
                if improper_resource_names == 0 or flags.DEBUG:
                    fire_event(SpacesInResourceNameDeprecation(unique_id=node.unique_id, level=level.value), level=level)
                improper_resource_names += 1
        if improper_resource_names > 0:
            if level == EventLevel.WARN:
                dbt.deprecations.warn('resource-names-with-spaces', count_invalid_names=improper_resource_names, show_debug_hint=not flags.DEBUG)
            else:
                raise DbtValidationError('Resource names cannot contain spaces')

    def check_for_microbatch_deprecations(self) -> None:
        if not get_flags().require_batched_execution_for_custom_microbatch_strategy:
            has_microbatch_model: bool = False
            for _, node in self.manifest.nodes.items():
                if isinstance(node, ModelNode) and node.config.materialized == 'incremental' and (node.config.incremental_strategy == 'microbatch'):
                    has_microbatch_model = True
                    break
            if has_microbatch_model and (not self.manifest._microbatch_macro_is_core(self.root_project.project_name)):
                dbt.deprecations.warn('microbatch-macro-outside-of-batches-deprecation')

    def load_and_parse_macros(self, project_parser_files: Dict[str, Dict[str, List[str]]]) -> None:
        for project in self.all_projects.values():
            if project.project_name not in project_parser_files:
                continue
            parser_files: Dict[str, List[str]] = project_parser_files[project.project_name]
            if 'MacroParser' in parser_files:
                parser = MacroParser(project, self.manifest)
                for file_id in parser_files['MacroParser']:
                    block = FileBlock(self.manifest.files[file_id])
                    parser.parse_file(block)
                    self._perf_info.parsed_path_count += 1
            if 'GenericTestParser' in parser_files:
                parser = GenericTestParser(project, self.manifest)
                for file_id in parser_files['GenericTestParser']:
                    block = FileBlock(self.manifest.files[file_id])
                    parser.parse_file(block)
                    self._perf_info.parsed_path_count += 1
        self.build_macro_resolver()
        self.macro_depends_on()

    def parse_project(
        self,
        project: Project,
        parser_files: Dict[str, List[str]],
        parser_types: List[Type[Parser]],
    ) -> None:
        project_loader_info: ProjectLoaderInfo = self._perf_info._project_index[project.project_name]
        start_timer: float = time.perf_counter()
        total_parsed_path_count: int = 0
        for parser_cls in parser_types:
            parser_name: str = parser_cls.__name__
            if parser_name not in parser_files or not parser_files[parser_name]:
                continue
            project_parsed_path_count: int = 0
            parser_start_timer: float = time.perf_counter()
            parser: Parser = parser_cls(project, self.manifest, self.root_project)
            for file_id in parser_files[parser_name]:
                block = FileBlock(self.manifest.files[file_id])
                if isinstance(parser, SchemaParser):
                    assert isinstance(block.file, SchemaSourceFile)
                    if self.partially_parsing:
                        dct = block.file.pp_dict
                    else:
                        dct = block.file.dict_from_yaml
                    parser.parse_file(block, dct=dct)
                else:
                    parser.parse_file(block)
                project_parsed_path_count += 1
            project_loader_info.parsers.append(ParserInfo(parser=parser_name, parsed_path_count=project_parsed_path_count, elapsed=time.perf_counter() - parser_start_timer))
            total_parsed_path_count += project_parsed_path_count
        if not self.partially_parsing and HookParser in parser_types:
            hook_parser = HookParser(project, self.manifest, self.root_project)
            path = hook_parser.get_path()
            file = load_source_file(path, ParseFileType.Hook, project.project_name, {})
            if file:
                file_block = FileBlock(file)
                hook_parser.parse_file(file_block)
        elapsed: float = time.perf_counter() - start_timer
        project_loader_info.parsed_path_count = project_loader_info.parsed_path_count + total_parsed_path_count
        project_loader_info.elapsed += elapsed
        self._perf_info.parsed_path_count = self._perf_info.parsed_path_count + total_parsed_path_count

    def build_macro_resolver(self) -> None:
        internal_package_names: List[str] = get_adapter_package_names(self.root_project.credentials.type)
        self.macro_resolver = MacroResolver(self.manifest.macros, self.root_project.project_name, internal_package_names)

    def macro_depends_on(self) -> None:
        macro_ctx: Dict[str, Any] = generate_macro_context(self.root_project)
        macro_namespace = TestMacroNamespace(self.macro_resolver, {}, None, MacroStack(), [])
        adapter = get_adapter(self.root_project)
        db_wrapper = ParseProvider().DatabaseWrapper(adapter, macro_namespace)
        for macro in self.manifest.macros.values():
            if macro.created_at < self.started_at:
                continue
            possible_macro_calls: List[str] = statically_extract_macro_calls(macro.macro_sql, macro_ctx, db_wrapper)
            for macro_name in possible_macro_calls:
                if macro_name == macro.name:
                    continue
                package_name: Optional[str] = macro.package_name
                if '.' in macro_name:
                    package_name, macro_name = macro_name.split('.')
                dep_macro_id: Optional[str] = self.macro_resolver.get_macro_id(package_name, macro_name)
                if dep_macro_id:
                    macro.depends_on.add_macro(dep_macro_id)

    def write_manifest_for_partial_parse(self) -> None:
        path: str = os.path.join(self.root_project.project_target_path, PARTIAL_PARSE_FILE_NAME)
        try:
            if self.manifest.metadata.dbt_version != __version__:
                fire_event(UnableToPartialParse(reason='saved manifest contained the wrong version'))
                self.manifest.metadata.dbt_version = __version__
            manifest_msgpack: bytes = self.manifest.to_msgpack(extended_mashumaro_encoder)
            make_directory(os.path.dirname(path))
            with open(path, 'wb') as fp:
                fp.write(manifest_msgpack)
        except Exception:
            raise

    def inject_external_nodes(self) -> bool:
        manifest_nodes_modified: bool = False
        for unique_id in self.manifest.external_node_unique_ids:
            remove_dependent_project_references(self.manifest, unique_id)
            manifest_nodes_modified = True
        for unique_id in self.manifest.external_node_unique_ids:
            self.manifest.nodes.pop(unique_id)
        pm = plugins.get_plugin_manager(self.root_project.project_name)
        plugin_model_nodes = pm.get_nodes().models
        for node_arg in plugin_model_nodes.values():
            node = ModelNode.from_args(node_arg)
            if node.unique_id not in self.manifest.nodes and node.unique_id not in self.manifest.disabled:
                self.manifest.add_node_nofile(node)
                manifest_nodes_modified = True
        return manifest_nodes_modified

    def is_partial_parsable(self, manifest: Manifest) -> Tuple[bool, Optional[ReparseReason]]:
        valid: bool = True
        reparse_reason: Optional[ReparseReason] = None
        if manifest.metadata.dbt_version != __version__:
            fire_event(UnableToPartialParse(reason='of a version mismatch'))
            return (False, ReparseReason.version_mismatch)
        if self.manifest.state_check.vars_hash != manifest.state_check.vars_hash:
            fire_event(UnableToPartialParse(reason='config vars, config profile, or config target have changed'))
            fire_event(Note(msg=f'previous checksum: {self.manifest.state_check.vars_hash.checksum}, current checksum: {manifest.state_check.vars_hash.checksum}'), level=EventLevel.DEBUG)
            valid = False
            reparse_reason = ReparseReason.vars_changed
        if self.manifest.state_check.profile_hash != manifest.state_check.profile_hash:
            fire_event(UnableToPartialParse(reason='profile has changed'))
            valid = False
            reparse_reason = ReparseReason.profile_changed
        if self.manifest.state_check.project_env_vars_hash != manifest.state_check.project_env_vars_hash:
            fire_event(UnableToPartialParse(reason='env vars used in dbt_project.yml have changed'))
            valid = False
            reparse_reason = ReparseReason.proj_env_vars_changed
        missing_keys = {k for k in self.manifest.state_check.project_hashes if k not in manifest.state_check.project_hashes}
        if missing_keys:
            fire_event(UnableToPartialParse(reason='a project dependency has been added'))
            valid = False
            reparse_reason = ReparseReason.deps_changed
        for key, new_value in self.manifest.state_check.project_hashes.items():
            if key in manifest.state_check.project_hashes:
                old_value = manifest.state_check.project_hashes[key]
                if new_value != old_value:
                    fire_event(UnableToPartialParse(reason='a project config has changed'))
                    valid = False
                    reparse_reason = ReparseReason.project_config_changed
        return (valid, reparse_reason)

    def skip_partial_parsing_because_of_macros(self) -> bool:
        if not self.partial_parser:
            return False
        if self.partial_parser.deleted_special_override_macro:
            return True
        for macro_name in special_override_macros:
            macro = self.macro_resolver.get_macro(None, macro_name)
            if macro and macro.package_name != 'dbt':
                if macro.file_id in self.partial_parser.file_diff['changed'] or macro.file_id in self.partial_parser.file_diff['added']:
                    return True
        return False

    def read_manifest_for_partial_parse(self) -> Optional[Manifest]:
        flags = get_flags()
        if not flags.PARTIAL_PARSE:
            fire_event(PartialParsingNotEnabled())
            return None
        path: str = flags.PARTIAL_PARSE_FILE_PATH or os.path.join(self.root_project.project_target_path, PARTIAL_PARSE_FILE_NAME)
        reparse_reason: Optional[ReparseReason] = None
        if os.path.exists(path):
            try:
                with open(path, 'rb') as fp:
                    manifest_mp: bytes = fp.read()
                manifest: Manifest = Manifest.from_msgpack(manifest_mp, decoder=extended_mashumuro_decoder)
                is_partial_parsable, reparse_reason = self.is_partial_parsable(manifest)
                if is_partial_parsable:
                    manifest.metadata.generated_at = datetime.datetime.utcnow()
                    manifest.metadata.invocation_id = get_invocation_id()
                    return manifest
            except Exception as exc:
                fire_event(ParsedFileLoadFailed(path=path, exc=str(exc), exc_info=traceback.format_exc()))
                reparse_reason = ReparseReason.load_file_failure
        else:
            fire_event(UnableToPartialParse(reason='saved manifest not found. Starting full parse.'))
            reparse_reason = ReparseReason.file_not_found
        if dbt.tracking.active_user is not None:
            dbt.tracking.track_partial_parser({'full_reparse_reason': reparse_reason})
        return None

    def build_perf_info(self) -> ManifestLoaderInfo:
        flags = get_flags()
        mli: ManifestLoaderInfo = ManifestLoaderInfo(
            is_partial_parse_enabled=flags.PARTIAL_PARSE,
            is_static_analysis_enabled=flags.STATIC_PARSER
        )
        for project in self.all_projects.values():
            project_info = ProjectLoaderInfo(project_name=project.project_name, elapsed=0)
            mli.projects.append(project_info)
            mli._project_index[project.project_name] = project_info
        return mli

    def build_manifest_state_check(self) -> ManifestStateCheck:
        config: Project = self.root_project
        all_projects: Mapping[str, Project] = self.all_projects
        secret_vars: List[str] = [v for k, v in config.cli_vars.items() if k.startswith(SECRET_ENV_PREFIX) and v.strip()]
        stringified_cli_vars: str = pprint.pformat(config.cli_vars)
        vars_hash = FileHash.from_contents('\x00'.join([stringified_cli_vars, getattr(config.args, 'profile', '') or '', getattr(config.args, 'target', '') or '', __version__]))
        fire_event(StateCheckVarsHash(checksum=vars_hash.checksum, vars=scrub_secrets(stringified_cli_vars, secret_vars), profile=config.args.profile, target=config.args.target, version=__version__))
        key_list: List[str] = list(config.project_env_vars.keys())
        key_list.sort()
        env_var_str: str = ''
        for key in key_list:
            env_var_str += f'{key}:{config.project_env_vars[key]}|'
        project_env_vars_hash = FileHash.from_contents(env_var_str)
        connection_keys: List[Any] = list(config.credentials.connection_info())
        connection_keys.sort()
        profile_hash = FileHash.from_contents(pprint.pformat(connection_keys))
        project_hashes: Dict[str, FileHash] = {}
        for name, project in all_projects.items():
            path = os.path.join(project.project_root, 'dbt_project.yml')
            with open(path) as fp:
                project_hashes[name] = FileHash.from_contents(fp.read())
        state_check = ManifestStateCheck(
            project_env_vars_hash=project_env_vars_hash,
            vars_hash=vars_hash,
            profile_hash=profile_hash,
            project_hashes=project_hashes,
        )
        return state_check

    def save_macros_to_adapter(self, adapter: Any) -> None:
        adapter.set_macro_resolver(self.manifest)
        query_header_context: Dict[str, Any] = generate_query_header_context(adapter.config, self.manifest)
        self.macro_hook(query_header_context)

    def create_macro_manifest(self) -> MacroManifest:
        for project in self.all_projects.values():
            macro_parser = MacroParser(project, self.manifest)
            for path in macro_parser.get_paths():
                source_file = load_source_file(path, ParseFileType.Macro, project.project_name, {})
                block = FileBlock(source_file)
                macro_parser.parse_file(block)
        macro_manifest = MacroManifest(self.manifest.macros)
        return macro_manifest

    @classmethod
    def load_macros(cls, root_config: RuntimeConfig, macro_hook: Callable[[Any], None], base_macros_only: bool = False) -> MacroManifest:
        projects = root_config.load_dependencies(base_only=base_macros_only)
        loader = cls(root_config, projects, macro_hook)
        return loader.create_macro_manifest()

    def track_project_load(self) -> None:
        invocation_id: str = get_invocation_id()
        dbt.tracking.track_project_load({
            'invocation_id': invocation_id,
            'project_id': self.root_project.hashed_name(),
            'path_count': self._perf_info.path_count,
            'parsed_path_count': self._perf_info.parsed_path_count,
            'read_files_elapsed': self._perf_info.read_files_elapsed,
            'load_macros_elapsed': self._perf_info.load_macros_elapsed,
            'parse_project_elapsed': self._perf_info.parse_project_elapsed,
            'patch_sources_elapsed': self._perf_info.patch_sources_elapsed,
            'process_manifest_elapsed': self._perf_info.process_manifest_elapsed,
            'load_all_elapsed': self._perf_info.load_all_elapsed,
            'is_partial_parse_enabled': self._perf_info.is_partial_parse_enabled,
            'is_static_analysis_enabled': self._perf_info.is_static_analysis_enabled,
            'static_analysis_path_count': self._perf_info.static_analysis_path_count,
            'static_analysis_parsed_path_count': self._perf_info.static_analysis_parsed_path_count,
        })

    def process_refs(self, current_project: str, dependencies: Any) -> None:
        for node in self.manifest.nodes.values():
            if node.created_at < self.started_at:
                continue
            _process_refs(self.manifest, current_project, node, dependencies)
        for exposure in self.manifest.exposures.values():
            if exposure.created_at < self.started_at:
                continue
            _process_refs(self.manifest, current_project, exposure, dependencies)
        for metric in self.manifest.metrics.values():
            if metric.created_at < self.started_at:
                continue
            _process_refs(self.manifest, current_project, metric, dependencies)
        for semantic_model in self.manifest.semantic_models.values():
            if semantic_model.created_at < self.started_at:
                continue
            _process_refs(self.manifest, current_project, semantic_model, dependencies)
            self.update_semantic_model(semantic_model)

    def process_metrics(self, config: Project) -> None:
        current_project: str = config.project_name
        for metric in self.manifest.metrics.values():
            if metric.created_at < self.started_at:
                continue
            _process_metric_node(self.manifest, current_project, metric)
            _process_metrics_for_node(self.manifest, current_project, metric)
        for node in self.manifest.nodes.values():
            if node.created_at < self.started_at:
                continue
            _process_metrics_for_node(self.manifest, current_project, node)
        for exposure in self.manifest.exposures.values():
            if exposure.created_at < self.started_at:
                continue
            _process_metrics_for_node(self.manifest, current_project, exposure)

    def process_saved_queries(self, config: Project) -> None:
        semantic_manifest_changed: bool = False
        semantic_manifest_nodes = chain(self.manifest.saved_queries.values(), self.manifest.semantic_models.values(), self.manifest.metrics.values())
        for node in semantic_manifest_nodes:
            if node.created_at > self.started_at:
                semantic_manifest_changed = True
                break
        if semantic_manifest_changed is False:
            return
        current_project: str = config.project_name
        for saved_query in self.manifest.saved_queries.values():
            _process_metrics_for_node(self.manifest, current_project, saved_query)

    def process_model_inferred_primary_keys(self) -> None:
        model_to_generic_test_map: Dict[str, List[GenericTestNode]] = {}
        for node in self.manifest.nodes.values():
            if not isinstance(node, ModelNode):
                continue
            if node.created_at < self.started_at:
                continue
            if not model_to_generic_test_map:
                model_to_generic_test_map = self.build_model_to_generic_tests_map()
            generic_tests: List[GenericTestNode] = []
            if node.unique_id in model_to_generic_test_map:
                generic_tests = model_to_generic_test_map[node.unique_id]
            primary_key = node.infer_primary_key(generic_tests)
            node.primary_key = sorted(primary_key)

    def update_semantic_model(self, semantic_model: SemanticModel) -> None:
        if semantic_model.depends_on_nodes[0]:
            refd_node = self.manifest.nodes[semantic_model.depends_on_nodes[0]]
            semantic_model.node_relation = NodeRelation(
                relation_name=refd_node.relation_name,
                alias=refd_node.alias,
                schema_name=refd_node.schema,
                database=refd_node.database,
            )

    def process_docs(self, config: Project) -> None:
        for node in self.manifest.nodes.values():
            if node.created_at < self.started_at:
                continue
            ctx: Dict[str, Any] = generate_runtime_docs_context(config, node, self.manifest, config.project_name)
            _process_docs_for_node(ctx, node, self.manifest)
        for source in self.manifest.sources.values():
            if source.created_at < self.started_at:
                continue
            ctx = generate_runtime_docs_context(config, source, self.manifest, config.project_name)
            _process_docs_for_source(ctx, source, self.manifest)
        for macro in self.manifest.macros.values():
            if macro.created_at < self.started_at:
                continue
            ctx = generate_runtime_docs_context(config, macro, self.manifest, config.project_name)
            _process_docs_for_macro(ctx, macro)
        for exposure in self.manifest.exposures.values():
            if exposure.created_at < self.started_at:
                continue
            ctx = generate_runtime_docs_context(config, exposure, self.manifest, config.project_name)
            _process_docs_for_exposure(ctx, exposure)
        for metric in self.manifest.metrics.values():
            if metric.created_at < self.started_at:
                continue
            ctx = generate_runtime_docs_context(config, metric, self.manifest, config.project_name)
            _process_docs_for_metrics(ctx, metric)
        for semantic_model in self.manifest.semantic_models.values():
            if semantic_model.created_at < self.started_at:
                continue
            ctx = generate_runtime_docs_context(config, semantic_model, self.manifest, config.project_name)
            _process_docs_for_semantic_model(ctx, semantic_model)
        for saved_query in self.manifest.saved_queries.values():
            if saved_query.created_at < self.started_at:
                continue
            ctx = generate_runtime_docs_context(config, saved_query, self.manifest, config.project_name)
            _process_docs_for_saved_query(ctx, saved_query)

    def process_sources(self, current_project: str) -> None:
        for node in self.manifest.nodes.values():
            if node.resource_type == NodeType.Source:
                continue
            assert not isinstance(node, SourceDefinition)
            if node.created_at < self.started_at:
                continue
            _process_sources_for_node(self.manifest, current_project, node)
        for exposure in self.manifest.exposures.values():
            if exposure.created_at < self.started_at:
                continue
            _process_sources_for_exposure(self.manifest, current_project, exposure)

    def process_unit_tests(self, current_project: str) -> None:
        models_to_versions = None
        unit_test_unique_ids: List[str] = list(self.manifest.unit_tests.keys())
        for unit_test_unique_id in unit_test_unique_ids:
            if unit_test_unique_id in self.manifest.unit_tests:
                unit_test = self.manifest.unit_tests[unit_test_unique_id]
            else:
                continue
            if unit_test.created_at < self.started_at:
                continue
            if not models_to_versions:
                models_to_versions = _build_model_names_to_versions(self.manifest)
            process_models_for_unit_test(self.manifest, current_project, unit_test, models_to_versions)

    def cleanup_disabled(self) -> None:
        disabled_nodes: List[str] = []
        for node in self.manifest.nodes.values():
            if not node.config.enabled:
                disabled_nodes.append(node.unique_id)
                self.manifest.add_disabled_nofile(node)
        for unique_id in disabled_nodes:
            self.manifest.nodes.pop(unique_id)
        disabled_copy = deepcopy(self.manifest.disabled)
        for disabled in disabled_copy.values():
            for node in disabled:
                if node.config.enabled:
                    for dis_index, dis_node in enumerate(disabled):
                        del self.manifest.disabled[node.unique_id][dis_index]
                        if not self.manifest.disabled[node.unique_id]:
                            self.manifest.disabled.pop(node.unique_id)
                    self.manifest.add_node_nofile(node)
        self.manifest.rebuild_ref_lookup()

    def check_valid_group_config(self) -> None:
        manifest = self.manifest
        group_names: Set[str] = {group.name for group in manifest.groups.values()}
        for metric in manifest.metrics.values():
            self.check_valid_group_config_node(metric, group_names)
        for semantic_model in manifest.semantic_models.values():
            self.check_valid_group_config_node(semantic_model, group_names)
        for saved_query in manifest.saved_queries.values():
            self.check_valid_group_config_node(saved_query, group_names)
        for node in manifest.nodes.values():
            self.check_valid_group_config_node(node, group_names)

    def check_valid_group_config_node(self, groupable_node: Any, valid_group_names: Set[str]) -> None:
        groupable_node_group: Optional[str] = groupable_node.group
        if groupable_node_group and groupable_node_group not in valid_group_names:
            raise dbt.exceptions.ParsingError(f"Invalid group '{groupable_node_group}', expected one of {sorted(list(valid_group_names))}", node=groupable_node)

    def check_valid_access_property(self) -> None:
        for node in self.manifest.nodes.values():
            if isinstance(node, ModelNode) and node.access == AccessType.Public and (node.get_materialization() == 'ephemeral'):
                raise InvalidAccessTypeError(unique_id=node.unique_id, field_value=node.access, materialization=node.get_materialization())

    def check_valid_snapshot_config(self) -> None:
        for node in self.manifest.nodes.values():
            if node.resource_type != NodeType.Snapshot:
                continue
            if node.created_at < self.started_at:
                continue
            node.config.final_validate()

    def check_valid_microbatch_config(self) -> None:
        if self.manifest.use_microbatch_batches(project_name=self.root_project.project_name):
            for node in self.manifest.nodes.values():
                if node.config.materialized == 'incremental' and node.config.incremental_strategy == 'microbatch':
                    event_time = node.config.event_time
                    if event_time is None:
                        raise dbt.exceptions.ParsingError(f"Microbatch model '{node.name}' must provide an 'event_time' (string) config that indicates the name of the event time column.")
                    if not isinstance(event_time, str):
                        raise dbt.exceptions.ParsingError(f"Microbatch model '{node.name}' must provide an 'event_time' config of type string, but got: {type(event_time)}.")
                    begin = node.config.begin
                    if begin is None:
                        raise dbt.exceptions.ParsingError(f"Microbatch model '{node.name}' must provide a 'begin' (datetime) config that indicates the earliest timestamp the microbatch model should be built from.")
                    if isinstance(begin, str):
                        try:
                            begin = datetime.datetime.fromisoformat(begin)
                            node.config.begin = begin
                        except Exception:
                            raise dbt.exceptions.ParsingError(f"Microbatch model '{node.name}' must provide a 'begin' config of valid datetime (ISO format), but got: {begin}.")
                    if not isinstance(begin, datetime.datetime):
                        raise dbt.exceptions.ParsingError(f"Microbatch model '{node.name}' must provide a 'begin' config of type datetime, but got: {type(begin)}.")
                    batch_size = node.config.batch_size
                    valid_batch_sizes = [size.value for size in BatchSize]
                    if batch_size not in valid_batch_sizes:
                        raise dbt.exceptions.ParsingError(f"Microbatch model '{node.name}' must provide a 'batch_size' config that is one of {valid_batch_sizes}, but got: {batch_size}.")
                    lookback = node.config.lookback
                    if not isinstance(lookback, int) and lookback is not None:
                        raise dbt.exceptions.ParsingError(f"Microbatch model '{node.name}' must provide the optional 'lookback' config as type int, but got: {type(lookback)}).")
                    concurrent_batches = node.config.concurrent_batches
                    if not isinstance(concurrent_batches, bool) and concurrent_batches is not None:
                        raise dbt.exceptions.ParsingError(f"Microbatch model '{node.name}' optional 'concurrent_batches' config must be of type `bool` if specified, but got: {type(concurrent_batches)}).")

    def check_forcing_batch_concurrency(self) -> None:
        if self.manifest.use_microbatch_batches(project_name=self.root_project.project_name):
            adapter = get_adapter(self.root_project)
            if not adapter.supports(Capability.MicrobatchConcurrency):
                models_forcing_concurrent_batches = 0
                for node in self.manifest.nodes.values():
                    if hasattr(node.config, 'concurrent_batches') and node.config.concurrent_batches is True:
                        models_forcing_concurrent_batches += 1
                if models_forcing_concurrent_batches > 0:
                    warn_or_error(InvalidConcurrentBatchesConfig(num_models=models_forcing_concurrent_batches, adapter_type=adapter.type()))

    def check_microbatch_model_has_a_filtered_input(self) -> None:
        if self.manifest.use_microbatch_batches(project_name=self.root_project.project_name):
            for node in self.manifest.nodes.values():
                if node.config.materialized == 'incremental' and node.config.incremental_strategy == 'microbatch':
                    has_input_with_event_time_config = False
                    for input_unique_id in node.depends_on.nodes:
                        input_node = self.manifest.expect(unique_id=input_unique_id)
                        input_event_time = input_node.config.event_time
                        if input_event_time:
                            if not isinstance(input_event_time, str):
                                raise dbt.exceptions.ParsingError(f"Microbatch model '{node.name}' depends on an input node '{input_node.name}' with an 'event_time' config of invalid (non-string) type: {type(input_event_time)}.")
                            has_input_with_event_time_config = True
                    if not has_input_with_event_time_config:
                        fire_event(MicrobatchModelNoEventTimeInputs(model_name=node.name))

    def write_perf_info(self, target_path: str) -> None:
        path: str = os.path.join(target_path, PERF_INFO_FILE_NAME)
        write_file(path, json.dumps(self._perf_info, cls=dbt.utils.JSONEncoder, indent=4))
        fire_event(ParsePerfInfoPath(path=path))

    def build_model_to_generic_tests_map(self) -> Dict[str, List[GenericTestNode]]:
        model_to_generic_tests_map: Dict[str, List[GenericTestNode]] = {}
        for _, node in self.manifest.nodes.items():
            if isinstance(node, GenericTestNode) and node.attached_node:
                if node.attached_node not in model_to_generic_tests_map:
                    model_to_generic_tests_map[node.attached_node] = []
                model_to_generic_tests_map[node.attached_node].append(node)
        for _, nodes in self.manifest.disabled.items():
            for disabled_node in nodes:
                if isinstance(disabled_node, GenericTestNode) and disabled_node.attached_node:
                    if disabled_node.attached_node not in model_to_generic_tests_map:
                        model_to_generic_tests_map[disabled_node.attached_node] = []
                    model_to_generic_tests_map[disabled_node.attached_node].append(disabled_node)
        return model_to_generic_tests_map

def invalid_target_fail_unless_test(
    node: ManifestNode,
    target_name: str,
    target_kind: str,
    target_package: Optional[str] = None,
    target_version: Optional[str] = None,
    disabled: Optional[Any] = None,
    should_warn_if_disabled: bool = True,
) -> None:
    if node.resource_type == NodeType.Test:
        if disabled:
            event = InvalidDisabledTargetInTestNode(
                resource_type_title=node.resource_type.title(),
                unique_id=node.unique_id,
                original_file_path=node.original_file_path,
                target_kind=target_kind,
                target_name=target_name,
                target_package=target_package if target_package else '',
            )
            fire_event(event, EventLevel.WARN if should_warn_if_disabled else None)
        else:
            warn_or_error(
                NodeNotFoundOrDisabled(
                    original_file_path=node.original_file_path,
                    unique_id=node.unique_id,
                    resource_type_title=node.resource_type.title(),
                    target_name=target_name,
                    target_kind=target_kind,
                    target_package=target_package if target_package else '',
                    disabled=str(disabled),
                )
            )
    else:
        raise TargetNotFoundError(node=node, target_name=target_name, target_kind=target_kind, target_package=target_package, target_version=target_version, disabled=disabled)

def _build_model_names_to_versions(manifest: Manifest) -> Dict[str, Dict[str, List[str]]]:
    model_names_to_versions: Dict[str, Dict[str, List[str]]] = {}
    for node in manifest.nodes.values():
        if node.resource_type != NodeType.Model:
            continue
        if not node.is_versioned:
            continue
        if node.package_name not in model_names_to_versions:
            model_names_to_versions[node.package_name] = {}
        if node.name not in model_names_to_versions[node.package_name]:
            model_names_to_versions[node.package_name][node.name] = []
        model_names_to_versions[node.package_name][node.name].append(node.unique_id)
    return model_names_to_versions

def _check_resource_uniqueness(manifest: Manifest, config: RuntimeConfig) -> None:
    alias_resources: Dict[str, Any] = {}
    name_resources: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for resource, node in manifest.nodes.items():
        if not node.is_relational:
            continue
        if node.package_name not in name_resources:
            name_resources[node.package_name] = {'ver': {}, 'unver': {}}
        if node.is_versioned:
            name_resources[node.package_name]['ver'][node.name] = node
        else:
            name_resources[node.package_name]['unver'][node.name] = node
        relation_cls = get_relation_class_by_name(config.credentials.type)
        relation = relation_cls.create_from(quoting=config, relation_config=node)
        full_node_name = str(relation)
        existing_alias = alias_resources.get(full_node_name)
        if existing_alias is not None:
            raise AmbiguousAliasError(node_1=existing_alias, node_2=node, duped_name=full_node_name)
        alias_resources[full_node_name] = node
    for ver_unver_dict in name_resources.values():
        versioned_names = ver_unver_dict['ver'].keys()
        unversioned_names = ver_unver_dict['unver'].keys()
        intersection_versioned = set(versioned_names).intersection(set(unversioned_names))
        if intersection_versioned:
            for name in intersection_versioned:
                versioned_node = ver_unver_dict['ver'][name]
                unversioned_node = ver_unver_dict['unver'][name]
                raise dbt.exceptions.DuplicateVersionedUnversionedError(versioned_node, unversioned_node)

def _warn_for_unused_resource_config_paths(manifest: Manifest, config: Project) -> None:
    resource_fqns = manifest.get_resource_fqns()
    disabled_fqns = frozenset((tuple(n.fqn) for n in list(chain.from_iterable(manifest.disabled.values()))))
    config.warn_for_unused_resource_config_paths(resource_fqns, disabled_fqns)

def _check_manifest(manifest: Manifest, config: RuntimeConfig) -> None:
    _check_resource_uniqueness(manifest, config)
    _warn_for_unused_resource_config_paths(manifest, config)

DocsContextCallback = Callable[[ResultNode], Dict[str, Any]]

def _get_doc_blocks(description: str, manifest: Manifest, node_package: str) -> List[str]:
    ast = parse(description)
    doc_blocks: List[str] = []
    if not hasattr(ast, 'body'):
        return doc_blocks
    for statement in ast.body:
        for node in statement.nodes:
            if isinstance(node, Call) and hasattr(node, 'node') and hasattr(node, 'args') and (node.node.name == 'doc'):
                doc_args = [arg.value for arg in node.args]
                if len(doc_args) == 1:
                    package, name = (None, doc_args[0])
                elif len(doc_args) == 2:
                    package, name = doc_args
                else:
                    continue
                if not manifest.metadata.project_name:
                    continue
                resolved_doc = manifest.resolve_doc(name, package, manifest.metadata.project_name, node_package)
                if resolved_doc:
                    doc_blocks.append(resolved_doc.unique_id)
    return doc_blocks

def _process_docs_for_node(context: Dict[str, Any], node: ManifestNode, manifest: Manifest) -> None:
    node.doc_blocks = _get_doc_blocks(node.description, manifest, node.package_name)
    node.description = get_rendered(node.description, context)
    for column_name, column in node.columns.items():
        column.doc_blocks = _get_doc_blocks(column.description, manifest, node.package_name)
        column.description = get_rendered(column.description, context)

def _process_docs_for_source(context: Dict[str, Any], source: Any, manifest: Manifest) -> None:
    source.doc_blocks = _get_doc_blocks(source.description, manifest, source.package_name)
    source.description = get_rendered(source.description, context)
    source.source_description = get_rendered(source.source_description, context)
    for column in source.columns.values():
        column.doc_blocks = _get_doc_blocks(column.description, manifest, source.package_name)
        column.description = get_rendered(column.description, context)

def _process_docs_for_macro(context: Dict[str, Any], macro: Macro) -> None:
    macro.description = get_rendered(macro.description, context)
    for arg in macro.arguments:
        arg.description = get_rendered(arg.description, context)

def _process_docs_for_exposure(context: Dict[str, Any], exposure: Exposure) -> None:
    exposure.description = get_rendered(exposure.description, context)

def _process_docs_for_metrics(context: Dict[str, Any], metric: Metric) -> None:
    metric.description = get_rendered(metric.description, context)

def _process_docs_for_semantic_model(context: Dict[str, Any], semantic_model: SemanticModel) -> None:
    if semantic_model.description:
        semantic_model.description = get_rendered(semantic_model.description, context)
    for dimension in semantic_model.dimensions:
        if dimension.description:
            dimension.description = get_rendered(dimension.description, context)
    for measure in semantic_model.measures:
        if measure.description:
            measure.description = get_rendered(measure.description, context)
    for entity in semantic_model.entities:
        if entity.description:
            entity.description = get_rendered(entity.description, context)

def _process_docs_for_saved_query(context: Dict[str, Any], saved_query: SavedQuery) -> None:
    if saved_query.description:
        saved_query.description = get_rendered(saved_query.description, context)

def _process_refs(manifest: Manifest, current_project: str, node: ManifestNode, dependencies: Any) -> None:
    dependencies = dependencies or {}
    if isinstance(node, SeedNode):
        return
    for ref in node.refs:
        target_model = None
        target_model_name: str = ref.name
        target_model_package: Optional[str] = ref.package
        target_model_version: Optional[str] = ref.version
        if len(ref.positional_args) < 1 or len(ref.positional_args) > 2:
            raise dbt.exceptions.DbtInternalError(f'Refs should always be 1 or 2 arguments - got {len(ref.positional_args)}')
        target_model = manifest.resolve_ref(node, target_model_name, target_model_package, target_model_version, current_project, node.package_name)
        if target_model is None or isinstance(target_model, Disabled):
            node.config.enabled = False
            invalid_target_fail_unless_test(
                node=node,
                target_name=target_model_name,
                target_kind='node',
                target_package=target_model_package,
                target_version=target_model_version,
                disabled=isinstance(target_model, Disabled),
                should_warn_if_disabled=False,
            )
            continue
        elif manifest.is_invalid_private_ref(node, target_model, dependencies):
            raise dbt.exceptions.DbtReferenceError(unique_id=node.unique_id, ref_unique_id=target_model.unique_id, access=AccessType.Private, scope=dbt_common.utils.cast_to_str(target_model.group))
        elif manifest.is_invalid_protected_ref(node, target_model, dependencies):
            raise dbt.exceptions.DbtReferenceError(unique_id=node.unique_id, ref_unique_id=target_model.unique_id, access=AccessType.Protected, scope=target_model.package_name)
        target_model_id: str = target_model.unique_id
        node.depends_on.add_node(target_model_id)

def _process_metric_depends_on(manifest: Manifest, current_project: str, metric: Metric) -> None:
    assert len(metric.type_params.input_measures) > 0
    for input_measure in metric.type_params.input_measures:
        target_semantic_model = manifest.resolve_semantic_model_for_measure(target_measure_name=input_measure.name, current_project=current_project, node_package=metric.package_name)
        if target_semantic_model is None:
            raise dbt.exceptions.ParsingError(f'A semantic model having a measure `{input_measure.name}` does not exist but was referenced.', node=metric)
        if target_semantic_model.config.enabled is False:
            raise dbt.exceptions.ParsingError(f'The measure `{input_measure.name}` is referenced on disabled semantic model `{target_semantic_model.name}`.', node=metric)
        metric.depends_on.add_node(target_semantic_model.unique_id)

def _process_metric_node(manifest: Manifest, current_project: str, metric: Metric) -> None:
    if len(metric.type_params.input_measures) > 0:
        return
    if metric.type is MetricType.SIMPLE or metric.type is MetricType.CUMULATIVE:
        assert metric.type_params.measure is not None, f'{metric} should have a measure defined, but it does not.'
        metric.add_input_measure(metric.type_params.measure)
        _process_metric_depends_on(manifest=manifest, current_project=current_project, metric=metric)
    elif metric.type is MetricType.CONVERSION:
        conversion_type_params = metric.type_params.conversion_type_params
        assert conversion_type_params, f'{metric.name} is a conversion metric and must have conversion_type_params defined.'
        metric.add_input_measure(conversion_type_params.base_measure)
        metric.add_input_measure(conversion_type_params.conversion_measure)
        _process_metric_depends_on(manifest=manifest, current_project=current_project, metric=metric)
    elif metric.type is MetricType.DERIVED or metric.type is MetricType.RATIO:
        input_metrics = metric.input_metrics
        if metric.type is MetricType.RATIO:
            if metric.type_params.numerator is None or metric.type_params.denominator is None:
                raise dbt.exceptions.ParsingError('Invalid ratio metric. Both a numerator and denominator must be specified', node=metric)
            input_metrics = [metric.type_params.numerator, metric.type_params.denominator]
        for input_metric in input_metrics:
            target_metric = manifest.resolve_metric(target_metric_name=input_metric.name, target_metric_package=None, current_project=current_project, node_package=metric.package_name)
            if target_metric is None:
                raise dbt.exceptions.ParsingError(f'The metric `{input_metric.name}` does not exist but was referenced.', node=metric)
            elif isinstance(target_metric, Disabled):
                raise dbt.exceptions.ParsingError(f'The metric `{input_metric.name}` is disabled and thus cannot be referenced.', node=metric)
            _process_metric_node(manifest=manifest, current_project=current_project, metric=target_metric)
            for input_measure in target_metric.type_params.input_measures:
                metric.add_input_measure(input_measure)
            metric.depends_on.add_node(target_metric.unique_id)
    else:
        assert_values_exhausted(metric.type)

def _process_metrics_for_node(manifest: Manifest, current_project: str, node: ManifestNode) -> None:
    if isinstance(node, SeedNode):
        return
    elif isinstance(node, SavedQuery):
        metrics = [[metric] for metric in node.metrics]
    else:
        metrics = node.metrics
    for metric in metrics:
        target_metric = None
        target_metric_package = None
        if len(metric) == 1:
            target_metric_name = metric[0]
        elif len(metric) == 2:
            target_metric_package, target_metric_name = metric
        else:
            raise dbt.exceptions.DbtInternalError(f'Metric references should always be 1 or 2 arguments - got {len(metric)}')
        target_metric = manifest.resolve_metric(target_metric_name, target_metric_package, current_project, node.package_name)
        if target_metric is None or isinstance(target_metric, Disabled):
            node.config.enabled = False
            invalid_target_fail_unless_test(
                node=node,
                target_name=target_metric_name,
                target_kind='metric',
                target_package=target_metric_package,
                disabled=isinstance(target_metric, Disabled)
            )
            continue
        target_metric_id: str = target_metric.unique_id
        node.depends_on.add_node(target_metric_id)

def remove_dependent_project_references(manifest: Manifest, external_node_unique_id: str) -> None:
    for child_id in manifest.child_map[external_node_unique_id]:
        node = manifest.expect(child_id)
        if external_node_unique_id in node.depends_on_nodes:
            node.depends_on_nodes.remove(external_node_unique_id)
        node.created_at = time.time()

def _process_sources_for_exposure(manifest: Manifest, current_project: str, exposure: Exposure) -> None:
    target_source = None
    for source_name, table_name in exposure.sources:
        target_source = manifest.resolve_source(source_name, table_name, current_project, exposure.package_name)
        if target_source is None or isinstance(target_source, Disabled):
            exposure.config.enabled = False
            invalid_target_fail_unless_test(
                node=exposure,
                target_name=f'{source_name}.{table_name}',
                target_kind='source',
                disabled=isinstance(target_source, Disabled)
            )
            continue
        target_source_id: str = target_source.unique_id
        exposure.depends_on.add_node(target_source_id)

def _process_sources_for_metric(manifest: Manifest, current_project: str, metric: Metric) -> None:
    target_source = None
    for source_name, table_name in metric.sources:
        target_source = manifest.resolve_source(source_name, table_name, current_project, metric.package_name)
        if target_source is None or isinstance(target_source, Disabled):
            metric.config.enabled = False
            invalid_target_fail_unless_test(
                node=metric,
                target_name=f'{source_name}.{table_name}',
                target_kind='source',
                disabled=isinstance(target_source, Disabled)
            )
            continue
        target_source_id: str = target_source.unique_id
        metric.depends_on.add_node(target_source_id)

def _process_sources_for_node(manifest: Manifest, current_project: str, node: ManifestNode) -> None:
    if isinstance(node, SeedNode):
        return
    target_source = None
    for source_name, table_name in node.sources:
        target_source = manifest.resolve_source(source_name, table_name, current_project, node.package_name)
        if target_source is None or isinstance(target_source, Disabled):
            node.config.enabled = False
            invalid_target_fail_unless_test(
                node=node,
                target_name=f'{source_name}.{table_name}',
                target_kind='source',
                disabled=isinstance(target_source, Disabled)
            )
            continue
        target_source_id: str = target_source.unique_id
        node.depends_on.add_node(target_source_id)

def process_macro(config: Project, manifest: Manifest, macro: Macro) -> None:
    ctx: Dict[str, Any] = generate_runtime_docs_context(config, macro, manifest, config.project_name)
    _process_docs_for_macro(ctx, macro)

def process_node(config: Project, manifest: Manifest, node: ManifestNode) -> None:
    _process_sources_for_node(manifest, config.project_name, node)
    _process_refs(manifest, config.project_name, node, config.dependencies)
    ctx: Dict[str, Any] = generate_runtime_docs_context(config, node, manifest, config.project_name)
    _process_docs_for_node(ctx, node, manifest)

def write_semantic_manifest(manifest: Manifest, target_path: str) -> None:
    path: str = os.path.join(target_path, SEMANTIC_MANIFEST_FILE_NAME)
    semantic_manifest = SemanticManifest(manifest)
    semantic_manifest.write_json_to_file(path)

def write_manifest(manifest: Manifest, target_path: str, which: Optional[Any] = None) -> None:
    file_name: str = MANIFEST_FILE_NAME
    path: str = os.path.join(target_path, file_name)
    manifest.write(path)
    write_semantic_manifest(manifest=manifest, target_path=target_path)

def parse_manifest(runtime_config: RuntimeConfig, write_perf_info: bool, write: bool, write_json: bool) -> Manifest:
    register_adapter(runtime_config, get_mp_context())
    adapter = get_adapter(runtime_config)
    adapter.set_macro_context_generator(generate_runtime_macro_context)
    manifest = ManifestLoader.get_full_manifest(runtime_config, write_perf_info=write_perf_info)
    if write and write_json:
        write_manifest(manifest, runtime_config.project_target_path)
        pm = plugins.get_plugin_manager(runtime_config.project_name)
        plugin_artifacts = pm.get_manifest_artifacts(manifest)
        for path, plugin_artifact in plugin_artifacts.items():
            plugin_artifact.write(path)
            fire_event(ArtifactWritten(artifact_type=plugin_artifact.__class__.__name__, artifact_path=path))
    return manifest