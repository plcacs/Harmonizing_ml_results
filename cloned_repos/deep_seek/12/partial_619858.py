import os
from copy import deepcopy
from typing import Callable, Dict, List, MutableMapping, Union, Set, Tuple, Optional, Any, cast
from dbt.constants import DEFAULT_ENV_PLACEHOLDER
from dbt.contracts.files import AnySourceFile, ParseFileType, SchemaSourceFile, parse_file_type_to_parser
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import AnalysisNode, ModelNode, SeedNode, SnapshotNode, Macro
from dbt.events.types import PartialParsingEnabled, PartialParsingFile
from dbt.node_types import NodeType
from dbt_common.context import get_invocation_context
from dbt_common.events.base_types import EventLevel
from dbt_common.events.functions import fire_event

mssat_files: Tuple[ParseFileType, ...] = (ParseFileType.Model, ParseFileType.Seed, ParseFileType.Snapshot, ParseFileType.Analysis, ParseFileType.SingularTest)
mg_files: Tuple[ParseFileType, ...] = (ParseFileType.Macro, ParseFileType.GenericTest)
key_to_prefix: Dict[str, str] = {'models': 'model', 'seeds': 'seed', 'snapshots': 'snapshot', 'analyses': 'analysis'}
parse_file_type_to_key: Dict[ParseFileType, str] = {ParseFileType.Model: 'models', ParseFileType.Seed: 'seeds', ParseFileType.Snapshot: 'snapshots', ParseFileType.Analysis: 'analyses'}
special_override_macros: List[str] = ['ref', 'source', 'config', 'generate_schema_name', 'generate_database_name', 'generate_alias_name']

class PartialParsing:
    def __init__(self, saved_manifest: Manifest, new_files: Dict[str, AnySourceFile]) -> None:
        self.saved_manifest: Manifest = saved_manifest
        self.new_files: Dict[str, AnySourceFile] = new_files
        self.project_parser_files: Dict[str, Dict[str, List[str]]] = {}
        self.saved_files: Dict[str, AnySourceFile] = self.saved_manifest.files
        self.macro_child_map: Dict[str, List[str]] = {}
        self.env_vars_changed_source_files: List[str]
        self.env_vars_changed_schema_files: Dict[str, Dict[str, List[str]]]
        self.env_vars_changed_source_files, self.env_vars_changed_schema_files = self.build_env_vars_to_files()
        self.file_diff: Dict[str, List[str]] = {}
        self.build_file_diff()
        self.processing_file: Optional[str] = None
        self.deleted_special_override_macro: bool = False
        self.disabled_by_file_id: Dict[str, List[str]] = self.saved_manifest.build_disabled_by_file_id()

    def skip_parsing(self) -> bool:
        return (not self.file_diff['deleted'] and 
                not self.file_diff['added'] and 
                not self.file_diff['changed'] and 
                not self.file_diff['changed_schema_files'] and 
                not self.file_diff['deleted_schema_files'])

    def build_file_diff(self) -> None:
        saved_file_ids: Set[str] = set(self.saved_files.keys())
        new_file_ids: Set[str] = set(self.new_files.keys())
        deleted_all_files: Set[str] = saved_file_ids.difference(new_file_ids)
        added: Set[str] = new_file_ids.difference(saved_file_ids)
        common: Set[str] = saved_file_ids.intersection(new_file_ids)
        changed_or_deleted_macro_file: bool = False
        deleted_schema_files: List[str] = []
        deleted: List[str] = []
        
        for file_id in deleted_all_files:
            if self.saved_files[file_id].parse_file_type == ParseFileType.Schema:
                deleted_schema_files.append(file_id)
            else:
                if self.saved_files[file_id].parse_file_type in mg_files:
                    changed_or_deleted_macro_file = True
                deleted.append(file_id)
        
        changed: List[str] = []
        changed_schema_files: List[str] = []
        unchanged: List[str] = []
        
        for file_id in common:
            if self.saved_files[file_id].checksum == self.new_files[file_id].checksum:
                unchanged.append(file_id)
            elif self.saved_files[file_id].parse_file_type == ParseFileType.Schema:
                sf = self.saved_files[file_id]
                if type(sf).__name__ != 'SchemaSourceFile':
                    raise Exception(f'Serialization failure for {file_id}')
                changed_schema_files.append(file_id)
            else:
                if self.saved_files[file_id].parse_file_type in mg_files:
                    changed_or_deleted_macro_file = True
                changed.append(file_id)
        
        for file_id in self.env_vars_changed_source_files:
            if file_id in deleted or file_id in changed:
                continue
            changed.append(file_id)
        
        for file_id in self.env_vars_changed_schema_files.keys():
            if file_id in deleted_schema_files or file_id in changed_schema_files:
                continue
            changed_schema_files.append(file_id)
        
        file_diff: Dict[str, List[str]] = {
            'deleted': deleted,
            'deleted_schema_files': deleted_schema_files,
            'added': added,
            'changed': changed,
            'changed_schema_files': changed_schema_files,
            'unchanged': unchanged
        }
        
        if changed_or_deleted_macro_file:
            self.macro_child_map = self.saved_manifest.build_macro_child_map()
        
        deleted_count: int = len(deleted) + len(deleted_schema_files)
        changed_count: int = len(changed) + len(changed_schema_files)
        event = PartialParsingEnabled(deleted=deleted_count, added=len(added), changed=changed_count)
        
        if get_invocation_context().env.get('DBT_PP_TEST'):
            fire_event(event, level=EventLevel.INFO)
        else:
            fire_event(event)
        
        self.file_diff = file_diff

    def get_parsing_files(self) -> Dict[str, Dict[str, List[str]]]:
        if self.skip_parsing():
            return {}
        
        for file_id in self.file_diff['added']:
            self.processing_file = file_id
            self.add_to_saved(file_id)
        
        for file_id in self.file_diff['changed_schema_files']:
            self.processing_file = file_id
            self.change_schema_file(file_id)
        
        for file_id in self.file_diff['deleted_schema_files']:
            self.processing_file = file_id
            self.delete_schema_file(file_id)
        
        for file_id in self.file_diff['deleted']:
            self.processing_file = file_id
            self.delete_from_saved(file_id)
        
        for file_id in self.file_diff['changed']:
            self.processing_file = file_id
            self.update_in_saved(file_id)
        
        return self.project_parser_files

    def add_to_pp_files(self, source_file: AnySourceFile) -> None:
        file_id = source_file.file_id
        parser_name = parse_file_type_to_parser[source_file.parse_file_type]
        project_name = source_file.project_name
        
        if not parser_name or not project_name:
            raise Exception(f'Did not find parse_file_type or project_name in SourceFile for {source_file.file_id}')
        
        if project_name not in self.project_parser_files:
            self.project_parser_files[project_name] = {}
        
        if parser_name not in self.project_parser_files[project_name]:
            self.project_parser_files[project_name][parser_name] = []
        
        if (file_id not in self.project_parser_files[project_name][parser_name] and 
            file_id not in self.file_diff['deleted'] and 
            file_id not in self.file_diff['deleted_schema_files']):
            self.project_parser_files[project_name][parser_name].append(file_id)

    def already_scheduled_for_parsing(self, source_file: AnySourceFile) -> bool:
        file_id = source_file.file_id
        project_name = source_file.project_name
        
        if project_name not in self.project_parser_files:
            return False
        
        parser_name = parse_file_type_to_parser[source_file.parse_file_type]
        if parser_name not in self.project_parser_files[project_name]:
            return False
        
        if file_id not in self.project_parser_files[project_name][parser_name]:
            return False
        
        return True

    def add_to_saved(self, file_id: str) -> None:
        source_file = deepcopy(self.new_files[file_id])
        if source_file.parse_file_type == ParseFileType.Schema:
            self.handle_added_schema_file(source_file)
        
        self.saved_files[file_id] = source_file
        self.add_to_pp_files(source_file)
        fire_event(PartialParsingFile(operation='added', file_id=file_id))

    def handle_added_schema_file(self, source_file: SchemaSourceFile) -> None:
        source_file.pp_dict = source_file.dict_from_yaml.copy()
        if 'sources' in source_file.pp_dict:
            for source in source_file.pp_dict['sources']:
                if 'overrides' in source:
                    self.remove_source_override_target(source)

    def delete_disabled(self, unique_id: str, file_id: str) -> Any:
        for dis_index, dis_node in enumerate(self.saved_manifest.disabled[unique_id]):
            if dis_node.file_id == file_id:
                node = dis_node
                index = dis_index
                break
        del self.saved_manifest.disabled[unique_id][index]
        if not self.saved_manifest.disabled[unique_id]:
            self.saved_manifest.disabled.pop(unique_id)
        return node

    def delete_from_saved(self, file_id: str) -> None:
        saved_source_file = self.saved_files[file_id]
        if saved_source_file.parse_file_type in mssat_files:
            self.remove_mssat_file(saved_source_file)
            self.saved_manifest.files.pop(file_id)
        if saved_source_file.parse_file_type in mg_files:
            self.delete_macro_file(saved_source_file, follow_references=True)
        if saved_source_file.parse_file_type == ParseFileType.Documentation:
            self.delete_doc_node(saved_source_file)
        if saved_source_file.parse_file_type == ParseFileType.Fixture:
            self.delete_fixture_node(saved_source_file)
        fire_event(PartialParsingFile(operation='deleted', file_id=file_id))

    def update_in_saved(self, file_id: str) -> None:
        new_source_file = deepcopy(self.new_files[file_id])
        old_source_file = self.saved_files[file_id]
        
        if new_source_file.parse_file_type in mssat_files:
            self.update_mssat_in_saved(new_source_file, old_source_file)
        elif new_source_file.parse_file_type in mg_files:
            self.update_macro_in_saved(new_source_file, old_source_file)
        elif new_source_file.parse_file_type == ParseFileType.Documentation:
            self.update_doc_in_saved(new_source_file, old_source_file)
        elif new_source_file.parse_file_type == ParseFileType.Fixture:
            self.update_fixture_in_saved(new_source_file, old_source_file)
        else:
            raise Exception(f'Invalid parse_file_type in source_file {file_id}')
        
        fire_event(PartialParsingFile(operation='updated', file_id=file_id))

    def update_mssat_in_saved(self, new_source_file: AnySourceFile, old_source_file: AnySourceFile) -> None:
        if self.already_scheduled_for_parsing(old_source_file):
            return
        
        unique_ids: List[str] = []
        if old_source_file.nodes:
            unique_ids = old_source_file.nodes
        
        file_id = new_source_file.file_id
        self.saved_files[file_id] = deepcopy(new_source_file)
        self.add_to_pp_files(new_source_file)
        
        for unique_id in unique_ids:
            self.remove_node_in_saved(new_source_file, unique_id)

    def remove_node_in_saved(self, source_file: AnySourceFile, unique_id: str) -> None:
        if unique_id in self.saved_manifest.nodes:
            node = self.saved_manifest.nodes.pop(unique_id)
        elif source_file.file_id in self.disabled_by_file_id and unique_id in self.saved_manifest.disabled:
            node = self.delete_disabled(unique_id, source_file.file_id)
        else:
            return
        
        if node.patch_path:
            file_id = node.patch_path
            if file_id not in self.file_diff['deleted'] and file_id in self.saved_files:
                schema_file = self.saved_files[file_id]
                dict_key = parse_file_type_to_key[source_file.parse_file_type]
                elem_patch = None
                
                if dict_key in schema_file.dict_from_yaml:
                    for elem in schema_file.dict_from_yaml[dict_key]:
                        if elem['name'] == node.name:
                            elem_patch = elem
                            break
                
                if elem_patch:
                    self.delete_schema_mssa_links(schema_file, dict_key, elem_patch)
                    self.merge_patch(schema_file, dict_key, elem_patch)
                    if unique_id in schema_file.node_patches:
                        schema_file.node_patches.remove(unique_id)
            
            if unique_id in self.saved_manifest.disabled:
                for node in self.saved_manifest.disabled[unique_id]:
                    node.patch_path = None

    def update_macro_in_saved(self, new_source_file: AnySourceFile, old_source_file: AnySourceFile) -> None:
        if self.already_scheduled_for_parsing(old_source_file):
            return
        
        self.handle_macro_file_links(old_source_file, follow_references=True)
        file_id = new_source_file.file_id
        self.saved_files[file_id] = deepcopy(new_source_file)
        self.add_to_pp_files(new_source_file)

    def update_doc_in_saved(self, new_source_file: AnySourceFile, old_source_file: AnySourceFile) -> None:
        if self.already_scheduled_for_parsing(old_source_file):
            return
        
        self.delete_doc_node(old_source_file)
        self.saved_files[new_source_file.file_id] = deepcopy(new_source_file)
        self.add_to_pp_files(new_source_file)

    def update_fixture_in_saved(self, new_source_file: AnySourceFile, old_source_file: AnySourceFile) -> None:
        if self.already_scheduled_for_parsing(old_source_file):
            return
        
        self.delete_fixture_node(old_source_file)
        self.saved_files[new_source_file.file_id] = deepcopy(new_source_file)
        self.add_to_pp_files(new_source_file)

    def remove_mssat_file(self, source_file: AnySourceFile) -> None:
        if not source_file.nodes:
            return
        
        for unique_id in source_file.nodes:
            self.remove_node_in_saved(source_file, unique_id)
            self.schedule_referencing_nodes_for_parsing(unique_id)

    def schedule_referencing_nodes_for_parsing(self, unique_id: str) -> None:
        if unique_id in self.saved_manifest.child_map:
            self.schedule_nodes_for_parsing(self.saved_manifest.child_map[unique_id])

    def schedule_nodes_for_parsing(self, unique_ids: List[str]) -> None:
        for unique_id in unique_ids:
            if unique_id in self.saved_manifest.nodes:
                node = self.saved_manifest.nodes[unique_id]
                if node.resource_type == NodeType.Test and node.test_node_type == 'generic':
                    continue
                
                file_id = node.file_id
                if file_id in self.saved_files and file_id not in self.file_diff['deleted']:
                    source_file = self.saved_files[file_id]
                    self.remove_mssat_file(source_file)
                    self.saved_files[file_id] = deepcopy(self.new_files[file_id])
                    self.add_to_pp_files(self.saved_files[file_id])
            elif unique_id in self.saved_manifest.sources:
                source = self.saved_manifest.sources[unique_id]
                self._schedule_for_parsing('sources', source, source.source_name, self.delete_schema_source)
            elif unique_id in self.saved_manifest.exposures:
                exposure = self.saved_manifest.exposures[unique_id]
                self._schedule_for_parsing('exposures', exposure, exposure.name, self.delete_schema_exposure)
            elif unique_id in self.saved_manifest.metrics:
                metric = self.saved_manifest.metrics[unique_id]
                self._schedule_for_parsing('metrics', metric, metric.name, self.delete_schema_metric)
            elif unique_id in self.saved_manifest.semantic_models:
                semantic_model = self.saved_manifest.semantic_models[unique_id]
                self._schedule_for_parsing('semantic_models', semantic_model, semantic_model.name, self.delete_schema_semantic_model)
            elif unique_id in self.saved_manifest.saved_queries:
                saved_query = self.saved_manifest.saved_queries[unique_id]
                self._schedule_for_parsing('saved_queries', saved_query, saved_query.name, self.delete_schema_saved_query)
            elif unique_id in self.saved_manifest.macros:
                macro = self.saved_manifest.macros[unique_id]
                file_id = macro.file_id
                if file_id in self.saved_files and file_id not in self.file_diff['deleted']:
                    source_file = self.saved_files[file_id]
                    self.delete_macro_file(source_file)
                    self.saved_files[file_id] = deepcopy(self.new_files[file_id])
                    self.add_to_pp_files(self.saved_files[file_id])
            elif unique_id in self.saved_manifest.unit_tests:
                unit_test = self.saved_manifest.unit_tests[unique_id]
                self._schedule_for_parsing('unit_tests', unit_test, unit_test.name, self.delete_schema_unit_test)

    def _schedule_for_parsing(self, dict_key: str, element: Any, name: str, delete: Callable) -> None:
        file_id = element.file_id
        if (file_id in self.saved_files