import os
from copy import deepcopy
from typing import Callable, Dict, List, MutableMapping, Union
from dbt.constants import DEFAULT_ENV_PLACEHOLDER
from dbt.contracts.files import AnySourceFile, ParseFileType, SchemaSourceFile, parse_file_type_to_parser
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import AnalysisNode, ModelNode, SeedNode, SnapshotNode
from dbt.events.types import PartialParsingEnabled, PartialParsingFile
from dbt.node_types import NodeType
from dbt_common.context import get_invocation_context
from dbt_common.events.base_types import EventLevel
from dbt_common.events.functions import fire_event
mssat_files = (ParseFileType.Model, ParseFileType.Seed, ParseFileType.
    Snapshot, ParseFileType.Analysis, ParseFileType.SingularTest)
mg_files = ParseFileType.Macro, ParseFileType.GenericTest
key_to_prefix = {'models': 'model', 'seeds': 'seed', 'snapshots':
    'snapshot', 'analyses': 'analysis'}
parse_file_type_to_key = {ParseFileType.Model: 'models', ParseFileType.Seed:
    'seeds', ParseFileType.Snapshot: 'snapshots', ParseFileType.Analysis:
    'analyses'}
special_override_macros = ['ref', 'source', 'config',
    'generate_schema_name', 'generate_database_name', 'generate_alias_name']


class PartialParsing:

    def __init__(self, saved_manifest, new_files):
        self.saved_manifest = saved_manifest
        self.new_files = new_files
        self.project_parser_files = {}
        self.saved_files = self.saved_manifest.files
        self.project_parser_files = {}
        self.macro_child_map = {}
        (self.env_vars_changed_source_files, self.env_vars_changed_schema_files
            ) = self.build_env_vars_to_files()
        self.build_file_diff()
        self.processing_file = None
        self.deleted_special_override_macro = False
        self.disabled_by_file_id = (self.saved_manifest.
            build_disabled_by_file_id())

    def func_5mnpu0xw(self):
        return not self.file_diff['deleted'] and not self.file_diff['added'
            ] and not self.file_diff['changed'] and not self.file_diff[
            'changed_schema_files'] and not self.file_diff[
            'deleted_schema_files']

    def func_9xaupb17(self):
        saved_file_ids = set(self.saved_files.keys())
        new_file_ids = set(self.new_files.keys())
        deleted_all_files = saved_file_ids.difference(new_file_ids)
        added = new_file_ids.difference(saved_file_ids)
        common = saved_file_ids.intersection(new_file_ids)
        changed_or_deleted_macro_file = False
        deleted_schema_files = []
        deleted = []
        for file_id in deleted_all_files:
            if self.saved_files[file_id
                ].parse_file_type == ParseFileType.Schema:
                deleted_schema_files.append(file_id)
            else:
                if self.saved_files[file_id].parse_file_type in mg_files:
                    changed_or_deleted_macro_file = True
                deleted.append(file_id)
        changed = []
        changed_schema_files = []
        unchanged = []
        for file_id in common:
            if self.saved_files[file_id].checksum == self.new_files[file_id
                ].checksum:
                unchanged.append(file_id)
            elif self.saved_files[file_id
                ].parse_file_type == ParseFileType.Schema:
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
            if (file_id in deleted_schema_files or file_id in
                changed_schema_files):
                continue
            changed_schema_files.append(file_id)
        file_diff = {'deleted': deleted, 'deleted_schema_files':
            deleted_schema_files, 'added': added, 'changed': changed,
            'changed_schema_files': changed_schema_files, 'unchanged':
            unchanged}
        if changed_or_deleted_macro_file:
            self.macro_child_map = self.saved_manifest.build_macro_child_map()
        deleted = len(deleted) + len(deleted_schema_files)
        changed = len(changed) + len(changed_schema_files)
        event = PartialParsingEnabled(deleted=deleted, added=len(added),
            changed=changed)
        if get_invocation_context().env.get('DBT_PP_TEST'):
            fire_event(event, level=EventLevel.INFO)
        else:
            fire_event(event)
        self.file_diff = file_diff

    def func_ppx7kalt(self):
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

    def func_78nm8q6j(self, source_file):
        file_id = source_file.file_id
        parser_name = parse_file_type_to_parser[source_file.parse_file_type]
        project_name = source_file.project_name
        if not parser_name or not project_name:
            raise Exception(
                f'Did not find parse_file_type or project_name in SourceFile for {source_file.file_id}'
                )
        if project_name not in self.project_parser_files:
            self.project_parser_files[project_name] = {}
        if parser_name not in self.project_parser_files[project_name]:
            self.project_parser_files[project_name][parser_name] = []
        if file_id not in self.project_parser_files[project_name][parser_name
            ] and file_id not in self.file_diff['deleted'
            ] and file_id not in self.file_diff['deleted_schema_files']:
            self.project_parser_files[project_name][parser_name].append(file_id
                )

    def func_mzee5xcv(self, source_file):
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

    def func_j06s3ynh(self, file_id):
        source_file = deepcopy(self.new_files[file_id])
        if source_file.parse_file_type == ParseFileType.Schema:
            self.handle_added_schema_file(source_file)
        self.saved_files[file_id] = source_file
        self.add_to_pp_files(source_file)
        fire_event(PartialParsingFile(operation='added', file_id=file_id))

    def func_bm7lezdl(self, source_file):
        source_file.pp_dict = source_file.dict_from_yaml.copy()
        if 'sources' in source_file.pp_dict:
            for source in source_file.pp_dict['sources']:
                if 'overrides' in source:
                    self.remove_source_override_target(source)

    def func_hz318dg7(self, unique_id, file_id):
        for dis_index, dis_node in enumerate(self.saved_manifest.disabled[
            unique_id]):
            if dis_node.file_id == file_id:
                node = dis_node
                index = dis_index
                break
        del self.saved_manifest.disabled[unique_id][index]
        if not self.saved_manifest.disabled[unique_id]:
            self.saved_manifest.disabled.pop(unique_id)
        return node

    def func_5cagafqf(self, file_id):
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

    def func_mdhy29hf(self, file_id):
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
            raise Exception(f'Invalid parse_file_type in source_file {file_id}'
                )
        fire_event(PartialParsingFile(operation='updated', file_id=file_id))

    def func_3ph9q2hi(self, new_source_file, old_source_file):
        if self.already_scheduled_for_parsing(old_source_file):
            return
        unique_ids = []
        if old_source_file.nodes:
            unique_ids = old_source_file.nodes
        file_id = new_source_file.file_id
        self.saved_files[file_id] = deepcopy(new_source_file)
        self.add_to_pp_files(new_source_file)
        for unique_id in unique_ids:
            self.remove_node_in_saved(new_source_file, unique_id)

    def func_cf6lg0ht(self, source_file, unique_id):
        if unique_id in self.saved_manifest.nodes:
            node = self.saved_manifest.nodes.pop(unique_id)
        elif source_file.file_id in self.disabled_by_file_id and unique_id in self.saved_manifest.disabled:
            node = self.delete_disabled(unique_id, source_file.file_id)
        else:
            return
        if node.patch_path:
            file_id = node.patch_path
            if file_id not in self.file_diff['deleted'
                ] and file_id in self.saved_files:
                schema_file = self.saved_files[file_id]
                dict_key = parse_file_type_to_key[source_file.parse_file_type]
                elem_patch = None
                if dict_key in schema_file.dict_from_yaml:
                    for elem in schema_file.dict_from_yaml[dict_key]:
                        if elem['name'] == node.name:
                            elem_patch = elem
                            break
                if elem_patch:
                    self.delete_schema_mssa_links(schema_file, dict_key,
                        elem_patch)
                    self.merge_patch(schema_file, dict_key, elem_patch)
                    if unique_id in schema_file.node_patches:
                        schema_file.node_patches.remove(unique_id)
            if unique_id in self.saved_manifest.disabled:
                for node in self.saved_manifest.disabled[unique_id]:
                    node.patch_path = None

    def func_g53x42iu(self, new_source_file, old_source_file):
        if self.already_scheduled_for_parsing(old_source_file):
            return
        self.handle_macro_file_links(old_source_file, follow_references=True)
        file_id = new_source_file.file_id
        self.saved_files[file_id] = deepcopy(new_source_file)
        self.add_to_pp_files(new_source_file)

    def func_2a0t8w8s(self, new_source_file, old_source_file):
        if self.already_scheduled_for_parsing(old_source_file):
            return
        self.delete_doc_node(old_source_file)
        self.saved_files[new_source_file.file_id] = deepcopy(new_source_file)
        self.add_to_pp_files(new_source_file)

    def func_frfihvxr(self, new_source_file, old_source_file):
        if self.already_scheduled_for_parsing(old_source_file):
            return
        self.delete_fixture_node(old_source_file)
        self.saved_files[new_source_file.file_id] = deepcopy(new_source_file)
        self.add_to_pp_files(new_source_file)

    def func_lvcsadq1(self, source_file):
        if not source_file.nodes:
            return
        for unique_id in source_file.nodes:
            self.remove_node_in_saved(source_file, unique_id)
            self.schedule_referencing_nodes_for_parsing(unique_id)

    def func_bj33qgby(self, unique_id):
        if unique_id in self.saved_manifest.child_map:
            self.schedule_nodes_for_parsing(self.saved_manifest.child_map[
                unique_id])

    def func_hoc5wg99(self, unique_ids):
        for unique_id in unique_ids:
            if unique_id in self.saved_manifest.nodes:
                node = self.saved_manifest.nodes[unique_id]
                if (node.resource_type == NodeType.Test and node.
                    test_node_type == 'generic'):
                    continue
                file_id = node.file_id
                if (file_id in self.saved_files and file_id not in self.
                    file_diff['deleted']):
                    source_file = self.saved_files[file_id]
                    self.remove_mssat_file(source_file)
                    self.saved_files[file_id] = deepcopy(self.new_files[
                        file_id])
                    self.add_to_pp_files(self.saved_files[file_id])
            elif unique_id in self.saved_manifest.sources:
                source = self.saved_manifest.sources[unique_id]
                self._schedule_for_parsing('sources', source, source.
                    source_name, self.delete_schema_source)
            elif unique_id in self.saved_manifest.exposures:
                exposure = self.saved_manifest.exposures[unique_id]
                self._schedule_for_parsing('exposures', exposure, exposure.
                    name, self.delete_schema_exposure)
            elif unique_id in self.saved_manifest.metrics:
                metric = self.saved_manifest.metrics[unique_id]
                self._schedule_for_parsing('metrics', metric, metric.name,
                    self.delete_schema_metric)
            elif unique_id in self.saved_manifest.semantic_models:
                semantic_model = self.saved_manifest.semantic_models[unique_id]
                self._schedule_for_parsing('semantic_models',
                    semantic_model, semantic_model.name, self.
                    delete_schema_semantic_model)
            elif unique_id in self.saved_manifest.saved_queries:
                saved_query = self.saved_manifest.saved_queries[unique_id]
                self._schedule_for_parsing('saved_queries', saved_query,
                    saved_query.name, self.delete_schema_saved_query)
            elif unique_id in self.saved_manifest.macros:
                macro = self.saved_manifest.macros[unique_id]
                file_id = macro.file_id
                if (file_id in self.saved_files and file_id not in self.
                    file_diff['deleted']):
                    source_file = self.saved_files[file_id]
                    self.delete_macro_file(source_file)
                    self.saved_files[file_id] = deepcopy(self.new_files[
                        file_id])
                    self.add_to_pp_files(self.saved_files[file_id])
            elif unique_id in self.saved_manifest.unit_tests:
                unit_test = self.saved_manifest.unit_tests[unique_id]
                self._schedule_for_parsing('unit_tests', unit_test,
                    unit_test.name, self.delete_schema_unit_test)

    def func_n6f7rthn(self, dict_key, element, name, delete):
        file_id = element.file_id
        if file_id in self.saved_files and file_id not in self.file_diff[
            'deleted'] and file_id not in self.file_diff['deleted_schema_files'
            ]:
            schema_file = self.saved_files[file_id]
            elements = []
            assert isinstance(schema_file, SchemaSourceFile)
            if dict_key in schema_file.dict_from_yaml:
                elements = schema_file.dict_from_yaml[dict_key]
            schema_element = self.get_schema_element(elements, name)
            if schema_element:
                delete(schema_file, schema_element)
                self.merge_patch(schema_file, dict_key, schema_element)

    def func_5cnurock(self, source_file, follow_references=False):
        self.check_for_special_deleted_macros(source_file)
        self.handle_macro_file_links(source_file, follow_references)
        file_id = source_file.file_id
        if file_id in self.saved_files:
            self.saved_files.pop(file_id)

    def func_eqpyh0il(self, source_file):
        for unique_id in source_file.macros:
            if unique_id in self.saved_manifest.macros:
                package_name = unique_id.split('.')[1]
                if package_name == 'dbt':
                    continue
                macro = self.saved_manifest.macros[unique_id]
                if macro.name in special_override_macros:
                    self.deleted_special_override_macro = True

    def func_nloqzkfv(self, macro_unique_id, referencing_nodes):
        for unique_id in self.macro_child_map[macro_unique_id]:
            if unique_id in referencing_nodes:
                continue
            referencing_nodes.append(unique_id)
            if unique_id.startswith('macro.'):
                self.recursively_gather_macro_references(unique_id,
                    referencing_nodes)

    def func_lohyapet(self, source_file, follow_references=False):
        macros = source_file.macros.copy()
        for unique_id in macros:
            if unique_id not in self.saved_manifest.macros:
                if unique_id in source_file.macros:
                    source_file.macros.remove(unique_id)
                continue
            base_macro = self.saved_manifest.macros.pop(unique_id)
            if self.macro_child_map and follow_references:
                referencing_nodes = []
                self.recursively_gather_macro_references(unique_id,
                    referencing_nodes)
                self.schedule_macro_nodes_for_parsing(referencing_nodes)
            if base_macro.patch_path:
                file_id = base_macro.patch_path
                if file_id in self.saved_files:
                    schema_file = self.saved_files[file_id]
                    macro_patches = []
                    if 'macros' in schema_file.dict_from_yaml:
                        macro_patches = schema_file.dict_from_yaml['macros']
                    macro_patch = self.get_schema_element(macro_patches,
                        base_macro.name)
                    self.delete_schema_macro_patch(schema_file, macro_patch)
                    self.merge_patch(schema_file, 'macros', macro_patch)
            if unique_id in source_file.macros:
                source_file.macros.remove(unique_id)

    def func_vi1c8wxl(self, unique_ids):
        for unique_id in unique_ids:
            if unique_id in self.saved_manifest.nodes:
                node = self.saved_manifest.nodes[unique_id]
                if (node.resource_type == NodeType.Test and node.
                    test_node_type == 'generic'):
                    schema_file_id = node.file_id
                    schema_file = self.saved_manifest.files[schema_file_id]
                    key, name = schema_file.get_key_and_name_for_test(node.
                        unique_id)
                    if key and name:
                        patch_list = []
                        if key in schema_file.dict_from_yaml:
                            patch_list = schema_file.dict_from_yaml[key]
                        patch = self.get_schema_element(patch_list, name)
                        if patch:
                            if key in ['models', 'seeds', 'snapshots']:
                                self.delete_schema_mssa_links(schema_file,
                                    key, patch)
                                self.merge_patch(schema_file, key, patch)
                                if unique_id in schema_file.node_patches:
                                    schema_file.node_patches.remove(unique_id)
                            elif key == 'sources':
                                if 'overrides' in patch:
                                    self.remove_source_override_target(patch)
                                self.delete_schema_source(schema_file, patch)
                                self.merge_patch(schema_file, 'sources', patch)
                else:
                    file_id = node.file_id
                    if (file_id in self.saved_files and file_id not in self
                        .file_diff['deleted']):
                        source_file = self.saved_files[file_id]
                        self.remove_mssat_file(source_file)
                        self.saved_files[file_id] = deepcopy(self.new_files
                            [file_id])
                        self.add_to_pp_files(self.saved_files[file_id])
            elif unique_id in self.saved_manifest.macros:
                macro = self.saved_manifest.macros[unique_id]
                file_id = macro.file_id
                if (file_id in self.saved_files and file_id not in self.
                    file_diff['deleted']):
                    source_file = self.saved_files[file_id]
                    self.delete_macro_file(source_file)
                    self.saved_files[file_id] = deepcopy(self.new_files[
                        file_id])
                    self.add_to_pp_files(self.saved_files[file_id])

    def func_g52cdbjd(self, source_file):
        docs = source_file.docs.copy()
        for unique_id in docs:
            self.saved_manifest.docs.pop(unique_id)
            source_file.docs.remove(unique_id)
        self.schedule_nodes_for_parsing(source_file.nodes)
        source_file.nodes = []
        self.saved_manifest.files.pop(source_file.file_id)

    def func_lav7tklo(self, source_file):
        fixture_unique_id = source_file.fixture
        self.saved_manifest.fixtures.pop(fixture_unique_id)
        unit_tests = source_file.unit_tests.copy()
        for unique_id in unit_tests:
            unit_test = self.saved_manifest.unit_tests.pop(unique_id)
            self._schedule_for_parsing('unit_tests', unit_test, unit_test.
                name, self.delete_schema_unit_test)
            source_file.unit_tests.remove(unique_id)
        self.saved_manifest.files.pop(source_file.file_id)

    def func_im9q2jgi(self, file_id):
        saved_schema_file = self.saved_files[file_id]
        new_schema_file = deepcopy(self.new_files[file_id])
        saved_yaml_dict = saved_schema_file.dict_from_yaml
        new_yaml_dict = new_schema_file.dict_from_yaml
        saved_schema_file.pp_dict = {}
        self.handle_schema_file_changes(saved_schema_file, saved_yaml_dict,
            new_yaml_dict)
        saved_schema_file.contents = new_schema_file.contents
        saved_schema_file.checksum = new_schema_file.checksum
        saved_schema_file.dfy = new_schema_file.dfy
        self.add_to_pp_files(saved_schema_file)
        fire_event(PartialParsingFile(operation='updated', file_id=file_id))

    def func_eucbxbb4(self, file_id):
        saved_schema_file = self.saved_files[file_id]
        saved_yaml_dict = saved_schema_file.dict_from_yaml
        new_yaml_dict = {}
        self.handle_schema_file_changes(saved_schema_file, saved_yaml_dict,
            new_yaml_dict)
        self.saved_manifest.files.pop(file_id)

    def func_m331gbk8(self, schema_file, saved_yaml_dict, new_yaml_dict):
        env_var_changes = {}
        if schema_file.file_id in self.env_vars_changed_schema_files:
            env_var_changes = self.env_vars_changed_schema_files[schema_file
                .file_id]
        for dict_key in ['models', 'seeds', 'snapshots', 'analyses']:
            key_diff = self.get_diff_for(dict_key, saved_yaml_dict,
                new_yaml_dict)
            if key_diff['changed']:
                for elem in key_diff['changed']:
                    if dict_key == 'snapshots' and 'relation' in elem:
                        self.delete_yaml_snapshot(schema_file, elem)
                    self.delete_schema_mssa_links(schema_file, dict_key, elem)
                    self.merge_patch(schema_file, dict_key, elem, True)
            if key_diff['deleted']:
                for elem in key_diff['deleted']:
                    if dict_key == 'snapshots' and 'relation' in elem:
                        self.delete_yaml_snapshot(schema_file, elem)
                    self.delete_schema_mssa_links(schema_file, dict_key, elem)
            if key_diff['added']:
                for elem in key_diff['added']:
                    self.merge_patch(schema_file, dict_key, elem, True)
            if dict_key in env_var_changes and dict_key in new_yaml_dict:
                for name in env_var_changes[dict_key]:
                    if name in key_diff['changed_or_deleted_names']:
                        continue
                    elem = self.get_schema_element(new_yaml_dict[dict_key],
                        name)
                    if elem:
                        if dict_key == 'snapshots' and 'relation' in elem:
                            self.delete_yaml_snapshot(schema_file, elem)
                        self.delete_schema_mssa_links(schema_file, dict_key,
                            elem)
                        self.merge_patch(schema_file, dict_key, elem, True)
        dict_key = 'sources'
        source_diff = self.get_diff_for(dict_key, saved_yaml_dict,
            new_yaml_dict)
        if source_diff['changed']:
            for source in source_diff['changed']:
                if 'overrides' in source:
                    self.remove_source_override_target(source)
                self.delete_schema_source(schema_file, source)
                self.merge_patch(schema_file, dict_key, source, True)
        if source_diff['deleted']:
            for source in source_diff['deleted']:
                if 'overrides' in source:
                    self.remove_source_override_target(source)
                self.delete_schema_source(schema_file, source)
        if source_diff['added']:
            for source in source_diff['added']:
                if 'overrides' in source:
                    self.remove_source_override_target(source)
                self.merge_patch(schema_file, dict_key, source, True)
        if dict_key in env_var_changes and dict_key in new_yaml_dict:
            for name in env_var_changes[dict_key]:
                if name in source_diff['changed_or_deleted_names']:
                    continue
                source = self.get_schema_element(new_yaml_dict[dict_key], name)
                if source:
                    if 'overrides' in source:
                        self.remove_source_override_target(source)
                    self.delete_schema_source(schema_file, source)
                    self.merge_patch(schema_file, dict_key, source, True)

        def func_b7oqloj3(key, delete):
            self._handle_element_change(schema_file, saved_yaml_dict,
                new_yaml_dict, env_var_changes, key, delete)
        func_b7oqloj3('macros', self.delete_schema_macro_patch)
        func_b7oqloj3('exposures', self.delete_schema_exposure)
        func_b7oqloj3('metrics', self.delete_schema_metric)
        func_b7oqloj3('groups', self.delete_schema_group)
        func_b7oqloj3('semantic_models', self.delete_schema_semantic_model)
        func_b7oqloj3('unit_tests', self.delete_schema_unit_test)
        func_b7oqloj3('saved_queries', self.delete_schema_saved_query)
        func_b7oqloj3('data_tests', self.delete_schema_data_test_patch)

    def func_r68479hd(self, schema_file, saved_yaml_dict, new_yaml_dict,
        env_var_changes, dict_key, delete):
        element_diff = self.get_diff_for(dict_key, saved_yaml_dict,
            new_yaml_dict)
        if element_diff['changed']:
            for element in element_diff['changed']:
                delete(schema_file, element)
                self.merge_patch(schema_file, dict_key, element, True)
        if element_diff['deleted']:
            for element in element_diff['deleted']:
                delete(schema_file, element)
        if element_diff['added']:
            for element in element_diff['added']:
                self.merge_patch(schema_file, dict_key, element, True)
        if dict_key in env_var_changes and dict_key in new_yaml_dict:
            for name in env_var_changes[dict_key]:
                if name in element_diff['changed_or_deleted_names']:
                    continue
                elem = self.get_schema_element(new_yaml_dict[dict_key], name)
                if elem:
                    delete(schema_file, elem)
                    self.merge_patch(schema_file, dict_key, elem, True)

    def func_d00q4feo(self, key, saved_yaml_dict, new_yaml_dict):
        if key in saved_yaml_dict or key in new_yaml_dict:
            saved_elements = saved_yaml_dict[key
                ] if key in saved_yaml_dict else []
            new_elements = new_yaml_dict[key] if key in new_yaml_dict else []
        else:
            return {'deleted': [], 'added': [], 'changed': []}
        saved_elements_by_name = {}
        new_elements_by_name = {}
        for element in saved_elements:
            saved_elements_by_name[element['name']] = element
        for element in new_elements:
            new_elements_by_name[element['name']] = element
        saved_element_names = set(saved_elements_by_name.keys())
        new_element_names = set(new_elements_by_name.keys())
        deleted = saved_element_names.difference(new_element_names)
        added = new_element_names.difference(saved_element_names)
        common = saved_element_names.intersection(new_element_names)
        changed = []
        for element_name in common:
            if saved_elements_by_name[element_name] != new_elements_by_name[
                element_name]:
                changed.append(element_name)
        deleted_elements = [saved_elements_by_name[name].copy() for name in
            deleted]
        added_elements = [new_elements_by_name[name].copy() for name in added]
        changed_elements = [new_elements_by_name[name].copy() for name in
            changed]
        diff = {'deleted': deleted_elements, 'added': added_elements,
            'changed': changed_elements, 'changed_or_deleted_names': list(
            changed) + list(deleted)}
        return diff

    def func_eepy6k5c(self, schema_file, key, patch, new_patch=False):
        if schema_file.pp_dict is None:
            schema_file.pp_dict = {}
        pp_dict = schema_file.pp_dict
        if key not in pp_dict:
            pp_dict[key] = [patch]
        else:
            found_elem = None
            for elem in pp_dict[key]:
                if elem['name'] == patch['name']:
                    found_elem = elem
            if not found_elem:
                pp_dict[key].append(patch)
            elif found_elem and new_patch:
                pp_dict[key].remove(found_elem)
                pp_dict[key].append(patch)
        schema_file.delete_from_env_vars(key, patch['name'])
        schema_file.delete_from_unrendered_configs(key, patch['name'])
        self.add_to_pp_files(schema_file)

    def func_61clvjl5(self, schema_file, dict_key, elem):
        prefix = key_to_prefix[dict_key]
        elem_unique_ids = []
        for unique_id in schema_file.node_patches:
            if not unique_id.startswith(prefix):
                continue
            parts = unique_id.split('.')
            elem_name = parts[2]
            if elem_name == elem['name']:
                elem_unique_ids.append(unique_id)
        for elem_unique_id in elem_unique_ids:
            if (elem_unique_id in self.saved_manifest.nodes or 
                elem_unique_id in self.saved_manifest.disabled):
                nodes = []
                if elem_unique_id in self.saved_manifest.nodes:
                    nodes = [self.saved_manifest.nodes.pop(elem_unique_id)]
                else:
                    nodes = self.saved_manifest.disabled.pop(elem_unique_id)
                for node in nodes:
                    file_id = node.file_id
                    if file_id in self.new_files:
                        self.saved_files[file_id] = deepcopy(self.new_files
                            [file_id])
                    if self.saved_files[file_id]:
                        source_file = self.saved_files[file_id]
                        self.add_to_pp_files(source_file)
                    if node.group != elem.get('group'):
                        self.schedule_referencing_nodes_for_parsing(node.
                            unique_id)
                    if node.is_versioned or elem.get('versions'):
                        self.schedule_referencing_nodes_for_parsing(node.
                            unique_id)
            schema_file.node_patches.remove(elem_unique_id)
        if dict_key in ['models', 'seeds', 'snapshots']:
            self.remove_tests(schema_file, dict_key, elem['name'])

    def func_g2fj5qxc(self, schema_file, dict_key, name):
        tests = schema_file.get_tests(dict_key, name)
        for test_unique_id in tests:
            if test_unique_id in self.saved_manifest.nodes:
                self.saved_manifest.nodes.pop(test_unique_id)
        schema_file.remove_tests(dict_key, name)

    def func_gaqln0am(self, schema_file, snapshot_dict):
        snapshot_name = snapshot_dict['name']
        snapshots = schema_file.snapshots.copy()
        for unique_id in snapshots:
            if unique_id in self.saved_manifest.nodes:
                snapshot = self.saved_manifest.nodes[unique_id]
                if snapshot.name == snapshot_name:
                    self.saved_manifest.nodes.pop(unique_id)
                    schema_file.snapshots.remove(unique_id)
            elif unique_id in self.saved_manifest.disabled:
                self.delete_disabled(unique_id, schema_file.file_id)
                schema_file.snapshots.remove(unique_id)

    def func_qmgnf9yw(self, schema_file, source_dict):
        source_name = source_dict['name']
        sources = schema_file.sources.copy()
        for unique_id in sources:
            if unique_id in self.saved_manifest.sources:
                source = self.saved_manifest.sources[unique_id]
                if source.source_name == source_name:
                    source = self.saved_manifest.sources.pop(unique_id)
                    schema_file.sources.remove(unique_id)
                    self.schedule_referencing_nodes_for_parsing(unique_id)
        self.remove_tests(schema_file, 'sources', source_name)

    def func_tc98n2yx(self, schema_file, macro):
        macro_unique_id = None
        if macro['name'] in schema_file.macro_patches:
            macro_unique_id = schema_file.macro_patches[macro['name']]
            del schema_file.macro_patches[macro['name']]
        if macro_unique_id and macro_unique_id in self.saved_manifest.macros:
            macro = self.saved_manifest.macros.pop(macro_unique_id)
            macro_file_id = macro.file_id
            if macro_file_id in self.new_files:
                self.saved_files[macro_file_id] = deepcopy(self.new_files[
                    macro_file_id])
                self.add_to_pp_files(self.saved_files[macro_file_id])

    def func_f6bj7nhq(self, schema_file, data_test):
        data_test_unique_id = None
        for unique_id in schema_file.node_patches:
            if not unique_id.startswith('test'):
                continue
            parts = unique_id.split('.')
            elem_name = parts[2]
            if elem_name == data_test['name']:
                data_test_unique_id = unique_id
                break
        if (data_test_unique_id and data_test_unique_id in self.
            saved_manifest.nodes):
            singular_data_test = self.saved_manifest.nodes.pop(
                data_test_unique_id)
            file_id = singular_data_test.file_id
            if file_id in self.new_files:
                self.saved_files[file_id] = deepcopy(self.new_files[file_id])
                self.add_to_pp_files(self.saved_files[file_id])

    def func_2743tw2d(self, schema_file, exposure_dict):
        exposure_name = exposure_dict['name']
        exposures = schema_file.exposures.copy()
        for unique_id in exposures:
            if unique_id in self.saved_manifest.exposures:
                exposure = self.saved_manifest.exposures[unique_id]
                if exposure.name == exposure_name:
                    self.saved_manifest.exposures.pop(unique_id)
                    schema_file.exposures.remove(unique_id)
            elif unique_id in self.saved_manifest.disabled:
                self.delete_disabled(unique_id, schema_file.file_id)

    def func_xrvgnoma(self, schema_file, group_dict):
        group_name = group_dict['name']
        groups = schema_file.groups.copy()
        for unique_id in groups:
            if unique_id in self.saved_manifest.groups:
                group = self.saved_manifest.groups[unique_id]
                if group.name == group_name:
                    self.schedule_nodes_for_parsing(self.saved_manifest.
                        group_map[group.name])
                    self.saved_manifest.groups.pop(unique_id)
                    schema_file.groups.remove(unique_id)

    def func_c0fdqg6v(self, schema_file, metric_dict):
        metric_name = metric_dict['name']
        metrics = schema_file.metrics.copy()
        for unique_id in metrics:
            if unique_id in self.saved_manifest.metrics:
                metric = self.saved_manifest.metrics[unique_id]
                if metric.name == metric_name:
                    if unique_id in self.saved_manifest.child_map:
                        self.schedule_nodes_for_parsing(self.saved_manifest
                            .child_map[unique_id])
                    self.saved_manifest.metrics.pop(unique_id)
                    schema_file.metrics.remove(unique_id)
            elif unique_id in self.saved_manifest.disabled:
                self.delete_disabled(unique_id, schema_file.file_id)

    def func_9aob2hlo(self, schema_file, saved_query_dict):
        saved_query_name = saved_query_dict['name']
        saved_queries = schema_file.saved_queries.copy()
        for unique_id in saved_queries:
            if unique_id in self.saved_manifest.saved_queries:
                saved_query = self.saved_manifest.saved_queries[unique_id]
                if saved_query.name == saved_query_name:
                    if unique_id in self.saved_manifest.child_map:
                        self.schedule_nodes_for_parsing(self.saved_manifest
                            .child_map[unique_id])
                    self.saved_manifest.saved_queries.pop(unique_id)
            elif unique_id in self.saved_manifest.disabled:
                self.delete_disabled(unique_id, schema_file.file_id)

    def func_aosge76z(self, schema_file, semantic_model_dict):
        semantic_model_name = semantic_model_dict['name']
        semantic_models = schema_file.semantic_models.copy()
        for unique_id in semantic_models:
            if unique_id in self.saved_manifest.semantic_models:
                semantic_model = self.saved_manifest.semantic_models[unique_id]
                if semantic_model.name == semantic_model_name:
                    if unique_id in self.saved_manifest.child_map:
                        self.schedule_nodes_for_parsing(self.saved_manifest
                            .child_map[unique_id])
                    self.saved_manifest.semantic_models.pop(unique_id)
                    schema_file.semantic_models.remove(unique_id)
            elif unique_id in self.saved_manifest.disabled:
                self.delete_disabled(unique_id, schema_file.file_id)
        if schema_file.generated_metrics:
            schema_file.fix_metrics_from_measures()
        if semantic_model_name in schema_file.metrics_from_measures:
            for unique_id in schema_file.metrics_from_measures[
                semantic_model_name]:
                if unique_id in self.saved_manifest.metrics:
                    self.saved_manifest.metrics.pop(unique_id)
                elif unique_id in self.saved_manifest.disabled:
                    self.delete_disabled(unique_id, schema_file.file_id)
            del schema_file.metrics_from_measures[semantic_model_name]

    def func_pfzyyfps(self, schema_file, unit_test_dict):
        unit_test_name = unit_test_dict['name']
        unit_tests = schema_file.unit_tests.copy()
        for unique_id in unit_tests:
            if unique_id in self.saved_manifest.unit_tests:
                unit_test = self.saved_manifest.unit_tests[unique_id]
                if unit_test.name == unit_test_name:
                    self.saved_manifest.unit_tests.pop(unique_id)
                    schema_file.unit_tests.remove(unique_id)

    def func_pw6qe825(self, elem_list, elem_name):
        for element in elem_list:
            if 'name' in element and element['name'] == elem_name:
                return element
        return None

    def func_yq0zgrwa(self, package_name, source_name):
        schema_file = None
        for source in self.saved_manifest.sources.values():
            if (source.package_name == package_name and source.source_name ==
                source_name):
                file_id = source.file_id
                if file_id in self.saved_files:
                    schema_file = self.saved_files[file_id]
                break
        return schema_file

    def func_gwvtthoz(self, source):
        package = source['overrides']
        source_name = source['name']
        orig_source_schema_file = self.get_schema_file_for_source(package,
            source_name)
        orig_sources = orig_source_schema_file.dict_from_yaml['sources']
        orig_source = self.get_schema_element(orig_sources, source_name)
        return orig_source_schema_file, orig_source

    def func_eb430beu(self, source_dict):
        orig_file, orig_source = self.get_source_override_file_and_dict(
            source_dict)
        if orig_source:
            self.delete_schema_source(orig_file, orig_source)
            self.merge_patch(orig_file, 'sources', orig_source)
            self.add_to_pp_files(orig_file)

    def func_cd5yqzs5(self):
        unchanged_vars = []
        changed_vars = []
        delete_vars = []
        for env_var in self.saved_manifest.env_vars:
            prev_value = self.saved_manifest.env_vars[env_var]
            current_value = os.getenv(env_var)
            if current_value is None:
                if prev_value == DEFAULT_ENV_PLACEHOLDER:
                    unchanged_vars.append(env_var)
                    continue
                delete_vars.append(env_var)
            if prev_value == current_value:
                unchanged_vars.append(env_var)
            else:
                changed_vars.append(env_var)
        for env_var in delete_vars:
            del self.saved_manifest.env_vars[env_var]
        env_vars_changed_source_files = []
        env_vars_changed_schema_files = {}
        for source_file in self.saved_files.values():
            if source_file.parse_file_type == ParseFileType.Fixture:
                continue
            file_id = source_file.file_id
            if not source_file.env_vars:
                continue
            if source_file.parse_file_type == ParseFileType.Schema:
                for yaml_key in source_file.env_vars.keys():
                    for name in source_file.env_vars[yaml_key].keys():
                        for env_var in source_file.env_vars[yaml_key][name]:
                            if env_var in changed_vars:
                                if (file_id not in
                                    env_vars_changed_schema_files):
                                    env_vars_changed_schema_files[file_id] = {}
                                if (yaml_key not in
                                    env_vars_changed_schema_files[file_id]):
                                    env_vars_changed_schema_files[file_id][
                                        yaml_key] = []
                                if name not in env_vars_changed_schema_files[
                                    file_id][yaml_key]:
                                    env_vars_changed_schema_files[file_id][
                                        yaml_key].append(name)
                                break
            else:
                for env_var in source_file.env_vars:
                    if env_var in changed_vars:
                        env_vars_changed_source_files.append(file_id)
                        break
        return env_vars_changed_source_files, env_vars_changed_schema_files
