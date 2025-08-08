saved_manifest: Manifest
new_files: Dict[str, AnySourceFile]

def skip_parsing(self) -> bool:
def build_file_diff(self) -> None:
def get_parsing_files(self) -> Dict[str, Dict[str, List[str]]]:
def add_to_pp_files(self, source_file: AnySourceFile) -> None:
def already_scheduled_for_parsing(self, source_file: AnySourceFile) -> bool:
def add_to_saved(self, file_id: str) -> None:
def handle_added_schema_file(self, source_file: SchemaSourceFile) -> None:
def delete_disabled(self, unique_id: str, file_id: str) -> MutableMapping:
def delete_from_saved(self, file_id: str) -> None:
def update_in_saved(self, file_id: str) -> None:
def update_mssat_in_saved(self, new_source_file: AnySourceFile, old_source_file: AnySourceFile) -> None:
def remove_node_in_saved(self, source_file: AnySourceFile, unique_id: str) -> None:
def update_macro_in_saved(self, new_source_file: AnySourceFile, old_source_file: AnySourceFile) -> None:
def update_doc_in_saved(self, new_source_file: AnySourceFile, old_source_file: AnySourceFile) -> None:
def update_fixture_in_saved(self, new_source_file: AnySourceFile, old_source_file: AnySourceFile) -> None:
def remove_mssat_file(self, source_file: AnySourceFile) -> None:
def schedule_referencing_nodes_for_parsing(self, unique_id: str) -> None:
def schedule_nodes_for_parsing(self, unique_ids: List[str]) -> None:
def _schedule_for_parsing(self, dict_key: str, element: MutableMapping, name: str, delete: Callable) -> None:
def delete_macro_file(self, source_file: AnySourceFile, follow_references: bool = False) -> None:
def check_for_special_deleted_macros(self, source_file: AnySourceFile) -> None:
def recursively_gather_macro_references(self, macro_unique_id: str, referencing_nodes: List[str]) -> None:
def handle_macro_file_links(self, source_file: AnySourceFile, follow_references: bool) -> None:
def schedule_macro_nodes_for_parsing(self, unique_ids: List[str]) -> None:
def delete_doc_node(self, source_file: AnySourceFile) -> None:
def delete_fixture_node(self, source_file: AnySourceFile) -> None:
def change_schema_file(self, file_id: str) -> None:
def delete_schema_file(self, file_id: str) -> None:
def handle_schema_file_changes(self, schema_file: SchemaSourceFile, saved_yaml_dict: Dict, new_yaml_dict: Dict) -> None:
def _handle_element_change(self, schema_file: SchemaSourceFile, saved_yaml_dict: Dict, new_yaml_dict: Dict, env_var_changes: Dict, dict_key: str, delete: Callable) -> None:
def get_diff_for(self, key: str, saved_yaml_dict: Dict, new_yaml_dict: Dict) -> Dict:
def merge_patch(self, schema_file: SchemaSourceFile, key: str, patch: MutableMapping, new_patch: bool = False) -> None:
def delete_schema_mssa_links(self, schema_file: SchemaSourceFile, dict_key: str, elem: MutableMapping) -> None:
def remove_tests(self, schema_file: SchemaSourceFile, dict_key: str, name: str) -> None:
def delete_yaml_snapshot(self, schema_file: SchemaSourceFile, snapshot_dict: MutableMapping) -> None:
def delete_schema_source(self, schema_file: SchemaSourceFile, source_dict: MutableMapping) -> None:
def delete_schema_macro_patch(self, schema_file: SchemaSourceFile, macro: MutableMapping) -> None:
def delete_schema_data_test_patch(self, schema_file: SchemaSourceFile, data_test: MutableMapping) -> None:
def delete_schema_exposure(self, schema_file: SchemaSourceFile, exposure_dict: MutableMapping) -> None:
def delete_schema_group(self, schema_file: SchemaSourceFile, group_dict: MutableMapping) -> None:
def delete_schema_metric(self, schema_file: SchemaSourceFile, metric_dict: MutableMapping) -> None:
def delete_schema_saved_query(self, schema_file: SchemaSourceFile, saved_query_dict: MutableMapping) -> None:
def delete_schema_semantic_model(self, schema_file: SchemaSourceFile, semantic_model_dict: MutableMapping) -> None:
def delete_schema_unit_test(self, schema_file: SchemaSourceFile, unit_test_dict: MutableMapping) -> None:
def get_schema_element(self, elem_list: List[MutableMapping], elem_name: str) -> MutableMapping:
def get_schema_file_for_source(self, package_name: str, source_name: str) -> SchemaSourceFile:
def get_source_override_file_and_dict(self, source: MutableMapping) -> Tuple[SchemaSourceFile, MutableMapping]:
def remove_source_override_target(self, source_dict: MutableMapping) -> None:
def build_env_vars_to_files(self) -> Tuple[List[str], Dict[str, Dict[str, List[str]]]:
