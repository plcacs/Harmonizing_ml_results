from typing import Any

# === Third-party dependency: dbt.tests.util ===
def run_dbt(args: Optional[List[str]] = ..., expect_pass: bool = ...) -> Any: ...
def run_dbt_and_capture(args: Optional[List[str]] = ..., expect_pass: bool = ...) -> Any: ...
def get_manifest(project_root) -> Optional[Manifest]: ...
def write_file(contents, *paths) -> Any: ...
def read_file(*paths) -> Any: ...
def relation_from_name(adapter, name: str) -> Any: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture

# === Internal dependency: tests.functional.adapter.constraints.fixtures ===
my_model_sql: str
foreign_key_model_sql: str
my_incremental_model_sql: str
my_model_wrong_order_sql: str
my_model_wrong_order_depends_on_fk_sql: str
my_model_view_wrong_order_sql: str
my_model_incremental_wrong_order_sql: str
my_model_incremental_wrong_order_depends_on_fk_sql: str
my_model_wrong_name_sql: str
my_model_view_wrong_name_sql: str
my_model_incremental_wrong_name_sql: str
my_model_data_type_sql: str
my_model_contract_sql_header_sql: str
my_model_incremental_contract_sql_header_sql: str
my_model_with_nulls_sql: str
my_model_incremental_with_nulls_sql: str
my_model_with_quoted_column_name_sql: str
model_schema_yml: str
model_fk_constraint_schema_yml: str
constrained_model_schema_yml: str
model_data_type_schema_yml: str
model_quoted_column_schema_yml: str
model_contract_header_schema_yml: str
create_table_macro_sql: str
incremental_foreign_key_schema_yml: str
incremental_foreign_key_model_raw_numbers_sql: str
incremental_foreign_key_model_stg_numbers_sql: str