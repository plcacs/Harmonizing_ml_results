from typing import Any

# === Third-party dependency: celery ===
# Used symbols: Celery

# === Third-party dependency: flask ===
# Used symbols: has_app_context

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark

# === Internal dependency: superset ===
# re-export: from superset.extensions import db

# === Internal dependency: superset.app ===
def create_app(superset_config_module: Optional[str] = ..., superset_app_root: Optional[str] = ...) -> Flask: ...

# === Internal dependency: superset.common.db_query_status ===
class QueryStatus(StrEnum): ...

# === Internal dependency: superset.db_engine_specs.base ===
class BaseEngineSpec:
    def expand_data(cls, columns: list[ResultSetColumnType], data: list[dict[Any, Any]]) -> tuple[list[ResultSetColumnType], list[dict[Any, Any]], list[ResultSetColumnType]]: ...

# === Internal dependency: superset.errors ===
class SupersetErrorType(StrEnum): ...
class ErrorLevel(StrEnum): ...

# === Internal dependency: superset.extensions ===
celery_app: Celery

# === Internal dependency: superset.models.sql_lab ===
class Query(CoreQuery, SqlTablesMixin, ExtraJSONMixin, ExploreMixin): ...

# === Internal dependency: superset.result_set ===
class SupersetResultSet:
    def __init__(self, data: DbapiResult, cursor_description: DbapiDescription, db_engine_spec: type[BaseEngineSpec]) -> Any: ...

# === Internal dependency: superset.sql.parse ===
class CTASMethod(enum.Enum): ...

# === Internal dependency: superset.sql_lab ===
def _serialize_payload(payload: dict[Any, Any], use_msgpack: Optional[bool] = ...) -> Union[bytes, str]: ...
def _serialize_and_expand_data(result_set: SupersetResultSet, db_engine_spec: BaseEngineSpec, use_msgpack: Optional[bool] = ..., expand_data: bool = ...) -> tuple[Union[bytes, str], list[Any], list[Any], list[Any]]: ...

# === Internal dependency: superset.utils.core ===
def backend() -> str: ...

# === Internal dependency: superset.utils.database ===
def get_example_database() -> Database: ...

# === Internal dependency: tests.integration_tests.conftest ===
CTAS_SCHEMA_NAME: str

# === Internal dependency: tests.integration_tests.test_app ===
app: create_app