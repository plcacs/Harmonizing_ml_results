# === Third-party dependency: celery ===
# Used symbols: Celery

# === Third-party dependency: flask ===
# Used symbols: has_app_context

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark

# === Internal dependency: superset ===
from superset.extensions import db

# === Internal dependency: superset.app ===
def create_app(superset_config_module=..., superset_app_root=...): ...

# === Internal dependency: superset.common.db_query_status ===
class QueryStatus(StrEnum): ...

# === Internal dependency: superset.db_engine_specs.base ===
class BaseEngineSpec:
    def expand_data(cls, columns, data): ...

# === Internal dependency: superset.errors ===
class SupersetErrorType(StrEnum): ...
class ErrorLevel(StrEnum): ...

# === Internal dependency: superset.extensions ===
celery_app = celery.Celery(...)

# === Internal dependency: superset.models.sql_lab ===
class Query(CoreQuery, SqlTablesMixin, ExtraJSONMixin, ExploreMixin): ...

# === Internal dependency: superset.result_set ===
class SupersetResultSet:
    def __init__(self, data, cursor_description, db_engine_spec): ...

# === Internal dependency: superset.sql.parse ===
class CTASMethod(enum.Enum): ...

# === Internal dependency: superset.sql_lab ===
def _serialize_payload(payload, use_msgpack=...): ...
def _serialize_and_expand_data(result_set, db_engine_spec, use_msgpack=..., expand_data=...): ...

# === Internal dependency: superset.utils.core ===
def backend(): ...

# === Internal dependency: superset.utils.database ===
def get_example_database(): ...

# === Internal dependency: tests.integration_tests.conftest ===
CTAS_SCHEMA_NAME = 'sqllab_test_db'

# === Internal dependency: tests.integration_tests.test_app ===
superset_config_module = environ.get(...)
app = create_app(...)