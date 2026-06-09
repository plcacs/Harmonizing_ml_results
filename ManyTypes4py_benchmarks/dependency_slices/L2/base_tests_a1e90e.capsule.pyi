from typing import Any

# === Third-party dependency: flask ===
# Used symbols: Response, g

# === Third-party dependency: flask_appbuilder.security.sqla ===
# Used symbols: models

# === Third-party dependency: flask_testing ===
# Used symbols: TestCase

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Unresolved dependency: rison ===
# Used unresolved symbols: dumps

# === Third-party dependency: sqlalchemy.dialects.mysql ===
# Used symbols: dialect

# === Third-party dependency: sqlalchemy.engine.interfaces ===
class Dialect(EventTarget): ...

# === Third-party dependency: sqlalchemy.sql ===
# Used symbols: func

# === Internal dependency: superset ===
# re-export: from superset.extensions import db
# re-export: from superset.extensions import security_manager

# === Internal dependency: superset.app ===
def create_app(superset_config_module: Optional[str] = ..., superset_app_root: Optional[str] = ...) -> Flask: ...

# === Internal dependency: superset.connectors.sqla.models ===
class BaseDatasource(AuditMixinNullable, ImportExportMixin): ...
class SqlaTable(CoreDataset, BaseDatasource, ExploreMixin): ...

# === Internal dependency: superset.models.core ===
class Database(CoreDatabase, AuditMixinNullable, ImportExportMixin): ...

# === Internal dependency: superset.models.dashboard ===
class Dashboard(CoreDashboard, AuditMixinNullable, ImportExportMixin):
    ...

# === Internal dependency: superset.models.slice ===
class Slice(CoreChart, AuditMixinNullable, ImportExportMixin): ...

# === Internal dependency: superset.sql.parse ===
class CTASMethod(enum.Enum): ...

# === Internal dependency: superset.utils.core ===
def get_example_default_schema() -> str | None: ...
def shortid() -> str: ...

# === Internal dependency: superset.utils.database ===
def get_example_database() -> Database: ...

# === Internal dependency: superset.utils.json ===
def loads(obj: Union[bytes, bytearray, str], encoding: Union[str, None] = ..., allow_nan: bool = ..., object_hook: Union[Callable[[dict[Any, Any]], Any], None] = ...) -> Any: ...

# === Internal dependency: superset.views.base_api ===
class BaseSupersetModelRestApi(BaseSupersetApiMixin, ModelRestApi): ...

# === Internal dependency: tests.integration_tests.constants ===
ADMIN_USERNAME: str

# === Internal dependency: tests.integration_tests.fixtures.importexport ===
database_config: dict[str, Any]
dataset_config: dict[str, Any]
chart_config: dict[str, Any]
dashboard_config: dict[str, Any]
metadata_files: Any

# === Internal dependency: tests.integration_tests.test_app ===
app: create_app

# === Third-party dependency: yaml ===
def safe_dump(data, stream = ..., **kwds) -> Any: ...