from typing import Any

# === Third-party dependency: flask_appbuilder.security.sqla.models ===
class User(Model): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture

# === Internal dependency: superset ===
# re-export: from superset.extensions import db

# === Internal dependency: superset.app ===
def create_app(superset_config_module: Optional[str] = ..., superset_app_root: Optional[str] = ...) -> Flask: ...

# === Internal dependency: superset.commands.dataset.exceptions ===
class DatasetAccessDeniedError(ForbiddenError):
    ...

# === Internal dependency: superset.commands.explore.form_data.state ===
class TemporaryExploreState(TypedDict): ...

# === Internal dependency: superset.connectors.sqla.models ===
class SqlaTable(CoreDataset, BaseDatasource, ExploreMixin): ...

# === Internal dependency: superset.extensions ===
cache_manager: CacheManager

# === Internal dependency: superset.models.slice ===
class Slice(CoreChart, AuditMixinNullable, ImportExportMixin): ...

# === Internal dependency: superset.utils.cache_manager ===
class CacheManager: ...

# === Internal dependency: superset.utils.core ===
class DatasourceType(StrEnum): ...

# === Internal dependency: superset.utils.json ===
def dumps(obj: Any, default: Optional[Callable[[Any], Any]] = ..., allow_nan: bool = ..., ignore_nan: bool = ..., sort_keys: bool = ..., indent: Union[str, int, None] = ..., separators: Union[tuple[str, str], None] = ..., cls: Union[type[simplejson.JSONEncoder], None] = ..., encoding: Optional[str] = ...) -> str: ...
def loads(obj: Union[bytes, bytearray, str], encoding: Union[str, None] = ..., allow_nan: bool = ..., object_hook: Union[Callable[[dict[Any, Any]], Any], None] = ...) -> Any: ...

# === Internal dependency: tests.integration_tests.test_app ===
app: create_app