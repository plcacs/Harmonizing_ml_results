# === Third-party dependency: flask_appbuilder.security.sqla.models ===
class User(Model): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture

# === Internal dependency: superset ===
from superset.extensions import db

# === Internal dependency: superset.app ===
def create_app(superset_config_module=..., superset_app_root=...): ...

# === Internal dependency: superset.commands.dataset.exceptions ===
class DatasetAccessDeniedError(ForbiddenError):
    ...

# === Internal dependency: superset.commands.explore.form_data.state ===
class TemporaryExploreState(TypedDict): ...

# === Internal dependency: superset.connectors.sqla.models ===
class SqlaTable(CoreDataset, BaseDatasource, ExploreMixin): ...

# === Internal dependency: superset.extensions ===
cache_manager = CacheManager(...)

# === Internal dependency: superset.models.slice ===
class Slice(CoreChart, AuditMixinNullable, ImportExportMixin): ...

# === Internal dependency: superset.utils.cache_manager ===
class CacheManager: ...

# === Internal dependency: superset.utils.core ===
class DatasourceType(StrEnum): ...

# === Internal dependency: superset.utils.json ===
def dumps(obj, default=..., allow_nan=..., ignore_nan=..., sort_keys=..., indent=..., separators=..., cls=..., encoding=...): ...
def loads(obj, encoding=..., allow_nan=..., object_hook=...): ...

# === Internal dependency: tests.integration_tests.test_app ===
superset_config_module = environ.get(...)
app = create_app(...)