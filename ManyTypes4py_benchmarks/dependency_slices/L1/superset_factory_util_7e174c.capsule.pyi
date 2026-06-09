# === Third-party dependency: sqlalchemy ===
# Used symbols: Table

# === Internal dependency: superset ===
from superset.extensions import db

# === Internal dependency: superset.connectors.sqla.models ===
class SqlaTable(CoreDataset, BaseDatasource, ExploreMixin): ...
metadata = Model.metadata
sqlatable_user = DBTable(...)

# === Internal dependency: superset.models.core ===
class Database(CoreDatabase, AuditMixinNullable, ImportExportMixin): ...

# === Internal dependency: superset.models.dashboard ===
class Dashboard(CoreDashboard, AuditMixinNullable, ImportExportMixin): ...
metadata = Model.metadata
dashboard_slices = Table(...)
dashboard_user = Table(...)
DashboardRoles = Table(...)

# === Internal dependency: superset.models.slice ===
class Slice(CoreChart, AuditMixinNullable, ImportExportMixin): ...
metadata = Model.metadata
slice_user = Table(...)

# === Internal dependency: tests.integration_tests.dashboards.dashboard_test_utils ===
def random_title(): ...
def random_slug(): ...
def random_str(): ...