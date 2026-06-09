from typing import Any

# === Third-party dependency: sqlalchemy ===
# Used symbols: Table

# === Internal dependency: superset ===
# re-export: from superset.extensions import db

# === Internal dependency: superset.connectors.sqla.models ===
class SqlaTable(CoreDataset, BaseDatasource, ExploreMixin): ...
sqlatable_user: Table

# === Internal dependency: superset.models.core ===
class Database(CoreDatabase, AuditMixinNullable, ImportExportMixin): ...

# === Internal dependency: superset.models.dashboard ===
class Dashboard(CoreDashboard, AuditMixinNullable, ImportExportMixin): ...
dashboard_slices: Table
dashboard_user: Table
DashboardRoles: Table

# === Internal dependency: superset.models.slice ===
class Slice(CoreChart, AuditMixinNullable, ImportExportMixin): ...
slice_user: Table

# === Internal dependency: tests.integration_tests.dashboards.dashboard_test_utils ===
def random_title() -> Any: ...
def random_slug() -> Any: ...
def random_str() -> Any: ...