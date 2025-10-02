import logging
from typing import Optional, List
from flask_appbuilder import Model
from flask_appbuilder.security.sqla.models import User
from superset import db
from superset.connectors.sqla.models import SqlaTable, sqlatable_user
from superset.models.core import Database
from superset.models.dashboard import Dashboard, dashboard_slices, dashboard_user, DashboardRoles
from superset.models.slice import Slice, slice_user
from tests.integration_tests.dashboards.dashboard_test_utils import random_slug, random_str, random_title

logger: logging.Logger = logging.getLogger(__name__)
inserted_dashboards_ids: List[int] = []
inserted_databases_ids: List[int] = []
inserted_sqltables_ids: List[int] = []
inserted_slices_ids: List[int] = []

def create_dashboard_to_db(dashboard_title: Optional[str] = None, slug: Optional[str] = None, published: bool = False, owners: Optional[List[User]] = None, slices: Optional[List[Slice]] = None, css: str = '', json_metadata: str = '', position_json: str = '') -> Dashboard:
    ...

def create_dashboard(dashboard_title: Optional[str] = None, slug: Optional[str] = None, published: bool = False, owners: Optional[List[User]] = None, slices: Optional[List[Slice]] = None, css: str = '', json_metadata: str = '', position_json: str = '') -> Dashboard:
    ...

def insert_model(dashboard: Model) -> None:
    ...

def create_slice_to_db(name: Optional[str] = None, datasource_id: Optional[int] = None, owners: Optional[List[User]] = None) -> Slice:
    ...

def create_slice(datasource_id: Optional[int] = None, datasource: Optional[SqlaTable] = None, name: Optional[str] = None, owners: Optional[List[User]] = None) -> Slice:
    ...

def create_datasource_table_to_db(name: Optional[str] = None, db_id: Optional[int] = None, owners: Optional[List[User]] = None) -> SqlaTable:
    ...

def create_datasource_table(name: Optional[str] = None, db_id: Optional[int] = None, database: Optional[Database] = None, owners: Optional[List[User]] = None) -> SqlaTable:
    ...

def create_database_to_db(name: Optional[str] = None) -> Database:
    ...

def create_database(name: Optional[str] = None) -> Database:
    ...

def delete_all_inserted_objects() -> None:
    ...

def delete_all_inserted_dashboards() -> None:
    ...

def delete_dashboard(dashboard: Dashboard, do_commit: bool = False) -> None:
    ...

def delete_dashboard_users_associations(dashboard: Dashboard) -> None:
    ...

def delete_dashboard_roles_associations(dashboard: Dashboard) -> None:
    ...

def delete_dashboard_slices_associations(dashboard: Dashboard) -> None:
    ...

def delete_all_inserted_slices() -> None:
    ...

def delete_slice(slice_: Slice, do_commit: bool = False) -> None:
    ...

def delete_slice_users_associations(slice_: Slice) -> None:
    ...

def delete_all_inserted_tables() -> None:
    ...

def delete_sqltable(table: SqlaTable, do_commit: bool = False) -> None:
    ...

def delete_table_users_associations(table: SqlaTable) -> None:
    ...

def delete_all_inserted_dbs() -> None:
    ...

def delete_database(database: Database, do_commit: bool = False) -> None:
    ...
