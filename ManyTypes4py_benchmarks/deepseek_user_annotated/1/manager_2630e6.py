# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=too-many-lines
"""A set of constants and methods to manage permissions and security"""

import logging
import re
import time
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, NamedTuple, Optional, Set, Tuple, TYPE_CHECKING, Type, Union

from flask import current_app, Flask, g, Request
from flask_appbuilder import Model
from flask_appbuilder.security.sqla.manager import SecurityManager
from flask_appbuilder.security.sqla.models import (
    assoc_permissionview_role,
    assoc_user_role,
    Permission,
    PermissionView,
    Role,
    User,
    ViewMenu,
)
from flask_appbuilder.security.views import (
    PermissionModelView,
    PermissionViewModelView,
    RoleModelView,
    UserModelView,
    ViewMenuModelView,
)
from flask_appbuilder.widgets import ListWidget
from flask_babel import lazy_gettext as _
from flask_login import AnonymousUserMixin, LoginManager
from jwt.api_jwt import _jwt_global_obj
from sqlalchemy import and_, inspect, or_
from sqlalchemy.engine.base import Connection
from sqlalchemy.orm import eagerload
from sqlalchemy.orm.mapper import Mapper
from sqlalchemy.orm.query import Query as SqlaQuery

from superset.constants import RouteMethod
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.exceptions import (
    DatasetInvalidPermissionEvaluationException,
    SupersetSecurityException,
)
from superset.security.guest_token import (
    GuestToken,
    GuestTokenResources,
    GuestTokenResourceType,
    GuestTokenRlsRule,
    GuestTokenUser,
    GuestUser,
)
from superset.sql_parse import extract_tables_from_jinja_sql, Table
from superset.tasks.utils import get_current_user
from superset.utils import json
from superset.utils.core import (
    DatasourceName,
    DatasourceType,
    get_user_id,
    RowLevelSecurityFilterType,
)
from superset.utils.filters import get_dataset_access_filters
from superset.utils.urls import get_url_host

if TYPE_CHECKING:
    from superset.common.query_context import QueryContext
    from superset.connectors.sqla.models import (
        BaseDatasource,
        RowLevelSecurityFilter,
        SqlaTable,
    )
    from superset.models.core import Database
    from superset.models.dashboard import Dashboard
    from superset.models.slice import Slice
    from superset.models.sql_lab import Query
    from superset.viz import BaseViz

logger = logging.getLogger(__name__)

DATABASE_PERM_REGEX = re.compile(r"^\[.+\]\.\(id\:(?P<id>\d+)\)$")


class DatabaseCatalogSchema(NamedTuple):
    database: str
    catalog: Optional[str]
    schema: str


class SupersetSecurityListWidget(ListWidget):  # pylint: disable=too-few-public-methods
    """
    Redeclaring to avoid circular imports
    """

    template: str = "superset/fab_overrides/list.html"


class SupersetRoleListWidget(ListWidget):  # pylint: disable=too-few-public-methods
    """
    Role model view from FAB already uses a custom list widget override
    So we override the override
    """

    template: str = "superset/fab_overrides/list_role.html"

    def __init__(self, **kwargs: Any) -> None:
        kwargs["appbuilder"] = current_app.appbuilder
        super().__init__(**kwargs)


UserModelView.list_widget = SupersetSecurityListWidget
RoleModelView.list_widget = SupersetRoleListWidget
PermissionViewModelView.list_widget = SupersetSecurityListWidget
PermissionModelView.list_widget = SupersetSecurityListWidget

# Limiting routes on FAB model views
UserModelView.include_route_methods: Set[Union[RouteMethod, str]] = RouteMethod.CRUD_SET | {
    RouteMethod.ACTION,
    RouteMethod.API_READ,
    RouteMethod.ACTION_POST,
    "userinfo",
}
RoleModelView.include_route_methods: Set[RouteMethod] = RouteMethod.CRUD_SET
PermissionViewModelView.include_route_methods: Set[RouteMethod] = {RouteMethod.LIST}
PermissionModelView.include_route_methods: Set[RouteMethod] = {RouteMethod.LIST}
ViewMenuModelView.include_route_methods: Set[RouteMethod] = {RouteMethod.LIST}

RoleModelView.list_columns: List[str] = ["name"]
RoleModelView.edit_columns: List[str] = ["name", "permissions", "user"]
RoleModelView.related_views: List[Any] = []


def freeze_value(value: Any) -> str:
    """
    Used to compare column and metric sets.
    """
    return json.dumps(value, sort_keys=True)


def query_context_modified(query_context: "QueryContext") -> bool:
    """
    Check if a query context has been modified.

    This is used to ensure guest users don't modify the payload and fetch data
    different from what was shared with them in dashboards.
    """
    form_data = query_context.form_data
    stored_chart = query_context.slice_

    # native filter requests
    if form_data is None or stored_chart is None:
        return False

    # cannot request a different chart
    if form_data.get("slice_id") != stored_chart.id:
        return True

    stored_query_context = (
        json.loads(cast(str, stored_chart.query_context))
        if stored_chart.query_context
        else None
    )

    # compare columns and metrics in form_data with stored values
    for key, equivalent in [
        ("metrics", ["metrics"]),
        ("columns", ["columns", "groupby"]),
        ("groupby", ["columns", "groupby"]),
        ("orderby", ["orderby"]),
    ]:
        requested_values = {freeze_value(value) for value in form_data.get(key) or []}
        stored_values = {
            freeze_value(value) for value in stored_chart.params_dict.get(key) or []
        }
        if not requested_values.issubset(stored_values):
            return True

        # compare queries in query_context
        queries_values = {
            freeze_value(value)
            for query in query_context.queries
            for value in getattr(query, key, []) or []
        }
        if stored_query_context:
            for query in stored_query_context.get("queries") or []:
                for key in equivalent:
                    stored_values.update(
                        {freeze_value(value) for value in query.get(key) or []}
                    )

        if not queries_values.issubset(stored_values):
            return True

    return False


class SupersetSecurityManager(SecurityManager):  # pylint: disable=too-many-public-methods
    userstatschartview: Optional[Any] = None
    READ_ONLY_MODEL_VIEWS: Set[str] = {"Database", "DynamicPlugin"}

    USER_MODEL_VIEWS: Set[str] = {
        "RegisterUserModelView",
        "UserDBModelView",
        "UserLDAPModelView",
        "UserInfoEditView",
        "UserOAuthModelView",
        "UserOIDModelView",
        "UserRemoteUserModelView",
    }

    GAMMA_READ_ONLY_MODEL_VIEWS: Set[str] = {
        "Dataset",
        "Datasource",
    } | READ_ONLY_MODEL_VIEWS

    ADMIN_ONLY_VIEW_MENUS: Set[str] = {
        "Access Requests",
        "Action Log",
        "Log",
        "List Users",
        "List Roles",
        "ResetPasswordView",
        "RoleModelView",
        "Row Level Security",
        "Row Level Security Filters",
        "RowLevelSecurityFiltersModelView",
        "Security",
        "SQL Lab",
        "User Registrations",
        "User's Statistics",
        # Guarding all AB_ADD_SECURITY_API = True REST APIs
        "Role",
        "Permission",
        "PermissionViewMenu",
        "ViewMenu",
        "User",
    } | USER_MODEL_VIEWS

    ALPHA_ONLY_VIEW_MENUS: Set[str] = {
        "Alerts & Report",
        "Annotation Layers",
        "Annotation",
        "CSS Templates",
        "ColumnarToDatabaseView",
        "CssTemplate",
        "ExcelToDatabaseView",
        "Import dashboards",
        "ImportExportRestApi",
        "Manage",
        "Queries",
        "ReportSchedule",
        "TableSchemaView",
    }

    ALPHA_ONLY_PMVS: Set[Tuple[str, str]] = {
        ("can_upload", "Database"),
    }

    ADMIN_ONLY_PERMISSIONS: Set[str] = {
        "can_update_role",
        "all_query_access",
        "can_grant_guest_token",
        "can_set_embedded",
        "can_warm_up_cache",
    }

    READ_ONLY_PERMISSION: Set[str] = {
        "can_show",
        "can_list",
        "can_get",
        "can_external_metadata",
        "can_external_metadata_by_name",
        "can_read",
    }

    ALPHA_ONLY_PERMISSIONS: Set[str] = {
        "muldelete",
        "all_database_access",
        "all_datasource_access",
    }

    OBJECT_SPEC_PERMISSIONS: Set[str] = {
        "database_access",
        "catalog_access",
        "schema_access",
        "datasource_access",
    }

    ACCESSIBLE_PERMS: Set[str] = {"can_userinfo", "resetmypassword", "can_recent_activity"}

    SQLLAB_ONLY_PERMISSIONS: Set[Tuple[str, str]] = {
        ("can_read", "SavedQuery"),
        ("can_write", "SavedQuery"),
        ("can_export", "SavedQuery"),
        ("can_read", "Query"),
        ("can_export_csv", "Query"),
        ("can_get_results", "SQLLab"),
        ("can_execute_sql_query", "SQLLab"),
        ("can_estimate_query_cost", "SQL Lab"),
        ("can_export_csv", "SQLLab"),
        ("can_read", "SQLLab"),
        ("can_sqllab_history", "Superset"),
        ("can_sqllab", "Superset"),
        ("can_test_conn", "Superset"),  # Deprecated permission remove on 3.0.0
        ("can_activate", "TabStateView"),
        ("can_get", "TabStateView"),
        ("can_delete_query", "TabStateView"),
        ("can_post", "TabStateView"),
        ("can_delete", "TabStateView"),
        ("can_put", "TabStateView"),
        ("can_migrate_query", "TabStateView"),
        ("menu_access", "SQL Lab"),
        ("menu_access", "SQL Editor"),
        ("menu_access", "Saved Queries"),
        ("menu_access", "Query Search"),
    }

    SQLLAB_EXTRA_PERMISSION_VIEWS: Set[Tuple[str, str]] = {
        ("can_csv", "Superset"),  # Deprecated permission remove on 3.0.0
        ("can_read", "Superset"),
        ("can_read", "Database"),
    }

    data_access_permissions: Tuple[str, ...] = (
        "database_access",
        "schema_access",
        "datasource_access",
        "all_datasource_access",
        "all_database_access",
        "all_query_access",
    )

    guest_user_cls: Type[GuestUser] = GuestUser
    pyjwt_for_guest_token: Any = _jwt_global_obj

    def create_login_manager(self, app: Flask) -> LoginManager:
        lm = super().create_login_manager(app)
        lm.request_loader(self.request_loader)
        return lm

    def request_loader(self, request: Request) -> Optional[User]:
        # pylint: disable=import-outside-toplevel
        from superset.extensions import feature_flag_manager

        if feature_flag_manager.is_feature_enabled("EMBEDDED_SUPERSET"):
            return self.get_guest_user_from_request(request)
        return None

    def get_catalog_perm(
        self,
        database: str,
        catalog: Optional[str] = None,
    ) -> Optional[str]:
        """
        Return the database specific catalog permission.

        :param database: The Superset database or database name
        :param catalog: The database catalog name
        :return: The database specific schema permission
        """
        if catalog is None:
            return None

        return f"[{database}].[{catalog}]"

    def get_schema_perm(
        self,
        database: str,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> Optional[str]:
        """
        Return the database specific schema permission.

        Catalogs were added in SIP-95, and not all databases support them. Because of
        this, the format used for permissions is different depending on whether a
        catalog is passed or not:

            [database].[schema]
            [database].[catalog].[schema]

        :param database: The database name
        :param catalog: The database catalog name
        :param schema: The database schema name
        :return: The database specific schema permission
        """
        if schema is None:
            return None

        if catalog:
            return f"[{database}].[{catalog}].[{schema}]"

        return f"[{database}].[{schema}]"

    @staticmethod
    def get_database_perm(database_id: int, database_name: str) -> Optional[str]:
        return f"[{database_name}].(id:{database_id})"

    @staticmethod
    def get_dataset_perm(
        dataset_id: int,
        dataset_name: str,
        database_name: str,
    ) -> Optional[str]:
        return f"[{database_name}].[{dataset_name}](id:{dataset_id})"

    def can_access(self, permission_name: str, view_name: str) -> bool:
        """
        Return True if the user can access the FAB permission/view, False otherwise.

        Note this method adds protection from has_access failing from missing
        permission/view entries.

        :param permission_name: The FAB permission name
        :param view_name: The FAB view-menu name
        :returns: Whether the user can access the FAB permission/view
        """

        user = g.user
        if user.is_anonymous:
            return self.is_item_public(permission_name, view_name)
        return self._has_view_access(user, permission_name, view_name)

    def can_access_all_queries(self) -> bool:
        """
        Return True if the user can access all SQL Lab queries, False otherwise.

        :returns: Whether the user can access all queries
        """

        return self.can_access("all_query_access", "all_query_access")

    def can_access_all_datasources(self) -> bool:
        """
        Return True if the user can access all the datasources, False otherwise.

        :returns: Whether the user can access all the datasources
        """

        return self.can_access_all_databases() or self.can_access(
            "all_datasource_access", "all_datasource_access"
        )

    def can_access_all_databases(self) -> bool:
        """
        Return True if the user can access all the databases, False otherwise.

        :returns: Whether the user can access all the databases
        """
        return self.can_access("all_database_access", "all_database_access")

    def can_access_database(self, database: "Database") -> bool:
        """
        Return True if the user can access the specified database, False otherwise.

        :param database: The database
        :returns: Whether the user can access the database
        """

        return (
            self.can_access_all_datasources()
            or self.can_access_all_databases()
            or self.can_access("database_access", database.perm)
        )

    def can_access_catalog(self, database: "Database", catalog: str) -> bool:
        """
        Return if the user can access the specified catalog.
        """
        catalog_perm = self.get_catalog_perm(database.database_name, catalog)
        return bool(
            self.can_access_all_datasources()
            or self.can_access_database(database)
            or (catalog_perm and self.can_access("catalog_access", catalog_perm))
        )

    def can_access_schema(self, datasource: "BaseDatasource") -> bool:
        """
        Return True if the user can access the schema associated with specified
        datasource, False otherwise.

        :param datasource: The datasource
        :returns: Whether the user can access the datasource's schema
        """

        return (
            self.can_access_all_datasources()
            or self.can_access_database(datasource.database)
            or (
                datasource.catalog
                and self.can_access_catalog(datasource.database, datasource.catalog)
            )
            or self.can_access("schema_access", datasource.schema_perm or "")
        )

    def can_access_datasource(self, datasource: "BaseDatasource") -> bool:
        """
        Return True if the user can access the specified datasource, False otherwise.

        :param datasource: The datasource
        :returns: Whether the user can access the datasource
        """

        try:
            self.raise_for_access(datasource=datasource)
        except SupersetSecurityException:
            return False

        return True

    def can_access_dashboard(self, dashboard: "Dashboard") -> bool:
        """
        Return True if the user can access the specified dashboard, False otherwise.

        :param dashboard: The dashboard
        :returns: Whether the user can access the dashboard
        """

        try:
            self.raise_for_access(dashboard=dashboard)
        except SupersetSecurity