from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, ContextManager, Optional, Type, TypeVar

from flask import Flask, Response
from flask.testing import FlaskClient
from flask_appbuilder.security.sqla import models as ab_models
from flask_testing import TestCase
from sqlalchemy.ext.declarative import DeclarativeMeta

from superset.connectors.sqla.models import SqlaTable
from superset.models.core import Database
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.sql_parse import CtasMethod
from superset.utils.core import get_example_default_schema, shortid
from superset.views.base_api import BaseSupersetModelRestApi
from tests.integration_tests.constants import ADMIN_USERNAME
from tests.integration_tests.test_app import app, login

FAKE_DB_NAME: str = ...
DEFAULT_PASSWORD: str = ...
test_client: FlaskClient = ...

def get_resp(
    client: FlaskClient,
    url: str,
    data: Optional[dict[str, Any]] = ...,
    follow_redirects: bool = ...,
    raise_on_error: bool = ...,
    json_: Optional[dict[str, Any]] = ...,
) -> str: ...
def post_assert_metric(client: FlaskClient, uri: str, data: dict[str, Any], func_name: str) -> Response: ...

_T = TypeVar("_T")

class SupersetTestCase(TestCase):
    default_schema_backend_map: dict[str, str]
    maxDiff: int

    def tearDown(self) -> None: ...
    def create_app(self) -> Flask: ...
    @staticmethod
    def get_nonexistent_numeric_id(model: DeclarativeMeta) -> int: ...
    @staticmethod
    def get_birth_names_dataset() -> SqlaTable: ...
    @staticmethod
    def create_user_with_roles(
        username: str, roles: list[str], should_create_roles: bool = ...
    ) -> ab_models.User: ...
    def temporary_user(
        self,
        clone_user: Optional[ab_models.User] = ...,
        username: Optional[str] = ...,
        extra_roles: Optional[list[ab_models.Role]] = ...,
        extra_pvms: Optional[list[Any]] = ...,
        login: bool = ...,
    ) -> ContextManager[ab_models.User]: ...
    @staticmethod
    def create_user(
        username: str,
        password: str,
        role_name: str,
        first_name: str = ...,
        last_name: str = ...,
        email: str = ...,
    ) -> ab_models.User: ...
    @staticmethod
    def get_user(username: str) -> Optional[ab_models.User]: ...
    @staticmethod
    def get_role(name: str) -> Optional[ab_models.Role]: ...
    @staticmethod
    def get_table_by_id(table_id: int) -> SqlaTable: ...
    @staticmethod
    def is_module_installed(module_name: str) -> bool: ...
    def get_or_create(self, cls: Type[_T], criteria: dict[str, Any], **kwargs: Any) -> _T: ...
    def login(self, username: str, password: str = ...) -> Response: ...
    def get_slice(self, slice_name: str) -> Slice: ...
    @staticmethod
    def get_table(name: str, database_id: Optional[int] = ..., schema: Optional[str] = ...) -> SqlaTable: ...
    @staticmethod
    def get_database_by_id(db_id: int) -> Database: ...
    @staticmethod
    def get_database_by_name(database_name: str = ...) -> Database: ...
    @staticmethod
    def get_datasource_mock() -> SqlaTable: ...
    def get_resp(
        self,
        url: str,
        data: Optional[dict[str, Any]] = ...,
        follow_redirects: bool = ...,
        raise_on_error: bool = ...,
        json_: Optional[dict[str, Any]] = ...,
    ) -> str: ...
    def get_json_resp(
        self,
        url: str,
        data: Optional[dict[str, Any]] = ...,
        follow_redirects: bool = ...,
        raise_on_error: bool = ...,
        json_: Optional[dict[str, Any]] = ...,
    ) -> Any: ...
    def logout(self) -> None: ...
    def grant_public_access_to_table(self, table: SqlaTable) -> None: ...
    def grant_role_access_to_table(self, table: SqlaTable, role_name: str) -> None: ...
    def revoke_public_access_to_table(self, table: SqlaTable) -> None: ...
    def revoke_role_access_to_table(self, role_name: str, table: SqlaTable) -> None: ...
    def run_sql(
        self,
        sql: str,
        client_id: Optional[str] = ...,
        username: Optional[str] = ...,
        raise_on_error: bool = ...,
        query_limit: Optional[int] = ...,
        database_name: str = ...,
        sql_editor_id: Optional[str] = ...,
        select_as_cta: bool = ...,
        tmp_table_name: Optional[str] = ...,
        schema: Optional[str] = ...,
        ctas_method: CtasMethod = ...,
        template_params: str = ...,
    ) -> Any: ...
    def create_fake_db(self) -> Database: ...
    def delete_fake_db(self) -> None: ...
    def create_fake_db_for_macros(self) -> Database: ...
    @staticmethod
    def delete_fake_db_for_macros() -> None: ...
    def get_dash_by_slug(self, dash_slug: str) -> Optional[Dashboard]: ...
    def get_assert_metric(self, uri: str, func_name: str) -> Response: ...
    def delete_assert_metric(self, uri: str, func_name: str) -> Response: ...
    def post_assert_metric(self, uri: str, data: dict[str, Any], func_name: str) -> Response: ...
    def put_assert_metric(self, uri: str, data: dict[str, Any], func_name: str) -> Response: ...
    @classmethod
    def get_dttm(cls) -> datetime: ...
    def insert_dashboard(
        self,
        dashboard_title: str,
        slug: str,
        owners: list[int],
        roles: list[int] = ...,
        created_by: Optional[Any] = ...,
        slices: Optional[list[Slice]] = ...,
        position_json: str = ...,
        css: str = ...,
        json_metadata: str = ...,
        published: bool = ...,
        certified_by: Optional[str] = ...,
        certification_details: Optional[str] = ...,
    ) -> Dashboard: ...
    def get_list(self, asset_type: str, filter: dict[str, Any] = ..., username: str = ...) -> Response: ...

@contextmanager
def db_insert_temp_object(obj: _T) -> ContextManager[_T]: ...