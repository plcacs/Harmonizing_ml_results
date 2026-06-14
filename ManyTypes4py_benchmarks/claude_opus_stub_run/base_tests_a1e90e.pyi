from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional, Union
from unittest.mock import MagicMock

from flask import Flask
from flask.testing import FlaskClient
from flask_appbuilder.security.sqla import models as ab_models
from flask_testing import TestCase

from superset.connectors.sqla.models import SqlaTable
from superset.models.core import Database
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.sql_parse import CtasMethod

FAKE_DB_NAME: str
DEFAULT_PASSWORD: str
test_client: FlaskClient

def get_resp(
    client: FlaskClient,
    url: str,
    data: Optional[Any] = ...,
    follow_redirects: bool = ...,
    raise_on_error: bool = ...,
    json_: Optional[Any] = ...,
) -> str: ...

def post_assert_metric(
    client: FlaskClient,
    uri: str,
    data: Any,
    func_name: str,
) -> Any: ...

class SupersetTestCase(TestCase):
    default_schema_backend_map: dict[str, str]
    maxDiff: int

    def tearDown(self) -> None: ...
    def create_app(self) -> Flask: ...

    @staticmethod
    def get_nonexistent_numeric_id(model: Any) -> int: ...

    @staticmethod
    def get_birth_names_dataset() -> SqlaTable: ...

    @staticmethod
    def create_user_with_roles(
        username: str,
        roles: list[str],
        should_create_roles: bool = ...,
    ) -> ab_models.User: ...

    @contextmanager
    def temporary_user(
        self,
        clone_user: Optional[ab_models.User] = ...,
        username: Optional[str] = ...,
        extra_roles: Optional[list[ab_models.Role]] = ...,
        extra_pvms: Optional[list[Any]] = ...,
        login: bool = ...,
    ) -> Generator[ab_models.User, None, None]: ...

    @staticmethod
    def create_user(
        username: str,
        password: str,
        role_name: str,
        first_name: str = ...,
        last_name: str = ...,
        email: str = ...,
    ) -> Any: ...

    @staticmethod
    def get_user(username: str) -> Optional[Any]: ...

    @staticmethod
    def get_role(name: str) -> Optional[Any]: ...

    @staticmethod
    def get_table_by_id(table_id: int) -> SqlaTable: ...

    @staticmethod
    def is_module_installed(module_name: str) -> bool: ...

    def get_or_create(self, cls: type, criteria: dict[str, Any], **kwargs: Any) -> Any: ...
    def login(self, username: str, password: str = ...) -> Any: ...
    def get_slice(self, slice_name: str) -> Slice: ...

    @staticmethod
    def get_table(
        name: str,
        database_id: Optional[int] = ...,
        schema: Optional[str] = ...,
    ) -> SqlaTable: ...

    @staticmethod
    def get_database_by_id(db_id: int) -> Database: ...

    @staticmethod
    def get_database_by_name(database_name: str = ...) -> Database: ...

    @staticmethod
    def get_datasource_mock() -> MagicMock: ...

    def get_resp(
        self,
        url: str,
        data: Optional[Any] = ...,
        follow_redirects: bool = ...,
        raise_on_error: bool = ...,
        json_: Optional[Any] = ...,
    ) -> str: ...

    def get_json_resp(
        self,
        url: str,
        data: Optional[Any] = ...,
        follow_redirects: bool = ...,
        raise_on_error: bool = ...,
        json_: Optional[Any] = ...,
    ) -> Any: ...

    def logout(self) -> None: ...
    def grant_public_access_to_table(self, table: Any) -> None: ...
    def grant_role_access_to_table(self, table: Any, role_name: str) -> None: ...
    def revoke_public_access_to_table(self, table: Any) -> None: ...
    def revoke_role_access_to_table(self, role_name: str, table: Any) -> None: ...

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
    def get_assert_metric(self, uri: str, func_name: str) -> Any: ...
    def delete_assert_metric(self, uri: str, func_name: str) -> Any: ...
    def post_assert_metric(self, uri: str, data: Any, func_name: str) -> Any: ...
    def put_assert_metric(self, uri: str, data: Any, func_name: str) -> Any: ...

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

    def get_list(
        self,
        asset_type: str,
        filter: dict[str, Any] = ...,
        username: str = ...,
    ) -> Any: ...

@contextmanager
def db_insert_temp_object(obj: Any) -> Generator[Any, None, None]: ...