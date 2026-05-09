"""Unit tests for Superset"""
from datetime import datetime
from typing import Any, Union, Optional, Generator, Type, TypeVar, Iterable, cast
from contextlib import contextmanager
from flask import Response
from flask_testing import TestCase
from unittest.mock import MagicMock
from superset.sql_parse import CtasMethod
from superset.connectors.sqla.models import SqlaTable
from superset.models.core import Database
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from flask_appbuilder.security.sqla import models as ab_models

T = TypeVar('T')

FAKE_DB_NAME: str = 'fake_db_100'
DEFAULT_PASSWORD: str = 'general'
test_client: Any = ...

def get_resp(
    client: Any,
    url: str,
    data: Optional[Any] = None,
    follow_redirects: bool = True,
    raise_on_error: bool = True,
    json_: Optional[Any] = None,
) -> str: ...

def post_assert_metric(
    client: Any,
    uri: str,
    data: Any,
    func_name: str,
) -> Response: ...

class SupersetTestCase(TestCase):
    default_schema_backend_map: dict[str, str]
    maxDiff: int

    def tearDown(self) -> None: ...
    def create_app(self) -> Any: ...

    @staticmethod
    def get_nonexistent_numeric_id(model: Any) -> int: ...

    @staticmethod
    def get_birth_names_dataset() -> SqlaTable: ...

    @staticmethod
    def create_user_with_roles(
        username: str,
        roles: Iterable[str],
        should_create_roles: bool = False,
    ) -> ab_models.User: ...

    @contextmanager
    def temporary_user(
        self,
        clone_user: Optional[ab_models.User] = None,
        username: Optional[str] = None,
        extra_roles: Optional[Iterable[ab_models.Role]] = None,
        extra_pvms: Optional[Iterable[Union[ab_models.PermissionView, tuple[Any, ...], list[Any]]]] = None,
        login: bool = False,
    ) -> Generator[ab_models.User, None, None]: ...

    @staticmethod
    def create_user(
        username: str,
        password: str,
        role_name: str,
        first_name: str = 'admin',
        last_name: str = 'user',
        email: str = 'admin@fab.org',
    ) -> ab_models.User: ...

    @staticmethod
    def get_user(username: str) -> Optional[ab_models.User]: ...

    @staticmethod
    def get_role(name: str) -> Optional[ab_models.Role]: ...

    @staticmethod
    def get_table_by_id(table_id: int) -> SqlaTable: ...

    @staticmethod
    def is_module_installed(module_name: str) -> bool: ...

    def get_or_create(self, cls: Type[T], criteria: dict[str, Any], **kwargs: Any) -> T: ...

    def login(self, username: str, password: str = DEFAULT_PASSWORD) -> Any: ...

    def get_slice(self, slice_name: str) -> Slice: ...

    @staticmethod
    def get_table(name: str, database_id: Optional[int] = None, schema: Optional[str] = None) -> SqlaTable: ...

    @staticmethod
    def get_database_by_id(db_id: int) -> Database: ...

    @staticmethod
    def get_database_by_name(database_name: str = 'main') -> Database: ...

    @staticmethod
    def get_datasource_mock() -> MagicMock: ...

    def get_resp(
        self,
        url: str,
        data: Optional[Any] = None,
        follow_redirects: bool = True,
        raise_on_error: bool = True,
        json_: Optional[Any] = None,
    ) -> str: ...

    def get_json_resp(
        self,
        url: str,
        data: Optional[Any] = None,
        follow_redirects: bool = True,
        raise_on_error: bool = True,
        json_: Optional[Any] = None,
    ) -> Any: ...

    def logout(self) -> None: ...

    def grant_public_access_to_table(self, table: SqlaTable) -> None: ...

    def grant_role_access_to_table(self, table: SqlaTable, role_name: str) -> None: ...

    def revoke_public_access_to_table(self, table: SqlaTable) -> None: ...

    def revoke_role_access_to_table(self, role_name: str, table: SqlaTable) -> None: ...

    def run_sql(
        self,
        sql: str,
        client_id: Optional[str] = None,
        username: Optional[str] = None,
        raise_on_error: bool = False,
        query_limit: Optional[int] = None,
        database_name: str = 'examples',
        sql_editor_id: Optional[int] = None,
        select_as_cta: bool = False,
        tmp_table_name: Optional[str] = None,
        schema: Optional[str] = None,
        ctas_method: CtasMethod = CtasMethod.TABLE,
        template_params: str = '{}',
    ) -> Any: ...

    def create_fake_db(self) -> Database: ...

    def delete_fake_db(self) -> None: ...

    def create_fake_db_for_macros(self) -> Database: ...

    @staticmethod
    def delete_fake_db_for_macros() -> None: ...

    def get_dash_by_slug(self, dash_slug: str) -> Optional[Dashboard]: ...

    def get_assert_metric(self, uri: str, func_name: str) -> Response: ...

    def delete_assert_metric(self, uri: str, func_name: str) -> Response: ...

    def post_assert_metric(self, uri: str, data: Any, func_name: str) -> Response: ...

    def put_assert_metric(self, uri: str, data: Any, func_name: str) -> Response: ...

    @classmethod
    def get_dttm(cls) -> datetime: ...

    def insert_dashboard(
        self,
        dashboard_title: str,
        slug: str,
        owners: Iterable[int],
        roles: Iterable[int] = [],
        created_by: Optional[int] = None,
        slices: Optional[Iterable[Slice]] = None,
        position_json: str = '',
        css: str = '',
        json_metadata: str = '',
        published: bool = False,
        certified_by: Optional[int] = None,
        certification_details: Optional[str] = None,
    ) -> Dashboard: ...

    def get_list(self, asset_type: str, filter: dict[str, Any] = {}, username: str = 'admin') -> Response: ...

@contextmanager
def db_insert_temp_object(obj: Any) -> Generator[Any, None, None]: ...