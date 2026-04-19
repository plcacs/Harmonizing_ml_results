from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeVar, Union
from flask import Response
from flask.testing import FlaskClient
from flask_appbuilder.security.sqla import models as ab_models
from flask_testing import TestCase
from superset.connectors.sqla.models import BaseDatasource, SqlaTable
from superset.models.core import Database
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.sql_parse import CtasMethod
from tests.integration_tests.constants import ADMIN_USERNAME

FAKE_DB_NAME: str
DEFAULT_PASSWORD: str
test_client: FlaskClient

def get_resp(
    client: FlaskClient,
    url: str,
    data: Optional[Dict[str, Any]] = ...,
    follow_redirects: bool = ...,
    raise_on_error: bool = ...,
    json_: Optional[Dict[str, Any]] = ...,
) -> str: ...
def post_assert_metric(client: FlaskClient, uri: str, data: Dict[str, Any], func_name: str) -> Response: ...

C = TypeVar("C")
T = TypeVar("T")

class SupersetTestCase(TestCase):
    default_schema_backend_map: Dict[str, str]
    maxDiff: int
    def tearDown(self) -> None: ...
    def create_app(self) -> Any: ...
    @staticmethod
    def get_nonexistent_numeric_id(model: Any) -> int: ...
    @staticmethod
    def get_birth_names_dataset() -> SqlaTable: ...
    @staticmethod
    def create_user_with_roles(username: str, roles: List[str], should_create_roles: bool = ...) -> ab_models.User: ...
    @contextmanager
    def temporary_user(
        self,
        clone_user: Optional[ab_models.User] = ...,
        username: Optional[str] = ...,
        extra_roles: Optional[List[ab_models.Role]] = ...,
        extra_pvms: Optional[List[Union[ab_models.PermissionView, Tuple[str, str]]]] = ...,
        login: bool = ...,
    ) -> Iterator[ab_models.User]: ...
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
    def get_or_create(self, cls: type[C], criteria: Dict[str, Any], **kwargs: Any) -> C: ...
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
        data: Optional[Dict[str, Any]] = ...,
        follow_redirects: bool = ...,
        raise_on_error: bool = ...,
        json_: Optional[Dict[str, Any]] = ...,
    ) -> str: ...
    def get_json_resp(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = ...,
        follow_redirects: bool = ...,
        raise_on_error: bool = ...,
        json_: Optional[Dict[str, Any]] = ...,
    ) -> Any: ...
    def logout(self) -> None: ...
    def grant_public_access_to_table(self, table: BaseDatasource) -> None: ...
    def grant_role_access_to_table(self, table: BaseDatasource, role_name: str) -> None: ...
    def revoke_public_access_to_table(self, table: BaseDatasource) -> None: ...
    def revoke_role_access_to_table(self, role_name: str, table: BaseDatasource) -> None: ...
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
    ) -> Dict[str, Any]: ...
    def create_fake_db(self) -> Database: ...
    def delete_fake_db(self) -> None: ...
    def create_fake_db_for_macros(self) -> Database: ...
    @staticmethod
    def delete_fake_db_for_macros() -> None: ...
    def get_dash_by_slug(self, dash_slug: str) -> Optional[Dashboard]: ...
    def get_assert_metric(self, uri: str, func_name: str) -> Response: ...
    def delete_assert_metric(self, uri: str, func_name: str) -> Response: ...
    def post_assert_metric(self, uri: str, data: Dict[str, Any], func_name: str) -> Response: ...
    def put_assert_metric(self, uri: str, data: Dict[str, Any], func_name: str) -> Response: ...
    @classmethod
    def get_dttm(cls) -> datetime: ...
    def insert_dashboard(
        self,
        dashboard_title: str,
        slug: str,
        owners: List[int],
        roles: List[int] = ...,
        created_by: Optional[ab_models.User] = ...,
        slices: Optional[List[Slice]] = ...,
        position_json: str = ...,
        css: str = ...,
        json_metadata: str = ...,
        published: bool = ...,
        certified_by: Optional[str] = ...,
        certification_details: Optional[str] = ...,
    ) -> Dashboard: ...
    def get_list(self, asset_type: str, filter: Dict[str, Any] = ..., username: str = ADMIN_USERNAME) -> Response: ...

@contextmanager
def db_insert_temp_object(obj: T) -> Iterator[T]: ...