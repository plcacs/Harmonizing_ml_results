```python
from typing import Any, Optional, Union, Dict, List, Iterator, ContextManager, TypeVar, overload
from datetime import datetime
from contextlib import contextmanager
from unittest.mock import Mock, MagicMock
import pandas as pd
import prison
from flask import Response
from flask_appbuilder.security.sqla import models as ab_models
from flask_testing import TestCase
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import dialect
from superset.sql_parse import CtasMethod
from superset import db, security_manager
from superset.connectors.sqla.models import BaseDatasource, SqlaTable
from superset.models import core as models
from superset.models.slice import Slice
from superset.models.core import Database
from superset.models.dashboard import Dashboard
from superset.utils.core import get_example_default_schema, shortid
from superset.utils import json
from superset.utils.database import get_example_database
from superset.views.base_api import BaseSupersetModelRestApi

FAKE_DB_NAME: str = ...
DEFAULT_PASSWORD: str = ...
test_client: Any = ...

def get_resp(
    client: Any,
    url: str,
    data: Any = ...,
    follow_redirects: bool = ...,
    raise_on_error: bool = ...,
    json_: Any = ...,
) -> str: ...

def post_assert_metric(
    client: Any,
    uri: str,
    data: Any,
    func_name: str,
) -> Response: ...

class SupersetTestCase(TestCase):
    default_schema_backend_map: Dict[str, str] = ...
    maxDiff: int = ...

    def tearDown(self) -> None: ...
    def create_app(self) -> Any: ...
    @staticmethod
    def get_nonexistent_numeric_id(model: Any) -> int: ...
    @staticmethod
    def get_birth_names_dataset() -> SqlaTable: ...
    @staticmethod
    def create_user_with_roles(
        username: str,
        roles: List[str],
        should_create_roles: bool = ...,
    ) -> Any: ...
    @contextmanager
    def temporary_user(
        self,
        clone_user: Any = ...,
        username: Optional[str] = ...,
        extra_roles: Any = ...,
        extra_pvms: Any = ...,
        login: bool = ...,
    ) -> Iterator[Any]: ...
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
    def get_user(username: str) -> Any: ...
    @staticmethod
    def get_role(name: str) -> Any: ...
    @staticmethod
    def get_table_by_id(table_id: int) -> SqlaTable: ...
    @staticmethod
    def is_module_installed(module_name: str) -> bool: ...
    def get_or_create(
        self,
        cls: Any,
        criteria: Dict[str, Any],
        **kwargs: Any,
    ) -> Any: ...
    def login(
        self,
        username: str,
        password: str = ...,
    ) -> Any: ...
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
        data: Any = ...,
        follow_redirects: bool = ...,
        raise_on_error: bool = ...,
        json_: Any = ...,
    ) -> str: ...
    def get_json_resp(
        self,
        url: str,
        data: Any = ...,
        follow_redirects: bool = ...,
        raise_on_error: bool = ...,
        json_: Any = ...,
    ) -> Any: ...
    def logout(self) -> None: ...
    def grant_public_access_to_table(self, table: Any) -> None: ...
    def grant_role_access_to_table(self, table: Any, role_name: str) -> None: ...
    def revoke_public_access_to_table(self, table: Any) -> None: ...
    def revoke_role_access_to_table(self, role_name: str, table: Any) -> None: ...
    def run_sql(
        self,
        sql: str,
        client_id: Any = ...,
        username: Optional[str] = ...,
        raise_on_error: bool = ...,
        query_limit: Any = ...,
        database_name: str = ...,
        sql_editor_id: Any = ...,
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
    def post_assert_metric(self, uri: str, data: Any, func_name: str) -> Response: ...
    def put_assert_metric(self, uri: str, data: Any, func_name: str) -> Response: ...
    @classmethod
    def get_dttm(cls) -> datetime: ...
    def insert_dashboard(
        self,
        dashboard_title: str,
        slug: str,
        owners: List[int],
        roles: List[int] = ...,
        created_by: Any = ...,
        slices: Any = ...,
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
        filter: Dict[str, Any] = ...,
        username: str = ...,
    ) -> Response: ...

@contextmanager
def db_insert_temp_object(obj: Any) -> Iterator[Any]: ...
```