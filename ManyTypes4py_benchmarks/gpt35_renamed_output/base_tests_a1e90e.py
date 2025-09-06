from datetime import datetime
from importlib.util import find_spec
from contextlib import contextmanager
from typing import Any, Union, Optional
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import prison
from flask import Response, g
from flask_appbuilder.security.sqla import models as ab_models
from flask_testing import TestCase
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import dialect
from tests.integration_tests.constants import ADMIN_USERNAME
from tests.integration_tests.test_app import app, login
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

FAKE_DB_NAME: str = 'fake_db_100'
DEFAULT_PASSWORD: str = 'general'
test_client = app.test_client()

def func_qrp8xmme(client: Any, url: str, data: Optional[dict] = None, follow_redirects: bool = True,
    raise_on_error: bool = True, json_: Optional[dict] = None) -> str:
    ...

def func_gfm8qvdk(client: Any, uri: str, data: dict, func_name: str) -> Response:
    ...

class SupersetTestCase(TestCase):
    default_schema_backend_map: dict = {'sqlite': 'main', 'mysql': 'superset',
        'postgresql': 'public', 'presto': 'default', 'hive': 'default'}
    maxDiff: int = -1

    def func_lgjvq24n(self) -> None:
        ...

    def func_pxv76r6a(self) -> Any:
        ...

    @staticmethod
    def func_va3hpslk(model: Any) -> int:
        ...

    @staticmethod
    def func_n3syhojf() -> SqlaTable:
        ...

    @staticmethod
    def func_486x0ydg(username: str, roles: list, should_create_roles: bool = False) -> ab_models.User:
        ...

    @contextmanager
    def func_63euqupl(self, clone_user: Optional[ab_models.User] = None, username: Optional[str] = None, extra_roles:
        Optional[list] = None, extra_pvms: Optional[list] = None, login: bool = False) -> ab_models.User:
        ...

    @staticmethod
    def func_j7rm3vgo(username: str, password: str, role_name: str, first_name: str = 'admin',
        last_name: str = 'user', email: str = 'admin@fab.org') -> ab_models.User:
        ...

    @staticmethod
    def func_zakrh315(username: str) -> Any:
        ...

    @staticmethod
    def func_03orbu2d(name: str) -> Any:
        ...

    @staticmethod
    def func_dy7rdlhx(table_id: int) -> SqlaTable:
        ...

    @staticmethod
    def func_2857aktd(module_name: str) -> bool:
        ...

    def func_39ta3sz1(self, cls: Any, criteria: dict, **kwargs) -> Any:
        ...

    def func_qs4bu81t(self, username: str, password: str = DEFAULT_PASSWORD) -> Any:
        ...

    def func_v2k7o352(self, slice_name: str) -> Slice:
        ...

    @staticmethod
    def func_p5bynajn(name: str, database_id: Optional[int] = None, schema: Optional[str] = None) -> SqlaTable:
        ...

    @staticmethod
    def func_ev36683v(db_id: int) -> Database:
        ...

    @staticmethod
    def func_ohireov9(database_name: str = 'main') -> Database:
        ...

    @staticmethod
    def func_p5ubh5f8() -> SqlaTable:
        ...

    def func_qrp8xmme(self, url: str, data: Optional[dict] = None, follow_redirects: bool = True,
        raise_on_error: bool = True, json_: Optional[dict] = None) -> str:
        ...

    def func_x5fs9be7(self, url: str, data: Optional[dict] = None, follow_redirects: bool = True,
        raise_on_error: bool = True, json_: Optional[dict] = None) -> dict:
        ...

    def func_4od4wikk(self) -> None:
        ...

    def func_zo6qbf9q(self, table: SqlaTable) -> None:
        ...

    def func_m3gdupz5(self, table: SqlaTable, role_name: str) -> None:
        ...

    def func_k91dafsv(self, table: SqlaTable) -> None:
        ...

    def func_0khkt52r(self, role_name: str, table: SqlaTable) -> None:
        ...

    def func_kmzo0kny(self, sql: str, client_id: Optional[str] = None, username: Optional[str] = None,
        raise_on_error: bool = False, query_limit: Optional[int] = None, database_name: str = 'examples',
        sql_editor_id: Optional[str] = None, select_as_cta: bool = False, tmp_table_name: Optional[str] = None,
        schema: Optional[str] = None, ctas_method: CtasMethod = CtasMethod.TABLE, template_params: str = '{}') -> dict:
        ...

    def func_0vbfv4pi(self) -> models.Database:
        ...

    def func_nizqqm93(self) -> None:
        ...

    def func_doixdm6w(self) -> models.Database:
        ...

    @staticmethod
    def func_7p24xdks() -> None:
        ...

    def func_lqp7mraq(self, dash_slug: str) -> Dashboard:
        ...

    def func_s6ikmvl1(self, uri: str, func_name: str) -> Response:
        ...

    def func_4xmbrt4d(self, uri: str, func_name: str) -> Response:
        ...

    def func_gfm8qvdk(self, uri: str, data: dict, func_name: str) -> Response:
        ...

    def func_ilm01yje(self, uri: str, data: dict, func_name: str) -> Response:
        ...

    @classmethod
    def func_oky6pfho(cls) -> datetime:
        ...

    def func_nqdedmpx(self, dashboard_title: str, slug: str, owners: list, roles: list = [],
        created_by: Optional[int] = None, slices: Optional[list] = None, position_json: str = '',
        css: str = '', json_metadata: str = '', published: bool = False, certified_by: Optional[int] = None,
        certification_details: Optional[str] = None) -> Dashboard:
        ...

    def func_ko4t9kyg(self, asset_type: str, filter: dict = {}, username: str = ADMIN_USERNAME) -> Any:
        ...

@contextmanager
def func_rq1rceii(obj: Any) -> Any:
    ...
