"""Unit tests for Superset"""
from datetime import datetime
from importlib.util import find_spec
from contextlib import contextmanager
from typing import Any, Union, Optional, List, Dict, Tuple, Type, TypeVar, cast, Callable, Iterator, Set
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

FAKE_DB_NAME = 'fake_db_100'
DEFAULT_PASSWORD = 'general'
test_client = app.test_client()

T = TypeVar('T')

def get_resp(client: Any, url: str, data: Optional[Dict[str, Any]] = None, follow_redirects: bool = True, 
             raise_on_error: bool = True, json_: Optional[Dict[str, Any]] = None) -> str:
    """Shortcut to get the parsed results while following redirects"""
    if data:
        resp = client.post(url, data=data, follow_redirects=follow_redirects)
    elif json_:
        resp = client.post(url, json=json_, follow_redirects=follow_redirects)
    else:
        resp = client.get(url, follow_redirects=follow_redirects)
    if raise_on_error and resp.status_code > 400:
        raise Exception(f'http request failed with code {resp.status_code}')
    return resp.data.decode('utf-8')

def post_assert_metric(client: Any, uri: str, data: Dict[str, Any], func_name: str) -> Response:
    """
    Simple client post with an extra assertion for statsd metrics

    :param client: test client for superset api requests
    :param uri: The URI to use for the HTTP POST
    :param data: The JSON data payload to be posted
    :param func_name: The function name that the HTTP POST triggers
    for the statsd metric assertion
    :return: HTTP Response
    """
    with patch.object(BaseSupersetModelRestApi, 'incr_stats', return_value=None) as mock_method:
        rv = client.post(uri, json=data)
    if 200 <= rv.status_code < 400:
        mock_method.assert_called_once_with('success', func_name)
    elif 400 <= rv.status_code < 500:
        mock_method.assert_called_once_with('warning', func_name)
    else:
        mock_method.assert_called_once_with('error', func_name)
    return rv

class SupersetTestCase(TestCase):
    default_schema_backend_map: Dict[str, str] = {'sqlite': 'main', 'mysql': 'superset', 'postgresql': 'public', 'presto': 'default', 'hive': 'default'}
    maxDiff: int = -1

    def tearDown(self) -> None:
        self.logout()

    def create_app(self) -> Any:
        return app

    @staticmethod
    def get_nonexistent_numeric_id(model: Type[T]) -> int:
        return (db.session.query(func.max(model.id)).scalar() or 0) + 1

    @staticmethod
    def get_birth_names_dataset() -> SqlaTable:
        return SupersetTestCase.get_table(name='birth_names')

    @staticmethod
    def create_user_with_roles(username: str, roles: List[str], should_create_roles: bool = False) -> ab_models.User:
        user_to_create = security_manager.find_user(username)
        if not user_to_create:
            security_manager.add_user(username, username, username, f'{username}@superset.com', security_manager.find_role('Gamma'), password=DEFAULT_PASSWORD)
            db.session.commit()
            user_to_create = security_manager.find_user(username)
            assert user_to_create
        user_to_create.roles = []
        for chosen_user_role in roles:
            if should_create_roles:
                security_manager.copy_role('Gamma', chosen_user_role, merge=False)
            user_to_create.roles.append(security_manager.find_role(chosen_user_role))
        db.session.commit()
        return user_to_create

    @contextmanager
    def temporary_user(self, clone_user: Optional[ab_models.User] = None, username: Optional[str] = None, 
                       extra_roles: Optional[List[ab_models.Role]] = None, extra_pvms: Optional[List[Any]] = None, 
                       login: bool = False) -> Iterator[ab_models.User]:
        """
        Create a temporary user for testing and delete it after the test

        with self.temporary_user(login=True, extra_roles=[Role(...)]) as user:
            user.do_something()

        # user is automatically logged out and deleted after the test
        """
        username = username or f'temp_user_{shortid()}'
        temp_user = ab_models.User(username=username, email=f'{username}@temp.com', active=True)
        if clone_user:
            temp_user.roles = clone_user.roles
            temp_user.first_name = clone_user.first_name
            temp_user.last_name = clone_user.last_name
            temp_user.password = clone_user.password
        else:
            temp_user.first_name = temp_user.last_name = username
        if clone_user:
            temp_user.roles = clone_user.roles
        if extra_roles:
            temp_user.roles.extend(extra_roles)
        pvms: List[Any] = []
        temp_role = None
        if extra_pvms:
            temp_role = ab_models.Role(name=f'tmp_role_{shortid()}')
            for pvm in extra_pvms:
                if isinstance(pvm, (tuple, list)):
                    pvms.append(security_manager.find_permission_view_menu(*pvm))
                else:
                    pvms.append(pvm)
            temp_role.permissions = pvms
            temp_user.roles.append(temp_role)
            db.session.add(temp_role)
            db.session.commit()
        db.session.add(temp_user)
        db.session.commit()
        previous_g_user = g.user if hasattr(g, 'user') else None
        try:
            if login:
                self.login(username=temp_user.username)
            else:
                g.user = temp_user
            yield temp_user
        finally:
            if temp_role:
                db.session.delete(temp_role)
            if login:
                self.logout()
            db.session.delete(temp_user)
            db.session.commit()
            g.user = previous_g_user

    @staticmethod
    def create_user(username: str, password: str, role_name: str, first_name: str = 'admin', 
                    last_name: str = 'user', email: str = 'admin@fab.org') -> ab_models.User:
        role_admin = security_manager.find_role(role_name)
        return security_manager.add_user(username, first_name, last_name, email, role_admin, password)

    @staticmethod
    def get_user(username: str) -> Optional[ab_models.User]:
        user = db.session.query(security_manager.user_model).filter_by(username=username).one_or_none()
        return user

    @staticmethod
    def get_role(name: str) -> Optional[ab_models.Role]:
        user = db.session.query(security_manager.role_model).filter_by(name=name).one_or_none()
        return user

    @staticmethod
    def get_table_by_id(table_id: int) -> SqlaTable:
        return db.session.query(SqlaTable).filter_by(id=table_id).one()

    @staticmethod
    def is_module_installed(module_name: str) -> bool:
        try:
            spec = find_spec(module_name)
            return spec is not None
        except (ModuleNotFoundError, ValueError, TypeError, ImportError):
            return False

    def get_or_create(self, cls: Type[T], criteria: Dict[str, Any], **kwargs: Any) -> T:
        obj = db.session.query(cls).filter_by(**criteria).first()
        if not obj:
            obj = cls(**criteria)
        obj.__dict__.update(**kwargs)
        db.session.add(obj)
        db.session.commit()
        return obj

    def login(self, username: str, password: str = DEFAULT_PASSWORD) -> Response:
        return login(self.client, username, password)

    def get_slice(self, slice_name: str) -> Slice:
        return db.session.query(Slice).filter_by(slice_name=slice_name).one()

    @staticmethod
    def get_table(name: str, database_id: Optional[int] = None, schema: Optional[str] = None) -> SqlaTable:
        schema = schema or get_example_default_schema()
        return db.session.query(SqlaTable).filter_by(database_id=database_id or SupersetTestCase.get_database_by_name('examples').id, schema=schema, table_name=name).one()

    @staticmethod
    def get_database_by_id(db_id: int) -> Database:
        return db.session.query(Database).filter_by(id=db_id).one()

    @staticmethod
    def get_database_by_name(database_name: str = 'main') -> Database:
        if database_name == 'examples':
            return get_example_database()
        else:
            raise ValueError("Database doesn't exist")

    @staticmethod
    def get_datasource_mock() -> MagicMock:
        datasource = MagicMock()
        results = Mock()
        results.query = Mock()
        results.status = Mock()
        results.error_message = None
        results.df = pd.DataFrame()
        datasource.type = 'table'
        datasource.query = Mock(return_value=results)
        mock_dttm_col = Mock()
        datasource.get_col = Mock(return_value=mock_dttm_col)
        datasource.query = Mock(return_value=results)
        datasource.database = Mock()
        datasource.database.db_engine_spec = Mock()
        datasource.database.perm = 'mock_database_perm'
        datasource.schema_perm = 'mock_schema_perm'
        datasource.perm = 'mock_datasource_perm'
        datasource.__class__ = SqlaTable
        datasource.database.db_engine_spec.mutate_expression_label = lambda x: x
        datasource.owners = MagicMock()
        datasource.id = 99999
        return datasource

    def get_resp(self, url: str, data: Optional[Dict[str, Any]] = None, follow_redirects: bool = True, 
                 raise_on_error: bool = True, json_: Optional[Dict[str, Any]] = None) -> str:
        return get_resp(self.client, url, data, follow_redirects, raise_on_error, json_)

    def get_json_resp(self, url: str, data: Optional[Dict[str, Any]] = None, follow_redirects: bool = True, 
                      raise_on_error: bool = True, json_: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Shortcut to get the parsed results while following redirects"""
        resp = self.get_resp(url, data, follow_redirects, raise_on_error, json_)
        return json.loads(resp)

    def logout(self) -> None:
        self.client.get('/logout/', follow_redirects=True)

    def grant_public_access_to_table(self, table: SqlaTable) -> None:
        role_name = 'Public'
        self.grant_role_access_to_table(table, role_name)

    def grant_role_access_to_table(self, table: SqlaTable, role_name: str) -> None:
        role = security_manager.find_role(role_name)
        perms = db.session.query(ab_models.PermissionView).all()
        for perm in perms:
            if perm.permission.name == 'datasource_access' and perm.view_menu and (table.perm in perm.view_menu.name):
                security_manager.add_permission_role(role, perm)

    def revoke_public_access_to_table(self, table: SqlaTable) -> None:
        role_name = 'Public'
        self.revoke_role_access_to_table(role_name, table)

    def revoke_role_access_to_table(self, role_name: str, table: SqlaTable) -> None:
        public_role = security_manager.find_role(role_name)
        perms = db.session.query(ab_models.PermissionView).all()
        for perm in perms:
            if perm.permission.name == 'datasource_access' and perm.view_menu and (table.perm in perm.view_menu.name):
                security_manager.del_permission_role(public_role, perm)

    def run_sql(self, sql: str, client_id: Optional[str] = None, username: Optional[str] = None, 
                raise_on_error: bool = False, query_limit: Optional[int] = None, database_name: str = 'examples', 
                sql_editor_id: Optional[str] = None, select_as_cta: bool = False, 
                tmp_table_name: Optional[str] = None, schema: Optional[str] = None, 
                ctas_method: CtasMethod = CtasMethod.TABLE, template_params: str = '{}') -> Dict[str, Any]:
        if username:
            self.logout()
            self.login(username)
        dbid = SupersetTestCase.get_database_by_name(database_name).id
        json_payload: Dict[str, Any] = {'database_id': dbid, 'sql': sql, 'client_id': client_id, 
                                        'queryLimit': query_limit, 'sql_editor_id': sql_editor_id, 
                                        'ctas_method': ctas_method, 'templateParams': template_params}
        if tmp_table_name:
            json_payload['tmp_table_name'] = tmp_table_name
        if select_as_cta:
            json_payload['select_as_cta'] = select_as_cta
        if schema:
            json_payload['schema'] = schema
        resp = self.get_json_resp('/api/v1/sqllab/execute/', raise_on_error=False, json_=json_payload)
        if username:
            self.logout()
        if raise_on_error and 'error' in resp:
            raise Exception('run_sql failed')
        return resp

    def create_fake_db(self) -> Database:
        database_name = FAKE_DB_NAME
        db_id = 100
        extra = '{\n            "schemas_allowed_for_file_upload":\n            ["this_schema_is_allowed", "this_schema_is_allowed_too"]\n        }'
        return self.get_or_create(cls=models.Database, criteria={'database_name': database_name}, sqlalchemy_uri='sqlite:///:memory:', id=db_id, extra=extra)

    def delete_fake_db(self) -> None:
        database = db.session.query(Database).filter(Database.database_name == FAKE_DB_NAME).scalar()
        if database:
            db.session.delete(database)
            db.session.commit()

    def create_fake_db_for_macros(self) -> Database:
        database_name = 'db_for_macros_testing'
        db_id = 200
        database = self.get_or_create(cls=models.Database, criteria={'database_name': database_name}, sqlalchemy_uri='db_for_macros_testing://user@host:8080/hive', id=db_id)

        def mock_get_dialect() -> Dialect:
            return dialect()
        database.get_dialect = mock_get_dialect
        return database

    @staticmethod
    def delete_fake_db_for_macros() -> None:
        database = db.session.query(Database).filter(Database.database_name == 'db_for_macros_testing').scalar()
        if database:
            db.session.delete(database)
            db.session.commit()

    def get_dash_by_slug(self, dash_slug: str) -> Optional[Dashboard]:
        return db.session.query(Dashboard).filter_by(slug=dash_slug).first()

    def get_assert_metric(self, uri: str, func_name: str) -> Response:
        """
        Simple client get with an extra assertion for statsd metrics

        :param uri: The URI to use for the HTTP GET
        :param func_name: The function name that the HTTP GET triggers
        for the statsd metric assertion
        :return: HTTP Response
        """
        with patch.object(BaseSupersetModelRestApi, 'incr_stats', return_value=None) as mock_method:
            rv = self.client.get(uri)
        if 200 <= rv.status_code < 400:
            mock_method.assert_called_once_with('success', func_name)
        elif 400 <= rv.status_code < 500:
            mock_method.assert_called_once_with('warning', func_name)
        else:
            mock_method.assert_called_once_with('error', func_name)
        return rv

    def delete_assert_metric(self, uri: str, func_name: str) -> Response:
        """
        Simple client delete with an extra assertion for statsd metrics

        :param uri: The URI to use for the HTTP DELETE
        :param func_name: The function name that the HTTP DELETE triggers
        for the statsd metric assertion
        :return: HTTP Response
        """
        with patch.object(BaseSupersetModelRestApi, 'incr_stats', return_value=None) as mock_method:
            rv = self.client.delete(uri)
        if 200 <= rv.status_code < 400:
            mock_method.assert_called_once_with('success', func_name)
        elif 400 <= rv.status_code < 500:
            mock_method.assert_called_once_with('warning', func_name)
        else:
            mock_method.assert_called_once_with('error', func_name)
        return rv

    def post_assert_metric(self, uri: str, data: Dict[str, Any], func_name: str) -> Response:
        return post_assert_metric(self.client, uri, data, func_name)

    def put_assert_metric(self, uri: str, data: Dict[str, Any], func_name: str) -> Response:
        """
        Simple client put with an extra assertion for statsd metrics

        :param uri: The URI to use for the HTTP PUT
        :param data: The JSON data payload to be posted
        :param func_name: The function name that the HTTP PUT triggers
        for the statsd metric assertion
        :return: HTTP Response
        """
        with patch.object(BaseSupersetModelRestApi, 'incr_stats', return_value=None) as mock_method:
            rv = self.client.put(uri, json=data)
        if 200 <= rv.status_code < 400:
            mock_method.assert_called_once_with('success', func_name)
        elif 400 <= rv.status_code < 500:
            mock_method.assert_called_once_with('warning', func_name)
        else:
            mock_method.assert_called_once_with('error', func_name)
        return rv

    @classmethod
    def get_dttm(cls) -> datetime:
        return datetime.strptime('2019-01-02 03:04:05.678900', '%Y-%m-%d %H:%M:%S.%f')

    def insert_dashboard(self, dashboard_title: str, slug: str, owners: List[int], roles: List[int] = [], 
                         created_by: Optional[int] = None, slices: Optional[List[Slice]] = None, 
                         position_json: str = '', css: str = '', json_metadata: str = '', 
                         published: bool = False, certified_by: Optional[str] = None, 
                         certification_details: Optional[str] = None) -> Dashboard:
        obj_owners: List[ab_models.User] = list()
        obj_roles: List[ab_models.Role] = list()
        slices = slices or []
        for owner in owners:
            user = db.session.query(security_manager.user_model).get(owner)
            obj_owners.append(user)
        for role in roles:
            role_obj = db.session.query(security_manager.role_model).get(role)
            obj_roles.append(role_obj)
        dashboard = Dashboard(dashboard_title=dashboard_title, slug=slug, owners=obj_owners, roles=obj_roles, 
                             position_json=position_json, css=css, json_metadata=json_metadata, slices=slices, 
                             published=published, created_by=created_by, certified_by=certified_by, 
                             certification_details=certification_details)
        db.session.add(dashboard)
        db.session.commit()
        return dashboard

    def get_list(self, asset_type: str, filter: Dict[str, Any] = {}, username: str = ADMIN_USERNAME) -> Response:
        """
        Get list of assets, by default using admin account. Can be filtered.
        """
        self.login(username)
        uri = f'api/v1/{asset_type}/?q={prison.dumps(filter)}'
        response = self.get_assert_metric(uri, 'get_list')
        return response

@contextmanager
def db_insert_temp_object(obj: Any) -> Iterator[Any]:
    """Insert a temporary object in database; delete when done."""
    try:
        db.session.add(obj)
        db.session.commit()
        yield obj
    finally:
        db.session.delete(obj)
        db.session.commit()
