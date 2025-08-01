from __future__ import annotations
import contextlib
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Type
import requests
from flask import copy_current_request_context, ctx, current_app, Flask, g
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import NoSuchTableError
from superset import db
from superset.constants import QUERY_CANCEL_KEY, QUERY_EARLY_CANCEL_KEY, USER_AGENT
from superset.databases.utils import make_url_safe
from superset.db_engine_specs.base import BaseEngineSpec, convert_inspector_columns
from superset.db_engine_specs.exceptions import (
    SupersetDBAPIConnectionError,
    SupersetDBAPIDatabaseError,
    SupersetDBAPIOperationalError,
    SupersetDBAPIProgrammingError,
)
from superset.db_engine_specs.presto import PrestoBaseEngineSpec
from superset.models.sql_lab import Query
from superset.sql_parse import Table
from superset.superset_typing import ResultSetColumnType
from superset.utils import core as utils, json
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from superset.models.core import Database
    with contextlib.suppress(ImportError):
        from trino.dbapi import Cursor
logger = logging.getLogger(__name__)
try:
    from trino.exceptions import HttpError
except ImportError:
    HttpError = Exception

class CustomTrinoAuthErrorMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        logger.info('is this being called?')
        return isinstance(instance, HttpError) and "error 401: b'Invalid credentials'" in str(instance)

class TrinoAuthError(HttpError, metaclass=CustomTrinoAuthErrorMeta):
    pass

class TrinoEngineSpec(PrestoBaseEngineSpec):
    engine: str = 'trino'
    engine_name: str = 'Trino'
    allows_alias_to_source_column: bool = False
    supports_oauth2: bool = True
    oauth2_exception: Type[Exception] = TrinoAuthError
    oauth2_token_request_type: str = 'data'

    @classmethod
    def get_extra_table_metadata(cls, database: Database, table: Table) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if (indexes := database.get_indexes(table)):
            col_names, latest_parts = cls.latest_partition(database, table, show_first=True, indexes=indexes)
            if not latest_parts:
                latest_parts = tuple([None] * len(col_names))
            metadata['partitions'] = {
                'cols': sorted(list({column_name for index in indexes if index.get('name') == 'partition' for column_name in index.get('column_names', [])})),
                'latest': dict(zip(col_names, latest_parts, strict=False)),
                'partitionQuery': cls._partition_query(table=table, indexes=indexes, database=database)
            }
        if database.has_view(Table(table.table, table.schema)):
            with database.get_inspector(catalog=table.catalog, schema=table.schema) as inspector:
                metadata['view'] = inspector.get_view_definition(table.table, table.schema)
        return metadata

    @classmethod
    def update_impersonation_config(
        cls,
        database: Database,
        connect_args: Dict[str, Any],
        uri: str,
        username: Optional[str],
        access_token: Optional[str],
    ) -> None:
        url = make_url_safe(uri)
        backend_name = url.get_backend_name()
        if backend_name == 'trino' and username is not None:
            connect_args['user'] = username
            if access_token is not None:
                http_session = requests.Session()
                http_session.headers.update({'Authorization': f'Bearer {access_token}'})
                connect_args['http_session'] = http_session

    @classmethod
    def get_url_for_impersonation(
        cls,
        url: URL,
        impersonate_user: bool,
        username: Optional[str],
        access_token: Optional[str],
    ) -> URL:
        return url

    @classmethod
    def get_allow_cost_estimate(cls, extra: Dict[str, Any]) -> bool:
        return True

    @classmethod
    def get_tracking_url(cls, cursor: Any) -> Optional[str]:
        try:
            return cursor.info_uri
        except AttributeError:
            with contextlib.suppress(AttributeError):
                conn = cursor.connection
                return f'{conn.http_scheme}://{conn.host}:{conn.port}/ui/query.html?{cursor._query.query_id}'
        return None

    @classmethod
    def handle_cursor(cls, cursor: Any, query: Query) -> None:
        cancel_query_id: str = cursor.query_id
        logger.debug('Query %d: queryId %s found in cursor', query.id, cancel_query_id)
        query.set_extra_json_key(key=QUERY_CANCEL_KEY, value=cancel_query_id)
        if (tracking_url := cls.get_tracking_url(cursor)):
            query.tracking_url = tracking_url
        db.session.commit()
        if query.extra.get(QUERY_EARLY_CANCEL_KEY):
            cls.cancel_query(cursor=cursor, query=query, cancel_query_id=cancel_query_id)
        super().handle_cursor(cursor=cursor, query=query)

    @classmethod
    def execute_with_cursor(cls, cursor: Any, sql: str, query: Query) -> None:
        query_id: int = query.id
        query_database: Database = query.database
        execute_result: Dict[str, Any] = {}
        execute_event: threading.Event = threading.Event()

        @copy_current_request_context
        def _execute(results: Dict[str, Any], event: threading.Event, app: Flask, g_copy: Any) -> None:
            logger.debug('Query %d: Running query: %s', query_id, sql)
            try:
                with app.app_context():
                    for key, value in g_copy.__dict__.items():
                        setattr(g, key, value)
                    cls.execute(cursor, sql, query_database)
            except Exception as ex:
                results['error'] = ex
            finally:
                event.set()
        execute_thread: threading.Thread = threading.Thread(
            target=_execute,
            args=(execute_result, execute_event, current_app._get_current_object(), g._get_current_object()),
        )
        execute_thread.start()
        time.sleep(0.1)
        while not cursor.query_id and (not execute_event.is_set()):
            time.sleep(0.1)
        logger.debug('Query %d: Handling cursor', query_id)
        cls.handle_cursor(cursor, query)
        logger.debug('Query %d: Waiting for query to complete', query_id)
        execute_event.wait()
        if (err := execute_result.get('error')):
            raise err

    @classmethod
    def prepare_cancel_query(cls, query: Query) -> None:
        if QUERY_CANCEL_KEY not in query.extra:
            query.set_extra_json_key(QUERY_EARLY_CANCEL_KEY, True)
            db.session.commit()

    @classmethod
    def cancel_query(cls, cursor: Any, query: Query, cancel_query_id: str) -> bool:
        try:
            cursor.execute(f"CALL system.runtime.kill_query(query_id => '{cancel_query_id}',message => 'Query cancelled by Superset')")
            cursor.fetchall()
        except Exception:
            return False
        return True

    @staticmethod
    def get_extra_params(database: Database) -> Dict[str, Any]:
        extra: Dict[str, Any] = BaseEngineSpec.get_extra_params(database)
        engine_params: Dict[str, Any] = extra.setdefault('engine_params', {})
        connect_args: Dict[str, Any] = engine_params.setdefault('connect_args', {})
        connect_args.setdefault('source', USER_AGENT)
        if database.server_cert:
            connect_args['http_scheme'] = 'https'
            connect_args['verify'] = utils.create_ssl_cert_file(database.server_cert)
        return extra

    @staticmethod
    def update_params_from_encrypted_extra(database: Database, params: Dict[str, Any]) -> None:
        if not database.encrypted_extra:
            return
        try:
            encrypted_extra: Dict[str, Any] = json.loads(database.encrypted_extra)
            auth_method: Optional[str] = encrypted_extra.pop('auth_method', None)
            auth_params: Dict[str, Any] = encrypted_extra.pop('auth_params', {})
            if not auth_method:
                return
            connect_args: Dict[str, Any] = params.setdefault('connect_args', {})
            connect_args['http_scheme'] = 'https'
            if auth_method == 'basic':
                from trino.auth import BasicAuthentication as trino_auth  # type: ignore
            elif auth_method == 'kerberos':
                from trino.auth import KerberosAuthentication as trino_auth  # type: ignore
            elif auth_method == 'certificate':
                from trino.auth import CertificateAuthentication as trino_auth  # type: ignore
            elif auth_method == 'jwt':
                from trino.auth import JWTAuthentication as trino_auth  # type: ignore
            else:
                allowed_extra_auths: Dict[str, Any] = current_app.config['ALLOWED_EXTRA_AUTHENTICATIONS'].get('trino', {})
                if auth_method in allowed_extra_auths:
                    trino_auth = allowed_extra_auths.get(auth_method)
                else:
                    raise ValueError(
                        f"For security reason, custom authentication '{auth_method}' must be listed in 'ALLOWED_EXTRA_AUTHENTICATIONS' config"
                    )
            connect_args['auth'] = trino_auth(**auth_params)
        except json.JSONDecodeError as ex:
            logger.error(ex, exc_info=True)
            raise

    @classmethod
    def get_dbapi_exception_mapping(cls) -> Dict[Type[Exception], Type[Exception]]:
        from requests import exceptions as requests_exceptions
        from trino import exceptions as trino_exceptions
        static_mapping: Dict[Type[Exception], Type[Exception]] = {
            requests_exceptions.ConnectionError: SupersetDBAPIConnectionError
        }

        class _CustomMapping(dict[Type[Exception], Type[Exception]]):
            def get(self, item: Type[Exception], default: Optional[Type[Exception]] = None) -> Optional[Type[Exception]]:
                if (static := static_mapping.get(item)):
                    return static
                if issubclass(item, trino_exceptions.InternalError):
                    return SupersetDBAPIDatabaseError
                if issubclass(item, trino_exceptions.OperationalError):
                    return SupersetDBAPIOperationalError
                if issubclass(item, trino_exceptions.ProgrammingError):
                    return SupersetDBAPIProgrammingError
                return default
        return _CustomMapping()

    @classmethod
    def _expand_columns(cls, col: ResultSetColumnType) -> List[ResultSetColumnType]:
        from trino.sqlalchemy import datatype
        cols: List[ResultSetColumnType] = [col]
        col_type: Any = col.get('type')
        if not isinstance(col_type, datatype.ROW):
            return cols
        for inner_name, inner_type in col_type.attr_types:
            outer_name: str = col['name']
            name: str = '.'.join([outer_name, inner_name])
            query_name: str = '.'.join([f'"{piece}"' for piece in name.split('.')])
            column_spec = cls.get_column_spec(str(inner_type))
            is_dttm: bool = column_spec.is_dttm if column_spec else False
            inner_col: ResultSetColumnType = ResultSetColumnType(
                name=name,
                column_name=name,
                type=inner_type,
                is_dttm=is_dttm,
                query_as=f'{query_name} AS "{name}"',
            )
            cols.extend(cls._expand_columns(inner_col))
        return cols

    @classmethod
    def get_columns(
        cls, inspector: Inspector, table: Table, options: Optional[Dict[str, Any]] = None
    ) -> List[ResultSetColumnType]:
        try:
            sqla_columns = inspector.get_columns(table.table, table.schema)
            base_cols: List[ResultSetColumnType] = convert_inspector_columns(sqla_columns)
        except NoSuchTableError:
            base_cols = super().get_columns(inspector, table, options)
        if not (options or {}).get('expand_rows'):
            return base_cols
        return [col for base in base_cols for col in cls._expand_columns(base)]

    @classmethod
    def get_indexes(cls, database: Database, inspector: Inspector, table: Table) -> List[Dict[str, Any]]:
        try:
            return super().get_indexes(database, inspector, table)
        except NoSuchTableError:
            return []