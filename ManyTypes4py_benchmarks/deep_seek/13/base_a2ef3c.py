from __future__ import annotations
import logging
import re
import warnings
from datetime import datetime
from re import Match, Pattern
from typing import Any, Callable, ContextManager, Dict, List, Optional, Set, Tuple, Type, Union, cast
from urllib.parse import urlencode, urljoin
from uuid import UUID, uuid4
import pandas as pd
import requests
import sqlparse
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from deprecation import deprecated
from flask import current_app, g
from flask_appbuilder.security.sqla.models import User
from flask_babel import gettext as __, lazy_gettext as _
from marshmallow import fields, Schema
from marshmallow.validate import Range
from sqlalchemy import column, select, types
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.interfaces import Compiled, Dialect
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import literal_column, quoted_name, text
from sqlalchemy.sql.expression import ColumnClause, Select, TextAsFrom, TextClause
from sqlalchemy.types import TypeEngine
from sqlparse.tokens import CTE
from superset import db, sql_parse
from superset.constants import QUERY_CANCEL_KEY, TimeGrain as TimeGrainConstants
from superset.databases.utils import get_table_metadata, make_url_safe
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.exceptions import DisallowedSQLFunction, OAuth2Error, OAuth2RedirectError
from superset.sql.parse import BaseSQLStatement, SQLScript, Table
from superset.sql_parse import ParsedQuery
from superset.superset_typing import OAuth2ClientConfig, OAuth2State, OAuth2TokenResponse, ResultSetColumnType, SQLAColumnType
from superset.utils import core as utils, json
from superset.utils.core import ColumnSpec, GenericDataType
from superset.utils.hashing import md5_sha_from_str
from superset.utils.json import redact_sensitive, reveal_sensitive
from superset.utils.network import is_hostname_valid, is_port_open
from superset.utils.oauth2 import encode_oauth2_state

if TYPE_CHECKING:
    from superset.connectors.sqla.models import TableColumn
    from superset.databases.schemas import TableMetadataResponse
    from superset.models.core import Database
    from superset.models.sql_lab import Query

ColumnTypeMapping = Tuple[Pattern[str], Union[TypeEngine, Callable[[Match[str]], TypeEngine], GenericDataType]
logger = logging.getLogger()
GenericDBException = Exception

def convert_inspector_columns(cols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result_set_columns = []
    for col in cols:
        result_set_columns.append({'column_name': col.get('name'), **col})
    return result_set_columns

class TimeGrain(NamedTuple):
    pass

builtin_time_grains = {
    TimeGrainConstants.SECOND: _('Second'),
    TimeGrainConstants.FIVE_SECONDS: _('5 second'),
    TimeGrainConstants.THIRTY_SECONDS: _('30 second'),
    TimeGrainConstants.MINUTE: _('Minute'),
    TimeGrainConstants.FIVE_MINUTES: _('5 minute'),
    TimeGrainConstants.TEN_MINUTES: _('10 minute'),
    TimeGrainConstants.FIFTEEN_MINUTES: _('15 minute'),
    TimeGrainConstants.THIRTY_MINUTES: _('30 minute'),
    TimeGrainConstants.HOUR: _('Hour'),
    TimeGrainConstants.SIX_HOURS: _('6 hour'),
    TimeGrainConstants.DAY: _('Day'),
    TimeGrainConstants.WEEK: _('Week'),
    TimeGrainConstants.MONTH: _('Month'),
    TimeGrainConstants.QUARTER: _('Quarter'),
    TimeGrainConstants.YEAR: _('Year'),
    TimeGrainConstants.WEEK_STARTING_SUNDAY: _('Week starting Sunday'),
    TimeGrainConstants.WEEK_STARTING_MONDAY: _('Week starting Monday'),
    TimeGrainConstants.WEEK_ENDING_SATURDAY: _('Week ending Saturday'),
    TimeGrainConstants.WEEK_ENDING_SUNDAY: _('Week ending Sunday'),
}

class TimestampExpression(ColumnClause):
    def __init__(self, expr: str, col: Any, **kwargs: Any) -> None:
        super().__init__(expr, **kwargs)
        self.col = col

    @property
    def _constructor(self) -> Type[ColumnClause]:
        return ColumnClause

@compiles(TimestampExpression)
def compile_timegrain_expression(element: TimestampExpression, compiler: Any, **kwargs: Any) -> str:
    return element.name.replace('{col}', compiler.process(element.col, **kwargs))

class LimitMethod:
    FETCH_MANY = 'fetch_many'
    WRAP_SQL = 'wrap_sql'
    FORCE_LIMIT = 'force_limit'

class MetricType(TypedDict, total=False):
    metric_name: str
    verbose_name: str
    metric_type: str
    expression: str

class BaseEngineSpec:
    engine_name: Optional[str] = None
    engine: str = 'base'
    engine_aliases: Set[str] = set()
    drivers: Dict[str, str] = {}
    default_driver: Optional[str] = None
    sqlalchemy_uri_placeholder: str = 'engine+driver://user:password@host:port/dbname[?key=value&key=value...]'
    disable_ssh_tunneling: bool = False
    _date_trunc_functions: Dict[str, str] = {}
    _time_grain_expressions: Dict[str, str] = {}
    _default_column_type_mappings: Tuple[ColumnTypeMapping, ...] = (
        (re.compile('^string', re.IGNORECASE), types.String(), GenericDataType.STRING),
        # ... (rest of the mappings)
    )
    column_type_mappings: Tuple[ColumnTypeMapping, ...] = ()
    column_type_mutators: Dict[Type[TypeEngine], Callable[[Any], Any]] = {}
    time_groupby_inline: bool = False
    limit_method: str = LimitMethod.FORCE_LIMIT
    supports_multivalues_insert: bool = False
    allows_joins: bool = True
    allows_subqueries: bool = True
    allows_alias_in_select: bool = True
    allows_alias_in_orderby: bool = True
    allows_sql_comments: bool = True
    allows_escaped_colons: bool = True
    allows_alias_to_source_column: bool = True
    allows_hidden_orderby_agg: bool = True
    allows_hidden_cc_in_orderby: bool = False
    allows_cte_in_subquery: bool = True
    cte_alias: str = '__cte'
    allow_limit_clause: bool = True
    select_keywords: Set[str] = {'SELECT'}
    top_keywords: Set[str] = {'TOP'}
    disallow_uri_query_params: Dict[str, Set[str]] = {}
    enforce_uri_query_params: Dict[str, Dict[str, str]] = {}
    force_column_alias_quotes: bool = False
    arraysize: int = 0
    max_column_name_length: Optional[int] = None
    try_remove_schema_from_table_name: bool = True
    run_multiple_statements_as_one: bool = False
    custom_errors: Dict[Pattern[str], Tuple[str, SupersetErrorType, Dict[str, Any]]] = {}
    encrypted_extra_sensitive_fields: Set[str] = {'$.*'}
    supports_file_upload: bool = True
    supports_dynamic_schema: bool = False
    supports_catalog: bool = False
    supports_dynamic_catalog: bool = False
    supports_oauth2: bool = False
    oauth2_scope: str = ''
    oauth2_authorization_request_uri: Optional[str] = None
    oauth2_token_request_uri: Optional[str] = None
    oauth2_token_request_type: str = 'data'
    oauth2_exception: Type[Exception] = OAuth2RedirectError
    has_query_id_before_execute: bool = True

    @classmethod
    def is_oauth2_enabled(cls) -> bool:
        return cls.supports_oauth2 and cls.engine_name in current_app.config['DATABASE_OAUTH2_CLIENTS']

    @classmethod
    def start_oauth2_dance(cls, database: Any) -> None:
        tab_id = str(uuid4())
        default_redirect_uri = url_for('DatabaseRestApi.oauth2', _external=True)
        state = {
            'database_id': database.id,
            'user_id': g.user.id,
            'default_redirect_uri': default_redirect_uri,
            'tab_id': tab_id,
        }
        oauth2_config = database.get_oauth2_config()
        if oauth2_config is None:
            raise OAuth2Error('No configuration found for OAuth2')
        oauth_url = cls.get_oauth2_authorization_uri(oauth2_config, state)
        raise OAuth2RedirectError(oauth_url, tab_id, default_redirect_uri)

    @classmethod
    def get_oauth2_config(cls) -> Optional[Dict[str, Any]]:
        oauth2_config = current_app.config['DATABASE_OAUTH2_CLIENTS']
        if cls.engine_name not in oauth2_config:
            return None
        db_engine_spec_config = oauth2_config[cls.engine_name]
        redirect_uri = current_app.config.get(
            'DATABASE_OAUTH2_REDIRECT_URI',
            url_for('DatabaseRestApi.oauth2', _external=True)
        )
        config = {
            'id': db_engine_spec_config['id'],
            'secret': db_engine_spec_config['secret'],
            'scope': db_engine_spec_config.get('scope') or cls.oauth2_scope,
            'redirect_uri': redirect_uri,
            'authorization_request_uri': db_engine_spec_config.get(
                'authorization_request_uri',
                cls.oauth2_authorization_request_uri
            ),
            'token_request_uri': db_engine_spec_config.get(
                'token_request_uri',
                cls.oauth2_token_request_uri
            ),
            'request_content_type': db_engine_spec_config.get(
                'request_content_type',
                cls.oauth2_token_request_type
            ),
        }
        return config

    @classmethod
    def get_oauth2_authorization_uri(cls, config: Dict[str, Any], state: Dict[str, Any]) -> str:
        uri = config['authorization_request_uri']
        params = {
            'scope': config['scope'],
            'access_type': 'offline',
            'include_granted_scopes': 'false',
            'response_type': 'code',
            'state': encode_oauth2_state(state),
            'redirect_uri': config['redirect_uri'],
            'client_id': config['id'],
            'prompt': 'consent',
        }
        return urljoin(uri, '?' + urlencode(params))

    @classmethod
    def get_oauth2_token(cls, config: Dict[str, Any], code: str) -> Dict[str, Any]:
        timeout = current_app.config['DATABASE_OAUTH2_TIMEOUT'].total_seconds()
        uri = config['token_request_uri']
        req_body = {
            'code': code,
            'client_id': config['id'],
            'client_secret': config['secret'],
            'redirect_uri': config['redirect_uri'],
            'grant_type': 'authorization_code',
        }
        if config['request_content_type'] == 'data':
            return requests.post(uri, data=req_body, timeout=timeout).json()
        return requests.post(uri, json=req_body, timeout=timeout).json()

    @classmethod
    def get_oauth2_fresh_token(cls, config: Dict[str, Any], refresh_token: str) -> Dict[str, Any]:
        timeout = current_app.config['DATABASE_OAUTH2_TIMEOUT'].total_seconds()
        uri = config['token_request_uri']
        req_body = {
            'client_id': config['id'],
            'client_secret': config['secret'],
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token',
        }
        if config['request_content_type'] == 'data':
            return requests.post(uri, data=req_body, timeout=timeout).json()
        return requests.post(uri, json=req_body, timeout=timeout).json()

    @classmethod
    def get_allows_alias_in_select(cls, database: Any) -> bool:
        return cls.allows_alias_in_select

    @classmethod
    def supports_url(cls, url: URL) -> bool:
        backend = url.get_backend_name()
        driver = url.get_driver_name()
        return cls.supports_backend(backend, driver)

    @classmethod
    def supports_backend(cls, backend: str, driver: Optional[str] = None) -> bool:
        if backend != cls.engine and backend not in cls.engine_aliases:
            return False
        if not cls.drivers or driver is None:
            return True
        return driver in cls.drivers

    @classmethod
    def get_default_catalog(cls, database: Any) -> Optional[str]:
        return None

    @classmethod
    def get_default_schema(cls, database: Any, catalog: Optional[str]) -> Optional[str]:
        with database.get_inspector(catalog=catalog) as inspector:
            return inspector.default_schema_name

    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: Dict[str, Any]) -> Optional[str]:
        return None

    @classmethod
    def get_default_schema_for_query(cls, database: Any, query: Any) -> Optional[str]:
        if cls.supports_dynamic_schema:
            return query.schema
        try:
            connect_args = database.get_extra()['engine_params']['connect_args']
        except KeyError:
            connect_args = {}
        sqlalchemy_uri = make_url_safe(database.sqlalchemy_uri)
        if (schema := cls.get_schema_from_engine_params(sqlalchemy_uri, connect_args)):
            return schema
        return cls.get_default_schema(database, query.catalog)

    @classmethod
    def get_dbapi_exception_mapping(cls) -> Dict[Type[Exception], Type[Exception]]:
        return {}

    @classmethod
    def parse_error_exception(cls, exception: Exception) -> Exception:
        return exception

    @classmethod
    def get_dbapi_mapped_exception(cls, exception: Exception) -> Exception:
        new_exception = cls.get_dbapi_exception_mapping().get(type(exception))
        if not new_exception:
            return cls.parse_error_exception(exception)
        return new_exception(str(exception))

    @classmethod
    def get_allow_cost_estimate(cls, extra: Dict[str, Any]) -> bool:
        return False

    @classmethod
    def get_text_clause(cls, clause: str) -> TextClause:
        if cls.allows_escaped_colons:
            clause = clause.replace(':', '\\:')
        return text(clause)

    @classmethod
    def get_engine(cls, database: Any, catalog: Optional[str] = None, schema: Optional[str] = None, source: Optional[str] = None) -> ContextManager[Engine]:
        return database.get_sqla_engine(catalog=catalog, schema=schema, source=source)

    @classmethod
    def get_timestamp_expr(cls, col: Any, pdf: str, time_grain: Optional[str]) -> TimestampExpression:
        if time_grain:
            type_ = str(getattr(col, 'type', ''))
            time_expr = cls.get_time_grain_expressions().get(time_grain)
            if not time_expr:
                raise NotImplementedError(f'No grain spec for {time_grain} for database {cls.engine}')
            if type_ and '{func}' in time_expr:
                date_trunc_function = cls._date_trunc_functions.get(type_)
                if date_trunc_function:
                    time_expr = time_expr.replace('{func}', date_trunc_function)
            if type_ and '{type}' in time_expr:
                date_trunc_function = cls._date_trunc_functions.get(type_)
                if date_trunc_function:
                    time_expr = time_expr.replace('{type}', type_)
        else:
            time_expr = '{col}'
        if pdf == 'epoch_s':
            time_expr = time_expr.replace('{col}', cls.epoch_to_dttm())
        elif pdf == 'epoch_ms':
            time_expr = time_expr.replace('{col}', cls.epoch_ms_to_dttm())
        return TimestampExpression(time_expr, col, type_=col.type)

    @classmethod
    def get_time_grains(cls) -> Tuple[TimeGrain, ...]:
        ret_list = []
        time_grains = builtin_time_grains.copy()
        time_grains.update(current_app.config['TIME_GRAIN_ADDONS'])
        for duration, func in cls.get_time_grain_expressions().items():
            if duration in time_grains:
                name = time_grains[duration]
                ret_list.append(TimeGrain(name, _(name), func, duration))
        return tuple(ret_list)

    @classmethod
    def _sort_time_grains(cls, val: Tuple[Optional[str], ...], index: int) -> int:
        pos = {'FIRST': 0, 'SECOND': 1, 'THIRD': 2, 'LAST': 3}
        if val[0] is None:
            return pos['FIRST']
        prog = re.compile('(.*\\/)?(P|PT)([0-9\\.]+)(S|M|H|D|W|M|Y)(\\/.*)?')
        result = prog.match(val[0