from __future__ import annotations
import logging
import re
import warnings
from datetime import datetime
from re import Match, Pattern
from typing import Any, Callable, cast, ContextManager, NamedTuple, TYPE_CHECKING, TypedDict, Union, Optional, List, Dict, Tuple
from urllib.parse import urlencode, urljoin
from uuid import uuid4

import pandas as pd
import requests
import sqlparse
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from deprecation import deprecated
from flask import current_app, g, url_for
from flask_appbuilder.security.sqla.models import User
from flask_babel import gettext as __, lazy_gettext as _
from marshmallow import fields, Schema
from marshmallow.validate import Range
from sqlalchemy import column, select, types, text
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.interfaces import Compiled, Dialect
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import literal_column, quoted_name
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

ColumnTypeMapping = tuple[Pattern[str], Union[TypeEngine, Callable[[Match[str]], TypeEngine]], GenericDataType]

logger = logging.getLogger()
GenericDBException = Exception


def convert_inspector_columns(cols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result_set_columns: List[Dict[str, Any]] = []
    for col in cols:
        result_set_columns.append({'column_name': col.get('name'), **col})
    return result_set_columns


class TimeGrain(NamedTuple):
    # Define fields as needed
    pass


builtin_time_grains: Dict[str, str] = {
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
        """
        Sqlalchemy class that can be used to render native column elements respecting
        engine-specific quoting rules as part of a string-based expression.

        :param expr: Sql expression with '{col}' denoting the locations where the col
        object will be rendered.
        :param col: the target column
        """
        super().__init__(expr, **kwargs)
        self.col = col

    @property
    def _constructor(self) -> Callable[..., ColumnClause]:
        return ColumnClause


@compiles(TimestampExpression)
def compile_timegrain_expression(element: TimestampExpression, compiler: Any, **kwargs: Any) -> str:
    return element.name.replace('{col}', compiler.process(element.col, **kwargs))


class LimitMethod:
    """Enum the ways that limits can be applied"""
    FETCH_MANY: str = 'fetch_many'
    WRAP_SQL: str = 'wrap_sql'
    FORCE_LIMIT: str = 'force_limit'


class MetricType(TypedDict, total=False):
    """
    Type for metrics return by `get_metrics`.
    """
    # Define metric fields as needed
    pass


class BaseEngineSpec:
    """
    Abstract class for database engine specific configurations
    """
    engine_name: Optional[str] = None
    engine: str = 'base'
    engine_aliases: set[str] = set()
    drivers: Dict[str, Any] = {}
    default_driver: Optional[str] = None
    sqlalchemy_uri_placeholder: str = 'engine+driver://user:password@host:port/dbname[?key=value&key=value...]'
    disable_ssh_tunneling: bool = False
    _date_trunc_functions: Dict[str, str] = {}
    _time_grain_expressions: Dict[str, str] = {}
    _default_column_type_mappings: Tuple[Tuple[Pattern[str], Any, GenericDataType], ...] = (
        (re.compile('^string', re.IGNORECASE), types.String(), GenericDataType.STRING),
        (re.compile('^n((var)?char|text)', re.IGNORECASE), types.UnicodeText(), GenericDataType.STRING),
        (re.compile('^(var)?char', re.IGNORECASE), types.String(), GenericDataType.STRING),
        (re.compile('^(tiny|medium|long)?text', re.IGNORECASE), types.String(), GenericDataType.STRING),
        (re.compile('^smallint', re.IGNORECASE), types.SmallInteger(), GenericDataType.NUMERIC),
        (re.compile('^int(eger)?', re.IGNORECASE), types.Integer(), GenericDataType.NUMERIC),
        (re.compile('^bigint', re.IGNORECASE), types.BigInteger(), GenericDataType.NUMERIC),
        (re.compile('^long', re.IGNORECASE), types.Float(), GenericDataType.NUMERIC),
        (re.compile('^decimal', re.IGNORECASE), types.Numeric(), GenericDataType.NUMERIC),
        (re.compile('^numeric', re.IGNORECASE), types.Numeric(), GenericDataType.NUMERIC),
        (re.compile('^float', re.IGNORECASE), types.Float(), GenericDataType.NUMERIC),
        (re.compile('^double', re.IGNORECASE), types.Float(), GenericDataType.NUMERIC),
        (re.compile('^real', re.IGNORECASE), types.REAL, GenericDataType.NUMERIC),
        (re.compile('^smallserial', re.IGNORECASE), types.SmallInteger(), GenericDataType.NUMERIC),
        (re.compile('^serial', re.IGNORECASE), types.Integer(), GenericDataType.NUMERIC),
        (re.compile('^bigserial', re.IGNORECASE), types.BigInteger(), GenericDataType.NUMERIC),
        (re.compile('^money', re.IGNORECASE), types.Numeric(), GenericDataType.NUMERIC),
        (re.compile('^timestamp', re.IGNORECASE), types.TIMESTAMP(), GenericDataType.TEMPORAL),
        (re.compile('^datetime', re.IGNORECASE), types.DateTime(), GenericDataType.TEMPORAL),
        (re.compile('^date', re.IGNORECASE), types.Date(), GenericDataType.TEMPORAL),
        (re.compile('^time', re.IGNORECASE), types.Time(), GenericDataType.TEMPORAL),
        (re.compile('^interval', re.IGNORECASE), types.Interval(), GenericDataType.TEMPORAL),
        (re.compile('^bool(ean)?', re.IGNORECASE), types.Boolean(), GenericDataType.BOOLEAN)
    )
    column_type_mappings: Tuple[Any, ...] = ()
    column_type_mutators: Dict[Any, Callable[[Any], Any]] = {}
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
    select_keywords: set[str] = {'SELECT'}
    top_keywords: set[str] = {'TOP'}
    disallow_uri_query_params: Dict[str, set[str]] = {}
    enforce_uri_query_params: Dict[str, Dict[str, Any]] = {}
    force_column_alias_quotes: bool = False
    arraysize: int = 0
    max_column_name_length: Optional[int] = None
    try_remove_schema_from_table_name: bool = True
    run_multiple_statements_as_one: bool = False
    custom_errors: Dict[Any, Tuple[str, Any, Dict[str, Any]]] = {}
    encrypted_extra_sensitive_fields: set[str] = {'$.*'}
    supports_file_upload: bool = True
    supports_dynamic_schema: bool = False
    supports_catalog: bool = False
    supports_dynamic_catalog: bool = False
    supports_oauth2: bool = False
    oauth2_scope: str = ''
    oauth2_authorization_request_uri: Optional[str] = None
    oauth2_token_request_uri: Optional[str] = None
    oauth2_token_request_type: str = 'data'
    oauth2_exception: Any = OAuth2RedirectError
    has_query_id_before_execute: bool = True

    @classmethod
    def is_oauth2_enabled(cls) -> bool:
        return cls.supports_oauth2 and cls.engine_name in current_app.config['DATABASE_OAUTH2_CLIENTS']

    @classmethod
    def start_oauth2_dance(cls, database: Any) -> None:
        tab_id: str = str(uuid4())
        default_redirect_uri: str = url_for('DatabaseRestApi.oauth2', _external=True)
        state: Dict[str, Any] = {
            'database_id': database.id,
            'user_id': g.user.id,
            'default_redirect_uri': default_redirect_uri,
            'tab_id': tab_id,
        }
        oauth2_config = database.get_oauth2_config()
        if oauth2_config is None:
            raise OAuth2Error('No configuration found for OAuth2')
        oauth_url: str = cls.get_oauth2_authorization_uri(oauth2_config, state)
        raise OAuth2RedirectError(oauth_url, tab_id, default_redirect_uri)

    @classmethod
    def get_oauth2_config(cls) -> Optional[Dict[str, Any]]:
        oauth2_config: Dict[str, Any] = current_app.config['DATABASE_OAUTH2_CLIENTS']
        if cls.engine_name not in oauth2_config:
            return None
        db_engine_spec_config: Dict[str, Any] = oauth2_config[cls.engine_name]
        redirect_uri: str = current_app.config.get('DATABASE_OAUTH2_REDIRECT_URI', url_for('DatabaseRestApi.oauth2', _external=True))
        config: Dict[str, Any] = {
            'id': db_engine_spec_config['id'],
            'secret': db_engine_spec_config['secret'],
            'scope': db_engine_spec_config.get('scope') or cls.oauth2_scope,
            'redirect_uri': redirect_uri,
            'authorization_request_uri': db_engine_spec_config.get('authorization_request_uri', cls.oauth2_authorization_request_uri),
            'token_request_uri': db_engine_spec_config.get('token_request_uri', cls.oauth2_token_request_uri),
            'request_content_type': db_engine_spec_config.get('request_content_type', cls.oauth2_token_request_type),
        }
        return config

    @classmethod
    def get_oauth2_authorization_uri(cls, config: Dict[str, Any], state: Dict[str, Any]) -> str:
        uri: str = config['authorization_request_uri']
        params: Dict[str, Any] = {
            'scope': config['scope'],
            'access_type': 'offline',
            'include_granted_scopes': 'false',
            'response_type': 'code',
            'state': encode_oauth2_state(state),
            'redirect_uri': config['redirect_uri'],
            'client_id': config['id'],
            'prompt': 'consent'
        }
        return urljoin(uri, '?' + urlencode(params))

    @classmethod
    def get_oauth2_token(cls, config: Dict[str, Any], code: str) -> Any:
        timeout: float = current_app.config['DATABASE_OAUTH2_TIMEOUT'].total_seconds()
        uri: str = config['token_request_uri']
        req_body: Dict[str, Any] = {
            'code': code,
            'client_id': config['id'],
            'client_secret': config['secret'],
            'redirect_uri': config['redirect_uri'],
            'grant_type': 'authorization_code'
        }
        if config['request_content_type'] == 'data':
            return requests.post(uri, data=req_body, timeout=timeout).json()
        return requests.post(uri, json=req_body, timeout=timeout).json()

    @classmethod
    def get_oauth2_fresh_token(cls, config: Dict[str, Any], refresh_token: str) -> Any:
        timeout: float = current_app.config['DATABASE_OAUTH2_TIMEOUT'].total_seconds()
        uri: str = config['token_request_uri']
        req_body: Dict[str, Any] = {
            'client_id': config['id'],
            'client_secret': config['secret'],
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token'
        }
        if config['request_content_type'] == 'data':
            return requests.post(uri, data=req_body, timeout=timeout).json()
        return requests.post(uri, json=req_body, timeout=timeout).json()

    @classmethod
    def get_allows_alias_in_select(cls, database: Any) -> bool:
        return cls.allows_alias_in_select

    @classmethod
    def supports_url(cls, url: URL) -> bool:
        backend: str = url.get_backend_name()
        driver: str = url.get_driver_name()
        return cls.supports_backend(backend, driver)

    @classmethod
    def supports_backend(cls, backend: str, driver: Optional[str] = None) -> bool:
        if backend != cls.engine and backend not in cls.engine_aliases:
            return False
        if not cls.drivers or driver is None:
            return True
        return driver in cls.drivers

    @classmethod
    def get_default_catalog(cls, database: Any) -> Any:
        return None

    @classmethod
    def get_default_schema(cls, database: Any, catalog: Any) -> Optional[str]:
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
    def get_dbapi_exception_mapping(cls) -> Dict[Any, Any]:
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
    def get_allow_cost_estimate(cls, extra: Any) -> bool:
        return False

    @classmethod
    def get_text_clause(cls, clause: str) -> TextClause:
        if cls.allows_escaped_colons:
            clause = clause.replace(':', '\\:')
        return text(clause)

    @classmethod
    def get_engine(cls, database: Any, catalog: Optional[Any] = None, schema: Optional[Any] = None, source: Optional[str] = None) -> ContextManager[Engine]:
        return database.get_sqla_engine(catalog=catalog, schema=schema, source=source)

    @classmethod
    def get_timestamp_expr(cls, col: Any, pdf: str, time_grain: Optional[str]) -> TimestampExpression:
        if time_grain:
            type_ = str(getattr(col, 'type', ''))
            time_expr: Optional[str] = cls.get_time_grain_expressions().get(time_grain)
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
        ret_list: List[TimeGrain] = []
        time_grains: Dict[str, str] = builtin_time_grains.copy()
        time_grains.update(current_app.config['TIME_GRAIN_ADDONS'])
        for duration, func in cls.get_time_grain_expressions().items():
            if duration in time_grains:
                name: str = time_grains[duration]
                ret_list.append(TimeGrain(name, _(name), func, duration))
        return tuple(ret_list)

    @classmethod
    def _sort_time_grains(cls, val: Tuple[Optional[str], ...], index: int) -> Any:
        pos: Dict[str, int] = {'FIRST': 0, 'SECOND': 1, 'THIRD': 2, 'LAST': 3}
        if val[0] is None:
            return pos['FIRST']
        prog = re.compile('(.*\\/)?(P|PT)([0-9\\.]+)(S|M|H|D|W|M|Y)(\\/.*)?')
        result = prog.match(val[0])
        if result is None:
            return pos['LAST']
        second_minute_hour = ['S', 'M', 'H']
        day_week_month_year = ['D', 'W', 'M', 'Y']
        is_less_than_day = result.group(2) == 'PT'
        interval = result.group(4)
        epoch_time_start_string = result.group(1) or result.group(5)
        has_starting_or_ending = bool(len(epoch_time_start_string or ''))

        def sort_day_week() -> int:
            if has_starting_or_ending:
                return pos['LAST']
            if is_less_than_day:
                return pos['SECOND']
            return pos['THIRD']

        def sort_interval() -> int:
            if is_less_than_day:
                return second_minute_hour.index(interval)
            return day_week_month_year.index(interval)
        plist: Dict[int, Any] = {0: sort_day_week(), 1: pos['SECOND'] if is_less_than_day else pos['THIRD'], 2: sort_interval(), 3: float(result.group(3))}
        return plist.get(index, 0)

    @classmethod
    def get_time_grain_expressions(cls) -> Dict[str, str]:
        time_grain_expressions: Dict[str, str] = cls._time_grain_expressions.copy()
        grain_addon_expressions: Dict[str, Dict[str, str]] = current_app.config['TIME_GRAIN_ADDON_EXPRESSIONS']
        time_grain_expressions.update(grain_addon_expressions.get(cls.engine, {}))
        denylist: List[str] = current_app.config['TIME_GRAIN_DENYLIST']
        for key in denylist:
            time_grain_expressions.pop(key, None)
        return dict(sorted(time_grain_expressions.items(), key=lambda x: (cls._sort_time_grains(x, 0), cls._sort_time_grains(x, 1), cls._sort_time_grains(x, 2), cls._sort_time_grains(x, 3))))

    @classmethod
    def fetch_data(cls, cursor: Any, limit: Optional[int] = None) -> Any:
        if cls.arraysize:
            cursor.arraysize = cls.arraysize
        try:
            if cls.limit_method == LimitMethod.FETCH_MANY and limit:
                return cursor.fetchmany(limit)
            data: Any = cursor.fetchall()
            description = cursor.description or []
            column_mutators: Dict[Any, Callable[[Any], Any]] = {row[0]: func for row in description if (func := cls.column_type_mutators.get(type(cls.get_sqla_column_type(cls.get_datatype(row[1])))))}
            if column_mutators:
                indexes: Dict[Any, int] = {row[0]: idx for idx, row in enumerate(description)}
                for row_idx, row in enumerate(data):
                    new_row: List[Any] = list(row)
                    for col, func in column_mutators.items():
                        col_idx = indexes[col]
                        new_row[col_idx] = func(row[col_idx])
                    data[row_idx] = tuple(new_row)
            return data
        except Exception as ex:
            raise cls.get_dbapi_mapped_exception(ex) from ex

    @classmethod
    def expand_data(cls, columns: List[Any], data: Any) -> Tuple[Any, Any, List[Any]]:
        return (columns, data, [])

    @classmethod
    def alter_new_orm_column(cls, orm_col: Any) -> None:
        pass

    @classmethod
    def epoch_to_dttm(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def epoch_ms_to_dttm(cls) -> str:
        return cls.epoch_to_dttm().replace('{col}', '({col}/1000)')

    @classmethod
    def get_datatype(cls, type_code: Any) -> Optional[str]:
        if isinstance(type_code, str) and type_code != '':
            return type_code.upper()
        return None

    @classmethod
    @deprecated(deprecated_in='3.0')
    def normalize_indexes(cls, indexes: Any) -> Any:
        return indexes

    @classmethod
    def get_table_metadata(cls, database: Any, table: Any) -> Any:
        return get_table_metadata(database, table)

    @classmethod
    def get_extra_table_metadata(cls, database: Any, table: Any) -> Any:
        if hasattr(cls, 'extra_table_metadata'):
            warnings.warn('The `extra_table_metadata` method is deprecated, please implement the `get_extra_table_metadata` method in the DB engine spec.', DeprecationWarning)
            if table.catalog:
                return {}
            return cls.extra_table_metadata(database, table.table, table.schema)
        return {}

    @classmethod
    def apply_limit_to_sql(cls, sql: str, limit: int, database: Any, force: bool = False) -> str:
        if cls.limit_method == LimitMethod.WRAP_SQL:
            sql = sql.strip('\t\n ;')
            qry: Select = select('*').select_from(TextAsFrom(text(sql), ['*']).alias('inner_qry')).limit(limit)
            return database.compile_sqla_query(qry)
        if cls.limit_method == LimitMethod.FORCE_LIMIT:
            parsed_query: ParsedQuery = sql_parse.ParsedQuery(sql, engine=cls.engine)
            sql = parsed_query.set_or_update_query_limit(limit, force=force)
        return sql

    @classmethod
    def apply_top_to_sql(cls, sql: str, limit: int) -> str:
        cte: Optional[str] = None
        sql_remainder: Optional[Any] = None
        sql = sql.strip(' \t\n;')
        query_limit: Optional[int] = sql_parse.extract_top_from_query(sql, cls.top_keywords)
        if not limit:
            final_limit: int = query_limit if query_limit is not None else limit
        elif int(query_limit or 0) < limit and query_limit is not None:
            final_limit = query_limit
        else:
            final_limit = limit
        if not cls.allows_cte_in_subquery:
            cte, sql_remainder = sql_parse.get_cte_remainder_query(sql)
        if cte:
            str_statement: str = str(sql_remainder)
            cte = cte + '\n'
        else:
            cte = ''
            str_statement = str(sql)
        str_statement = str_statement.replace('\n', ' ').replace('\r', '')
        tokens: List[str] = [token for token in str_statement.rstrip().split(' ') if token]
        if cls.top_not_in_sql(str_statement):
            selects: List[int] = [i for i, word in enumerate(tokens) if word.upper() in cls.select_keywords]
            first_select: int = selects[0]
            if tokens[first_select + 1].upper() == 'DISTINCT':
                first_select += 1
            tokens.insert(first_select + 1, 'TOP')
            tokens.insert(first_select + 2, str(final_limit))
        next_is_limit_token: bool = False
        new_tokens: List[str] = []
        for token in tokens:
            if token in cls.top_keywords:
                next_is_limit_token = True
            elif next_is_limit_token:
                if token.isdigit():
                    token = str(final_limit)
                    next_is_limit_token = False
            new_tokens.append(token)
        sql = ' '.join(new_tokens)
        return cte + sql

    @classmethod
    def top_not_in_sql(cls, sql: str) -> bool:
        for top_word in cls.top_keywords:
            if top_word.upper() in sql.upper():
                return False
        return True

    @classmethod
    def get_limit_from_sql(cls, sql: str) -> Optional[int]:
        parsed_query: ParsedQuery = sql_parse.ParsedQuery(sql, engine=cls.engine)
        return parsed_query.limit

    @classmethod
    def set_or_update_query_limit(cls, sql: str, limit: int) -> str:
        parsed_query: ParsedQuery = sql_parse.ParsedQuery(sql, engine=cls.engine)
        return parsed_query.set_or_update_query_limit(limit)

    @classmethod
    def get_cte_query(cls, sql: str) -> Optional[str]:
        if not cls.allows_cte_in_subquery:
            stmt = sqlparse.parse(sql)[0]
            idx, token = stmt.token_next(-1, skip_ws=True, skip_cm=True)
            if not (token and token.ttype == CTE):
                return None
            idx, token = stmt.token_next(idx)
            idx = stmt.token_index(token) + 1
            remainder: str = ''.join((str(token) for token in stmt.tokens[idx:])).strip()
            return f'WITH {token.value},\n{cls.cte_alias} AS (\n{remainder}\n)'
        return None

    @classmethod
    def df_to_sql(cls, database: Any, table: Any, df: pd.DataFrame, to_sql_kwargs: Dict[str, Any]) -> None:
        to_sql_kwargs['name'] = table.table
        if table.schema:
            to_sql_kwargs['schema'] = table.schema
        with cls.get_engine(database, catalog=table.catalog, schema=table.schema) as engine:
            if engine.dialect.supports_multivalues_insert or cls.supports_multivalues_insert:
                to_sql_kwargs['method'] = 'multi'
            df.to_sql(con=engine, **to_sql_kwargs)

    @classmethod
    def convert_dttm(cls, target_type: Any, dttm: datetime, db_extra: Optional[Any] = None) -> Any:
        return None

    @classmethod
    def handle_cursor(cls, cursor: Any, query: Any) -> None:
        pass

    @classmethod
    def execute_with_cursor(cls, cursor: Any, sql: str, query: Any) -> None:
        logger.debug('Query %d: Running query: %s', query.id, sql)
        cls.execute(cursor, sql, query.database, async_=True)
        if not cls.has_query_id_before_execute:
            cancel_query_id = query.database.db_engine_spec.get_cancel_query_id(cursor, query)
            if cancel_query_id is not None:
                query.set_extra_json_key(QUERY_CANCEL_KEY, cancel_query_id)
                db.session.commit()
        logger.debug('Query %d: Handling cursor', query.id)
        cls.handle_cursor(cursor, query)

    @classmethod
    def extract_error_message(cls, ex: Exception) -> str:
        return f'{cls.engine} error: {cls._extract_error_message(ex)}'

    @classmethod
    def _extract_error_message(cls, ex: Exception) -> str:
        return utils.error_msg_from_exception(ex)

    @classmethod
    def extract_errors(cls, ex: Exception, context: Optional[Dict[str, Any]] = None) -> List[SupersetError]:
        raw_message: str = cls._extract_error_message(ex)
        context = context or {}
        for regex, (message, error_type, extra) in cls.custom_errors.items():
            if (match := regex.search(raw_message)):
                params = {**context, **match.groupdict()}
                extra['engine_name'] = cls.engine_name
                return [SupersetError(error_type=error_type, message=message % params, level=ErrorLevel.ERROR, extra=extra)]
        return [SupersetError(error_type=SupersetErrorType.GENERIC_DB_ENGINE_ERROR, message=cls._extract_error_message(ex), level=ErrorLevel.ERROR, extra={'engine_name': cls.engine_name})]

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: Dict[str, Any], catalog: Optional[Any] = None, schema: Optional[Any] = None) -> Tuple[URL, Dict[str, Any]]:
        return (uri, {**connect_args, **cls.enforce_uri_query_params.get(uri.get_driver_name(), {})})

    @classmethod
    def get_prequeries(cls, database: Any, catalog: Optional[Any] = None, schema: Optional[Any] = None) -> List[str]:
        return []

    @classmethod
    def patch(cls) -> None:
        pass

    @classmethod
    def get_catalog_names(cls, database: Any, inspector: Inspector) -> set[str]:
        return set()

    @classmethod
    def get_schema_names(cls, inspector: Inspector) -> set[str]:
        return set(inspector.get_schema_names())

    @classmethod
    def get_table_names(cls, database: Any, inspector: Inspector, schema: Any) -> set[str]:
        try:
            tables = set(inspector.get_table_names(schema))
        except Exception as ex:
            raise cls.get_dbapi_mapped_exception(ex) from ex
        if schema and cls.try_remove_schema_from_table_name:
            tables = {re.sub(f'^{schema}\\.', '', table) for table in tables}
        return tables

    @classmethod
    def get_view_names(cls, database: Any, inspector: Inspector, schema: Any) -> set[str]:
        try:
            views = set(inspector.get_view_names(schema))
        except Exception as ex:
            raise cls.get_dbapi_mapped_exception(ex) from ex
        if schema and cls.try_remove_schema_from_table_name:
            views = {re.sub(f'^{schema}\\.', '', view) for view in views}
        return views

    @classmethod
    def get_indexes(cls, database: Any, inspector: Inspector, table: Any) -> Any:
        return inspector.get_indexes(table.table, table.schema)

    @classmethod
    def get_table_comment(cls, inspector: Inspector, table: Any) -> Optional[str]:
        comment: Optional[str] = None
        try:
            comment_obj = inspector.get_table_comment(table.table, table.schema)
            comment = comment_obj.get('text') if isinstance(comment_obj, dict) else None
        except NotImplementedError:
            pass
        except Exception as ex:
            logger.error('Unexpected error while fetching table comment', exc_info=True)
            logger.exception(ex)
        return comment

    @classmethod
    def get_columns(cls, inspector: Inspector, table: Any, options: Optional[Any] = None) -> List[Dict[str, Any]]:
        return convert_inspector_columns(cast(List[SQLAColumnType], inspector.get_columns(table.table, table.schema)))

    @classmethod
    def get_metrics(cls, database: Any, inspector: Inspector, table: Any) -> List[Dict[str, Any]]:
        return [{'metric_name': 'count', 'verbose_name': 'COUNT(*)', 'metric_type': 'count', 'expression': 'COUNT(*)'}]

    @classmethod
    def where_latest_partition(cls, database: Any, table: Any, query: Any, columns: Optional[List[Any]] = None) -> Any:
        return None

    @classmethod
    def _get_fields(cls, cols: List[Dict[str, Any]]) -> List[Any]:
        return [literal_column(query_as) if (query_as := c.get('query_as')) else column(c['column_name']) for c in cols]

    @classmethod
    def select_star(cls, database: Any, table: Any, engine: Engine, limit: int = 100, show_cols: bool = False, indent: bool = True, latest_partition: bool = True, cols: Optional[List[Any]] = None) -> str:
        fields: Union[str, List[Any]] = '*'
        cols = cols or []
        if (show_cols or latest_partition) and (not cols):
            cols = database.get_columns(table)
        if show_cols:
            fields = cls._get_fields(cols)
        full_table_name: str = cls.quote_table(table, engine.dialect)
        qry: Select = select(fields).select_from(text(full_table_name))
        if limit and cls.allow_limit_clause:
            qry = qry.limit(limit)
        if latest_partition:
            partition_query = cls.where_latest_partition(database, table, qry, columns=cols)
            if partition_query is not None:
                qry = partition_query
        sql: str = database.compile_sqla_query(qry, table.catalog, table.schema)
        if indent:
            sql = SQLScript(sql, engine=cls.engine).format()
        return sql

    @classmethod
    def estimate_statement_cost(cls, database: Any, statement: str, cursor: Any) -> Any:
        raise Exception('Database does not support cost estimation')

    @classmethod
    def query_cost_formatter(cls, raw_cost: Any) -> Any:
        raise Exception('Database does not support cost estimation')

    @classmethod
    def process_statement(cls, statement: Any, database: Any) -> Any:
        return database.mutate_sql_based_on_config(str(statement), is_split=True)

    @classmethod
    def estimate_query_cost(cls, database: Any, catalog: Any, schema: Any, sql: str, source: Optional[str] = None) -> List[Any]:
        extra: Any = database.get_extra() or {}
        if not cls.get_allow_cost_estimate(extra):
            raise Exception('Database does not support cost estimation')
        parsed_script = SQLScript(sql, engine=cls.engine)
        with database.get_raw_connection(catalog=catalog, schema=schema, source=source) as conn:
            cursor = conn.cursor()
            return [cls.estimate_statement_cost(database, cls.process_statement(statement, database), cursor) for statement in parsed_script.statements]

    @classmethod
    def get_url_for_impersonation(cls, url: URL, impersonate_user: bool, username: Optional[str], access_token: Optional[str]) -> URL:
        if impersonate_user and username is not None:
            url = url.set(username=username)
        return url

    @classmethod
    def update_impersonation_config(cls, database: Any, connect_args: Dict[str, Any], uri: URL, username: Optional[str], access_token: Optional[str]) -> None:
        pass

    @classmethod
    def execute(cls, cursor: Any, query: str, database: Any, **kwargs: Any) -> None:
        if not cls.allows_sql_comments:
            query = sql_parse.strip_comments_from_sql(query, engine=cls.engine)
        disallowed_functions: set[str] = current_app.config['DISALLOWED_SQL_FUNCTIONS'].get(cls.engine, set())
        if sql_parse.check_sql_functions_exist(query, disallowed_functions, cls.engine):
            raise DisallowedSQLFunction(disallowed_functions)
        if cls.arraysize:
            cursor.arraysize = cls.arraysize
        try:
            cursor.execute(query, **kwargs)
        except Exception as ex:
            if database.is_oauth2_enabled() and cls.needs_oauth2(ex):
                cls.start_oauth2_dance(database)
            raise cls.get_dbapi_mapped_exception(ex) from ex

    @classmethod
    def needs_oauth2(cls, ex: Exception) -> bool:
        return g and hasattr(g, 'user') and isinstance(ex, cls.oauth2_exception)

    @classmethod
    def make_label_compatible(cls, label: str) -> Any:
        label_mutated: str = cls._mutate_label(label)
        if cls.max_column_name_length and len(label_mutated) > cls.max_column_name_length:
            label_mutated = cls._truncate_label(label)
        if cls.force_column_alias_quotes:
            label_mutated = quoted_name(label_mutated, True)
        return label_mutated

    @classmethod
    def get_column_types(cls, column_type: Optional[str]) -> Optional[Tuple[Any, GenericDataType]]:
        if not column_type:
            return None
        for regex, sqla_type, generic_type in cls.column_type_mappings + cls._default_column_type_mappings:
            match = regex.match(column_type)
            if not match:
                continue
            if callable(sqla_type):
                return (sqla_type(match), generic_type)
            return (sqla_type, generic_type)
        return None

    @staticmethod
    def _mutate_label(label: str) -> str:
        return label

    @classmethod
    def _truncate_label(cls, label: str) -> str:
        label = md5_sha_from_str(label)
        if cls.max_column_name_length and len(label) > cls.max_column_name_length:
            label = label[:cls.max_column_name_length]
        return label

    @classmethod
    def column_datatype_to_string(cls, sqla_column_type: TypeEngine, dialect: Dialect) -> str:
        sqla_column_type = sqla_column_type.copy()
        if hasattr(sqla_column_type, 'collation'):
            sqla_column_type.collation = None
        if hasattr(sqla_column_type, 'charset'):
            sqla_column_type.charset = None
        return sqla_column_type.compile(dialect=dialect).upper()

    @classmethod
    def get_function_names(cls, database: Any) -> List[str]:
        return []

    @staticmethod
    def pyodbc_rows_to_tuples(data: List[Any]) -> List[Any]:
        if data and type(data[0]).__name__ == 'Row':
            data = [tuple(row) for row in data]
        return data

    @staticmethod
    def mutate_db_for_connection_test(database: Any) -> Any:
        return None

    @staticmethod
    def get_extra_params(database: Any) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        if database.extra:
            try:
                extra = json.loads(database.extra)
            except json.JSONDecodeError as ex:
                logger.error(ex, exc_info=True)
                raise
        return extra

    @staticmethod
    def update_params_from_encrypted_extra(database: Any, params: Dict[str, Any]) -> None:
        if not database.encrypted_extra:
            return
        try:
            encrypted_extra = json.loads(database.encrypted_extra)
            params.update(encrypted_extra)
        except json.JSONDecodeError as ex:
            logger.error(ex, exc_info=True)
            raise

    @classmethod
    def is_select_query(cls, parsed_query: ParsedQuery) -> bool:
        return parsed_query.is_select()

    @classmethod
    def get_column_spec(cls, native_type: str, db_extra: Optional[Any] = None, source: utils.ColumnTypeSource = utils.ColumnTypeSource.GET_TABLE) -> Optional[ColumnSpec]:
        if (col_types := cls.get_column_types(native_type)):
            column_type, generic_type = col_types
            is_dttm: bool = generic_type == GenericDataType.TEMPORAL
            return ColumnSpec(sqla_type=column_type, generic_type=generic_type, is_dttm=is_dttm)
        return None

    @classmethod
    def get_sqla_column_type(cls, native_type: str, db_extra: Optional[Any] = None, source: utils.ColumnTypeSource = utils.ColumnTypeSource.GET_TABLE) -> Optional[Any]:
        column_spec = cls.get_column_spec(native_type=native_type, db_extra=db_extra, source=source)
        return column_spec.sqla_type if column_spec else None

    @classmethod
    def prepare_cancel_query(cls, query: Any) -> Any:
        return None

    @classmethod
    def has_implicit_cancel(cls) -> bool:
        return False

    @classmethod
    def get_cancel_query_id(cls, cursor: Any, query: Any) -> Any:
        return None

    @classmethod
    def cancel_query(cls, cursor: Any, query: Any, cancel_query_id: Any) -> bool:
        return False

    @classmethod
    def parse_sql(cls, sql: str) -> List[str]:
        return [str(s).strip(' ;') for s in sqlparse.parse(sql)]

    @classmethod
    def get_impersonation_key(cls, user: Optional[User]) -> Optional[str]:
        return user.username if user else None

    @classmethod
    def mask_encrypted_extra(cls, encrypted_extra: Optional[str]) -> Optional[str]:
        if encrypted_extra is None or not cls.encrypted_extra_sensitive_fields:
            return encrypted_extra
        try:
            config = json.loads(encrypted_extra)
        except (TypeError, json.JSONDecodeError):
            return encrypted_extra
        masked_encrypted_extra = redact_sensitive(config, cls.encrypted_extra_sensitive_fields)
        return json.dumps(masked_encrypted_extra)

    @classmethod
    def unmask_encrypted_extra(cls, old: Optional[str], new: Optional[str]) -> Optional[str]:
        if old is None or new is None:
            return new
        try:
            old_config = json.loads(old)
            new_config = json.loads(new)
        except (TypeError, json.JSONDecodeError):
            return new
        new_config = reveal_sensitive(old_config, new_config, cls.encrypted_extra_sensitive_fields)
        return json.dumps(new_config)

    @classmethod
    def get_public_information(cls) -> Dict[str, Any]:
        return {
            'supports_file_upload': cls.supports_file_upload,
            'disable_ssh_tunneling': cls.disable_ssh_tunneling,
            'supports_dynamic_catalog': cls.supports_dynamic_catalog,
            'supports_oauth2': cls.supports_oauth2
        }

    @classmethod
    def validate_database_uri(cls, sqlalchemy_uri: URL) -> None:
        if (db_engine_uri_validator := current_app.config['DB_SQLA_URI_VALIDATOR']):
            db_engine_uri_validator(sqlalchemy_uri)
        if (existing_disallowed := cls.disallow_uri_query_params.get(sqlalchemy_uri.get_driver_name(), set()).intersection(sqlalchemy_uri.query)):
            raise ValueError(f'Forbidden query parameter(s): {existing_disallowed}')

    @classmethod
    def denormalize_name(cls, dialect: Dialect, name: str) -> str:
        if hasattr(dialect, 'requires_name_normalize') and dialect.requires_name_normalize:
            return dialect.denormalize_name(name)
        return name

    @classmethod
    def quote_table(cls, table: Any, dialect: Dialect) -> str:
        quoters: Dict[str, Callable[[Any], str]] = {
            'catalog': dialect.identifier_preparer.quote_schema,
            'schema': dialect.identifier_preparer.quote_schema,
            'table': dialect.identifier_preparer.quote
        }
        return '.'.join((function(getattr(table, key)) for key, function in quoters.items() if getattr(table, key)))


class BasicParametersSchema(Schema):
    username = fields.String(required=True, allow_none=True, metadata={'description': __('Username')})
    password = fields.String(allow_none=True, metadata={'description': __('Password')})
    host = fields.String(required=True, metadata={'description': __('Hostname or IP address')})
    port = fields.Integer(required=True, metadata={'description': __('Database port')}, validate=Range(min=0, max=2 ** 16, max_inclusive=False))
    database = fields.String(required=True, metadata={'description': __('Database name')})
    query = fields.Dict(keys=fields.Str(), values=fields.Raw(), metadata={'description': __('Additional parameters')})
    encryption = fields.Boolean(required=False, metadata={'description': __('Use an encrypted connection to the database')})
    ssh = fields.Boolean(required=False, metadata={'description': __('Use an ssh tunnel connection to the database')})


class BasicParametersType(TypedDict, total=False):
    pass


class BasicPropertiesType(TypedDict):
    pass


class BasicParametersMixin:
    parameters_schema: Schema = BasicParametersSchema()
    default_driver: str = ''
    encryption_parameters: Dict[str, Any] = {}

    @classmethod
    def build_sqlalchemy_uri(cls, parameters: Dict[str, Any], encrypted_extra: Optional[str] = None) -> str:
        query: Dict[str, Any] = parameters.get('query', {}).copy()
        if parameters.get('encryption'):
            if not cls.encryption_parameters:
                raise Exception('Unable to build a URL with encryption enabled')
            query.update(cls.encryption_parameters)
        return str(URL.create(f'{cls.engine}+{cls.default_driver}'.rstrip('+'),
                              username=parameters.get('username'),
                              password=parameters.get('password'),
                              host=parameters['host'],
                              port=parameters['port'],
                              database=parameters['database'],
                              query=query))

    @classmethod
    def get_parameters_from_uri(cls, uri: str, encrypted_extra: Optional[str] = None) -> Dict[str, Any]:
        url_obj: URL = make_url_safe(uri)
        query: Dict[str, Any] = {key: value for key, value in url_obj.query.items() if (key, value) not in cls.encryption_parameters.items()}
        encryption: bool = all((item in url_obj.query.items() for item in cls.encryption_parameters.items()))
        return {
            'username': url_obj.username,
            'password': url_obj.password,
            'host': url_obj.host,
            'port': url_obj.port,
            'database': url_obj.database,
            'query': query,
            'encryption': encryption
        }

    @classmethod
    def validate_parameters(cls, properties: Dict[str, Any]) -> List[SupersetError]:
        errors: List[SupersetError] = []
        required: set[str] = {'host', 'port', 'username', 'database'}
        parameters: Dict[str, Any] = properties.get('parameters', {})
        present: set[str] = {key for key in parameters if parameters.get(key, ())}
        if (missing := sorted(required - present)):
            errors.append(SupersetError(message=f'One or more parameters are missing: {", ".join(missing)}', error_type=SupersetErrorType.CONNECTION_MISSING_PARAMETERS_ERROR, level=ErrorLevel.WARNING, extra={'missing': missing}))
        host: Optional[str] = parameters.get('host', None)
        if not host:
            return errors
        if not is_hostname_valid(host):
            errors.append(SupersetError(message="The hostname provided can't be resolved.", error_type=SupersetErrorType.CONNECTION_INVALID_HOSTNAME_ERROR, level=ErrorLevel.ERROR, extra={'invalid': ['host']}))
            return errors
        port: Any = parameters.get('port', None)
        if not port:
            return errors
        try:
            port = int(port)
        except (ValueError, TypeError):
            errors.append(SupersetError(message='Port must be a valid integer.', error_type=SupersetErrorType.CONNECTION_INVALID_PORT_ERROR, level=ErrorLevel.ERROR, extra={'invalid': ['port']}))
        if not (isinstance(port, int) and 0 <= port < 2 ** 16):
            errors.append(SupersetError(message='The port must be an integer between 0 and 65535 (inclusive).', error_type=SupersetErrorType.CONNECTION_INVALID_PORT_ERROR, level=ErrorLevel.ERROR, extra={'invalid': ['port']}))
        elif not is_port_open(host, port):
            errors.append(SupersetError(message='The port is closed.', error_type=SupersetErrorType.CONNECTION_PORT_CLOSED_ERROR, level=ErrorLevel.ERROR, extra={'invalid': ['port']}))
        return errors

    @classmethod
    def parameters_json_schema(cls) -> Optional[Dict[str, Any]]:
        if not cls.parameters_schema:
            return None
        spec = APISpec(title='Database Parameters', version='1.0.0', openapi_version='3.0.2', plugins=[MarshmallowPlugin()])
        spec.components.schema(cls.__name__, schema=cls.parameters_schema)
        return spec.to_dict()['components']['schemas'][cls.__name__]