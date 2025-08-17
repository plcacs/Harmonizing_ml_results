from __future__ import annotations

import logging
import re
import warnings
from datetime import datetime
from re import Match, Pattern
from typing import (
    Any,
    Callable,
    cast,
    ContextManager,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    TypedDict,
    Union,
)
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
from superset.superset_typing import (
    OAuth2ClientConfig,
    OAuth2State,
    OAuth2TokenResponse,
    ResultSetColumnType,
    SQLAColumnType,
)
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


ColumnTypeMapping = Tuple[
    Pattern[str],
    Union[TypeEngine, Callable[[Match[str]], TypeEngine]],
    GenericDataType,
]

logger = logging.getLogger()

GenericDBException = Exception


def convert_inspector_columns(cols: List[SQLAColumnType]) -> List[ResultSetColumnType]:
    result_set_columns: List[ResultSetColumnType] = []
    for col in cols:
        result_set_columns.append({"column_name": col.get("name"), **col})  # type: ignore
    return result_set_columns


class TimeGrain(NamedTuple):
    name: str
    label: str
    function: str
    duration: Optional[str]


builtin_time_grains: Dict[Optional[str], str] = {
    TimeGrainConstants.SECOND: _("Second"),
    TimeGrainConstants.FIVE_SECONDS: _("5 second"),
    TimeGrainConstants.THIRTY_SECONDS: _("30 second"),
    TimeGrainConstants.MINUTE: _("Minute"),
    TimeGrainConstants.FIVE_MINUTES: _("5 minute"),
    TimeGrainConstants.TEN_MINUTES: _("10 minute"),
    TimeGrainConstants.FIFTEEN_MINUTES: _("15 minute"),
    TimeGrainConstants.THIRTY_MINUTES: _("30 minute"),
    TimeGrainConstants.HOUR: _("Hour"),
    TimeGrainConstants.SIX_HOURS: _("6 hour"),
    TimeGrainConstants.DAY: _("Day"),
    TimeGrainConstants.WEEK: _("Week"),
    TimeGrainConstants.MONTH: _("Month"),
    TimeGrainConstants.QUARTER: _("Quarter"),
    TimeGrainConstants.YEAR: _("Year"),
    TimeGrainConstants.WEEK_STARTING_SUNDAY: _("Week starting Sunday"),
    TimeGrainConstants.WEEK_STARTING_MONDAY: _("Week starting Monday"),
    TimeGrainConstants.WEEK_ENDING_SATURDAY: _("Week ending Saturday"),
    TimeGrainConstants.WEEK_ENDING_SUNDAY: _("Week ending Sunday"),
}


class TimestampExpression(ColumnClause):
    def __init__(self, expr: str, col: ColumnClause, **kwargs: Any) -> None:
        super().__init__(expr, **kwargs)
        self.col = col

    @property
    def _constructor(self) -> ColumnClause:
        return ColumnClause


@compiles(TimestampExpression)
def compile_timegrain_expression(
    element: TimestampExpression, compiler: Compiled, **kwargs: Any
) -> str:
    return element.name.replace("{col}", compiler.process(element.col, **kwargs))


class LimitMethod:
    FETCH_MANY = "fetch_many"
    WRAP_SQL = "wrap_sql"
    FORCE_LIMIT = "force_limit"


class MetricType(TypedDict, total=False):
    metric_name: str
    expression: str
    verbose_name: Optional[str]
    metric_type: Optional[str]
    description: Optional[str]
    d3format: Optional[str]
    currency: Optional[str]
    warning_text: Optional[str]
    extra: Optional[str]


class BaseEngineSpec:
    engine_name: Optional[str] = None
    engine = "base"
    engine_aliases: Set[str] = set()
    drivers: Dict[str, str] = {}
    default_driver: Optional[str] = None
    sqlalchemy_uri_placeholder = (
        "engine+driver://user:password@host:port/dbname[?key=value&key=value...]"
    )
    disable_ssh_tunneling = False
    _date_trunc_functions: Dict[str, str] = {}
    _time_grain_expressions: Dict[Optional[str], str] = {}
    _default_column_type_mappings: Tuple[ColumnTypeMapping, ...] = (
        (
            re.compile(r"^string", re.IGNORECASE),
            types.String(),
            GenericDataType.STRING,
        ),
        (
            re.compile(r"^n((var)?char|text)", re.IGNORECASE),
            types.UnicodeText(),
            GenericDataType.STRING,
        ),
        (
            re.compile(r"^(var)?char", re.IGNORECASE),
            types.String(),
            GenericDataType.STRING,
        ),
        (
            re.compile(r"^(tiny|medium|long)?text", re.IGNORECASE),
            types.String(),
            GenericDataType.STRING,
        ),
        (
            re.compile(r"^smallint", re.IGNORECASE),
            types.SmallInteger(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^int(eger)?", re.IGNORECASE),
            types.Integer(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^bigint", re.IGNORECASE),
            types.BigInteger(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^long", re.IGNORECASE),
            types.Float(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^decimal", re.IGNORECASE),
            types.Numeric(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^numeric", re.IGNORECASE),
            types.Numeric(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^float", re.IGNORECASE),
            types.Float(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^double", re.IGNORECASE),
            types.Float(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^real", re.IGNORECASE),
            types.REAL,
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^smallserial", re.IGNORECASE),
            types.SmallInteger(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^serial", re.IGNORECASE),
            types.Integer(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^bigserial", re.IGNORECASE),
            types.BigInteger(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^money", re.IGNORECASE),
            types.Numeric(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^timestamp", re.IGNORECASE),
            types.TIMESTAMP(),
            GenericDataType.TEMPORAL,
        ),
        (
            re.compile(r"^datetime", re.IGNORECASE),
            types.DateTime(),
            GenericDataType.TEMPORAL,
        ),
        (
            re.compile(r"^date", re.IGNORECASE),
            types.Date(),
            GenericDataType.TEMPORAL,
        ),
        (
            re.compile(r"^time", re.IGNORECASE),
            types.Time(),
            GenericDataType.TEMPORAL,
        ),
        (
            re.compile(r"^interval", re.IGNORECASE),
            types.Interval(),
            GenericDataType.TEMPORAL,
        ),
        (
            re.compile(r"^bool(ean)?", re.IGNORECASE),
            types.Boolean(),
            GenericDataType.BOOLEAN,
        ),
    )
    column_type_mappings: Tuple[ColumnTypeMapping, ...] = ()
    column_type_mutators: Dict[TypeEngine, Callable[[Any], Any]] = {}
    time_groupby_inline = False
    limit_method = LimitMethod.FORCE_LIMIT
    supports_multivalues_insert = False
    allows_joins = True
    allows_subqueries = True
    allows_alias_in_select = True
    allows_alias_in_orderby = True
    allows_sql_comments = True
    allows_escaped_colons = True
    allows_alias_to_source_column = True
    allows_hidden_orderby_agg = True
    allows_hidden_cc_in_orderby = False
    allows_cte_in_subquery = True
    cte_alias = "__cte"
    allow_limit_clause = True
    select_keywords: Set[str] = {"SELECT"}
    top_keywords: Set[str] = {"TOP"}
    disallow_uri_query_params: Dict[str, Set[str]] = {}
    enforce_uri_query_params: Dict[str, Dict[str, Any]] = {}
    force_column_alias_quotes = False
    arraysize = 0
    max_column_name_length: Optional[int] = None
    try_remove_schema_from_table_name = True
    run_multiple_statements_as_one = False
    custom_errors: Dict[
        Pattern[str], Tuple[str, SupersetErrorType, Dict[str, Any]]
    ] = {}
    encrypted_extra_sensitive_fields: Set[str] = {"$.*"}
    supports_file_upload = True
    supports_dynamic_schema = False
    supports_catalog = False
    supports_dynamic_catalog = False
    supports_oauth2 = False
    oauth2_scope = ""
    oauth2_authorization_request_uri: Optional[str] = None
    oauth2_token_request_uri: Optional[str] = None
    oauth2_token_request_type = "data"
    oauth2_exception = OAuth2RedirectError
    has_query_id_before_execute = True

    @classmethod
    def is_oauth2_enabled(cls) -> bool:
        return (
            cls.supports_oauth2
            and cls.engine_name in current_app.config["DATABASE_OAUTH2_CLIENTS"]
        )

    @classmethod
    def start_oauth2_dance(cls, database: Database) -> None:
        tab_id = str(uuid4())
        default_redirect_uri = url_for("DatabaseRestApi.oauth2", _external=True)

        state: OAuth2State = {
            "database_id": database.id,
            "user_id": g.user.id,
            "default_redirect_uri": default_redirect_uri,
            "tab_id": tab_id,
        }
        oauth2_config = database.get_oauth2_config()
        if oauth2_config is None:
            raise OAuth2Error("No configuration found for OAuth2")

        oauth_url = cls.get_oauth2_authorization_uri(oauth2_config, state)

        raise OAuth2RedirectError(oauth_url, tab_id, default_redirect_uri)

    @classmethod
    def get_oauth2_config(cls) -> Optional[OAuth2ClientConfig]:
        oauth2_config = current_app.config["DATABASE_OAUTH2_CLIENTS"]
        if cls.engine_name not in oauth2_config:
            return None

        db_engine_spec_config = oauth2_config[cls.engine_name]
        redirect_uri = current_app.config.get(
            "DATABASE_OAUTH2_REDIRECT_URI",
            url_for("DatabaseRestApi.oauth2", _external=True),
        )

        config: OAuth2ClientConfig = {
            "id": db_engine_spec_config["id"],
            "secret": db_engine_spec_config["secret"],
            "scope": db_engine_spec_config.get("scope") or cls.oauth2_scope,
            "redirect_uri": redirect_uri,
            "authorization_request_uri": db_engine_spec_config.get(
                "authorization_request_uri",
                cls.oauth2_authorization_request_uri,
            ),
            "token_request_uri": db_engine_spec_config.get(
                "token_request_uri",
                cls.oauth2_token_request_uri,
            ),
            "request_content_type": db_engine_spec_config.get(
                "request_content_type", cls.oauth2_token_request_type
            ),
        }

        return config

    @classmethod
    def get_oauth2_authorization_uri(
        cls,
        config: OAuth2ClientConfig,
        state: OAuth2State,
    ) -> str:
        uri = config["authorization_request_uri"]
        params = {
            "scope": config["scope"],
            "access_type": "offline",
            "include_granted_scopes": "false",
            "response_type": "code",
            "state": encode_oauth2_state(state),
            "redirect_uri": config["redirect_uri"],
            "client_id": config["id"],
            "prompt": "consent",
        }
        return urljoin(uri, "?" + urlencode(params))

    @classmethod
    def get_oauth2_token(
        cls,
        config: OAuth2ClientConfig,
        code: str,
    ) -> OAuth2TokenResponse:
        timeout = current_app.config["DATABASE_OAUTH2_TIMEOUT"].total_seconds()
        uri = config["token_request_uri"]
        req_body = {
            "code": code,
            "client_id": config["id"],
            "client_secret": config["secret"],
            "redirect_uri": config["redirect_uri"],
            "grant_type": "authorization_code",
        }
        if config["request_content_type"] == "data":
            return requests.post(uri, data=req_body, timeout=timeout).json()
        return requests.post(uri, json=req_body, timeout=timeout).json()

    @classmethod
    def get_oauth2_fresh_token(
        cls,
        config: OAuth2ClientConfig,
        refresh_token: str,
    ) -> OAuth2TokenResponse:
        timeout = current_app.config["DATABASE_OAUTH2_TIMEOUT"].total_seconds()
        uri = config["token_request_uri"]
        req_body = {
            "client_id": config["id"],
            "client_secret": config["secret"],
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        if config["request_content_type"] == "data":
            return requests.post(uri, data=req_body, timeout=timeout).json()
        return requests.post(uri, json=req_body, timeout=timeout).json()

    @classmethod
    def get_allows_alias_in_select(
        cls,
        database: Database,
    ) -> bool:
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
    def get_default_catalog(
        cls,
        database: Database,
    ) -> Optional[str]:
        return None

    @classmethod
    def get_default_schema(cls, database: Database, catalog: Optional[str]) -> Optional[str]:
        with database.get_inspector(catalog=catalog) as inspector:
            return inspector.default_schema_name

    @classmethod
    def get_schema_from_engine_params(
        cls,
        sqlalchemy_uri: URL,
        connect_args: Dict[str, Any],
    ) -> Optional[str]:
        return None

    @classmethod
    def get_default_schema_for_query(
        cls,
        database: Database,
        query: Query,
    ) -> Optional[str]:
        if cls.supports_dynamic_schema:
            return query.schema

        try:
            connect_args = database.get_extra()["engine_params"]["connect_args"]
        except KeyError:
            connect_args = {}
        sqlalchemy_uri = make_url_safe(database.sqlalchemy_uri)
        if schema := cls.get_schema_from_engine_params(sqlalchemy_uri, connect_args):
            return schema

        return cls.get