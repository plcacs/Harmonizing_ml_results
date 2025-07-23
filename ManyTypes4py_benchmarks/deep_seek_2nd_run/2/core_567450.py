"""Utility functions used across Superset"""
from __future__ import annotations
import _thread
import collections
import errno
import logging
import os
import platform
import re
import signal
import smtplib
import sqlite3
import ssl
import tempfile
import threading
import traceback
import uuid
import zlib
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import timedelta
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from enum import Enum, IntEnum
from io import BytesIO
from timeit import default_timer
from types import TracebackType
from typing import Any, Callable, cast, Dict, List, NamedTuple, Optional, Set, Tuple, TYPE_CHECKING, TypedDict, TypeVar, Union
from urllib.parse import unquote_plus
from zipfile import ZipFile
import markdown as md
import nh3
import pandas as pd
import sqlalchemy as sa
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import Certificate, load_pem_x509_certificate
from flask import current_app, g, request
from flask_appbuilder import SQLA
from flask_appbuilder.security.sqla.models import User
from flask_babel import gettext as __
from markupsafe import Markup
from pandas.api.types import infer_dtype
from pandas.core.dtypes.common import is_numeric_dtype
from sqlalchemy import event, exc, inspect, select, Text
from sqlalchemy.dialects.mysql import LONGTEXT, MEDIUMTEXT
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.sql.type_api import Variant
from sqlalchemy.types import TypeEngine
from typing_extensions import TypeGuard
from superset.constants import EXTRA_FORM_DATA_APPEND_KEYS, EXTRA_FORM_DATA_OVERRIDE_EXTRA_KEYS, EXTRA_FORM_DATA_OVERRIDE_REGULAR_MAPPINGS, NO_TIME_RANGE
from superset.errors import ErrorLevel, SupersetErrorType
from superset.exceptions import CertificateException, SupersetException, SupersetTimeoutException
from superset.sql_parse import sanitize_clause
from superset.superset_typing import AdhocColumn, AdhocMetric, AdhocMetricColumn, Column, FilterValues, FlaskResponse, FormData, Metric
from superset.utils.backports import StrEnum
from superset.utils.database import get_example_database
from superset.utils.date_parser import parse_human_timedelta
from superset.utils.hashing import md5_sha_from_dict, md5_sha_from_str

if TYPE_CHECKING:
    from superset.connectors.sqla.models import BaseDatasource, TableColumn
    from superset.models.sql_lab import Query

logging.getLogger('MARKDOWN').setLevel(logging.INFO)
logger = logging.getLogger(__name__)
DTTM_ALIAS = '__timestamp'
TIME_COMPARISON = '__'
JS_MAX_INTEGER = 9007199254740991
InputType = TypeVar('InputType')
ADHOC_FILTERS_REGEX = re.compile('^adhoc_filters')

class AdhocMetricExpressionType(StrEnum):
    SIMPLE = 'SIMPLE'
    SQL = 'SQL'

class AnnotationType(StrEnum):
    FORMULA = 'FORMULA'
    INTERVAL = 'INTERVAL'
    EVENT = 'EVENT'
    TIME_SERIES = 'TIME_SERIES'

class GenericDataType(IntEnum):
    """
    Generic database column type that fits both frontend and backend.
    """
    NUMERIC = 0
    STRING = 1
    TEMPORAL = 2
    BOOLEAN = 3

class DatasourceType(StrEnum):
    TABLE = 'table'
    DATASET = 'dataset'
    QUERY = 'query'
    SAVEDQUERY = 'saved_query'
    VIEW = 'view'

class LoggerLevel(StrEnum):
    INFO = 'info'
    WARNING = 'warning'
    EXCEPTION = 'exception'

class HeaderDataType(TypedDict):
    pass

class DatasourceDict(TypedDict):
    pass

class AdhocFilterClause(TypedDict, total=False):
    pass

class QueryObjectFilterClause(TypedDict, total=False):
    pass

class ExtraFiltersTimeColumnType(StrEnum):
    TIME_COL = '__time_col'
    TIME_GRAIN = '__time_grain'
    TIME_ORIGIN = '__time_origin'
    TIME_RANGE = '__time_range'

class ExtraFiltersReasonType(StrEnum):
    NO_TEMPORAL_COLUMN = 'no_temporal_column'
    COL_NOT_IN_DATASOURCE = 'not_in_datasource'

class FilterOperator(StrEnum):
    """
    Operators used filter controls
    """
    EQUALS = '=='
    NOT_EQUALS = '!='
    GREATER_THAN = '>'
    LESS_THAN = '<'
    GREATER_THAN_OR_EQUALS = '>='
    LESS_THAN_OR_EQUALS = '<='
    LIKE = 'LIKE'
    NOT_LIKE = 'NOT LIKE'
    ILIKE = 'ILIKE'
    IS_NULL = 'IS NULL'
    IS_NOT_NULL = 'IS NOT NULL'
    IN = 'IN'
    NOT_IN = 'NOT IN'
    IS_TRUE = 'IS TRUE'
    IS_FALSE = 'IS FALSE'
    TEMPORAL_RANGE = 'TEMPORAL_RANGE'

class FilterStringOperators(StrEnum):
    EQUALS = ('EQUALS',)
    NOT_EQUALS = ('NOT_EQUALS',)
    LESS_THAN = ('LESS_THAN',)
    GREATER_THAN = ('GREATER_THAN',)
    LESS_THAN_OR_EQUAL = ('LESS_THAN_OR_EQUAL',)
    GREATER_THAN_OR_EQUAL = ('GREATER_THAN_OR_EQUAL',)
    IN = ('IN',)
    NOT_IN = ('NOT_IN',)
    ILIKE = ('ILIKE',)
    LIKE = ('LIKE',)
    IS_NOT_NULL = ('IS_NOT_NULL',)
    IS_NULL = ('IS_NULL',)
    LATEST_PARTITION = ('LATEST_PARTITION',)
    IS_TRUE = ('IS_TRUE',)
    IS_FALSE = ('IS_FALSE',)

class PostProcessingBoxplotWhiskerType(StrEnum):
    """
    Calculate cell contribution to row/column total
    """
    TUKEY = 'tukey'
    MINMAX = 'min/max'
    PERCENTILE = 'percentile'

class PostProcessingContributionOrientation(StrEnum):
    """
    Calculate cell contribution to row/column total
    """
    ROW = 'row'
    COLUMN = 'column'

class QuerySource(Enum):
    """
    The source of a SQL query.
    """
    CHART = 0
    DASHBOARD = 1
    SQL_LAB = 2

class QueryStatus(StrEnum):
    """Enum-type class for query statuses"""
    STOPPED = 'stopped'
    FAILED = 'failed'
    PENDING = 'pending'
    RUNNING = 'running'
    SCHEDULED = 'scheduled'
    SUCCESS = 'success'
    FETCHING = 'fetching'
    TIMED_OUT = 'timed_out'

class DashboardStatus(StrEnum):
    """Dashboard status used for frontend filters"""
    PUBLISHED = 'published'
    DRAFT = 'draft'

class ReservedUrlParameters(StrEnum):
    """
    Reserved URL parameters that are used internally by Superset. These will not be
    passed to chart queries, as they control the behavior of the UI.
    """
    STANDALONE = 'standalone'
    EDIT_MODE = 'edit'

    @staticmethod
    def is_standalone_mode() -> bool:
        standalone_param = request.args.get(ReservedUrlParameters.STANDALONE.value)
        standalone = bool(standalone_param and standalone_param != 'false' and (standalone_param != '0'))
        return standalone

class RowLevelSecurityFilterType(StrEnum):
    REGULAR = 'Regular'
    BASE = 'Base'

class ColumnTypeSource(Enum):
    GET_TABLE = 1
    CURSOR_DESCRIPTION = 2

class ColumnSpec(NamedTuple):
    python_date_format: Optional[str] = None

def parse_js_uri_path_item(item: Optional[str], unquote: bool = True, eval_undefined: bool = False) -> Optional[str]:
    """Parse an uri path item made with js.

    :param item: an uri path component
    :param unquote: Perform unquoting of string using urllib.parse.unquote_plus()
    :param eval_undefined: When set to True and item is either 'null' or 'undefined',
    assume item is undefined and return None.
    :return: Either None, the original item or unquoted item
    """
    item = None if eval_undefined and item in ('null', 'undefined') else item
    return unquote_plus(item) if unquote and item else item

def cast_to_num(value: Optional[Union[str, int, float]]) -> Optional[Union[int, float]]:
    """Casts a value to an int/float

    >>> cast_to_num('1 ')
    1.0
    >>> cast_to_num(' 2')
    2.0
    >>> cast_to_num('5')
    5
    >>> cast_to_num('5.2')
    5.2
    >>> cast_to_num(10)
    10
    >>> cast_to_num(10.1)
    10.1
    >>> cast_to_num(None) is None
    True
    >>> cast_to_num('this is not a string') is None
    True

    :param value: value to be converted to numeric representation
    :returns: value cast to `int` if value is all digits, `float` if `value` is
              decimal value and `None`` if it can't be converted
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return None

def cast_to_boolean(value: Optional[Union[str, int, float, bool]]) -> Optional[bool]:
    """Casts a value to an int/float

    >>> cast_to_boolean(1)
    True
    >>> cast_to_boolean(0)
    False
    >>> cast_to_boolean(0.5)
    True
    >>> cast_to_boolean('true')
    True
    >>> cast_to_boolean('false')
    False
    >>> cast_to_boolean('False')
    False
    >>> cast_to_boolean(None)

    :param value: value to be converted to boolean representation
    :returns: value cast to `bool`. when value is 'true' or value that are not 0
              converted into True. Return `None` if value is `None`
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() == 'true'
    return False

def error_msg_from_exception(ex: Exception) -> str:
    """Translate exception into error message

    Database have different ways to handle exception. This function attempts
    to make sense of the exception object and construct a human readable
    sentence.

    TODO(bkyryliuk): parse the Presto error message from the connection
                     created via create_engine.
    engine = create_engine('presto://localhost:3506/silver') -
      gives an e.message as the str(dict)
    presto.connect('localhost', port=3506, catalog='silver') - as a dict.
    The latter version is parsed correctly by this function.
    """
    msg = ''
    if hasattr(ex, 'message'):
        if isinstance(ex.message, dict):
            msg = ex.message.get('message')
        elif ex.message:
            msg = ex.message
    return str(msg) or str(ex)

def markdown(raw: str, markup_wrap: bool = False) -> Union[str, Markup]:
    safe_markdown_tags = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'b', 'i', 'strong', 'em', 'tt', 'p', 'br', 'span', 'div', 'blockquote', 'code', 'hr', 'ul', 'ol', 'li', 'dd', 'dt', 'img', 'a'}
    safe_markdown_attrs = {'img': {'src', 'alt', 'title'}, 'a': {'href', 'alt', 'title'}}
    safe = md.markdown(raw or '', extensions=['markdown.extensions.tables', 'markdown.extensions.fenced_code', 'markdown.extensions.codehilite'])
    safe = nh3.clean(safe, tags=safe_markdown_tags, attributes=safe_markdown_attrs)
    if markup_wrap:
        safe = Markup(safe)
    return safe

def readfile(file_path: str) -> str:
    with open(file_path) as f:
        content = f.read()
    return content

def generic_find_constraint_name(table: str, columns: Set[str], referenced: str, database: SQLA) -> Optional[str]:
    """Utility to find a constraint name in alembic migrations"""
    tbl = sa.Table(table, database.metadata, autoload=True, autoload_with=database.engine)
    for fk in tbl.foreign_key_constraints:
        if fk.referred_table.name == referenced and set(fk.column_keys) == columns:
            return fk.name
    return None

def generic_find_fk_constraint_name(table: str, columns: Set[str], referenced: str, insp: Inspector) -> Optional[str]:
    """Utility to find a foreign-key constraint name in alembic migrations"""
    for fk in insp.get_foreign_keys(table):
        if fk['referred_table'] == referenced and set(fk['referred_columns']) == columns:
            return fk['name']
    return None

def generic_find_fk_constraint_names(table: str, columns: Set[str], referenced: str, insp: Inspector) -> Set[str]:
    """Utility to find foreign-key constraint names in alembic migrations"""
    names = set()
    for fk in insp.get_foreign_keys(table):
        if fk['referred_table'] == referenced and set(fk['referred_columns']) == columns:
            names.add(fk['name'])
    return names

def generic_find_uq_constraint_name(table: str, columns: Set[str], insp: Inspector) -> Optional[str]:
    """Utility to find a unique constraint name in alembic migrations"""
    for uq in insp.get_unique_constraints(table):
        if columns == set(uq['column_names']):
            return uq['name']
    return None

def get_datasource_full_name(database_name: str, datasource_name: str, catalog: Optional[str] = None, schema: Optional[str] = None) -> str:
    parts = [database_name, catalog, schema, datasource_name]
    return '.'.join([f'[{part}]' for part in parts if part])

class SigalrmTimeout:
    """
    To be used in a ``with`` block and timeout its content.
    """

    def __init__(self, seconds: int = 1, error_message: str = 'Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum: int, frame: Any) -> None:
        logger.error('Process timed out', exc_info=True)
        raise SupersetTimeoutException(error_type=SupersetErrorType.BACKEND_TIMEOUT_ERROR, message=self.error_message, level=ErrorLevel.ERROR, extra={'timeout': self.seconds})

    def __enter__(self) -> None:
        try:
            if threading.current_thread() == threading.main_thread():
                signal.signal(signal.SIGALRM, self.handle_timeout)
                signal.alarm(self.seconds)
        except ValueError as ex:
            logger.warning("timeout can't be used in the current context")
            logger.exception(ex)

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        try:
            signal.alarm(0)
        except ValueError as ex:
            logger.warning("timeout can't be used in the current context")
            logger.exception(ex)

class TimerTimeout:

    def __init__(self, seconds: int = 1, error_message: str = 'Timeout'):
        self.seconds = seconds
        self.error_message = error_message
        self.timer = threading.Timer(seconds, _thread.interrupt_main)

    def __enter__(self) -> None:
        self.timer.start()

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.timer.cancel()
        if type is KeyboardInterrupt:
            raise SupersetTimeoutException(error_type=SupersetErrorType.BACKEND_TIMEOUT_ERROR, message=self.error_message, level=ErrorLevel.ERROR, extra={'timeout': self.seconds})

timeout = TimerTimeout if platform.system() == 'Windows' else SigalrmTimeout

def pessimistic_connection_handling(some_engine: Engine) -> None:

    @event.listens_for(some_engine, 'engine_connect')
    def ping_connection(connection: Connection, branch: bool) -> None:
        if branch:
            return
        save_should_close_with_result = connection.should_close_with_result
        connection.should_close_with_result = False
        try:
            connection.scalar(select([1]))
        except exc.DBAPIError as err:
            if err.connection_invalidated:
                connection.scalar(select([1]))
            else:
                raise
        finally:
            connection.should_close_with_result = save_should_close_with_result
    if some_engine.dialect.name == 'sqlite':

        @event.listens_for(some_engine, 'connect')
        def set_sqlite_pragma(connection: Connection, *args: Any) -> None:
            """
            Enable foreign key support for