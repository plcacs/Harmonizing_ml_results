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
from typing import Any, Callable, cast, Dict, List, NamedTuple, Optional, TypedDict, TypeVar, Union
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
DTTM_ALIAS: str = '__timestamp'
TIME_COMPARISON: str = '__'
JS_MAX_INTEGER: int = 9007199254740991
InputType = TypeVar('InputType')
ADHOC_FILTERS_REGEX: re.Pattern = re.compile('^adhoc_filters')


class AdhocMetricExpressionType(StrEnum):
    SIMPLE: str = 'SIMPLE'
    SQL: str = 'SQL'


class AnnotationType(StrEnum):
    FORMULA: str = 'FORMULA'
    INTERVAL: str = 'INTERVAL'
    EVENT: str = 'EVENT'
    TIME_SERIES: str = 'TIME_SERIES'


class GenericDataType(IntEnum):
    """
    Generic database column type that fits both frontend and backend.
    """
    NUMERIC: int = 0
    STRING: int = 1
    TEMPORAL: int = 2
    BOOLEAN: int = 3


class DatasourceType(StrEnum):
    TABLE: str = 'table'
    DATASET: str = 'dataset'
    QUERY: str = 'query'
    SAVEDQUERY: str = 'saved_query'
    VIEW: str = 'view'


class LoggerLevel(StrEnum):
    INFO: str = 'info'
    WARNING: str = 'warning'
    EXCEPTION: str = 'exception'


class HeaderDataType(TypedDict):
    notification_format: str
    owners: List[int]
    notification_type: str
    notification_source: Optional[str]
    chart_id: Optional[int]
    dashboard_id: Optional[int]
    slack_channels: Optional[List[str]]


class DatasourceDict(TypedDict):
    type: str
    id: int


class AdhocFilterClause(TypedDict, total=(False)):
    clause: str
    expressionType: str
    filterOptionName: Optional[str]
    comparator: Optional[FilterValues]
    operator: str
    subject: str
    isExtra: Optional[bool]
    sqlExpression: Optional[str]


class QueryObjectFilterClause(TypedDict, total=(False)):
    col: Column
    op: str
    val: Optional[FilterValues]
    grain: Optional[str]
    isExtra: Optional[bool]


class ExtraFiltersTimeColumnType(StrEnum):
    TIME_COL: str = '__time_col'
    TIME_GRAIN: str = '__time_grain'
    TIME_ORIGIN: str = '__time_origin'
    TIME_RANGE: str = '__time_range'


class ExtraFiltersReasonType(StrEnum):
    NO_TEMPORAL_COLUMN: str = 'no_temporal_column'
    COL_NOT_IN_DATASOURCE: str = 'not_in_datasource'


class FilterOperator(StrEnum):
    """
    Operators used filter controls
    """
    EQUALS: str = '=='
    NOT_EQUALS: str = '!='
    GREATER_THAN: str = '>'
    LESS_THAN: str = '<'
    GREATER_THAN_OR_EQUALS: str = '>='
    LESS_THAN_OR_EQUALS: str = '<='
    LIKE: str = 'LIKE'
    NOT_LIKE: str = 'NOT LIKE'
    ILIKE: str = 'ILIKE'
    IS_NULL: str = 'IS NULL'
    IS_NOT_NULL: str = 'IS NOT NULL'
    IN: str = 'IN'
    NOT_IN: str = 'NOT IN'
    IS_TRUE: str = 'IS TRUE'
    IS_FALSE: str = 'IS FALSE'
    TEMPORAL_RANGE: str = 'TEMPORAL_RANGE'


class FilterStringOperators(StrEnum):
    EQUALS: tuple[str, ...] = ('EQUALS',)
    NOT_EQUALS: tuple[str, ...] = ('NOT_EQUALS',)
    LESS_THAN: tuple[str, ...] = ('LESS_THAN',)
    GREATER_THAN: tuple[str, ...] = ('GREATER_THAN',)
    LESS_THAN_OR_EQUAL: tuple[str, ...] = ('LESS_THAN_OR_EQUAL',)
    GREATER_THAN_OR_EQUAL: tuple[str, ...] = ('GREATER_THAN_OR_EQUAL',)
    IN: tuple[str, ...] = ('IN',)
    NOT_IN: tuple[str, ...] = ('NOT_IN',)
    ILIKE: tuple[str, ...] = ('ILIKE',)
    LIKE: tuple[str, ...] = ('LIKE',)
    IS_NOT_NULL: tuple[str, ...] = ('IS_NOT_NULL',)
    IS_NULL: tuple[str, ...] = ('IS_NULL',)
    LATEST_PARTITION: tuple[str, ...] = ('LATEST_PARTITION',)
    IS_TRUE: tuple[str, ...] = ('IS_TRUE',)
    IS_FALSE: tuple[str, ...] = ('IS_FALSE',)


class PostProcessingBoxplotWhiskerType(StrEnum):
    """
    Calculate cell contribution to row/column total
    """
    TUKEY: str = 'tukey'
    MINMAX: str = 'min/max'
    PERCENTILE: str = 'percentile'


class PostProcessingContributionOrientation(StrEnum):
    """
    Calculate cell contribution to row/column total
    """
    ROW: str = 'row'
    COLUMN: str = 'column'


class QuerySource(Enum):
    """
    The source of a SQL query.
    """
    CHART: int = 0
    DASHBOARD: int = 1
    SQL_LAB: int = 2


class QueryStatus(StrEnum):
    """Enum-type class for query statuses"""
    STOPPED: str = 'stopped'
    FAILED: str = 'failed'
    PENDING: str = 'pending'
    RUNNING: str = 'running'
    SCHEDULED: str = 'scheduled'
    SUCCESS: str = 'success'
    FETCHING: str = 'fetching'
    TIMED_OUT: str = 'timed_out'


class DashboardStatus(StrEnum):
    """Dashboard status used for frontend filters"""
    PUBLISHED: str = 'published'
    DRAFT: str = 'draft'


class ReservedUrlParameters(StrEnum):
    """
    Reserved URL parameters that are used internally by Superset. These will not be
    passed to chart queries, as they control the behavior of the UI.
    """
    STANDALONE: str = 'standalone'
    EDIT_MODE: str = 'edit'

    @staticmethod
    def is_standalone_mode():
        standalone_param: Optional[str] = request.args.get(
            ReservedUrlParameters.STANDALONE.value)
        standalone: Optional[bool] = bool(standalone_param and 
            standalone_param != 'false' and standalone_param != '0')
        return standalone


class RowLevelSecurityFilterType(StrEnum):
    REGULAR: str = 'Regular'
    BASE: str = 'Base'


class ColumnTypeSource(Enum):
    GET_TABLE: int = 1
    CURSOR_DESCRIPTION: int = 2


class ColumnSpec(NamedTuple):
    sqla_type: Union[TypeEngine, str]
    generic_type: GenericDataType
    is_dttm: bool
    python_date_format: Optional[str] = None


def parse_js_uri_path_item(item, unquote=True, eval_undefined=False):
    """Parse an uri path item made with js.

    :param item: an uri path component
    :param unquote: Perform unquoting of string using urllib.parse.unquote_plus()
    :param eval_undefined: When set to True and item is either 'null' or 'undefined',
    assume item is undefined and return None.
    :return: Either None, the original item or unquoted item
    """
    item = None if eval_undefined and item in ('null', 'undefined') else item
    return unquote_plus(item) if unquote and item else item


def cast_to_num(value):
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
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def cast_to_boolean(value):
    """Casts a value to a boolean

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


def error_msg_from_exception(ex):
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
    msg: str = ''
    if hasattr(ex, 'message'):
        if isinstance(ex.message, dict):
            msg = ex.message.get('message', '')
        elif ex.message:
            msg = ex.message
    return str(msg) or str(ex)


def markdown(raw, markup_wrap=False):
    safe_markdown_tags: set[str] = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'b',
        'i', 'strong', 'em', 'tt', 'p', 'br', 'span', 'div', 'blockquote',
        'code', 'hr', 'ul', 'ol', 'li', 'dd', 'dt', 'img', 'a'}
    safe_markdown_attrs: Dict[str, set[str]] = {'img': {'src', 'alt',
        'title'}, 'a': {'href', 'alt', 'title'}}
    safe: str = md.markdown(raw or '', extensions=[
        'markdown.extensions.tables', 'markdown.extensions.fenced_code',
        'markdown.extensions.codehilite'])
    safe = nh3.clean(safe, tags=safe_markdown_tags, attributes=
        safe_markdown_attrs)
    if markup_wrap:
        safe = Markup(safe)
    return safe


def readfile(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content: str = f.read()
    return content


def generic_find_constraint_name(table, columns, referenced, database):
    """Utility to find a constraint name in alembic migrations"""
    tbl: sa.Table = sa.Table(table, database.metadata, autoload_with=
        database.engine)
    for fk in tbl.foreign_key_constraints:
        if fk.referred_table.name == referenced and set(fk.column_keys
            ) == columns:
            return fk.name
    return None


def generic_find_fk_constraint_name(table, columns, referenced, insp):
    """Utility to find a foreign-key constraint name in alembic migrations"""
    for fk in insp.get_foreign_keys(table):
        if fk['referred_table'] == referenced and set(fk['referred_columns']
            ) == columns:
            return fk['name']
    return None


def generic_find_fk_constraint_names(table, columns, referenced, insp):
    """Utility to find foreign-key constraint names in alembic migrations"""
    names: set[str] = set()
    for fk in insp.get_foreign_keys(table):
        if fk['referred_table'] == referenced and set(fk['referred_columns']
            ) == columns:
            names.add(fk['name'])
    return names


def generic_find_uq_constraint_name(table, columns, insp):
    """Utility to find a unique constraint name in alembic migrations"""
    for uq in insp.get_unique_constraints(table):
        if columns == set(uq['column_names']):
            return uq['name']
    return None


def get_datasource_full_name(database_name, datasource_name, catalog=None,
    schema=None):
    parts: List[str] = [database_name, catalog, schema, datasource_name]
    return '.'.join([f'[{part}]' for part in parts if part])


class SigalrmTimeout:
    """
    To be used in a ``with`` block and timeout its content.
    """

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds: int = seconds
        self.error_message: str = error_message

    def handle_timeout(self, signum, frame):
        logger.error('Process timed out', exc_info=True)
        raise SupersetTimeoutException(error_type=SupersetErrorType.
            BACKEND_TIMEOUT_ERROR, message=self.error_message, level=
            ErrorLevel.ERROR, extra={'timeout': self.seconds})

    def __enter__(self):
        try:
            if threading.current_thread() == threading.main_thread():
                signal.signal(signal.SIGALRM, self.handle_timeout)
                signal.alarm(self.seconds)
        except ValueError as ex:
            logger.warning("timeout can't be used in the current context")
            logger.exception(ex)

    def __exit__(self, type, value, tb):
        try:
            signal.alarm(0)
        except ValueError as ex:
            logger.warning("timeout can't be used in the current context")
            logger.exception(ex)


class TimerTimeout:

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds: int = seconds
        self.error_message: str = error_message
        self.timer: threading.Timer = threading.Timer(seconds, _thread.
            interrupt_main)

    def __enter__(self):
        self.timer.start()

    def __exit__(self, type, value, tb):
        self.timer.cancel()
        if type is KeyboardInterrupt:
            raise SupersetTimeoutException(error_type=SupersetErrorType.
                BACKEND_TIMEOUT_ERROR, message=self.error_message, level=
                ErrorLevel.ERROR, extra={'timeout': self.seconds})


timeout: Union[type[TimerTimeout], type[SigalrmTimeout]
    ] = TimerTimeout if platform.system() == 'Windows' else SigalrmTimeout


def pessimistic_connection_handling(some_engine):

    @event.listens_for(some_engine, 'engine_connect')
    def ping_connection(connection, branch):
        if branch:
            return
        save_should_close_with_result: bool = (connection.
            should_close_with_result)
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
        def set_sqlite_pragma(connection, *args: Any):
            """
            Enable foreign key support for SQLite.

            :param connection: The SQLite connection
            :param \\*args: Additional positional arguments
            :see: https://docs.sqlalchemy.org/en/latest/dialects/sqlite.html
            """
            with closing(connection.cursor()) as cursor:
                cursor.execute('PRAGMA foreign_keys=ON')


def send_email_smtp(to, subject, html_content, config, files=None, data=
    None, pdf=None, images=None, dryrun=False, cc=None, bcc=None,
    mime_subtype='mixed', header_data=None):
    """
    Send an email with html content, eg:
    send_email_smtp(
        'test@example.com', 'foo', '<b>Foo</b> bar',['/dev/null'], dryrun=True)
    """
    smtp_mail_from: str = config['SMTP_MAIL_FROM']
    smtp_mail_to: List[str] = get_email_address_list(to)
    msg: MIMEMultipart = MIMEMultipart(mime_subtype)
    msg['Subject'] = subject
    msg['From'] = smtp_mail_from
    msg['To'] = ', '.join(smtp_mail_to)
    msg.preamble = 'This is a multi-part message in MIME format.'
    recipients: List[str] = smtp_mail_to
    if cc:
        smtp_mail_cc: List[str] = get_email_address_list(cc)
        msg['Cc'] = ', '.join(smtp_mail_cc)
        recipients = recipients + smtp_mail_cc
    smtp_mail_bcc: List[str] = []
    if bcc:
        smtp_mail_bcc = get_email_address_list(bcc)
        recipients = recipients + smtp_mail_bcc
    msg['Date'] = formatdate(localtime=True)
    mime_text: MIMEText = MIMEText(html_content, 'html')
    msg.attach(mime_text)
    for fname in (files or []):
        basename: str = os.path.basename(fname)
        with open(fname, 'rb') as f:
            msg.attach(MIMEApplication(f.read(), Content_Disposition=
                f"attachment; filename='{basename}'", Name=basename))
    for name, body in (data or {}).items():
        msg.attach(MIMEApplication(body, Content_Disposition=
            f"attachment; filename='{name}'", Name=name))
    for name, body_pdf in (pdf or {}).items():
        msg.attach(MIMEApplication(body_pdf, Content_Disposition=
            f"attachment; filename='{name}'", Name=name))
    for msgid, imgdata in (images or {}).items():
        formatted_time: str = formatdate(localtime=True)
        file_name: str = f'{subject} {formatted_time}'
        image: MIMEImage = MIMEImage(imgdata, name=file_name)
        image.add_header('Content-ID', f'<{msgid}>')
        image.add_header('Content-Disposition', 'inline')
        msg.attach(image)
    msg_mutator: Callable[..., MIMEMultipart] = config['EMAIL_HEADER_MUTATOR']
    new_msg: MIMEMultipart = msg_mutator(msg, **header_data or {})
    new_to: List[str] = new_msg['To'].split(', ') if 'To' in new_msg else []
    new_cc: List[str] = new_msg['Cc'].split(', ') if 'Cc' in new_msg else []
    new_recipients: List[str] = new_to + new_cc + smtp_mail_bcc
    if set(new_recipients) != set(recipients):
        recipients = new_recipients
    send_mime_email(smtp_mail_from, recipients, new_msg, config, dryrun=dryrun)


def send_mime_email(e_from, e_to, mime_msg, config, dryrun=False):
    smtp_host: str = config['SMTP_HOST']
    smtp_port: int = config['SMTP_PORT']
    smtp_user: str = config['SMTP_USER']
    smtp_password: str = config['SMTP_PASSWORD']
    smtp_starttls: bool = config['SMTP_STARTTLS']
    smtp_ssl: bool = config['SMTP_SSL']
    smtp_ssl_server_auth: bool = config['SMTP_SSL_SERVER_AUTH']
    if dryrun:
        logger.info('Dryrun enabled, email notification content is below:')
        logger.info(mime_msg.as_string())
        return
    ssl_context: Optional[ssl.SSLContext] = ssl.create_default_context(
        ) if smtp_ssl_server_auth else None
    smtp: Union[smtplib.SMTP_SSL, smtplib.SMTP] = smtplib.SMTP_SSL(smtp_host,
        smtp_port, context=ssl_context) if smtp_ssl else smtplib.SMTP(smtp_host
        , smtp_port)
    if smtp_starttls:
        smtp.starttls(context=ssl_context)
    if smtp_user and smtp_password:
        smtp.login(smtp_user, smtp_password)
    logger.debug('Sent an email to %s', str(e_to))
    smtp.sendmail(e_from, e_to, mime_msg.as_string())
    smtp.quit()


def get_email_address_list(address_string):
    address_string_list: List[str] = []
    if isinstance(address_string, str):
        address_string_list = re.split(',|\\s|;', address_string)
    return [x.strip() for x in address_string_list if x.strip()]


def choicify(values):
    """Takes an iterable and makes an iterable of tuples with it"""
    return [(v, v) for v in values]


def zlib_compress(data):
    """
    Compress things in a py2/3 safe fashion
    >>> json_str = '{"test": 1}'
    >>> blob = zlib_compress(json_str)
    """
    if isinstance(data, str):
        return zlib.compress(bytes(data, 'utf-8'))
    return zlib.compress(data)


def zlib_decompress(blob, decode=True):
    """
    Decompress things to a string in a py2/3 safe fashion
    >>> json_str = '{"test": 1}'
    >>> blob = zlib_compress(json_str)
    >>> got_str = zlib_decompress(blob)
    >>> got_str == json_str
    True
    """
    if isinstance(blob, bytes):
        decompressed: bytes = zlib.decompress(blob)
    else:
        decompressed = zlib.decompress(bytes(blob, 'utf-8'))
    return decompressed.decode('utf-8') if decode else decompressed


def simple_filter_to_adhoc(filter_clause, clause='where'):
    result: AdhocFilterClause = {'clause': clause.upper(), 'expressionType':
        'SIMPLE', 'comparator': filter_clause.get('val'), 'operator':
        filter_clause['op'], 'subject': cast(str, filter_clause['col'])}
    if filter_clause.get('isExtra') is not None:
        result['isExtra'] = True
    result['filterOptionName'] = md5_sha_from_dict(cast(Dict[Any, Any], result)
        )
    return result


def form_data_to_adhoc(form_data, clause):
    if clause not in ('where', 'having'):
        raise ValueError(__('Unsupported clause type: %(clause)s', clause=
            clause))
    result: AdhocFilterClause = {'clause': clause.upper(), 'expressionType':
        'SQL', 'sqlExpression': form_data.get(clause)}
    result['filterOptionName'] = md5_sha_from_dict(cast(Dict[Any, Any], result)
        )
    return result


def merge_extra_form_data(form_data):
    """
    Merge extra form data (appends and overrides) into the main payload
    and add applied time extras to the payload.
    """
    filter_keys: List[str] = ['filters', 'adhoc_filters']
    extra_form_data: Dict[str, Any] = form_data.pop('extra_form_data', {})
    append_filters: Optional[List[QueryObjectFilterClause]
        ] = extra_form_data.get('filters', None)
    for key in [key for key in EXTRA_FORM_DATA_APPEND_KEYS if key not in
        filter_keys]:
        extra_value: Dict[str, Any] = extra_form_data.get(key, {})
        form_value: Dict[str, Any] = form_data.get(key, {})
        form_value.update(extra_value)
        if form_value:
            form_data['key'] = extra_value
    for src_key, target_key in EXTRA_FORM_DATA_OVERRIDE_REGULAR_MAPPINGS.items(
        ):
        value: Any = extra_form_data.get(src_key)
        if value is not None:
            form_data[target_key] = value
    extras: Dict[str, Any] = form_data.get('extras', {})
    for key in EXTRA_FORM_DATA_OVERRIDE_EXTRA_KEYS:
        value: Any = extra_form_data.get(key)
        if value is not None:
            extras[key] = value
    if extras:
        form_data['extras'] = extras
    adhoc_filters: List[AdhocFilterClause] = form_data.get('adhoc_filters', [])
    form_data['adhoc_filters'] = adhoc_filters
    append_adhoc_filters: Optional[List[AdhocFilterClause]
        ] = extra_form_data.get('adhoc_filters', [])
    adhoc_filters.extend({'isExtra': True, **adhoc_filter} for adhoc_filter in
        append_adhoc_filters)
    if append_filters:
        for key, value in form_data.items():
            if re.match('adhoc_filter.*', key):
                value.extend(simple_filter_to_adhoc({'isExtra': True, **
                    fltr}) for fltr in append_filters if fltr)
    if form_data.get('time_range') and not form_data.get('granularity_sqla'):
        for adhoc_filter in form_data.get('adhoc_filters', []):
            if adhoc_filter.get('operator') == 'TEMPORAL_RANGE':
                adhoc_filter['comparator'] = form_data['time_range']


def merge_extra_filters(form_data):
    form_data.setdefault('applied_time_extras', {})
    adhoc_filters: List[AdhocFilterClause] = form_data.get('adhoc_filters', [])
    form_data['adhoc_filters'] = adhoc_filters
    merge_extra_form_data(form_data)
    if 'extra_filters' in form_data:
        date_options: Dict[str, str] = {'__time_range': 'time_range',
            '__time_col': 'granularity_sqla', '__time_grain': 'time_grain_sqla'
            }

        def get_filter_key(f):
            if 'expressionType' in f:
                return f"{f['subject']}__{f['operator']}"
            return f"{f['col']}__{f['op']}"
        existing_filters: Dict[str, Any] = {}
        for existing in adhoc_filters:
            if existing['expressionType'] == 'SIMPLE' and existing.get(
                'comparator') is not None and existing.get('subject'
                ) is not None:
                existing_filters[get_filter_key(existing)] = existing[
                    'comparator']
        for filtr in form_data['extra_filters']:
            filtr['isExtra'] = True
            filter_column: str = filtr['col']
            if (time_extra := date_options.get(filter_column)):
                time_extra_value: Optional[str] = filtr.get('val')
                if time_extra_value and time_extra_value != NO_TIME_RANGE:
                    form_data[time_extra] = time_extra_value
                    form_data['applied_time_extras'][filter_column
                        ] = time_extra_value
            elif filtr['val']:
                filter_key: Optional[str] = get_filter_key(filtr)
                if filter_key and filter_key in existing_filters:
                    if isinstance(filtr['val'], list):
                        if isinstance(existing_filters[filter_key], list):
                            if set(existing_filters[filter_key]) != set(filtr
                                ['val']):
                                adhoc_filters.append(simple_filter_to_adhoc
                                    (filtr))
                        else:
                            adhoc_filters.append(simple_filter_to_adhoc(filtr))
                    elif filtr['val'] != existing_filters[filter_key]:
                        adhoc_filters.append(simple_filter_to_adhoc(filtr))
                else:
                    adhoc_filters.append(simple_filter_to_adhoc(filtr))
        del form_data['extra_filters']


def merge_request_params(form_data, params):
    """
    Merge request parameters to the key `url_params` in form_data. Only updates
    or appends parameters to `form_data` that are defined in `params; preexisting
    parameters not defined in params are left unchanged.

    :param form_data: object to be updated
    :param params: request parameters received via query string
    """
    url_params: Dict[str, Any] = form_data.get('url_params', {})
    for key, value in params.items():
        if key in ('form_data', 'r'):
            continue
        url_params[key] = value
    form_data['url_params'] = url_params


def user_label(user):
    """Given a user ORM FAB object, returns a label"""
    if user:
        if user.first_name and user.last_name:
            return user.first_name + ' ' + user.last_name
        return user.username
    return None


def get_example_default_schema():
    """
    Return the default schema of the examples database, if any.
    """
    database = get_example_database()
    with database.get_sqla_engine() as engine:
        return inspect(engine).default_schema_name


def backend():
    return get_example_database().backend


def is_adhoc_metric(metric):
    return isinstance(metric, dict) and 'expressionType' in metric


def is_adhoc_column(column):
    return isinstance(column, dict) and {'label', 'sqlExpression'}.issubset(
        column.keys())


def is_base_axis(column):
    return is_adhoc_column(column) and column.get('columnType') == 'BASE_AXIS'


def get_base_axis_columns(columns):
    return [column for column in columns or [] if is_base_axis(column)]


def get_non_base_axis_columns(columns):
    return [column for column in columns or [] if not is_base_axis(column)]


def get_base_axis_labels(columns):
    return tuple(get_column_name(column) for column in
        get_base_axis_columns(columns))


def get_x_axis_label(columns):
    labels: tuple[str, ...] = get_base_axis_labels(columns)
    return labels[0] if labels else None


def get_column_name(column, verbose_map=None):
    """
    Extract label from column

    :param column: object to extract label from
    :param verbose_map: verbose_map from dataset for optional mapping from
                        raw name to verbose name
    :return: String representation of column
    :raises ValueError: if metric object is invalid
    """
    if isinstance(column, dict):
        if (label := column.get('label')):
            return label
        if (expr := column.get('sqlExpression')):
            return expr
    if isinstance(column, str):
        verbose_map = verbose_map or {}
        return verbose_map.get(column, column)
    raise ValueError('Missing label')


def get_metric_name(metric, verbose_map=None):
    """
    Extract label from metric

    :param metric: object to extract label from
    :param verbose_map: verbose_map from dataset for optional mapping from
                        raw name to verbose name
    :return: String representation of metric
    :raises ValueError: if metric object is invalid
    """
    if is_adhoc_metric(metric):
        adhoc_metric: AdhocMetric = cast(AdhocMetric, metric)
        if (label := adhoc_metric.get('label')):
            return label
        if (expression_type := adhoc_metric.get('expressionType')
            ) == AdhocMetricExpressionType.SQL:
            if (sql_expression := adhoc_metric.get('sqlExpression')):
                return sql_expression
        if expression_type == AdhocMetricExpressionType.SIMPLE:
            column: AdhocMetricColumn = adhoc_metric.get('column') or {}
            column_name: Optional[str] = column.get('column_name')
            aggregate: Optional[str] = adhoc_metric.get('aggregate')
            if column and aggregate:
                return f'{aggregate}({column_name})'
            if column_name:
                return column_name
    if isinstance(metric, str):
        verbose_map = verbose_map or {}
        return verbose_map.get(metric, metric)
    raise ValueError(__('Invalid metric object: %(metric)s', metric=str(
        metric)))


def get_column_names(columns, verbose_map=None):
    return [column for column in [get_column_name(column, verbose_map) for
        column in columns or []] if column]


def get_metric_names(metrics, verbose_map=None):
    return [metric for metric in [get_metric_name(metric, verbose_map) for
        metric in metrics or []] if metric]


def get_first_metric_name(metrics, verbose_map=None):
    metric_labels: List[str] = get_metric_names(metrics, verbose_map)
    return metric_labels[0] if metric_labels else None


def ensure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as ex:
        if not (os.path.isdir(path) and ex.errno == errno.EEXIST):
            raise


def convert_legacy_filters_into_adhoc(form_data):
    if not form_data.get('adhoc_filters'):
        adhoc_filters: List[AdhocFilterClause] = []
        form_data['adhoc_filters'] = adhoc_filters
        for clause in ('having', 'where'):
            if clause in form_data and form_data[clause] != '':
                adhoc_filters.append(form_data_to_adhoc(form_data, clause))
        if 'filters' in form_data:
            adhoc_filters.extend(simple_filter_to_adhoc(fltr, 'where') for
                fltr in form_data['filters'] if fltr is not None)
    for key in ('filters', 'having', 'where'):
        if key in form_data:
            del form_data[key]


def split_adhoc_filters_into_base_filters(form_data):
    """
    Mutates form data to restructure the adhoc filters in the form of the three base
    filters, `where`, `having`, and `filters` which represent free form where sql,
    free form having sql, and structured where clauses.
    """
    adhoc_filters: Any = form_data.get('adhoc_filters')
    if isinstance(adhoc_filters, list):
        simple_where_filters: List[Dict[str, Any]] = []
        sql_where_filters: List[str] = []
        sql_having_filters: List[str] = []
        for adhoc_filter in adhoc_filters:
            expression_type: Optional[str] = adhoc_filter.get('expressionType')
            clause: Optional[str] = adhoc_filter.get('clause')
            if expression_type == 'SIMPLE':
                if clause == 'WHERE':
                    simple_where_filters.append({'col': adhoc_filter.get(
                        'subject'), 'op': adhoc_filter.get('operator'),
                        'val': adhoc_filter.get('comparator')})
            elif expression_type == 'SQL':
                sql_expression: Optional[str] = adhoc_filter.get(
                    'sqlExpression')
                sql_expression = sanitize_clause(sql_expression)
                if clause == 'WHERE':
                    sql_where_filters.append(sql_expression)
                elif clause == 'HAVING':
                    sql_having_filters.append(sql_expression)
        form_data['where'] = ' AND '.join([f'({sql})' for sql in
            sql_where_filters])
        form_data['having'] = ' AND '.join([f'({sql})' for sql in
            sql_having_filters])
        form_data['filters'] = simple_where_filters


def get_user():
    """
    Get the current user (if defined).

    :returns: The current user
    """
    return g.user if hasattr(g, 'user') else None


def get_username():
    """
    Get username (if defined) associated with the current user.

    :returns: The username
    """
    try:
        return g.user.username
    except Exception:
        return None


def get_user_id():
    """
    Get the user identifier (if defined) associated with the current user.

    Though the Flask-AppBuilder `User` and Flask-Login  `AnonymousUserMixin` and
    `UserMixin` models provide a convenience `get_id` method, for generality, the
    identifier is encoded as a `str` whereas in Superset all identifiers are encoded as
    an `int`.

    returns: The user identifier
    """
    try:
        return g.user.id
    except Exception:
        return None


def get_user_email():
    """
    Get the email (if defined) associated with the current user.

    :returns: The email
    """
    try:
        return g.user.email
    except Exception:
        return None


@contextmanager
def override_user(user, force=True):
    """
    Temporarily override the current user per `flask.g` with the specified user.

    Sometimes, often in the context of async Celery tasks, it is useful to switch the
    current user (which may be undefined) to different one, execute some SQLAlchemy
    tasks et al. and then revert back to the original one.

    :param user: The override user
    :param force: Whether to override the current user if set
    """
    if hasattr(g, 'user'):
        if force or g.user is None:
            current: Optional[User] = g.user
            g.user = user
            yield
            g.user = current
        else:
            yield
    else:
        g.user = user
        yield
        delattr(g, 'user')


def parse_ssl_cert(certificate):
    """
    Parses the contents of a certificate and returns a valid certificate object
    if valid.

    :param certificate: Contents of certificate file
    :return: Valid certificate instance
    :raises CertificateException: If certificate is not valid/unparseable
    """
    try:
        return load_pem_x509_certificate(certificate.encode('utf-8'),
            default_backend())
    except ValueError as ex:
        raise CertificateException('Invalid certificate') from ex


def create_ssl_cert_file(certificate):
    """
    This creates a certificate file that can be used to validate HTTPS
    sessions. A certificate is only written to disk once; on subsequent calls,
    only the path of the existing certificate is returned.

    :param certificate: The contents of the certificate
    :return: The path to the certificate file
    :raises CertificateException: If certificate is not valid/unparseable
    """
    filename: str = f'{md5_sha_from_str(certificate)}.crt'
    cert_dir: str = current_app.config['SSL_CERT_PATH']
    path: str = cert_dir if cert_dir else tempfile.gettempdir()
    path = os.path.join(path, filename)
    if not os.path.exists(path):
        parse_ssl_cert(certificate)
        with open(path, 'w', encoding='utf-8') as cert_file:
            cert_file.write(certificate)
    return path


def time_function(func, *args: Any, **kwargs: Any):
    """
    Measures the amount of time a function takes to execute in ms

    :param func: The function execution time to measure
    :param args: args to be passed to the function
    :param kwargs: kwargs to be passed to the function
    :return: A tuple with the duration and response from the function
    """
    start: float = default_timer()
    response: FlaskResponse = func(*args, **kwargs)
    stop: float = default_timer()
    return (stop - start) * 1000.0, response


def MediumText():
    return Text().with_variant(MEDIUMTEXT(), 'mysql')


def LongText():
    return Text().with_variant(LONGTEXT(), 'mysql')


def shortid():
    return f'{uuid.uuid4()}'[-12:]


class DatasourceName(NamedTuple):
    table: str
    schema: str
    catalog: Optional[str] = None


def get_stacktrace():
    if current_app.config['SHOW_STACKTRACE']:
        return traceback.format_exc()
    return None


def split(string, delimiter=' ', quote='"', escaped_quote='\\"'):
    """
    A split function that is aware of quotes and parentheses.

    :param string: string to split
    :param delimiter: string defining where to split, usually a comma or space
    :param quote: string, either a single or a double quote
    :param escaped_quote: string representing an escaped quote
    :return: list of strings
    """
    parens: int = 0
    quotes: bool = False
    i: int = 0
    for j, character in enumerate(string):
        complete: bool = parens == 0 and not quotes
        if complete and character == delimiter:
            yield string[i:j]
            i = j + len(delimiter)
        elif character == '(':
            parens += 1
        elif character == ')':
            parens -= 1
        elif character == quote:
            if quotes and string[j - len(escaped_quote):j] != escaped_quote:
                quotes = False
            elif not quotes:
                quotes = True
    yield string[i:]


T = TypeVar('T')


def as_list(x):
    """
    Wrap an object in a list if it's not a list.

    :param x: The object
    :returns: A list wrapping the object if it's not already a list
    """
    return x if isinstance(x, list) else [x]


def get_form_data_token(form_data):
    """
    Return the token contained within form data or generate a new one.

    :param form_data: chart form data
    :return: original token if predefined, otherwise new uuid4 based token
    """
    return form_data.get('token') or 'token_' + uuid.uuid4().hex[:8]


def get_column_name_from_column(column):
    """
    Extract the physical column that a column is referencing. If the column is
    an adhoc column, always returns `None`.

    :param column: Physical and ad-hoc column
    :return: column name if physical column, otherwise None
    """
    if is_adhoc_column(column):
        return None
    if isinstance(column, str):
        return column
    return None


def get_column_names_from_columns(columns):
    """
    Extract the physical columns that a list of columns are referencing. Ignore
    adhoc columns

    :param columns: Physical and adhoc columns
    :return: column names of all physical columns
    """
    return [col for col in map(get_column_name_from_column, columns) if col]


def get_column_name_from_metric(metric):
    """
    Extract the column that a metric is referencing. If the metric isn't
    a simple metric, always returns `None`.

    :param metric: Ad-hoc metric
    :return: column name if simple metric, otherwise None
    """
    if is_adhoc_metric(metric):
        adhoc_metric: AdhocMetric = cast(AdhocMetric, metric)
        if adhoc_metric['expressionType'] == AdhocMetricExpressionType.SIMPLE:
            column_dict: Dict[str, Any] = cast(Dict[str, Any], adhoc_metric
                .get('column', {}))
            return column_dict.get('column_name')
    return None


def get_column_names_from_metrics(metrics):
    """
    Extract the columns that a list of metrics are referencing. Excludes all
    SQL metrics.

    :param metrics: Ad-hoc metric
    :return: column name if simple metric, otherwise None
    """
    return [col for col in map(get_column_name_from_metric, metrics) if col]


def extract_dataframe_dtypes(df, datasource=None):
    """Serialize pandas/numpy dtypes to generic types"""
    inferred_type_map: Dict[str, GenericDataType] = {'floating':
        GenericDataType.NUMERIC, 'integer': GenericDataType.NUMERIC,
        'mixed-integer-float': GenericDataType.NUMERIC, 'decimal':
        GenericDataType.NUMERIC, 'boolean': GenericDataType.BOOLEAN,
        'datetime64': GenericDataType.TEMPORAL, 'datetime': GenericDataType
        .TEMPORAL, 'date': GenericDataType.TEMPORAL}
    columns_by_name: Dict[str, Any] = {}
    if datasource:
        for column in datasource.columns:
            if isinstance(column, dict):
                columns_by_name[column.get('column_name') or ''] = column
            else:
                columns_by_name[column.column_name] = column
    generic_types: List[GenericDataType] = []
    for column in df.columns:
        column_object: Any = columns_by_name.get(column)
        series: pd.Series = df[column]
        inferred_type: str = infer_dtype(series)
        if isinstance(column_object, dict):
            generic_type: GenericDataType = (GenericDataType.TEMPORAL if 
                column_object and column_object.get('is_dttm') else
                inferred_type_map.get(inferred_type, GenericDataType.STRING))
        else:
            generic_type = (GenericDataType.TEMPORAL if column_object and
                getattr(column_object, 'is_dttm', False) else
                inferred_type_map.get(inferred_type, GenericDataType.STRING))
        generic_types.append(generic_type)
    return generic_types


def extract_column_dtype(col):
    if col.is_temporal:
        return GenericDataType.TEMPORAL
    if col.is_numeric:
        return GenericDataType.NUMERIC
    return GenericDataType.STRING


def is_test():
    return parse_boolean_string(os.environ.get('SUPERSET_TESTENV', 'false'))


def get_time_filter_status(datasource, applied_time_extras):
    temporal_columns: set[Any] = {col.column_name for col in datasource.
        columns if col.is_dttm}
    applied: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []
    if (time_column := applied_time_extras.get(ExtraFiltersTimeColumnType.
        TIME_COL.value)):
        if time_column in temporal_columns:
            applied.append({'column': ExtraFiltersTimeColumnType.TIME_COL.
                value})
        else:
            rejected.append({'reason': ExtraFiltersReasonType.
                COL_NOT_IN_DATASOURCE.value, 'column':
                ExtraFiltersTimeColumnType.TIME_COL.value})
    if ExtraFiltersTimeColumnType.TIME_GRAIN.value in applied_time_extras:
        if temporal_columns:
            applied.append({'column': ExtraFiltersTimeColumnType.TIME_GRAIN
                .value})
        else:
            rejected.append({'reason': ExtraFiltersReasonType.
                NO_TEMPORAL_COLUMN.value, 'column':
                ExtraFiltersTimeColumnType.TIME_GRAIN.value})
    if applied_time_extras.get(ExtraFiltersTimeColumnType.TIME_RANGE.value):
        if temporal_columns:
            applied.append({'column': ExtraFiltersTimeColumnType.TIME_RANGE
                .value})
        else:
            rejected.append({'reason': ExtraFiltersReasonType.
                NO_TEMPORAL_COLUMN.value, 'column':
                ExtraFiltersTimeColumnType.TIME_RANGE.value})
    return applied, rejected


def format_list(items, sep=', ', quote='"'):
    quote_escaped: str = '\\' + quote
    return sep.join(f'{quote}{x.replace(quote, quote_escaped)}{quote}' for
        x in items)


def find_duplicates(items):
    """Find duplicate items in an iterable."""
    return [item for item, count in collections.Counter(items).items() if 
        count > 1]


def remove_duplicates(items, key=None):
    """Remove duplicate items in an iterable."""
    if not key:
        return list(dict.fromkeys(items).keys())
    seen: set[Any] = set()
    result: List[InputType] = []
    for item in items:
        item_key: Any = key(item)
        if item_key not in seen:
            seen.add(item_key)
            result.append(item)
    return result


@dataclass(frozen=True)
class DateColumn:
    col_label: str
    timestamp_format: Optional[str] = None
    offset: Optional[int] = None
    time_shift: Optional[str] = None

    def __hash__(self):
        return hash(self.col_label)

    def __eq__(self, other):
        return isinstance(other, DateColumn) and hash(self) == hash(other)

    @classmethod
    def get_legacy_time_column(cls, timestamp_format, offset, time_shift):
        return cls(timestamp_format=timestamp_format, offset=offset,
            time_shift=time_shift, col_label=DTTM_ALIAS)


def normalize_dttm_col(df, dttm_cols=tuple()):
    for _col in dttm_cols:
        if _col.col_label not in df.columns:
            continue
        if _col.timestamp_format in ('epoch_s', 'epoch_ms'):
            dttm_series: pd.Series = df[_col.col_label]
            if is_numeric_dtype(dttm_series):
                unit: str = _col.timestamp_format.replace('epoch_', '')
                df[_col.col_label] = pd.to_datetime(dttm_series, utc=False,
                    unit=unit, origin='unix', errors='raise', exact=False)
            else:
                df[_col.col_label] = dttm_series.apply(pd.Timestamp)
        else:
            df[_col.col_label] = pd.to_datetime(df[_col.col_label], utc=
                False, format=_col.timestamp_format, errors='raise', exact=
                False)
        if _col.offset:
            df[_col.col_label] += timedelta(hours=_col.offset)
        if _col.time_shift is not None:
            df[_col.col_label] += parse_human_timedelta(_col.time_shift)


def parse_boolean_string(bool_str):
    """
    Convert a string representation of a true/false value into a boolean

    >>> parse_boolean_string(None)
    False
    >>> parse_boolean_string('false')
    False
    >>> parse_boolean_string('true')
    True
    >>> parse_boolean_string('False')
    False
    >>> parse_boolean_string('True')
    True
    >>> parse_boolean_string('foo')
    False
    >>> parse_boolean_string('0')
    False
    >>> parse_boolean_string('1')
    True

    :param bool_str: string representation of a value that is assumed to be boolean
    :return: parsed boolean value
    """
    if bool_str is None:
        return False
    return bool_str.lower() in ('y', 'yes', 't', 'true', 'on', '1')


def apply_max_row_limit(limit, max_limit=None):
    """
    Override row limit if max global limit is defined

    :param limit: requested row limit
    :param max_limit: Maximum allowed row limit
    :return: Capped row limit

    >>> apply_max_row_limit(100000, 10)
    10
    >>> apply_max_row_limit(10, 100000)
    10
    >>> apply_max_row_limit(0, 10000)
    10000
    """
    if max_limit is None:
        max_limit = current_app.config['SQL_MAX_ROW']
    if limit != 0:
        return min(max_limit, limit)
    return max_limit


def create_zip(files):
    buf: BytesIO = BytesIO()
    with ZipFile(buf, 'w') as bundle:
        for filename, contents in files.items():
            with bundle.open(filename, 'w') as fp:
                fp.write(contents)
    buf.seek(0)
    return buf


def check_is_safe_zip(zip_file):
    """
    Checks whether a ZIP file is safe, raises SupersetException if not.

    :param zip_file:
    :return:
    """
    uncompress_size: int = 0
    compress_size: int = 0
    for zip_file_element in zip_file.infolist():
        if zip_file_element.file_size > current_app.config[
            'ZIPPED_FILE_MAX_SIZE']:
            raise SupersetException(
                'Found file with size above allowed threshold')
        uncompress_size += zip_file_element.file_size
        compress_size += zip_file_element.compress_size
    compress_ratio: float = uncompress_size / compress_size
    if compress_ratio > current_app.config['ZIP_FILE_MAX_COMPRESS_RATIO']:
        raise SupersetException('Zip compress ratio above allowed threshold')


def remove_extra_adhoc_filters(form_data):
    """
    Remove filters from slice data that originate from a filter box or native filter
    """
    adhoc_filters: Dict[str, Any] = {key: value for key, value in form_data
        .items() if ADHOC_FILTERS_REGEX.match(key)}
    for key, value in adhoc_filters.items():
        form_data[key] = [filter_ for filter_ in value or [] if not filter_
            .get('isExtra')]


def to_int(v, value_if_invalid=0):
    try:
        return int(v)
    except (ValueError, TypeError):
        return value_if_invalid


@dataclass
class DateColumn:
    col_label: str
    timestamp_format: Optional[str] = None
    offset: Optional[int] = None
    time_shift: Optional[str] = None

    def __hash__(self):
        return hash(self.col_label)

    def __eq__(self, other):
        return isinstance(other, DateColumn) and hash(self) == hash(other)

    @classmethod
    def get_legacy_time_column(cls, timestamp_format, offset, time_shift):
        return cls(timestamp_format=timestamp_format, offset=offset,
            time_shift=time_shift, col_label=DTTM_ALIAS)


def get_stacktrace():
    if current_app.config['SHOW_STACKTRACE']:
        return traceback.format_exc()
    return None


def split(string, delimiter=' ', quote='"', escaped_quote='\\"'):
    """
    A split function that is aware of quotes and parentheses.

    :param string: string to split
    :param delimiter: string defining where to split, usually a comma or space
    :param quote: string, either a single or a double quote
    :param escaped_quote: string representing an escaped quote
    :return: list of strings
    """
    parens: int = 0
    quotes: bool = False
    i: int = 0
    for j, character in enumerate(string):
        complete: bool = parens == 0 and not quotes
        if complete and character == delimiter:
            yield string[i:j]
            i = j + len(delimiter)
        elif character == '(':
            parens += 1
        elif character == ')':
            parens -= 1
        elif character == quote:
            if quotes and string[j - len(escaped_quote):j] != escaped_quote:
                quotes = False
            elif not quotes:
                quotes = True
    yield string[i:]


def as_list(x):
    """
    Wrap an object in a list if it's not a list.

    :param x: The object
    :returns: A list wrapping the object if it's not already a list
    """
    return x if isinstance(x, list) else [x]


def get_form_data_token(form_data):
    """
    Return the token contained within form data or generate a new one.

    :param form_data: chart form data
    :return: original token if predefined, otherwise new uuid4 based token
    """
    return form_data.get('token') or 'token_' + uuid.uuid4().hex[:8]


def get_column_name_from_column(column):
    """
    Extract the physical column that a column is referencing. If the column is
    an adhoc column, always returns `None`.

    :param column: Physical and ad-hoc column
    :return: column name if physical column, otherwise None
    """
    if is_adhoc_column(column):
        return None
    if isinstance(column, str):
        return column
    return None


def get_column_names_from_columns(columns):
    """
    Extract the physical columns that a list of columns are referencing. Ignore
    adhoc columns

    :param columns: Physical and adhoc columns
    :return: column names of all physical columns
    """
    return [col for col in map(get_column_name_from_column, columns) if col]


def get_column_name_from_metric(metric):
    """
    Extract the column that a metric is referencing. If the metric isn't
    a simple metric, always returns `None`.

    :param metric: Ad-hoc metric
    :return: column name if simple metric, otherwise None
    """
    if is_adhoc_metric(metric):
        adhoc_metric: AdhocMetric = cast(AdhocMetric, metric)
        if adhoc_metric['expressionType'] == AdhocMetricExpressionType.SIMPLE:
            column_dict: Dict[str, Any] = cast(Dict[str, Any], adhoc_metric
                .get('column', {}))
            return column_dict.get('column_name')
    return None


def get_column_names_from_metrics(metrics):
    """
    Extract the columns that a list of metrics are referencing. Excludes all
    SQL metrics.

    :param metrics: Ad-hoc metric
    :return: column name if simple metric, otherwise None
    """
    return [col for col in map(get_column_name_from_metric, metrics) if col]


def extract_dataframe_dtypes(df, datasource=None):
    """Serialize pandas/numpy dtypes to generic types"""
    inferred_type_map: Dict[str, GenericDataType] = {'floating':
        GenericDataType.NUMERIC, 'integer': GenericDataType.NUMERIC,
        'mixed-integer-float': GenericDataType.NUMERIC, 'decimal':
        GenericDataType.NUMERIC, 'boolean': GenericDataType.BOOLEAN,
        'datetime64': GenericDataType.TEMPORAL, 'datetime': GenericDataType
        .TEMPORAL, 'date': GenericDataType.TEMPORAL}
    columns_by_name: Dict[str, Any] = {}
    if datasource:
        for column in datasource.columns:
            if isinstance(column, dict):
                columns_by_name[column.get('column_name') or ''] = column
            else:
                columns_by_name[column.column_name] = column
    generic_types: List[GenericDataType] = []
    for column in df.columns:
        column_object: Any = columns_by_name.get(column)
        series: pd.Series = df[column]
        inferred_type: str = infer_dtype(series)
        if isinstance(column_object, dict):
            generic_type: GenericDataType = (GenericDataType.TEMPORAL if 
                column_object and column_object.get('is_dttm') else
                inferred_type_map.get(inferred_type, GenericDataType.STRING))
        else:
            generic_type = (GenericDataType.TEMPORAL if column_object and
                getattr(column_object, 'is_dttm', False) else
                inferred_type_map.get(inferred_type, GenericDataType.STRING))
        generic_types.append(generic_type)
    return generic_types


def extract_column_dtype(col):
    if col.is_temporal:
        return GenericDataType.TEMPORAL
    if col.is_numeric:
        return GenericDataType.NUMERIC
    return GenericDataType.STRING


def is_test():
    return parse_boolean_string(os.environ.get('SUPERSET_TESTENV', 'false'))


def get_time_filter_status(datasource, applied_time_extras):
    temporal_columns: set[Any] = {col.column_name for col in datasource.
        columns if col.is_dttm}
    applied: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []
    if (time_column := applied_time_extras.get(ExtraFiltersTimeColumnType.
        TIME_COL.value)):
        if time_column in temporal_columns:
            applied.append({'column': ExtraFiltersTimeColumnType.TIME_COL.
                value})
        else:
            rejected.append({'reason': ExtraFiltersReasonType.
                COL_NOT_IN_DATASOURCE.value, 'column':
                ExtraFiltersTimeColumnType.TIME_COL.value})
    if ExtraFiltersTimeColumnType.TIME_GRAIN.value in applied_time_extras:
        if temporal_columns:
            applied.append({'column': ExtraFiltersTimeColumnType.TIME_GRAIN
                .value})
        else:
            rejected.append({'reason': ExtraFiltersReasonType.
                NO_TEMPORAL_COLUMN.value, 'column':
                ExtraFiltersTimeColumnType.TIME_GRAIN.value})
    if applied_time_extras.get(ExtraFiltersTimeColumnType.TIME_RANGE.value):
        if temporal_columns:
            applied.append({'column': ExtraFiltersTimeColumnType.TIME_RANGE
                .value})
        else:
            rejected.append({'reason': ExtraFiltersReasonType.
                NO_TEMPORAL_COLUMN.value, 'column':
                ExtraFiltersTimeColumnType.TIME_RANGE.value})
    return applied, rejected


def format_list(items, sep=', ', quote='"'):
    quote_escaped: str = '\\' + quote
    return sep.join(f'{quote}{x.replace(quote, quote_escaped)}{quote}' for
        x in items)


def find_duplicates(items):
    """Find duplicate items in an iterable."""
    return [item for item, count in collections.Counter(items).items() if 
        count > 1]


def remove_duplicates(items, key=None):
    """Remove duplicate items in an iterable."""
    if not key:
        return list(dict.fromkeys(items).keys())
    seen: set[Any] = set()
    result: List[InputType] = []
    for item in items:
        item_key: Any = key(item)
        if item_key not in seen:
            seen.add(item_key)
            result.append(item)
    return result


@dataclass(frozen=True)
class DateColumn:
    col_label: str
    timestamp_format: Optional[str] = None
    offset: Optional[int] = None
    time_shift: Optional[str] = None

    def __hash__(self):
        return hash(self.col_label)

    def __eq__(self, other):
        return isinstance(other, DateColumn) and hash(self) == hash(other)

    @classmethod
    def get_legacy_time_column(cls, timestamp_format, offset, time_shift):
        return cls(timestamp_format=timestamp_format, offset=offset,
            time_shift=time_shift, col_label=DTTM_ALIAS)


def normalize_dttm_col(df, dttm_cols=tuple()):
    for _col in dttm_cols:
        if _col.col_label not in df.columns:
            continue
        if _col.timestamp_format in ('epoch_s', 'epoch_ms'):
            dttm_series: pd.Series = df[_col.col_label]
            if is_numeric_dtype(dttm_series):
                unit: str = _col.timestamp_format.replace('epoch_', '')
                df[_col.col_label] = pd.to_datetime(dttm_series, utc=False,
                    unit=unit, origin='unix', errors='raise', exact=False)
            else:
                df[_col.col_label] = dttm_series.apply(pd.Timestamp)
        else:
            df[_col.col_label] = pd.to_datetime(df[_col.col_label], utc=
                False, format=_col.timestamp_format, errors='raise', exact=
                False)
        if _col.offset:
            df[_col.col_label] += timedelta(hours=_col.offset)
        if _col.time_shift is not None:
            df[_col.col_label] += parse_human_timedelta(_col.time_shift)


def parse_boolean_string(bool_str):
    """
    Convert a string representation of a true/false value into a boolean

    >>> parse_boolean_string(None)
    False
    >>> parse_boolean_string('false')
    False
    >>> parse_boolean_string('true')
    True
    >>> parse_boolean_string('False')
    False
    >>> parse_boolean_string('True')
    True
    >>> parse_boolean_string('foo')
    False
    >>> parse_boolean_string('0')
    False
    >>> parse_boolean_string('1')
    True

    :param bool_str: string representation of a value that is assumed to be boolean
    :return: parsed boolean value
    """
    if bool_str is None:
        return False
    return bool_str.lower() in ('y', 'yes', 't', 'true', 'on', '1')


def apply_max_row_limit(limit, max_limit=None):
    """
    Override row limit if max global limit is defined

    :param limit: requested row limit
    :param max_limit: Maximum allowed row limit
    :return: Capped row limit

    >>> apply_max_row_limit(100000, 10)
    10
    >>> apply_max_row_limit(10, 100000)
    10
    >>> apply_max_row_limit(0, 10000)
    10000
    """
    if max_limit is None:
        max_limit = current_app.config['SQL_MAX_ROW']
    if limit != 0:
        return min(max_limit, limit)
    return max_limit


def create_zip(files):
    buf: BytesIO = BytesIO()
    with ZipFile(buf, 'w') as bundle:
        for filename, contents in files.items():
            with bundle.open(filename, 'w') as fp:
                fp.write(contents)
    buf.seek(0)
    return buf


def check_is_safe_zip(zip_file):
    """
    Checks whether a ZIP file is safe, raises SupersetException if not.

    :param zip_file:
    :return:
    """
    uncompress_size: int = 0
    compress_size: int = 0
    for zip_file_element in zip_file.infolist():
        if zip_file_element.file_size > current_app.config[
            'ZIPPED_FILE_MAX_SIZE']:
            raise SupersetException(
                'Found file with size above allowed threshold')
        uncompress_size += zip_file_element.file_size
        compress_size += zip_file_element.compress_size
    compress_ratio: float = uncompress_size / compress_size
    if compress_ratio > current_app.config['ZIP_FILE_MAX_COMPRESS_RATIO']:
        raise SupersetException('Zip compress ratio above allowed threshold')


def remove_extra_adhoc_filters(form_data):
    """
    Remove filters from slice data that originate from a filter box or native filter
    """
    adhoc_filters: Dict[str, Any] = {key: value for key, value in form_data
        .items() if ADHOC_FILTERS_REGEX.match(key)}
    for key, value in adhoc_filters.items():
        form_data[key] = [filter_ for filter_ in value or [] if not filter_
            .get('isExtra')]


def to_int(v, value_if_invalid=0):
    try:
        return int(v)
    except (ValueError, TypeError):
        return value_if_invalid


@dataclass
class DateColumn:
    col_label: str
    timestamp_format: Optional[str] = None
    offset: Optional[int] = None
    time_shift: Optional[str] = None

    def __hash__(self):
        return hash(self.col_label)

    def __eq__(self, other):
        return isinstance(other, DateColumn) and hash(self) == hash(other)

    @classmethod
    def get_legacy_time_column(cls, timestamp_format, offset, time_shift):
        return cls(timestamp_format=timestamp_format, offset=offset,
            time_shift=time_shift, col_label=DTTM_ALIAS)


def normalize_dttm_col(df, dttm_cols=tuple()):
    for _col in dttm_cols:
        if _col.col_label not in df.columns:
            continue
        if _col.timestamp_format in ('epoch_s', 'epoch_ms'):
            dttm_series: pd.Series = df[_col.col_label]
            if is_numeric_dtype(dttm_series):
                unit: str = _col.timestamp_format.replace('epoch_', '')
                df[_col.col_label] = pd.to_datetime(dttm_series, utc=False,
                    unit=unit, origin='unix', errors='raise', exact=False)
            else:
                df[_col.col_label] = dttm_series.apply(pd.Timestamp)
        else:
            df[_col.col_label] = pd.to_datetime(df[_col.col_label], utc=
                False, format=_col.timestamp_format, errors='raise', exact=
                False)
        if _col.offset:
            df[_col.col_label] += timedelta(hours=_col.offset)
        if _col.time_shift is not None:
            df[_col.col_label] += parse_human_timedelta(_col.time_shift)


def parse_boolean_string(bool_str):
    """
    Convert a string representation of a true/false value into a boolean

    >>> parse_boolean_string(None)
    False
    >>> parse_boolean_string('false')
    False
    >>> parse_boolean_string('true')
    True
    >>> parse_boolean_string('False')
    False
    >>> parse_boolean_string('True')
    True
    >>> parse_boolean_string('foo')
    False
    >>> parse_boolean_string('0')
    False
    >>> parse_boolean_string('1')
    True

    :param bool_str: string representation of a value that is assumed to be boolean
    :return: parsed boolean value
    """
    if bool_str is None:
        return False
    return bool_str.lower() in ('y', 'yes', 't', 'true', 'on', '1')


def apply_max_row_limit(limit, max_limit=None):
    """
    Override row limit if max global limit is defined

    :param limit: requested row limit
    :param max_limit: Maximum allowed row limit
    :return: Capped row limit

    >>> apply_max_row_limit(100000, 10)
    10
    >>> apply_max_row_limit(10, 100000)
    10
    >>> apply_max_row_limit(0, 10000)
    10000
    """
    if max_limit is None:
        max_limit = current_app.config['SQL_MAX_ROW']
    if limit != 0:
        return min(max_limit, limit)
    return max_limit


def create_zip(files):
    buf: BytesIO = BytesIO()
    with ZipFile(buf, 'w') as bundle:
        for filename, contents in files.items():
            with bundle.open(filename, 'w') as fp:
                fp.write(contents)
    buf.seek(0)
    return buf


def check_is_safe_zip(zip_file):
    """
    Checks whether a ZIP file is safe, raises SupersetException if not.

    :param zip_file:
    :return:
    """
    uncompress_size: int = 0
    compress_size: int = 0
    for zip_file_element in zip_file.infolist():
        if zip_file_element.file_size > current_app.config[
            'ZIPPED_FILE_MAX_SIZE']:
            raise SupersetException(
                'Found file with size above allowed threshold')
        uncompress_size += zip_file_element.file_size
        compress_size += zip_file_element.compress_size
    compress_ratio: float = uncompress_size / compress_size
    if compress_ratio > current_app.config['ZIP_FILE_MAX_COMPRESS_RATIO']:
        raise SupersetException('Zip compress ratio above allowed threshold')


def remove_extra_adhoc_filters(form_data):
    """
    Remove filters from slice data that originate from a filter box or native filter
    """
    adhoc_filters: Dict[str, Any] = {key: value for key, value in form_data
        .items() if ADHOC_FILTERS_REGEX.match(key)}
    for key, value in adhoc_filters.items():
        form_data[key] = [filter_ for filter_ in value or [] if not filter_
            .get('isExtra')]


def to_int(v, value_if_invalid=0):
    try:
        return int(v)
    except (ValueError, TypeError):
        return value_if_invalid
