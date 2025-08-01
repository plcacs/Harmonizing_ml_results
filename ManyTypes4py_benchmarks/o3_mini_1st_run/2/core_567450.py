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
from typing import Any, Callable, Dict, Iterator as TypingIterator, List, Optional, Tuple, Type, TypeVar, Union
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
from typing_extensions import TypedDict

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
        standalone_param: Optional[str] = request.args.get(ReservedUrlParameters.STANDALONE.value)
        standalone: bool = bool(standalone_param and standalone_param != 'false' and (standalone_param != '0'))
        return standalone


class RowLevelSecurityFilterType(StrEnum):
    REGULAR = 'Regular'
    BASE = 'Base'


class ColumnTypeSource(Enum):
    GET_TABLE = 1
    CURSOR_DESCRIPTION = 2


class ColumnSpec(Tuple):
    python_date_format = None


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
              decimal value and `None` if it can't be converted
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return None


def cast_to_boolean(value: Any) -> Optional[bool]:
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


def error_msg_from_exception(ex: Exception) -> str:
    """Translate exception into error message

    Database have different ways to handle exception. This function attempts
    to make sense of the exception object and construct a human readable
    sentence.
    """
    msg: str = ''
    if hasattr(ex, 'message'):
        if isinstance(ex.message, dict):
            msg = ex.message.get('message', '')
        elif ex.message:
            msg = ex.message
    return str(msg) or str(ex)


def markdown(raw: Optional[str], markup_wrap: bool = False) -> Union[str, Markup]:
    safe_markdown_tags = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'b', 'i', 'strong', 'em', 'tt', 'p', 'br', 'span', 'div', 'blockquote', 'code', 'hr', 'ul', 'ol', 'li', 'dd', 'dt', 'img', 'a'}
    safe_markdown_attrs = {'img': {'src', 'alt', 'title'}, 'a': {'href', 'alt', 'title'}}
    safe: str = md.markdown(raw or '', extensions=['markdown.extensions.tables', 'markdown.extensions.fenced_code', 'markdown.extensions.codehilite'])
    safe = nh3.clean(safe, tags=safe_markdown_tags, attributes=safe_markdown_attrs)
    if markup_wrap:
        safe = Markup(safe)
    return safe


def readfile(file_path: str) -> str:
    with open(file_path) as f:
        content: str = f.read()
    return content


def generic_find_constraint_name(table: str, columns: Iterable[str], referenced: str, database: Any) -> Optional[str]:
    """Utility to find a constraint name in alembic migrations"""
    tbl = sa.Table(table, database.metadata, autoload=True, autoload_with=database.engine)
    for fk in tbl.foreign_key_constraints:
        if fk.referred_table.name == referenced and set(fk.column_keys) == set(columns):
            return fk.name
    return None


def generic_find_fk_constraint_name(table: str, columns: Iterable[str], referenced: str, insp: Inspector) -> Optional[str]:
    """Utility to find a foreign-key constraint name in alembic migrations"""
    for fk in insp.get_foreign_keys(table):
        if fk['referred_table'] == referenced and set(fk['referred_columns']) == set(columns):
            return fk['name']
    return None


def generic_find_fk_constraint_names(table: str, columns: Iterable[str], referenced: str, insp: Inspector) -> Set[str]:
    """Utility to find foreign-key constraint names in alembic migrations"""
    names: Set[str] = set()
    for fk in insp.get_foreign_keys(table):
        if fk['referred_table'] == referenced and set(fk['referred_columns']) == set(columns):
            names.add(fk['name'])
    return names


def generic_find_uq_constraint_name(table: str, columns: Iterable[str], insp: Inspector) -> Optional[str]:
    """Utility to find a unique constraint name in alembic migrations"""
    for uq in insp.get_unique_constraints(table):
        if set(uq['column_names']) == set(columns):
            return uq['name']
    return None


def get_datasource_full_name(database_name: str, datasource_name: str, catalog: Optional[str] = None, schema: Optional[str] = None) -> str:
    parts: List[str] = [database_name, catalog, schema, datasource_name]
    return '.'.join([f'[{part}]' for part in parts if part])


class SigalrmTimeout:
    """
    To be used in a ``with`` block and timeout its content.
    """

    def __init__(self, seconds: int = 1, error_message: str = 'Timeout') -> None:
        self.seconds: int = seconds
        self.error_message: str = error_message

    def handle_timeout(self, signum: int, frame: Any) -> None:
        logger.error('Process timed out', exc_info=True)
        raise SupersetTimeoutException(
            error_type=SupersetErrorType.BACKEND_TIMEOUT_ERROR,
            message=self.error_message,
            level=ErrorLevel.ERROR,
            extra={'timeout': self.seconds}
        )

    def __enter__(self) -> None:
        try:
            if threading.current_thread() == threading.main_thread():
                signal.signal(signal.SIGALRM, self.handle_timeout)
                signal.alarm(self.seconds)
        except ValueError as ex:
            logger.warning("timeout can't be used in the current context")
            logger.exception(ex)

    def __exit__(self, type: Optional[Type[BaseException]], value: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        try:
            signal.alarm(0)
        except ValueError as ex:
            logger.warning("timeout can't be used in the current context")
            logger.exception(ex)


class TimerTimeout:
    def __init__(self, seconds: int = 1, error_message: str = 'Timeout') -> None:
        self.seconds: int = seconds
        self.error_message: str = error_message
        self.timer: threading.Timer = threading.Timer(seconds, _thread.interrupt_main)

    def __enter__(self) -> None:
        self.timer.start()

    def __exit__(self, type: Optional[Type[BaseException]], value: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        self.timer.cancel()
        if type is KeyboardInterrupt:
            raise SupersetTimeoutException(
                error_type=SupersetErrorType.BACKEND_TIMEOUT_ERROR,
                message=self.error_message,
                level=ErrorLevel.ERROR,
                extra={'timeout': self.seconds}
            )


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
        def set_sqlite_pragma(connection: Any, *args: Any) -> None:
            """
            Enable foreign key support for SQLite.
            :param connection: The SQLite connection
            :param *args: Additional positional arguments
            :see: https://docs.sqlalchemy.org/en/latest/dialects/sqlite.html
            """
            with closing(connection.cursor()) as cursor:
                cursor.execute('PRAGMA foreign_keys=ON')


def send_email_smtp(
    to: Union[str, List[str]],
    subject: str,
    html_content: str,
    config: Dict[str, Any],
    files: Optional[List[str]] = None,
    data: Optional[Dict[str, bytes]] = None,
    pdf: Optional[Dict[str, bytes]] = None,
    images: Optional[Dict[str, bytes]] = None,
    dryrun: bool = False,
    cc: Optional[Union[str, List[str]]] = None,
    bcc: Optional[Union[str, List[str]]] = None,
    mime_subtype: str = 'mixed',
    header_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Send an email with html content.
    """
    smtp_mail_from: str = config['SMTP_MAIL_FROM']
    smtp_mail_to: List[str] = get_email_address_list(to)  # type: ignore
    msg: MIMEMultipart = MIMEMultipart(mime_subtype)
    msg['Subject'] = subject
    msg['From'] = smtp_mail_from
    msg['To'] = ', '.join(smtp_mail_to)
    msg.preamble = 'This is a multi-part message in MIME format.'
    recipients: List[str] = smtp_mail_to.copy()
    if cc:
        smtp_mail_cc: List[str] = get_email_address_list(cc)
        msg['Cc'] = ', '.join(smtp_mail_cc)
        recipients.extend(smtp_mail_cc)
    smtp_mail_bcc: List[str] = []
    if bcc:
        smtp_mail_bcc = get_email_address_list(bcc)
        recipients.extend(smtp_mail_bcc)
    msg['Date'] = formatdate(localtime=True)
    mime_text: MIMEText = MIMEText(html_content, 'html')
    msg.attach(mime_text)
    for fname in files or []:
        basename: str = os.path.basename(fname)
        with open(fname, 'rb') as f:
            msg.attach(MIMEApplication(f.read(), Content_Disposition=f"attachment; filename='{basename}'", Name=basename))
    for name, body in (data or {}).items():
        msg.attach(MIMEApplication(body, Content_Disposition=f"attachment; filename='{name}'", Name=name))
    for name, body_pdf in (pdf or {}).items():
        msg.attach(MIMEApplication(body_pdf, Content_Disposition=f"attachment; filename='{name}'", Name=name))
    for msgid, imgdata in (images or {}).items():
        formatted_time: str = formatdate(localtime=True)
        file_name: str = f'{subject} {formatted_time}'
        image: MIMEImage = MIMEImage(imgdata, name=file_name)
        image.add_header('Content-ID', f'<{msgid}>')
        image.add_header('Content-Disposition', 'inline')
        msg.attach(image)
    msg_mutator: Callable[[MIMEMultipart, Any], MIMEMultipart] = config['EMAIL_HEADER_MUTATOR']
    new_msg: MIMEMultipart = msg_mutator(msg, **(header_data or {}))
    new_to: List[str] = new_msg['To'].split(', ') if 'To' in new_msg else []
    new_cc: List[str] = new_msg['Cc'].split(', ') if 'Cc' in new_msg else []
    new_recipients: List[str] = new_to + new_cc + smtp_mail_bcc
    if set(new_recipients) != set(recipients):
        recipients = new_recipients
    send_mime_email(smtp_mail_from, recipients, new_msg, config, dryrun=dryrun)


def send_mime_email(e_from: str, e_to: List[str], mime_msg: MIMEMultipart, config: Dict[str, Any], dryrun: bool = False) -> None:
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
    ssl_context: Optional[ssl.SSLContext] = ssl.create_default_context() if smtp_ssl_server_auth else None
    smtp: Union[smtplib.SMTP, smtplib.SMTP_SSL] = smtplib.SMTP_SSL(smtp_host, smtp_port, context=ssl_context) if smtp_ssl else smtplib.SMTP(smtp_host, smtp_port)
    if smtp_starttls:
        smtp.starttls(context=ssl_context)
    if smtp_user and smtp_password:
        smtp.login(smtp_user, smtp_password)
    logger.debug('Sent an email to %s', str(e_to))
    smtp.sendmail(e_from, e_to, mime_msg.as_string())
    smtp.quit()


def get_email_address_list(address_string: Union[str, Any]) -> List[str]:
    address_string_list: List[str] = []
    if isinstance(address_string, str):
        address_string_list = re.split(r',|\s|;', address_string)
    return [x.strip() for x in address_string_list if x.strip()]


T = TypeVar('T')


def choicify(values: Iterable[T]) -> List[Tuple[T, T]]:
    """Takes an iterable and makes an iterable of tuples with it"""
    return [(v, v) for v in values]


def zlib_compress(data: Union[str, bytes]) -> bytes:
    """
    Compress things in a py2/3 safe fashion
    >>> json_str = '{"test": 1}'
    >>> blob = zlib_compress(json_str)
    """
    if isinstance(data, str):
        return zlib.compress(data.encode('utf-8'))
    return zlib.compress(data)


def zlib_decompress(blob: Union[bytes, str], decode: bool = True) -> Union[str, bytes]:
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
        decompressed = zlib.decompress(blob.encode('utf-8'))
    return decompressed.decode('utf-8') if decode else decompressed


def simple_filter_to_adhoc(filter_clause: Dict[str, Any], clause: str = 'where') -> Dict[str, Any]:
    result: Dict[str, Any] = {
        'clause': clause.upper(),
        'expressionType': 'SIMPLE',
        'comparator': filter_clause.get('val'),
        'operator': filter_clause['op'],
        'subject': cast(str, filter_clause['col'])
    }
    if filter_clause.get('isExtra'):
        result['isExtra'] = True
    result['filterOptionName'] = md5_sha_from_dict(cast(Dict[Any, Any], result))
    return result


def form_data_to_adhoc(form_data: Dict[str, Any], clause: str) -> Dict[str, Any]:
    if clause not in ('where', 'having'):
        raise ValueError(__('Unsupported clause type: %(clause)s', clause=clause))
    result: Dict[str, Any] = {
        'clause': clause.upper(),
        'expressionType': 'SQL',
        'sqlExpression': form_data.get(clause)
    }
    result['filterOptionName'] = md5_sha_from_dict(cast(Dict[Any, Any], result))
    return result


def merge_extra_form_data(form_data: Dict[str, Any]) -> None:
    """
    Merge extra form data (appends and overrides) into the main payload
    and add applied time extras to the payload.
    """
    filter_keys = ['filters', 'adhoc_filters']
    extra_form_data: Dict[str, Any] = form_data.pop('extra_form_data', {})
    append_filters: Optional[Any] = extra_form_data.get('filters', None)
    for key in [key for key in EXTRA_FORM_DATA_APPEND_KEYS if key not in filter_keys]:
        extra_value: Dict[Any, Any] = getattr(extra_form_data, key, {})  # type: ignore
        form_value: Dict[Any, Any] = getattr(form_data, key, {})  # type: ignore
        form_value.update(extra_value)
        if form_value:
            form_data['key'] = extra_value
    for src_key, target_key in EXTRA_FORM_DATA_OVERRIDE_REGULAR_MAPPINGS.items():
        value = extra_form_data.get(src_key)
        if value is not None:
            form_data[target_key] = value
    extras: Dict[Any, Any] = form_data.get('extras', {})
    for key in EXTRA_FORM_DATA_OVERRIDE_EXTRA_KEYS:
        value = extra_form_data.get(key)
        if value is not None:
            extras[key] = value
    if extras:
        form_data['extras'] = extras
    adhoc_filters: List[Any] = form_data.get('adhoc_filters', [])
    form_data['adhoc_filters'] = adhoc_filters
    append_adhoc_filters = extra_form_data.get('adhoc_filters', [])
    adhoc_filters.extend(({'isExtra': True, **adhoc_filter} for adhoc_filter in append_adhoc_filters))
    if append_filters:
        for key, value in form_data.items():
            if re.match('adhoc_filter.*', key):
                value.extend((simple_filter_to_adhoc({'isExtra': True, **fltr}, clause='where') for fltr in append_filters if fltr))


    if form_data.get('time_range') and (not form_data.get('granularity_sqla')):
        for adhoc_filter in form_data.get('adhoc_filters', []):
            if adhoc_filter.get('operator') == 'TEMPORAL_RANGE':
                adhoc_filter['comparator'] = form_data['time_range']


def merge_extra_filters(form_data: Dict[str, Any]) -> None:
    form_data.setdefault('applied_time_extras', {})
    adhoc_filters: List[Any] = form_data.get('adhoc_filters', [])
    form_data['adhoc_filters'] = adhoc_filters
    merge_extra_form_data(form_data)
    if 'extra_filters' in form_data:
        date_options: Dict[str, str] = {'__time_range': 'time_range', '__time_col': 'granularity_sqla', '__time_grain': 'time_grain_sqla'}

        def get_filter_key(f: Dict[str, Any]) -> str:
            if 'expressionType' in f:
                return f"{f['subject']}__{f['operator']}"
            return f"{f['col']}__{f['op']}"

        existing_filters: Dict[str, Any] = {}
        for existing in adhoc_filters:
            if existing.get('expressionType') == 'SIMPLE' and existing.get('comparator') is not None and (existing.get('subject') is not None):
                existing_filters[get_filter_key(existing)] = existing['comparator']
        for filtr in form_data['extra_filters']:
            filtr['isExtra'] = True
            filter_column = filtr['col']
            if (time_extra := date_options.get(filter_column)):
                time_extra_value = filtr.get('val')
                if time_extra_value and time_extra_value != NO_TIME_RANGE:
                    form_data[time_extra] = time_extra_value
                    form_data['applied_time_extras'][filter_column] = time_extra_value
            elif filtr['val']:
                if (filter_key := get_filter_key(filtr)) in existing_filters:
                    if isinstance(filtr['val'], list):
                        if isinstance(existing_filters[filter_key], list):
                            if set(existing_filters[filter_key]) != set(filtr['val']):
                                adhoc_filters.append(simple_filter_to_adhoc(filtr, clause='where'))
                        else:
                            adhoc_filters.append(simple_filter_to_adhoc(filtr, clause='where'))
                    elif filtr['val'] != existing_filters[filter_key]:
                        adhoc_filters.append(simple_filter_to_adhoc(filtr, clause='where'))
                else:
                    adhoc_filters.append(simple_filter_to_adhoc(filtr, clause='where'))
        del form_data['extra_filters']


def merge_request_params(form_data: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Merge request parameters to the key `url_params` in form_data.
    Only updates parameters defined in params.
    """
    url_params: Dict[str, Any] = form_data.get('url_params', {})
    for key, value in params.items():
        if key in ('form_data', 'r'):
            continue
        url_params[key] = value
    form_data['url_params'] = url_params


def user_label(user: Optional[User]) -> Optional[str]:
    """Given a user ORM FAB object, returns a label"""
    if user:
        if user.first_name and user.last_name:
            return user.first_name + ' ' + user.last_name
        return user.username
    return None


def get_example_default_schema() -> Optional[str]:
    """
    Return the default schema of the examples database, if any.
    """
    database = get_example_database()
    with database.get_sqla_engine() as engine:
        return inspect(engine).default_schema_name


def backend() -> Any:
    return get_example_database().backend


def is_adhoc_metric(metric: Any) -> bool:
    return isinstance(metric, dict) and 'expressionType' in metric


def is_adhoc_column(column: Any) -> bool:
    return isinstance(column, dict) and {'label', 'sqlExpression'}.issubset(column.keys())


def is_base_axis(column: Any) -> bool:
    return is_adhoc_column(column) and column.get('columnType') == 'BASE_AXIS'


def get_base_axis_columns(columns: Optional[Iterable[Any]]) -> List[Any]:
    return [column for column in (columns or []) if is_base_axis(column)]


def get_non_base_axis_columns(columns: Optional[Iterable[Any]]) -> List[Any]:
    return [column for column in (columns or []) if not is_base_axis(column)]


def get_base_axis_labels(columns: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    return tuple(get_column_name(column) for column in get_base_axis_columns(columns))


def get_x_axis_label(columns: Optional[Iterable[Any]]) -> Optional[str]:
    labels: Tuple[str, ...] = get_base_axis_labels(columns)
    return labels[0] if labels else None


def get_column_name(column: Any, verbose_map: Optional[Dict[str, str]] = None) -> str:
    """
    Extract label from column
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


def get_metric_name(metric: Any, verbose_map: Optional[Dict[str, str]] = None) -> str:
    """
    Extract label from metric
    """
    if is_adhoc_metric(metric):
        if (label := metric.get('label')):
            return label
        if (expression_type := metric.get('expressionType')) == 'SQL':
            if (sql_expression := metric.get('sqlExpression')):
                return sql_expression
        if expression_type == 'SIMPLE':
            column = metric.get('column') or {}
            column_name = column.get('column_name')
            aggregate = metric.get('aggregate')
            if column and aggregate:
                return f'{aggregate}({column_name})'
            if column_name:
                return column_name
    if isinstance(metric, str):
        verbose_map = verbose_map or {}
        return verbose_map.get(metric, metric)
    raise ValueError(__('Invalid metric object: %(metric)s', metric=str(metric)))


def get_column_names(columns: Optional[Iterable[Any]], verbose_map: Optional[Dict[str, str]] = None) -> List[str]:
    return [col for col in (get_column_name(column, verbose_map) for column in (columns or [])) if col]


def get_metric_names(metrics: Optional[Iterable[Any]], verbose_map: Optional[Dict[str, str]] = None) -> List[str]:
    return [metric for metric in (get_metric_name(metric, verbose_map) for metric in (metrics or [])) if metric]


def get_first_metric_name(metrics: Optional[Iterable[Any]], verbose_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    metric_labels: List[str] = get_metric_names(metrics, verbose_map)
    return metric_labels[0] if metric_labels else None


def ensure_path_exists(path: str) -> None:
    try:
        os.makedirs(path)
    except OSError as ex:
        if not (os.path.isdir(path) and ex.errno == errno.EEXIST):
            raise


def convert_legacy_filters_into_adhoc(form_data: Dict[str, Any]) -> None:
    if not form_data.get('adhoc_filters'):
        adhoc_filters: List[Any] = []
        form_data['adhoc_filters'] = adhoc_filters
        for clause in ('having', 'where'):
            if clause in form_data and form_data[clause] != '':
                adhoc_filters.append(form_data_to_adhoc(form_data, clause))
        if 'filters' in form_data:
            adhoc_filters.extend((simple_filter_to_adhoc(fltr, clause='where') for fltr in form_data['filters'] if fltr is not None))
    for key in ('filters', 'having', 'where'):
        if key in form_data:
            del form_data[key]


def split_adhoc_filters_into_base_filters(form_data: Dict[str, Any]) -> None:
    """
    Mutates form data to restructure the adhoc filters in the form of the three base
    filters, `where`, `having`, and `filters`.
    """
    adhoc_filters = form_data.get('adhoc_filters')
    if isinstance(adhoc_filters, list):
        simple_where_filters: List[Dict[str, Any]] = []
        sql_where_filters: List[str] = []
        sql_having_filters: List[str] = []
        for adhoc_filter in adhoc_filters:
            expression_type = adhoc_filter.get('expressionType')
            clause = adhoc_filter.get('clause')
            if expression_type == 'SIMPLE':
                if clause == 'WHERE':
                    simple_where_filters.append({
                        'col': adhoc_filter.get('subject'),
                        'op': adhoc_filter.get('operator'),
                        'val': adhoc_filter.get('comparator')
                    })
            elif expression_type == 'SQL':
                sql_expression = adhoc_filter.get('sqlExpression')
                sql_expression = sanitize_clause(sql_expression)
                if clause == 'WHERE':
                    sql_where_filters.append(sql_expression)
                elif clause == 'HAVING':
                    sql_having_filters.append(sql_expression)
        form_data['where'] = ' AND '.join([f'({sql})' for sql in sql_where_filters])
        form_data['having'] = ' AND '.join([f'({sql})' for sql in sql_having_filters])
        form_data['filters'] = simple_where_filters


def get_user() -> Optional[Any]:
    """
    Get the current user (if defined).
    """
    return g.user if hasattr(g, 'user') else None


def get_username() -> Optional[str]:
    """
    Get username (if defined) associated with the current user.
    """
    try:
        return g.user.username
    except Exception:
        return None


def get_user_id() -> Optional[int]:
    """
    Get the user identifier (if defined) associated with the current user.
    """
    try:
        return g.user.id
    except Exception:
        return None


def get_user_email() -> Optional[str]:
    """
    Get the email (if defined) associated with the current user.
    """
    try:
        return g.user.email
    except Exception:
        return None


@contextmanager
def override_user(user: Any, force: bool = True) -> TypingIterator[None]:
    """
    Temporarily override the current user per `flask.g` with the specified user.
    """
    if hasattr(g, 'user'):
        if force or g.user is None:
            current = g.user
            g.user = user
            try:
                yield
            finally:
                g.user = current
        else:
            yield
    else:
        g.user = user
        try:
            yield
        finally:
            delattr(g, 'user')


def parse_ssl_cert(certificate: str) -> Certificate:
    """
    Parses the contents of a certificate and returns a valid certificate object
    if valid.
    """
    try:
        return load_pem_x509_certificate(certificate.encode('utf-8'), default_backend())
    except ValueError as ex:
        raise CertificateException('Invalid certificate') from ex


def create_ssl_cert_file(certificate: str) -> str:
    """
    Creates a certificate file that can be used to validate HTTPS sessions.
    """
    filename: str = f'{md5_sha_from_str(certificate)}.crt'
    cert_dir: str = current_app.config['SSL_CERT_PATH']
    path: str = cert_dir if cert_dir else tempfile.gettempdir()
    path = os.path.join(path, filename)
    if not os.path.exists(path):
        parse_ssl_cert(certificate)
        with open(path, 'w') as cert_file:
            cert_file.write(certificate)
    return path


def time_function(func: Callable[..., T], *args: Any, **kwargs: Any) -> Tuple[float, T]:
    """
    Measures the amount of time a function takes to execute in ms
    """
    start: float = default_timer()
    response: T = func(*args, **kwargs)
    stop: float = default_timer()
    return ((stop - start) * 1000.0, response)


def MediumText() -> TypeEngine:
    return Text().with_variant(MEDIUMTEXT(), 'mysql')


def LongText() -> TypeEngine:
    return Text().with_variant(LONGTEXT(), 'mysql')


def shortid() -> str:
    return f'{uuid.uuid4()}'[-12:]


class DatasourceName(NamedTuple):
    catalog: Optional[str] = None


def get_stacktrace() -> Optional[str]:
    if current_app.config['SHOW_STACKTRACE']:
        return traceback.format_exc()
    return None


def split(string: str, delimiter: str = ' ', quote: str = '"', escaped_quote: str = '\\"') -> Iterator[str]:
    """
    A split function that is aware of quotes and parentheses.
    """
    parens: int = 0
    quotes: bool = False
    i: int = 0
    for j, character in enumerate(string):
        complete: bool = parens == 0 and (not quotes)
        if complete and character == delimiter:
            yield string[i:j]
            i = j + len(delimiter)
        elif character == '(':
            parens += 1
        elif character == ')':
            parens -= 1
        elif character == quote:
            if quotes and string[j - len(escaped_quote) + 1:j + 1] != escaped_quote:
                quotes = False
            elif not quotes:
                quotes = True
    yield string[i:]


def as_list(x: Union[T, List[T]]) -> List[T]:
    """
    Wrap an object in a list if it's not already a list.
    """
    return x if isinstance(x, list) else [x]


def get_form_data_token(form_data: Dict[str, Any]) -> str:
    """
    Return the token contained within form data or generate a new one.
    """
    return form_data.get('token') or 'token_' + uuid.uuid4().hex[:8]


def get_column_name_from_column(column: Any) -> Optional[Any]:
    """
    Extract the physical column that a column is referencing.
    """
    if is_adhoc_column(column):
        return None
    return column


def get_column_names_from_columns(columns: Iterable[Any]) -> List[Any]:
    """
    Extract the physical columns that a list of columns are referencing.
    """
    return [col for col in map(get_column_name_from_column, columns) if col]


def get_column_name_from_metric(metric: Any) -> Optional[str]:
    """
    Extract the column that a metric is referencing.
    """
    if is_adhoc_metric(metric):
        metric = cast(AdhocMetric, metric)
        if metric['expressionType'] == AdhocMetricExpressionType.SIMPLE:
            return cast(dict[str, Any], metric['column'])['column_name']
    return None


def get_column_names_from_metrics(metrics: Iterable[Any]) -> List[str]:
    """
    Extract the columns that a list of metrics are referencing.
    """
    return [col for col in map(get_column_name_from_metric, metrics) if col]


def extract_dataframe_dtypes(df: pd.DataFrame, datasource: Optional[Any] = None) -> List[GenericDataType]:
    """Serialize pandas/numpy dtypes to generic types"""
    inferred_type_map: Dict[str, GenericDataType] = {
        'floating': GenericDataType.NUMERIC,
        'integer': GenericDataType.NUMERIC,
        'mixed-integer-float': GenericDataType.NUMERIC,
        'decimal': GenericDataType.NUMERIC,
        'boolean': GenericDataType.BOOLEAN,
        'datetime64': GenericDataType.TEMPORAL,
        'datetime': GenericDataType.TEMPORAL,
        'date': GenericDataType.TEMPORAL
    }
    columns_by_name: Dict[str, Any] = {}
    if datasource:
        for column in datasource.columns:
            if isinstance(column, dict):
                columns_by_name[column.get('column_name')] = column
            else:
                columns_by_name[column.column_name] = column
    generic_types: List[GenericDataType] = []
    for column in df.columns:
        column_object = columns_by_name.get(column)
        series = df[column]
        inferred_type: str = infer_dtype(series)
        if isinstance(column_object, dict):
            generic_type = GenericDataType.TEMPORAL if column_object and column_object.get('is_dttm') else inferred_type_map.get(inferred_type, GenericDataType.STRING)
        else:
            generic_type = GenericDataType.TEMPORAL if column_object and column_object.is_dttm else inferred_type_map.get(inferred_type, GenericDataType.STRING)
        generic_types.append(generic_type)
    return generic_types


def extract_column_dtype(col: Any) -> GenericDataType:
    if col.is_temporal:
        return GenericDataType.TEMPORAL
    if col.is_numeric:
        return GenericDataType.NUMERIC
    return GenericDataType.STRING


def is_test() -> bool:
    return parse_boolean_string(os.environ.get('SUPERSET_TESTENV', 'false'))


def get_time_filter_status(datasource: Any, applied_time_extras: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    temporal_columns = {col.column_name for col in datasource.columns if col.is_dttm}
    applied: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    if (time_column := applied_time_extras.get(ExtraFiltersTimeColumnType.TIME_COL)):
        if time_column in temporal_columns:
            applied.append({'column': ExtraFiltersTimeColumnType.TIME_COL})
        else:
            rejected.append({'reason': ExtraFiltersReasonType.COL_NOT_IN_DATASOURCE, 'column': ExtraFiltersTimeColumnType.TIME_COL})
    if ExtraFiltersTimeColumnType.TIME_GRAIN in applied_time_extras:
        if temporal_columns:
            applied.append({'column': ExtraFiltersTimeColumnType.TIME_GRAIN})
        else:
            rejected.append({'reason': ExtraFiltersReasonType.NO_TEMPORAL_COLUMN, 'column': ExtraFiltersTimeColumnType.TIME_GRAIN})
    if applied_time_extras.get(ExtraFiltersTimeColumnType.TIME_RANGE):
        if temporal_columns:
            applied.append({'column': ExtraFiltersTimeColumnType.TIME_RANGE})
        else:
            rejected.append({'reason': ExtraFiltersReasonType.NO_TEMPORAL_COLUMN, 'column': ExtraFiltersTimeColumnType.TIME_RANGE})
    return (applied, rejected)


def format_list(items: Iterable[str], sep: str = ', ', quote: str = '"') -> str:
    quote_escaped: str = '\\' + quote
    return sep.join((f'{quote}{x.replace(quote, quote_escaped)}{quote}' for x in items))


def find_duplicates(items: Iterable[T]) -> List[T]:
    """Find duplicate items in an iterable."""
    return [item for item, count in collections.Counter(items).items() if count > 1]


def remove_duplicates(items: Iterable[T], key: Optional[Callable[[T], Any]] = None) -> List[T]:
    """Remove duplicate items in an iterable."""
    if key is None:
        return list(dict.fromkeys(items).keys())
    seen: set = set()
    result: List[T] = []
    for item in items:
        item_key: Any = key(item)
        if item_key not in seen:
            seen.add(item_key)
            result.append(item)
    return result


@dataclass
class DateColumn:
    timestamp_format: Optional[str] = None
    offset: Optional[float] = None
    time_shift: Optional[str] = None
    col_label: str = DTTM_ALIAS

    def __hash__(self) -> int:
        return hash(self.col_label)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, DateColumn) and hash(self) == hash(other)

    @classmethod
    def get_legacy_time_column(cls, timestamp_format: Optional[str], offset: Optional[float], time_shift: Optional[str]) -> DateColumn:
        return cls(timestamp_format=timestamp_format, offset=offset, time_shift=time_shift, col_label=DTTM_ALIAS)


def normalize_dttm_col(df: pd.DataFrame, dttm_cols: Iterable[DateColumn] = ()) -> None:
    for _col in dttm_cols:
        if _col.col_label not in df.columns:
            continue
        if _col.timestamp_format in ('epoch_s', 'epoch_ms'):
            dttm_series = df[_col.col_label]
            if is_numeric_dtype(dttm_series):
                unit: str = _col.timestamp_format.replace('epoch_', '')
                df[_col.col_label] = pd.to_datetime(dttm_series, utc=False, unit=unit, origin='unix', errors='raise', exact=False)
            else:
                df[_col.col_label] = dttm_series.apply(pd.Timestamp)
        else:
            df[_col.col_label] = pd.to_datetime(df[_col.col_label], utc=False, format=_col.timestamp_format, errors='raise', exact=False)
        if _col.offset:
            df[_col.col_label] += timedelta(hours=_col.offset)
        if _col.time_shift is not None:
            df[_col.col_label] += parse_human_timedelta(_col.time_shift)


def parse_boolean_string(bool_str: Optional[str]) -> bool:
    """
    Convert a string representation of a true/false value into a boolean
    """
    if bool_str is None:
        return False
    return bool_str.lower() in ('y', 'yes', 'true', 't', 'on', '1')


def apply_max_row_limit(limit: int, max_limit: Optional[int] = None) -> int:
    """
    Override row limit if max global limit is defined
    """
    if max_limit is None:
        max_limit = current_app.config['SQL_MAX_ROW']
    if limit != 0:
        return min(max_limit, limit)
    return max_limit


def create_zip(files: Dict[str, bytes]) -> BytesIO:
    buf: BytesIO = BytesIO()
    with ZipFile(buf, 'w') as bundle:
        for filename, contents in files.items():
            with bundle.open(filename, 'w') as fp:
                fp.write(contents)
    buf.seek(0)
    return buf


def check_is_safe_zip(zip_file: ZipFile) -> None:
    """
    Checks whether a ZIP file is safe, raises SupersetException if not.
    """
    uncompress_size: int = 0
    compress_size: int = 0
    for zip_file_element in zip_file.infolist():
        if zip_file_element.file_size > current_app.config['ZIPPED_FILE_MAX_SIZE']:
            raise SupersetException('Found file with size above allowed threshold')
        uncompress_size += zip_file_element.file_size
        compress_size += zip_file_element.compress_size
    compress_ratio: float = uncompress_size / compress_size
    if compress_ratio > current_app.config['ZIP_FILE_MAX_COMPRESS_RATIO']:
        raise SupersetException('Zip compress ratio above allowed threshold')


def remove_extra_adhoc_filters(form_data: Dict[str, Any]) -> None:
    """
    Remove filters from slice data that originate from a filter box or native filter
    """
    adhoc_filters = {key: value for key, value in form_data.items() if ADHOC_FILTERS_REGEX.match(key)}
    for key, value in adhoc_filters.items():
        form_data[key] = [filter_ for filter_ in value or [] if not filter_.get('isExtra')]


def to_int(v: Any, value_if_invalid: int = 0) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return value_if_invalid