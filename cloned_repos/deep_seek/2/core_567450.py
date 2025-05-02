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

logging.getLogger('MARKDOWN').setLevel(logging.INFO)
logger = logging.getLogger(__name__)
DTTM_ALIAS = '__timestamp'
TIME_COMPARISON = '__'
JS_MAX_INTEGER = 9007199254740991

InputType = TypeVar('InputType')
T = TypeVar('T')

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
    TUKEY = 'tukey'
    MINMAX = 'min/max'
    PERCENTILE = 'percentile'

class PostProcessingContributionOrientation(StrEnum):
    ROW = 'row'
    COLUMN = 'column'

class QuerySource(Enum):
    CHART = 0
    DASHBOARD = 1
    SQL_LAB = 2

class QueryStatus(StrEnum):
    STOPPED = 'stopped'
    FAILED = 'failed'
    PENDING = 'pending'
    RUNNING = 'running'
    SCHEDULED = 'scheduled'
    SUCCESS = 'success'
    FETCHING = 'fetching'
    TIMED_OUT = 'timed_out'

class DashboardStatus(StrEnum):
    PUBLISHED = 'published'
    DRAFT = 'draft'

class ReservedUrlParameters(StrEnum):
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
    item = None if eval_undefined and item in ('null', 'undefined') else item
    return unquote_plus(item) if unquote and item else item

def cast_to_num(value: Optional[Union[str, int, float]]) -> Optional[Union[int, float]]:
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

def cast_to_boolean(value: Optional[Union[str, int, float, bool]]) -> Optional[bool]:
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
    msg = ''
    if hasattr(ex, 'message'):
        if isinstance(ex.message, dict):
            msg = ex.message.get('message')
        elif ex.message:
            msg = ex.message
    return str(msg) or str(ex)

def markdown(raw: Optional[str], markup_wrap: bool = False) -> Union[str, Markup]:
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
    tbl = sa.Table(table, database.metadata, autoload=True, autoload_with=database.engine)
    for fk in tbl.foreign_key_constraints:
        if fk.referred_table.name == referenced and set(fk.column_keys) == columns:
            return fk.name
    return None

def generic_find_fk_constraint_name(table: str, columns: Set[str], referenced: str, insp: Inspector) -> Optional[str]:
    for fk in insp.get_foreign_keys(table):
        if fk['referred_table'] == referenced and set(fk['referred_columns']) == columns:
            return fk['name']
    return None

def generic_find_fk_constraint_names(table: str, columns: Set[str], referenced: str, insp: Inspector) -> Set[str]:
    names = set()
    for fk in insp.get_foreign_keys(table):
        if fk['referred_table'] == referenced and set(fk['referred_columns']) == columns:
            names.add(fk['name'])
    return names

def generic_find_uq_constraint_name(table: str, columns: Set[str], insp: Inspector) -> Optional[str]:
    for uq in insp.get_unique_constraints(table):
        if columns == set(uq['column_names']):
            return uq['name']
    return None

def get_datasource_full_name(database_name: str, datasource_name: str, catalog: Optional[str] = None, schema: Optional[str] = None) -> str:
    parts = [database_name, catalog, schema, datasource_name]
    return '.'.join([f'[{part}]' for part in parts if part])

class SigalrmTimeout:
    def __init__(self, seconds: int = 1, error_message: str = 'Timeout') -> None:
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
    def __init__(self, seconds: int = 1, error_message: str = 'Timeout') -> None:
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
    header_data: Optional[Dict[str, Any]] = None,
) -> None:
    smtp_mail_from = config['SMTP_MAIL_FROM']
    smtp_mail_to = get_email_address_list(to)
    msg = MIMEMultipart(mime_subtype)
    msg['Subject'] = subject
    msg['From'] = smtp_mail_from
    msg['To'] = ', '.join(smtp_mail_to)
    msg.preamble = 'This is a multi-part message in MIME format.'
    recipients = smtp_mail_to
    if cc:
        smtp_mail_cc = get_email_address_list(cc)
        msg['Cc'] = ', '.join(smtp_mail_cc)
        recipients = recipients + smtp_mail_cc
    smtp_mail_bcc = []
    if bcc:
        smtp_mail_bcc = get_email_address_list(bcc)
        recipients = recipients + smtp_mail_bcc
    msg['Date'] = formatdate(localtime=True)
    mime_text = MIMEText(html_content, 'html')
    msg.attach(mime_text)
    for fname in files or []:
        basename = os.path.basename(fname)
        with open(fname, 'rb') as f:
            msg.attach(MIMEApplication(f.read(), Content_Disposition=f"attachment; filename='{basename}'", Name=basename))
    for name, body in (data or {}).items():
        msg.attach(MIMEApplication(body, Content_Disposition=f"attachment; filename='{name}'", Name=name))
    for name, body_pdf in (pdf or {}).items():
        msg.attach(MIMEApplication(body_pdf, Content_Disposition=f"attachment; filename='{name}'", Name=name))
    for msgid, imgdata in (images or {}).items():
        formatted_time = formatdate(localtime=True)
        file_name = f'{subject} {formatted_time}'
        image = MIMEImage(imgdata, name=file_name)
        image.add_header('Content-ID', f'<{msgid}>')
        image.add_header('Content-Disposition', 'inline')
        msg.attach(image)
    msg_mutator = config['EMAIL_HEADER_MUTATOR']
    new_msg = msg_mutator(msg, **header_data or {})
    new_to = new_msg['To'].split(', ') if 'To' in new_msg else []
    new_cc = new_msg['Cc'].split(', ') if 'Cc' in new_msg else []
    new_recipients = new_to + new_cc + smtp_mail_bcc
    if set(new_recipients) != set(recipients):
        recipients = new_recipients
    send_mime_email(smtp_mail_from, recipients, new_msg, config, dryrun=dryrun)

def send_mime_email(
    e_from: str,
    e_to: List[str],
    mime_msg: MIMEMultipart,
    config: Dict[str, Any],
    dryrun: bool = False,
) -> None:
    smtp_host = config