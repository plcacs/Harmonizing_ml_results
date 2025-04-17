# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Utility functions used across Superset"""

# pylint: disable=too-many-lines
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
from typing import Any, Callable, cast, NamedTuple, TYPE_CHECKING, TypedDict, TypeVar, Optional, Union, List, Dict, Set, Tuple
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

from superset.constants import (
    EXTRA_FORM_DATA_APPEND_KEYS,
    EXTRA_FORM_DATA_OVERRIDE_EXTRA_KEYS,
    EXTRA_FORM_DATA_OVERRIDE_REGULAR_MAPPINGS,
    NO_TIME_RANGE,
)
from superset.errors import ErrorLevel, SupersetErrorType
from superset.exceptions import (
    CertificateException,
    SupersetException,
    SupersetTimeoutException,
)
from superset.sql_parse import sanitize_clause
from superset.superset_typing import (
    AdhocColumn,
    AdhocMetric,
    AdhocMetricColumn,
    Column,
    FilterValues,
    FlaskResponse,
    FormData,
    Metric,
)
from superset.utils.backports import StrEnum
from superset.utils.database import get_example_database
from superset.utils.date_parser import parse_human_timedelta
from superset.utils.hashing import md5_sha_from_dict, md5_sha_from_str

if TYPE_CHECKING:
    from superset.connectors.sqla.models import BaseDatasource, TableColumn
    from superset.models.sql_lab import Query

logging.getLogger("MARKDOWN").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

DTTM_ALIAS = "__timestamp"

TIME_COMPARISON = "__"

JS_MAX_INTEGER = 9007199254740991  # Largest int Java Script can handle 2^53-1

InputType = TypeVar("InputType")  # pylint: disable=invalid-name

ADHOC_FILTERS_REGEX = re.compile("^adhoc_filters")


class AdhocMetricExpressionType(StrEnum):
    SIMPLE = "SIMPLE"
    SQL = "SQL"


class AnnotationType(StrEnum):
    FORMULA = "FORMULA"
    INTERVAL = "INTERVAL"
    EVENT = "EVENT"
    TIME_SERIES = "TIME_SERIES"


class GenericDataType(IntEnum):
    """
    Generic database column type that fits both frontend and backend.
    """

    NUMERIC = 0
    STRING = 1
    TEMPORAL = 2
    BOOLEAN = 3
    # ARRAY = 4     # Mapping all the complex data types to STRING for now
    # JSON = 5      # and leaving these as a reminder.
    # MAP = 6
    # ROW = 7


class DatasourceType(StrEnum):
    TABLE = "table"
    DATASET = "dataset"
    QUERY = "query"
    SAVEDQUERY = "saved_query"
    VIEW = "view"


class LoggerLevel(StrEnum):
    INFO = "info"
    WARNING = "warning"
    EXCEPTION = "exception"


class HeaderDataType(TypedDict):
    notification_format: str
    owners: List[int]
    notification_type: str
    notification_source: Optional[str]
    chart_id: Optional[int]
    dashboard_id: Optional[int]
    slack_channels: Optional[List[str]]


class DatasourceDict(TypedDict):
    type: str  # todo(hugh): update this to be DatasourceType
    id: int


class AdhocFilterClause(TypedDict, total=False):
    clause: str
    expressionType: str
    filterOptionName: Optional[str]
    comparator: Optional[FilterValues]
    operator: str
    subject: str
    isExtra: Optional[bool]
    sqlExpression: Optional[str]


class QueryObjectFilterClause(TypedDict, total=False):
    col: Column
    op: str  # pylint: disable=invalid-name
    val: Optional[FilterValues]
    grain: Optional[str]
    isExtra: Optional[bool]


class ExtraFiltersTimeColumnType(StrEnum):
    TIME_COL = "__time_col"
    TIME_GRAIN = "__time_grain"
    TIME_ORIGIN = "__time_origin"
    TIME_RANGE = "__time_range"


class ExtraFiltersReasonType(StrEnum):
    NO_TEMPORAL_COLUMN = "no_temporal_column"
    COL_NOT_IN_DATASOURCE = "not_in_datasource"


class FilterOperator(StrEnum):
    """
    Operators used filter controls
    """

    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUALS = ">="
    LESS_THAN_OR_EQUALS = "<="
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    ILIKE = "ILIKE"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    IN = "IN"
    NOT_IN = "NOT IN"
    IS_TRUE = "IS TRUE"
    IS_FALSE = "IS FALSE"
    TEMPORAL_RANGE = "TEMPORAL_RANGE"


class FilterStringOperators(StrEnum):
    EQUALS = ("EQUALS",)
    NOT_EQUALS = ("NOT_EQUALS",)
    LESS_THAN = ("LESS_THAN",)
    GREATER_THAN = ("GREATER_THAN",)
    LESS_THAN_OR_EQUAL = ("LESS_THAN_OR_EQUAL",)
    GREATER_THAN_OR_EQUAL = ("GREATER_THAN_OR_EQUAL",)
    IN = ("IN",)
    NOT_IN = ("NOT_IN",)
    ILIKE = ("ILIKE",)
    LIKE = ("LIKE",)
    IS_NOT_NULL = ("IS_NOT_NULL",)
    IS_NULL = ("IS_NULL",)
    LATEST_PARTITION = ("LATEST_PARTITION",)
    IS_TRUE = ("IS_TRUE",)
    IS_FALSE = ("IS_FALSE",)


class PostProcessingBoxplotWhiskerType(StrEnum):
    """
    Calculate cell contribution to row/column total
    """

    TUKEY = "tukey"
    MINMAX = "min/max"
    PERCENTILE = "percentile"


class PostProcessingContributionOrientation(StrEnum):
    """
    Calculate cell contribution to row/column total
    """

    ROW = "row"
    COLUMN = "column"


class QuerySource(Enum):
    """
    The source of a SQL query.
    """

    CHART = 0
    DASHBOARD = 1
    SQL_LAB = 2


class QueryStatus(StrEnum):
    """Enum-type class for query statuses"""

    STOPPED: str = "stopped"
    FAILED: str = "failed"
    PENDING: str = "pending"
    RUNNING: str = "running"
    SCHEDULED: str = "scheduled"
    SUCCESS: str = "success"
    FETCHING: str = "fetching"
    TIMED_OUT: str = "timed_out"


class DashboardStatus(StrEnum):
    """Dashboard status used for frontend filters"""

    PUBLISHED = "published"
    DRAFT = "draft"


class ReservedUrlParameters(StrEnum):
    """
    Reserved URL parameters that are used internally by Superset. These will not be
    passed to chart queries, as they control the behavior of the UI.
    """

    STANDALONE = "standalone"
    EDIT_MODE = "edit"

    @staticmethod
    def is_standalone_mode() -> Optional[bool]:
        standalone_param = request.args.get(ReservedUrlParameters.STANDALONE.value)
        standalone: Optional[bool] = bool(
            standalone_param and standalone_param != "false" and standalone_param != "0"
        )
        return standalone


class RowLevelSecurityFilterType(StrEnum):
    REGULAR = "Regular"
    BASE = "Base"


class ColumnTypeSource(Enum):
    GET_TABLE = 1
    CURSOR_DESCRIPTION = 2


class ColumnSpec(NamedTuple):
    sqla_type: Union[TypeEngine, str]
    generic_type: GenericDataType
    is_dttm: bool
    python_date_format: Optional[str] = None


def parse_js_uri_path_item(
    item: Optional[str], unquote: bool = True, eval_undefined: bool = False
) -> Optional[str]:
    """Parse an uri path item made with js.

    :param item: an uri path component
    :param unquote: Perform unquoting of string using urllib.parse.unquote_plus()
    :param eval_undefined: When set to True and item is either 'null' or 'undefined',
    assume item is undefined and return None.
    :return: Either None, the original item or unquoted item
    """
    item = None if eval_undefined and item in ("null", "undefined") else item
    return unquote_plus(item) if unquote and item else item


def cast_to_num(value: Union[float, int, str, None]) -> Union[float, int, None]:
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
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return None


def cast_to_boolean(value: Any) -> Optional[bool]:
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
        return value.strip().lower() == "true"
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
    msg = ""
    if hasattr(ex, "message"):
        if isinstance(ex.message, dict):
            msg = ex.message.get("message")  # type: ignore
        elif ex.message:
            msg = ex.message
    return str(msg) or str(ex)


def markdown(raw: str, markup_wrap: Optional[bool] = False) -> str:
    safe_markdown_tags = {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "b",
        "i",
        "strong",
        "em",
        "tt",
        "p",
        "br",
        "span",
        "div",
        "blockquote",
        "code",
        "hr",
        "ul",
        "ol",
        "li",
        "dd",
        "dt",
        "img",
        "a",
    }
    safe_markdown_attrs = {
        "img": {"src", "alt", "title"},
        "a": {"href", "alt", "title"},
    }
    safe = md.markdown(
        raw or "",
        extensions=[
            "markdown.extensions.tables",
            "markdown.extensions.fenced_code",
            "markdown.extensions.codehilite",
        ],
    )
    # pylint: disable=no-member
    safe = nh3.clean(safe, tags=safe_markdown_tags, attributes=safe_markdown_attrs)
    if markup_wrap:
        safe = Markup(safe)
    return safe


def readfile(file_path: str) -> Optional[str]:
    with open(file_path) as f:
        content = f.read()
    return content


def generic_find_constraint_name(
    table: str, columns: Set[str], referenced: str, database: SQLA
) -> Optional[str]:
    """Utility to find a constraint name in alembic migrations"""
    tbl = sa.Table(
        table, database.metadata, autoload=True, autoload_with=database.engine
    )

    for fk in tbl.foreign_key_constraints:
        if fk.referred_table.name == referenced and set(fk.column_keys) == columns:
            return fk.name

    return None


def generic_find_fk_constraint_name(
    table: str, columns: Set[str], referenced: str, insp: Inspector
) -> Optional[str]:
    """Utility to find a foreign-key constraint name in alembic migrations"""
    for fk in insp.get_foreign_keys(table):
        if (
            fk["referred_table"] == referenced
            and set(fk["referred_columns"]) == columns
        ):
            return fk["name"]

    return None


def generic_find_fk_constraint_names(  # pylint: disable=invalid-name
    table: str, columns: Set[str], referenced: str, insp: Inspector
) -> Set[str]:
    """Utility to find foreign-key constraint names in alembic migrations"""
    names = set()

    for fk in insp.get_foreign_keys(table):
        if (
            fk["referred_table"] == referenced
            and set(fk["referred_columns"]) == columns
        ):
            names.add(fk["name"])

    return names


def generic_find_uq_constraint_name(
    table: str, columns: Set[str], insp: Inspector
) -> Optional[str]:
    """Utility to find a unique constraint name in alembic migrations"""

    for uq in insp.get_unique_constraints(table):
        if columns == set(uq["column_names"]):
            return uq["name"]

    return None


def get_datasource_full_name(
    database_name: str,
    datasource_name: str,
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
) -> str:
    parts = [database_name, catalog, schema, datasource_name]
    return ".".join([f"[{part}]" for part in parts if part])


class SigalrmTimeout:
    """
    To be used in a ``with`` block and timeout its content.
    """

    def __init__(self, seconds: int = 1, error_message: str = "Timeout") -> None:
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(  # pylint: disable=unused-argument
        self, signum: int, frame: Any
    ) -> None:
        logger.error("Process timed out", exc_info=True)
        raise SupersetTimeoutException(
            error_type=SupersetErrorType.BACKEND_TIMEOUT_ERROR,
            message=self.error_message,
            level=ErrorLevel.ERROR,
            extra={"timeout": self.seconds},
        )

    def __enter__(self) -> None:
        try:
            if threading.current_thread() == threading.main_thread():
                signal.signal(signal.SIGALRM, self.handle_timeout)
                signal.alarm(self.seconds)
        except ValueError as ex:
            logger.warning("timeout can't be used in the current context")
            logger.exception(ex)

    def __exit__(  # pylint: disable=redefined-outer-name,redefined-builtin
        self, type: Any, value: Any, traceback: TracebackType
    ) -> None:
        try:
            signal.alarm(0)
        except ValueError as ex:
            logger.warning("timeout can't be used in the current context")
            logger.exception(ex)


class TimerTimeout:
    def __init__(self, seconds: int = 1, error_message: str = "Timeout") -> None:
        self.seconds = seconds
        self.error_message = error_message
        self.timer = threading.Timer(seconds, _thread.interrupt_main)

    def __enter__(self) -> None:
        self.timer.start()

    def __exit__(  # pylint: disable=redefined-outer-name,redefined-builtin
        self, type: Any, value: Any, traceback: TracebackType
    ) -> None:
        self.timer.cancel()
        if type is KeyboardInterrupt:  # raised by _thread.interrupt_main
            raise SupersetTimeoutException(
                error_type=SupersetErrorType.BACKEND_TIMEOUT_ERROR,
                message=self.error_message,
                level=ErrorLevel.ERROR,
                extra={"timeout": self.seconds},
            )


# Windows has no support for SIGALRM, so we use the timer based timeout
timeout: Union[Type[TimerTimeout], Type[SigalrmTimeout]] = (
    TimerTimeout if platform.system() == "Windows" else SigalrmTimeout
)


def pessimistic_connection_handling(some_engine: Engine) -> None:
    @event.listens_for(some_engine, "engine_connect")
    def ping_connection(connection: Connection, branch: bool) -> None:
        if branch:
            # 'branch' refers to a sub-connection of a connection,
            # we don't want to bother pinging on these.
            return

        # turn off 'close with result'.  This flag is only used with
        # 'connectionless' execution, otherwise will be False in any case
        save_should_close_with_result = connection.should_close_with_result
        connection.should_close_with_result = False

        try:
            # run a SELECT 1.   use a core select() so that
            # the SELECT of a scalar value without a table is
            # appropriately formatted for the backend
            connection.scalar(select([1]))
        except exc.DBAPIError as err:
            # catch SQLAlchemy's DBAPIError, which is a wrapper
            # for the DBAPI's exception.  It includes a .connection_invalidated
            # attribute which specifies if this connection is a 'disconnect'
            # condition, which is based on inspection of the original exception
            # by the dialect in use.
            if err.connection_invalidated:
                # run the same SELECT again - the connection will re-validate
                # itself and establish a new connection.  The disconnect detection
                # here also causes the whole connection pool to be invalidated
                # so that all stale connections are discarded.
                connection.scalar(select([1]))
            else:
                raise
        finally:
            # restore 'close with result'
            connection.should_close_with_result = save_should_close_with_result

    if some_engine.dialect.name == "sqlite":

        @event.listens_for(some_engine, "connect")
        def set_sqlite_pragma(  # pylint: disable=unused-argument
            connection: sqlite3.Connection,
            *args: Any,
        ) -> None:
            r"""
            Enable foreign key support for SQLite.

            :param connection: The SQLite connection
            :param \*args: Additional positional arguments
            :see: https://docs.sqlalchemy.org/en/latest/dialects/sqlite.html
            """

            with closing(connection.cursor()) as cursor:
                cursor.execute("PRAGMA foreign_keys=ON")


def send_email_smtp(  # pylint: disable=invalid-name,too-many-arguments,too-many-locals
    to: str,
    subject: str,
    html_content: str,
    config: Dict[str, Any],
    files: Optional[List[str]] = None,
    data: Optional[Dict[str, str]] = None,
    pdf: Optional[Dict[str, bytes]] = None,
    images: Optional[Dict[str, bytes]] = None,
    dryrun: bool = False,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    mime_subtype: str = "mixed",
    header_data: Optional[HeaderDataType] = None,
) -> None:
    """
    Send an email with html content, eg:
    send_email_smtp(
        'test@example.com', 'foo', '<b>Foo</b> bar',['/dev/null'], dryrun=True)
    """
    smtp_mail_from = config["SMTP_MAIL_FROM"]
    smtp_mail_to = get_email_address_list(to)

    msg = MIMEMultipart(mime_subtype)
    msg["Subject"] = subject
    msg["From"] = smtp_mail_from
    msg["To"] = ", ".join(smtp_mail_to)

    msg.preamble = "This is a multi-part message in MIME format."

    recipients = smtp_mail_to
    if cc:
        smtp_mail_cc = get_email_address_list(cc)
        msg["Cc"] = ", ".join(smtp_mail_cc)
        recipients = recipients + smtp_mail_cc

    smtp_mail_bcc = []
    if bcc:
        # don't add bcc in header
        smtp_mail_bcc = get_email_address_list(bcc)
        recipients = recipients + smtp_mail_bcc

    msg["Date"] = formatdate(localtime=True)
    mime_text = MIMEText(html_content, "html")
    msg.attach(mime_text)

    # Attach files by reading them from disk
    for fname in files or []:
        basename = os.path.basename(fname)
        with open(fname, "rb") as f:
            msg.attach(
                MIMEApplication(
                    f.read(),
                    Content_Disposition=f"attachment; filename='{basename}'",
                    Name=basename,
                )
            )

    # Attach any files passed directly
    for name, body in (data or {}).items():
        msg.attach(
            MIMEApplication(
                body, Content_Disposition=f"attachment; filename='{name}'", Name=name
            )
        )

    for name, body_pdf in (pdf or {}).items():
        msg.attach(
            MIMEApplication(
                body_pdf,
                Content_Disposition=f"attachment; filename='{name}'",
                Name=name,
            )
        )

    # Attach any inline images, which may be required for display in
    # HTML content (inline)
    for msgid, imgdata in (images or {}).items():
        formatted_time = formatdate(localtime=True)
        file_name = f"{subject} {formatted_time}"
        image = MIMEImage(imgdata, name=file_name)
        image.add_header("Content-ID", f"<{msgid}>")
        image.add_header("Content-Disposition", "inline")
        msg.attach(image)
    msg_mutator = config["EMAIL_HEADER_MUTATOR"]
    # the base notification returns the message without any editing.
    new_msg = msg_mutator(msg, **(header_data or {}))
    new_to = new_msg["To"].split(", ") if "To" in new_msg else []
    new_cc = new_msg["Cc"].split(", ") if "Cc" in new_msg else []
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
    smtp_host = config["SMTP_HOST"]
    smtp_port = config["SMTP_PORT"]
    smtp_user = config["SMTP_USER"]
    smtp_password = config["SMTP_PASSWORD"]
    smtp_starttls = config["SMTP_STARTTLS"]
    smtp_ssl = config["SMTP_SSL"]
    smtp_ssl_server_auth = config["SMTP_SSL_SERVER_AUTH"]

    if dryrun:
        logger.info("Dryrun enabled, email notification content is below:")
        logger.info(mime_msg.as_string())
        return

    # Default ssl context is SERVER_AUTH using the default system
    # root CA certificates
    ssl_context = ssl.create_default_context() if smtp_ssl_server_auth else None
    smtp = (
        smtplib.SMTP_SSL(smtp_host, smtp_port, context=ssl_context)
        if smtp_ssl
        else smtplib.SMTP(smtp_host, smtp_port)
    )
    if smtp_starttls:
        smtp.starttls(context=ssl_context)
    if smtp_user and smtp_password:
        smtp.login(smtp_user, smtp_password)
    logger.debug("Sent an email to %s", str(e_to))
    smtp.sendmail(e_from, e_to, mime_msg.as_string())
    smtp.quit()


def get_email_address_list(address_string: str) -> List[str]:
    address_string_list: List[str] = []
    if isinstance(address_string, str):
        address_string_list = re.split(r",|\s|;", address_string)
    return [x.strip() for x in address_string_list if x.strip()]


def choicify(values: Iterable[Any]) -> List[Tuple[Any, Any]]:
    """Takes an iterable and makes an iterable of tuples with it"""
    return [(v, v) for v in values]


def zlib_compress(data: Union[bytes, str]) -> bytes:
    """
    Compress things in a py2/3 safe fashion
    >>> json_str = '{"test": 1}'
    >>> blob = zlib_compress(json_str)
    """
    if isinstance(data, str):
        return zlib.compress(bytes(data, "utf-8"))
    return zlib.compress(data)


def zlib_decompress(blob: bytes, decode: Optional[bool] = True) -> Union[bytes, str]:
    """
    Decompress things to a string in a py2/3 safe fashion
    >>> json_str = '{"test": 1}'
    >>> blob = zlib_compress(json_str)
    >>> got_str = zlib_decompress(blob)
    >>> got_str == json_str
    True
    """
    if isinstance(blob, bytes):
        decompressed = zlib.decompress(blob)
    else:
        decompressed = zlib.decompress(bytes(blob, "utf-8"))
    return decompressed.decode("utf-8") if decode else decompressed


def simple_filter_to_adhoc(
    filter_clause: QueryObjectFilterClause,
    clause: str = "where",
) -> AdhocFilterClause:
    result: AdhocFilterClause = {
        "clause": clause.upper(),
        "expressionType": "SIMPLE",
        "comparator": filter_clause.get("val"),
        "operator": filter_clause["op"],
        "subject": cast(str, filter_clause["col"]),
    }
    if filter_clause.get("isExtra"):
        result["isExtra"] = True
    result["filterOptionName"] = md5_sha_from_dict(cast(Dict[Any, Any], result))

    return result


def form_data_to_adhoc(form_data: Dict[str, Any], clause: str) -> AdhocFilterClause:
    if clause not in ("where", "having"):
        raise ValueError(__("Unsupported clause type: %(clause)s", clause=clause))
    result: AdhocFilterClause = {
        "clause": clause.upper(),
        "expressionType": "SQL",
        "sqlExpression": form_data.get(clause),
    }
    result["filterOptionName"] = md5_sha_from_dict(cast(Dict[Any, Any], result))

    return result


def merge_extra_form_data(form_data: Dict[str, Any]) -> None:  # noqa: C901
    """
    Merge extra form data (appends and overrides) into the main payload
    and add applied time extras to the payload.
    """
    filter_keys = ["filters", "adhoc_filters"]
    extra_form_data = form_data.pop("extra_form_data", {})
    append_filters: List[QueryObjectFilterClause] = extra_form_data.get("filters", None)

    # merge append extras
    for key in [key for key in EXTRA_FORM_DATA_APPEND_KEYS if key not in filter_keys]:
        extra_value = getattr(extra_form_data, key, {})
        form_value = getattr(form_data, key, {})
        form_value.update(extra_value)
        if form_value:
            form_data["key"] = extra_value

    # map regular extras that apply to form data properties
    for src_key, target_key in EXTRA_FORM_DATA_OVERRIDE_REGULAR_MAPPINGS.items():
        value = extra_form_data.get(src_key)
        if value is not None:
            form_data[target_key] = value

    # map extras that apply to form data extra properties
    extras = form_data.get("extras", {})
    for key in EXTRA_FORM_DATA_OVERRIDE_EXTRA_KEYS:
        value = extra_form_data.get(key)
        if value is not None:
            extras[key] = value
    if extras:
        form_data["extras"] = extras

    adhoc_filters: List[AdhocFilterClause] = form_data.get("adhoc_filters", [])
    form_data["adhoc_filters"] = adhoc_filters
    append_adhoc_filters: List[AdhocFilterClause] = extra_form_data.get(
        "adhoc_filters", []
    )
    adhoc_filters.extend(
        {"isExtra": True, **adhoc_filter} for adhoc_filter in append_adhoc_filters
    )
    if append_filters:
        for key, value in form_data.items():
            if re.match("adhoc_filter.*", key):
                value.extend(
                    simple_filter_to_adhoc({"isExtra": True, **fltr})
                    for fltr in append_filters
                    if fltr
                )
    if form_data.get("time_range") and not form_data.get("granularity_sqla"):
        for adhoc_filter in form_data.get("adhoc_filters", []):
            if adhoc_filter.get("operator") == "TEMPORAL_RANGE":
                adhoc_filter["comparator"] = form_data["time_range"]


def merge_extra_filters(form_data: Dict[str, Any]) -> None:  # noqa: C901
    # extra_filters are temporary/contextual filters (using the legacy constructs)
    # that are external to the slice definition. We use those for dynamic
    # interactive filters.
    # Note extra_filters only support simple filters.
    form_data.setdefault("applied_time_extras", {})
    adhoc_filters = form_data.get("adhoc_filters", [])
    form_data["adhoc_filters"] = adhoc_filters
    merge_extra_form_data(form_data)
    if "extra_filters" in form_data:
        # __form and __to are special extra_filters that target time
        # boundaries. The rest of extra_filters are simple
        # [column_name in list_of_values]. `__` prefix is there to avoid
        # potential conflicts with column that would be named `from` or `to`
        date_options = {
            "__time_range": "time_range",
            "__time_col": "granularity_sqla",
            "__time_grain": "time_grain_sqla",
        }

        # Grab list of existing filters 'keyed' on the column and operator

        def get_filter_key(f: Dict[str, Any]) -> str:
            if "expressionType" in f:
                return f"{f['subject']}__{f['operator']}"

            return f"{f['col']}__{f['op']}"

        existing_filters = {}
        for existing in adhoc_filters:
            if (
                existing["expressionType"] == "SIMPLE"
                and existing.get("comparator") is not None
                and existing.get("subject") is not None
            ):
                existing_filters[get_filter