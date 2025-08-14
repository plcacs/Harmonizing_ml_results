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
from typing import Any, Callable, cast, NamedTuple, TYPE_CHECKING, TypedDict, TypeVar, Optional, Union, Dict, List, Tuple, Set
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
    type: str
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
    op: str
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
    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"
    LESS_THAN_OR_EQUAL = "LESS_THAN_OR_EQUAL"
    GREATER_THAN_OR_EQUAL = "GREATER_THAN_OR_EQUAL"
    IN = "IN"
    NOT_IN = "NOT_IN"
    ILIKE = "ILIKE"
    LIKE = "LIKE"
    IS_NOT_NULL = "IS_NOT_NULL"
    IS_NULL = "IS_NULL"
    LATEST_PARTITION = "LATEST_PARTITION"
    IS_TRUE = "IS_TRUE"
    IS_FALSE = "IS_FALSE"


class PostProcessingBoxplotWhiskerType(StrEnum):
    TUKEY = "tukey"
    MINMAX = "min/max"
    PERCENTILE = "percentile"


class PostProcessingContributionOrientation(StrEnum):
    ROW = "row"
    COLUMN = "column"


class QuerySource(Enum):
    CHART = 0
    DASHBOARD = 1
    SQL_LAB = 2


class QueryStatus(StrEnum):
    STOPPED: str = "stopped"
    FAILED: str = "failed"
    PENDING: str = "pending"
    RUNNING: str = "running"
    SCHEDULED: str = "scheduled"
    SUCCESS: str = "success"
    FETCHING: str = "fetching"
    TIMED_OUT: str = "timed_out"


class DashboardStatus(StrEnum):
    PUBLISHED = "published"
    DRAFT = "draft"


class ReservedUrlParameters(StrEnum):
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
    return unquote_plus(item) if unquote and item else item


def cast_to_num(value: Union[float, int, str, None]) -> Union[float, int, None]:
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
    msg = ""
    if hasattr(ex, "message"):
        if isinstance(ex.message, dict):
            msg = ex.message.get("message")
        elif ex.message:
            msg = ex.message
    return str(msg) or str(ex)


def markdown(raw: str, markup_wrap: Optional[bool] = False) -> str:
    safe_markdown_tags = {
        "h1", "h2", "h3", "h4", "h5", "h6", "b", "i", "strong", "em", "tt", "p", "br",
        "span", "div", "blockquote", "code", "hr", "ul", "ol", "li", "dd", "dt", "img", "a"
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
    for fk in insp.get_foreign_keys(table):
        if (
            fk["referred_table"] == referenced
            and set(fk["referred_columns"]) == columns
        ):
            return fk["name"]
    return None


def generic_find_fk_constraint_names(
    table: str, columns: Set[str], referenced: str, insp: Inspector
) -> Set[str]:
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
    def __init__(self, seconds: int = 1, error_message: str = "Timeout") -> None:
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum: int, frame: Any) -> None:
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

    def __exit__(
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

    def __exit__(
        self, type: Any, value: Any, traceback: TracebackType
    ) -> None:
        self.timer.cancel()
        if type is KeyboardInterrupt:
            raise SupersetTimeoutException(
                error_type=SupersetErrorType.BACKEND_TIMEOUT_ERROR,
                message=self.error_message,
                level=ErrorLevel.ERROR,
                extra={"timeout": self.seconds},
            )


timeout: Union[Type[TimerTimeout], Type[SigalrmTimeout]] = (
    TimerTimeout if platform.system() == "Windows" else SigalrmTimeout
)


def pessimistic_connection_handling(some_engine: Engine) -> None:
    @event.listens_for(some_engine, "engine_connect")
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

    if some_engine.dialect.name == "sqlite":
        @event.listens_for(some_engine, "connect")
        def set_sqlite_pragma(connection: sqlite3.Connection, *args: Any) -> None:
            with closing(connection.cursor()) as cursor:
                cursor.execute("PRAGMA foreign_keys=ON")


def send_email_smtp(
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
        smtp_mail_bcc = get_email_address_list(bcc)
        recipients = recipients + smtp_mail_bcc
    msg["Date"] = formatdate(localtime=True)
    mime_text = MIMEText(html_content, "html")
    msg.attach(mime_text)
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
    for name, body in (data or {}).