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
# pylint: disable=consider-using-transaction,too-many-lines
from __future__ import annotations

import contextlib
import logging
import re
import time
from abc import ABCMeta
from collections import defaultdict, deque
from datetime import datetime
from re import Pattern
from textwrap import dedent
from typing import Any, cast, Optional, TYPE_CHECKING, Union, List, Dict, Set, Tuple, Deque
from urllib import parse

import pandas as pd
from flask import current_app
from flask_babel import gettext as __, lazy_gettext as _
from packaging.version import Version
from sqlalchemy import Column, literal_column, types
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.result import Row as ResultRow
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.expression import ColumnClause, Select

from superset import cache_manager, db, is_feature_enabled
from superset.common.db_query_status import QueryStatus
from superset.constants import TimeGrain
from superset.databases.utils import make_url_safe
from superset.db_engine_specs.base import BaseEngineSpec
from superset.errors import SupersetErrorType
from superset.exceptions import SupersetTemplateException
from superset.models.sql_lab import Query
from superset.models.sql_types.presto_sql_types import (
    Array,
    Date,
    Interval,
    Map,
    Row,
    TimeStamp,
    TinyInteger,
)
from superset.result_set import destringify
from superset.superset_typing import ResultSetColumnType
from superset.utils import core as utils, json
from superset.utils.core import GenericDataType

if TYPE_CHECKING:
    # prevent circular imports
    from superset.models.core import Database
    from superset.sql_parse import Table

    with contextlib.suppress(ImportError):  # pyhive may not be installed
        from pyhive.presto import Cursor

COLUMN_DOES_NOT_EXIST_REGEX = re.compile(
    "line (?P<location>.+?): .*Column '(?P<column_name>.+?)' cannot be resolved"
)
TABLE_DOES_NOT_EXIST_REGEX = re.compile(".*Table (?P<table_name>.+?) does not exist")
SCHEMA_DOES_NOT_EXIST_REGEX = re.compile(
    "line (?P<location>.+?): .*Schema '(?P<schema_name>.+?)' does not exist"
)
CONNECTION_ACCESS_DENIED_REGEX = re.compile("Access Denied: Invalid credentials")
CONNECTION_INVALID_HOSTNAME_REGEX = re.compile(
    r"Failed to establish a new connection: \[Errno 8\] nodename nor servname "
    "provided, or not known"
)
CONNECTION_HOST_DOWN_REGEX = re.compile(
    r"Failed to establish a new connection: \[Errno 60\] Operation timed out"
)
CONNECTION_PORT_CLOSED_REGEX = re.compile(
    r"Failed to establish a new connection: \[Errno 61\] Connection refused"
)
CONNECTION_UNKNOWN_DATABASE_ERROR = re.compile(
    r"line (?P<location>.+?): Catalog '(?P<catalog_name>.+?)' does not exist"
)

logger = logging.getLogger(__name__)


def get_children(column: ResultSetColumnType) -> List[ResultSetColumnType]:
    """
    Get the children of a complex Presto type (row or array).
    """
    pattern = re.compile(r"(?P<type>\w+)\((?P<children>.*)\)")
    if not column["type"]:
        raise ValueError
    match = pattern.match(cast(str, column["type"]))
    if not match:
        raise Exception(
            f"Unable to parse column type {column['type']}"
        )

    group = match.groupdict()
    type_ = group["type"].upper()
    children_type = group["children"]
    if type_ == "ARRAY":
        return [
            {
                "column_name": column["column_name"],
                "name": column["column_name"],
                "type": children_type,
                "is_dttm": False,
            }
        ]

    if type_ == "ROW":
        nameless_columns = 0
        columns = []
        for child in utils.split(children_type, ","):
            parts = list(utils.split(child.strip(), " "))
            if len(parts) == 2:
                name, type_ = parts
                name = name.strip('"')
            else:
                name = f"_col{nameless_columns}"
                type_ = parts[0]
                nameless_columns += 1
            _column: ResultSetColumnType = {
                "column_name": f"{column['column_name']}.{name.lower()}",
                "name": f"{column['column_name']}.{name.lower()}",
                "type": type_,
                "is_dttm": False,
            }
            columns.append(_column)
        return columns

    raise Exception(f"Unknown type {type_}!")


class PrestoBaseEngineSpec(BaseEngineSpec, metaclass=ABCMeta):
    """
    A base class that share common functions between Presto and Trino
    """

    supports_dynamic_schema: bool = True
    supports_catalog: bool = True
    supports_dynamic_catalog: bool = True

    column_type_mappings: Tuple[Tuple[Pattern[str], Any, GenericDataType], ...] = (
        (
            re.compile(r"^boolean.*", re.IGNORECASE),
            types.BOOLEAN(),
            GenericDataType.BOOLEAN,
        ),
        (
            re.compile(r"^tinyint.*", re.IGNORECASE),
            TinyInteger(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^smallint.*", re.IGNORECASE),
            types.SmallInteger(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^integer.*", re.IGNORECASE),
            types.INTEGER(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^bigint.*", re.IGNORECASE),
            types.BigInteger(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^real.*", re.IGNORECASE),
            types.FLOAT(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^double.*", re.IGNORECASE),
            types.FLOAT(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^decimal.*", re.IGNORECASE),
            types.DECIMAL(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^varchar(\((\d+)\))*$", re.IGNORECASE),
            lambda match: types.VARCHAR(int(match[2])) if match[2] else types.String(),
            GenericDataType.STRING,
        ),
        (
            re.compile(r"^char(\((\d+)\))*$", re.IGNORECASE),
            lambda match: types.CHAR(int(match[2])) if match[2] else types.String(),
            GenericDataType.STRING,
        ),
        (
            re.compile(r"^varbinary.*", re.IGNORECASE),
            types.VARBINARY(),
            GenericDataType.STRING,
        ),
        (
            re.compile(r"^json.*", re.IGNORECASE),
            types.JSON(),
            GenericDataType.STRING,
        ),
        (
            re.compile(r"^date.*", re.IGNORECASE),
            types.Date(),
            GenericDataType.TEMPORAL,
        ),
        (
            re.compile(r"^timestamp.*", re.IGNORECASE),
            types.TIMESTAMP(),
            GenericDataType.TEMPORAL,
        ),
        (
            re.compile(r"^interval.*", re.IGNORECASE),
            Interval(),
            GenericDataType.TEMPORAL,
        ),
        (
            re.compile(r"^time.*", re.IGNORECASE),
            types.Time(),
            GenericDataType.TEMPORAL,
        ),
        (re.compile(r"^array.*", re.IGNORECASE), Array(), GenericDataType.STRING),
        (re.compile(r"^map.*", re.IGNORECASE), Map(), GenericDataType.STRING),
        (re.compile(r"^row.*", re.IGNORECASE), Row(), GenericDataType.STRING),
    )

    _time_grain_expressions: Dict[Optional[TimeGrain], str] = {
        None: "{col}",
        TimeGrain.SECOND: "date_trunc('second', CAST({col} AS TIMESTAMP))",
        TimeGrain.FIVE_SECONDS: "date_trunc('second', CAST({col} AS TIMESTAMP)) - interval '1' second * (second(CAST({col} AS TIMESTAMP)) % 5)",
        TimeGrain.THIRTY_SECONDS: "date_trunc('second', CAST({col} AS TIMESTAMP)) - interval '1' second * (second(CAST({col} AS TIMESTAMP)) % 30)",
        TimeGrain.MINUTE: "date_trunc('minute', CAST({col} AS TIMESTAMP))",
        TimeGrain.FIVE_MINUTES: "date_trunc('minute', CAST({col} AS TIMESTAMP)) - interval '1' minute * (minute(CAST({col} AS TIMESTAMP)) % 5)",
        TimeGrain.TEN_MINUTES: "date_trunc('minute', CAST({col} AS TIMESTAMP)) - interval '1' minute * (minute(CAST({col} AS TIMESTAMP)) % 10)",
        TimeGrain.FIFTEEN_MINUTES: "date_trunc('minute', CAST({col} AS TIMESTAMP)) - interval '1' minute * (minute(CAST({col} AS TIMESTAMP)) % 15)",
        TimeGrain.HALF_HOUR: "date_trunc('minute', CAST({col} AS TIMESTAMP)) - interval '1' minute * (minute(CAST({col} AS TIMESTAMP)) % 30)",
        TimeGrain.HOUR: "date_trunc('hour', CAST({col} AS TIMESTAMP))",
        TimeGrain.SIX_HOURS: "date_trunc('hour', CAST({col} AS TIMESTAMP)) - interval '1' hour * (hour(CAST({col} AS TIMESTAMP)) % 6)",
        TimeGrain.DAY: "date_trunc('day', CAST({col} AS TIMESTAMP))",
        TimeGrain.WEEK: "date_trunc('week', CAST({col} AS TIMESTAMP))",
        TimeGrain.MONTH: "date_trunc('month', CAST({col} AS TIMESTAMP))",
        TimeGrain.QUARTER: "date_trunc('quarter', CAST({col} AS TIMESTAMP))",
        TimeGrain.YEAR: "date_trunc('year', CAST({col} AS TIMESTAMP))",
        TimeGrain.WEEK_STARTING_SUNDAY: "date_trunc('week', CAST({col} AS TIMESTAMP) + interval '1' day) - interval '1' day",
        TimeGrain.WEEK_STARTING_MONDAY: "date_trunc('week', CAST({col} AS TIMESTAMP))",
        TimeGrain.WEEK_ENDING_SATURDAY: "date_trunc('week', CAST({col} AS TIMESTAMP) + interval '1' day) + interval '5' day",
        TimeGrain.WEEK_ENDING_SUNDAY: "date_trunc('week', CAST({col} AS TIMESTAMP)) + interval '6' day",
    }

    @classmethod
    def convert_dttm(
        cls, target_type: str, dttm: datetime, db_extra: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Convert a Python `datetime` object to a SQL expression.
        """
        sqla_type = cls.get_sqla_column_type(target_type)

        if isinstance(sqla_type, types.Date):
            return f"DATE '{dttm.date().isoformat()}'"
        if isinstance(sqla_type, types.TIMESTAMP):
            return f"""TIMESTAMP '{dttm.isoformat(timespec="microseconds", sep=" ")}'"""

        return None

    @classmethod
    def epoch_to_dttm(cls) -> str:
        return "from_unixtime({col})"

    @classmethod
    def get_default_catalog(cls, database: "Database") -> Optional[str]:
        """
        Return the default catalog.
        """
        if database.url_object.database is None:
            return None

        return database.url_object.database.split("/")[0]

    @classmethod
    def get_catalog_names(
        cls,
        database: "Database",
        inspector: Inspector,
    ) -> Set[str]:
        """
        Get all catalogs.
        """
        return {catalog for (catalog,) in inspector.bind.execute("SHOW CATALOGS")}

    @classmethod
    def adjust_engine_params(
        cls,
        uri: URL,
        connect_args: Dict[str, Any],
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> Tuple[URL, Dict[str, Any]]:
        if uri.database and "/" in uri.database:
            current_catalog, current_schema = uri.database.split("/", 1)
        else:
            current_catalog, current_schema = uri.database, None

        if schema:
            schema = parse.quote(schema, safe="")

        adjusted_database = "/".join(
            [
                catalog or current_catalog or "",
                schema or current_schema or "",
            ]
        ).rstrip("/")

        uri = uri.set(database=adjusted_database)

        return uri, connect_args

    @classmethod
    def get_schema_from_engine_params(
        cls,
        sqlalchemy_uri: URL,
        connect_args: Dict[str, Any],
    ) -> Optional[str]:
        """
        Return the configured schema.
        """
        database = sqlalchemy_uri.database.strip("/")

        if "/" not in database:
            return None

        return parse.unquote(database.split("/")[1])

    @classmethod
    def estimate_statement_cost(
        cls, database: "Database", statement: str, cursor: Any
    ) -> Dict[str, Any]:
        """
        Run a SQL query that estimates the cost of a given statement.
        """
        sql = f"EXPLAIN (TYPE IO, FORMAT JSON) {statement}"
        cursor.execute(sql)

        result = json.loads(cursor.fetchone()[0])
        return result

    @classmethod
    def query_cost_formatter(
        cls, raw_cost: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Format cost estimate.
        """

        def humanize(value: Any, suffix: str) -> str:
            try:
                value = int(value)
            except ValueError:
                return str(value)

            prefixes = ["K", "M", "G", "T", "P", "E", "Z", "Y"]
            prefix = ""
            to_next_prefix = 1000
            while value > to_next_prefix and prefixes:
                prefix = prefixes.pop(0)
                value //= to_next_prefix

            return f"{value} {prefix}{suffix}"

        cost = []
        columns = [
            ("outputRowCount", "Output count", " rows"),
            ("outputSizeInBytes", "Output size", "B"),
            ("cpuCost", "CPU cost", ""),
            ("maxMemory", "Max memory", "B"),
            ("networkCost", "Network cost", ""),
        ]
        for row in raw_cost:
            estimate: Dict[str, float] = row.get("estimate", {})
            statement_cost = {}
            for key, label, suffix in columns:
                if key in estimate:
                    statement_cost[label] = humanize(estimate[key], suffix).strip()
            cost.append(statement_cost)

        return cost

    @classmethod
    @cache_manager.data_cache.memoize()
    def get_function_names(cls, database: "Database") -> List[str]:
        """
        Get a list of function names that are able to be called on the database.
        """
        return database.get_df("SHOW FUNCTIONS")["Function"].tolist()

    @classmethod
    def _partition_query(
        cls,
        table: "Table",
        indexes: List[Dict[str, Any]],
        database: "Database",
        limit: int = 0,
        order_by: Optional[List[Tuple[str, bool]]] = None,
        filters: Optional[Dict[Any, Any]] = None,
    ) -> str:
        """
        Return a partition query.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        order_by_clause = ""
        if order_by:
            l = []
            for field, desc in order_by:
                l.append(field + " DESC" if desc else "")
            order_by_clause = "ORDER BY " + ", ".join(l)

        where_clause = ""
        if filters:
            l = []
            for field, value in filters.items():
                l.append(f"{field} = '{value}'")
            where_clause = "WHERE " + " AND ".join(l)

        presto_version = database.get_extra().get("version")

        if presto_version and Version(presto_version) < Version("0.199"):
            full_table_name = (
                f"{table.schema}.{table.table}" if table.schema else table.table
            )
            partition_select_clause = f"SHOW PARTITIONS FROM {full_table_name}"
        else:
            system_table_name = f'"{table.table}$partitions"'
            full_table_name = (
                f"{table.schema