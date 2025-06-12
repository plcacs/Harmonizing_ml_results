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
from typing import Any, cast, Optional, TYPE_CHECKING, Dict, List, Tuple, Set, Deque, Union, DefaultDict
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
from superset.models.sql_types.presto_sql_types import Array, Date, Interval, Map, Row, TimeStamp, TinyInteger
from superset.result_set import destringify
from superset.superset_typing import ResultSetColumnType
from superset.utils import core as utils, json
from superset.utils.core import GenericDataType

if TYPE_CHECKING:
    from superset.models.core import Database
    from superset.sql_parse import Table
    from pyhive.presto import Cursor

COLUMN_DOES_NOT_EXIST_REGEX: Pattern[str] = re.compile("line (?P<location>.+?): .*Column '(?P<column_name>.+?)' cannot be resolved")
TABLE_DOES_NOT_EXIST_REGEX: Pattern[str] = re.compile('.*Table (?P<table_name>.+?) does not exist')
SCHEMA_DOES_NOT_EXIST_REGEX: Pattern[str] = re.compile("line (?P<location>.+?): .*Schema '(?P<schema_name>.+?)' does not exist")
CONNECTION_ACCESS_DENIED_REGEX: Pattern[str] = re.compile('Access Denied: Invalid credentials')
CONNECTION_INVALID_HOSTNAME_REGEX: Pattern[str] = re.compile('Failed to establish a new connection: \\[Errno 8\\] nodename nor servname provided, or not known')
CONNECTION_HOST_DOWN_REGEX: Pattern[str] = re.compile('Failed to establish a new connection: \\[Errno 60\\] Operation timed out')
CONNECTION_PORT_CLOSED_REGEX: Pattern[str] = re.compile('Failed to establish a new connection: \\[Errno 61\\] Connection refused')
CONNECTION_UNKNOWN_DATABASE_ERROR: Pattern[str] = re.compile("line (?P<location>.+?): Catalog '(?P<catalog_name>.+?)' does not exist")
logger: logging.Logger = logging.getLogger(__name__)

def get_children(column: Dict[str, Any]) -> List[Dict[str, Any]]:
    pattern = re.compile('(?P<type>\\w+)\\((?P<children>.*)\\)')
    if not column['type']:
        raise ValueError
    match = pattern.match(cast(str, column['type']))
    if not match:
        raise Exception(f'Unable to parse column type {column['type']}')
    group = match.groupdict()
    type_ = group['type'].upper()
    children_type = group['children']
    if type_ == 'ARRAY':
        return [{'column_name': column['column_name'], 'name': column['column_name'], 'type': children_type, 'is_dttm': False}]
    if type_ == 'ROW':
        nameless_columns = 0
        columns = []
        for child in utils.split(children_type, ','):
            parts = list(utils.split(child.strip(), ' '))
            if len(parts) == 2:
                name, type_ = parts
                name = name.strip('"')
            else:
                name = f'_col{nameless_columns}'
                type_ = parts[0]
                nameless_columns += 1
            _column = {'column_name': f'{column['column_name']}.{name.lower()}', 'name': f'{column['column_name']}.{name.lower()}', 'type': type_, 'is_dttm': False}
            columns.append(_column)
        return columns
    raise Exception(f'Unknown type {type_}!')

class PrestoBaseEngineSpec(BaseEngineSpec, metaclass=ABCMeta):
    supports_dynamic_schema: bool = True
    supports_catalog: bool = supports_dynamic_catalog: bool = True
    column_type_mappings: Tuple[Tuple[Pattern[str], Any, GenericDataType], ...] = (
        (re.compile('^boolean.*', re.IGNORECASE), types.BOOLEAN(), GenericDataType.BOOLEAN),
        (re.compile('^tinyint.*', re.IGNORECASE), TinyInteger(), GenericDataType.NUMERIC),
        (re.compile('^smallint.*', re.IGNORECASE), types.SmallInteger(), GenericDataType.NUMERIC),
        (re.compile('^integer.*', re.IGNORECASE), types.INTEGER(), GenericDataType.NUMERIC),
        (re.compile('^bigint.*', re.IGNORECASE), types.BigInteger(), GenericDataType.NUMERIC),
        (re.compile('^real.*', re.IGNORECASE), types.FLOAT(), GenericDataType.NUMERIC),
        (re.compile('^double.*', re.IGNORECASE), types.FLOAT(), GenericDataType.NUMERIC),
        (re.compile('^decimal.*', re.IGNORECASE), types.DECIMAL(), GenericDataType.NUMERIC),
        (re.compile('^varchar(\\((\\d+)\\))*$', re.IGNORECASE), lambda match: types.VARCHAR(int(match[2])) if match[2] else types.String(), GenericDataType.STRING),
        (re.compile('^char(\\((\\d+)\\))*$', re.IGNORECASE), lambda match: types.CHAR(int(match[2])) if match[2] else types.String(), GenericDataType.STRING),
        (re.compile('^varbinary.*', re.IGNORECASE), types.VARBINARY(), GenericDataType.STRING),
        (re.compile('^json.*', re.IGNORECASE), types.JSON(), GenericDataType.STRING),
        (re.compile('^date.*', re.IGNORECASE), types.Date(), GenericDataType.TEMPORAL),
        (re.compile('^timestamp.*', re.IGNORECASE), types.TIMESTAMP(), GenericDataType.TEMPORAL),
        (re.compile('^interval.*', re.IGNORECASE), Interval(), GenericDataType.TEMPORAL),
        (re.compile('^time.*', re.IGNORECASE), types.Time(), GenericDataType.TEMPORAL),
        (re.compile('^array.*', re.IGNORECASE), Array(), GenericDataType.STRING),
        (re.compile('^map.*', re.IGNORECASE), Map(), GenericDataType.STRING),
        (re.compile('^row.*', re.IGNORECASE), Row(), GenericDataType.STRING),
    )
    _time_grain_expressions: Dict[Optional[TimeGrain], str] = {
        None: '{col}',
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
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: Optional[Dict[str, Any]] = None) -> Optional[str]:
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, types.Date):
            return f"DATE '{dttm.date().isoformat()}'"
        if isinstance(sqla_type, types.TIMESTAMP):
            return f"TIMESTAMP '{dttm.isoformat(timespec='microseconds', sep=' ')}'"
        return None

    @classmethod
    def epoch_to_dttm(cls) -> str:
        return 'from_unixtime({col})'

    @classmethod
    def get_default_catalog(cls, database: 'Database') -> Optional[str]:
        if database.url_object.database is None:
            return None
        return database.url_object.database.split('/')[0]

    @classmethod
    def get_catalog_names(cls, database: 'Database', inspector: Inspector) -> Set[str]:
        return {catalog for catalog, in inspector.bind.execute('SHOW CATALOGS')}

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: Dict[str, Any], catalog: Optional[str] = None, schema: Optional[str] = None) -> Tuple[URL, Dict[str, Any]]:
        if uri.database and '/' in uri.database:
            current_catalog, current_schema = uri.database.split('/', 1)
        else:
            current_catalog, current_schema = (uri.database, None)
        if schema:
            schema = parse.quote(schema, safe='')
        adjusted_database = '/'.join([catalog or current_catalog or '', schema or current_schema or '']).rstrip('/')
        uri = uri.set(database=adjusted_database)
        return (uri, connect_args)

    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: Dict[str, Any]) -> Optional[str]:
        database = sqlalchemy_uri.database.strip('/')
        if '/' not in database:
            return None
        return parse.unquote(database.split('/')[1])

    @classmethod
    def estimate_statement_cost(cls, database: 'Database', statement: str, cursor: 'Cursor') -> Dict[str, Any]:
        sql = f'EXPLAIN (TYPE IO, FORMAT JSON) {statement}'
        cursor.execute(sql)
        result = json.loads(cursor.fetchone()[0])
        return result

    @classmethod
    def query_cost_formatter(cls, raw_cost: Dict[str, Any]) -> List[Dict[str, str]]:
        def humanize(value: Any, suffix: str) -> str:
            try:
                value = int(value)
            except ValueError:
                return str(value)
            prefixes = ['K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
            prefix = ''
            to_next_prefix = 1000
            while value > to_next_prefix and prefixes:
                prefix = prefixes.pop(0)
                value //= to_next_prefix
            return f'{value} {prefix}{suffix}'
        cost = []
        columns = [('outputRowCount', 'Output count', ' rows'), ('outputSizeInBytes', 'Output size', 'B'), ('cpuCost', 'CPU cost', ''), ('maxMemory', 'Max memory', 'B'), ('networkCost', 'Network cost', '')]
        for row in raw_cost:
            estimate = row.get('estimate', {})
            statement_cost = {}
            for key, label, suffix in columns:
                if key in estimate:
                    statement_cost[label] = humanize(estimate[key], suffix).strip()
            cost.append(statement_cost)
        return cost

    @classmethod
    @cache_manager.data_cache.memoize()
    def get_function_names(cls, database: 'Database') -> List[str]:
        return database.get_df('SHOW FUNCTIONS')['Function'].tolist()

    @classmethod
    def _partition_query(cls, table: 'Table', indexes: List[Dict[str, Any]], database: 'Database', limit: int = 0, order_by: Optional[List[Tuple[str, bool]]] = None, filters: Optional[Dict[str, Any]] = None) -> str:
        limit_clause = f'LIMIT {limit}' if limit else ''
        order_by_clause = ''
        if order_by:
            l = []
            for field, desc in order_by:
                l.append(field + ' DESC' if desc else '')
            order_by_clause = 'ORDER BY ' + ', '.join(l)
        where_clause = ''
        if filters:
            l = []
            for field, value in filters.items():
                l.append(f"{field} = '{value}'")
            where_clause = 'WHERE ' + ' AND '.join(l)
        presto_version = database.get_extra().get('version')
        if presto_version and Version(presto_version) < Version('0.199'):
            full_table_name = f'{table.schema}.{table.table}' if table.schema else table.table
            partition_select_clause = f'SHOW PARTITIONS FROM {full_table_name}'
        else:
            system_table_name = f'"{table.table}$partitions"'
            full_table_name = f'{table.schema}.{system_table_name}' if table.schema else system_table_name
            partition_select_clause = f'SELECT * FROM {full_table_name}'
        sql = dedent(f'            {partition_select_clause}\n            {where_clause}\n            {order_by_clause}\n            {limit_clause}\n        ')
        return sql

    @classmethod
    def where_latest_partition(cls, database: 'Database', table: 'Table', query: Select, columns: Optional[List[Dict[str, Any]]] = None) -> Optional[Select]:
        try:
            col_names, values = cls.latest_partition(database, table, show_first=True)
        except Exception:
            return None
        if values is None:
            return None
        column_type_by_name = {column.get('column_name'): column.get('type') for column in columns or []}
        for col_name, value in zip(col_names, values, strict=False):
            col_type = column_type_by_name.get(col_name)
            if isinstance(col_type, str):
                col_type_class = getattr(types, col_type, None)
                col_type = col_type_class() if col_type_class else None
            if isinstance(col_type, types.DATE):
                col_type = Date()
            elif isinstance(col_type, types.TIMESTAMP):
                col_type = TimeStamp()
            query = query.where(Column(col_name, col_type) == value)
        return query

    @classmethod
    def _latest_partition_from_df(cls, df: pd.DataFrame) -> Any:
        if not df.empty:
            return df.to_records(index=False)[0].item()
        return None

    @classmethod
    @cache_manager.data_cache.memoize(timeout=60)
    def latest_partition(cls, database: 'Database', table: 'Table', show_first: bool = False, indexes: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[str], Optional[Tuple[Any, ...]]]:
        if indexes is None:
            indexes = database.get_indexes(table)
        if not indexes:
            raise SupersetTemplateException(f'Error getting partition for {table}. Verify that this table has a partition.')
        if len(indexes[0]['column_names']) < 1:
            raise SupersetTemplateException('The table should have one partitioned field')
        if not show_first and len(indexes[0]['column_names']) > 1:
            raise SupersetTemplateException('The table should have a single partitioned field to use this function. You may want to use