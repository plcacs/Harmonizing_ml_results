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
from typing import Any, cast, Deque, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
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
    with contextlib.suppress(ImportError):
        from pyhive.presto import Cursor

COLUMN_DOES_NOT_EXIST_REGEX: Pattern[str] = re.compile(
    "line (?P<location>.+?): .*Column '(?P<column_name>.+?)' cannot be resolved"
)
TABLE_DOES_NOT_EXIST_REGEX: Pattern[str] = re.compile(".*Table (?P<table_name>.+?) does not exist")
SCHEMA_DOES_NOT_EXIST_REGEX: Pattern[str] = re.compile("line (?P<location>.+?): .*Schema '(?P<schema_name>.+?)' does not exist")
CONNECTION_ACCESS_DENIED_REGEX: Pattern[str] = re.compile('Access Denied: Invalid credentials')
CONNECTION_INVALID_HOSTNAME_REGEX: Pattern[str] = re.compile(
    'Failed to establish a new connection: \\[Errno 8\\] nodename nor servname provided, or not known'
)
CONNECTION_HOST_DOWN_REGEX: Pattern[str] = re.compile(
    'Failed to establish a new connection: \\[Errno 60\\] Operation timed out'
)
CONNECTION_PORT_CLOSED_REGEX: Pattern[str] = re.compile(
    'Failed to establish a new connection: \\[Errno 61\\] Connection refused'
)
CONNECTION_UNKNOWN_DATABASE_ERROR: Pattern[str] = re.compile(
    "line (?P<location>.+?): Catalog '(?P<catalog_name>.+?)' does not exist"
)
logger = logging.getLogger(__name__)


def get_children(column: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get the children of a complex Presto type (row or array).

    For arrays, we return a single list with the base type:

        >>> get_children(dict(name="a", type="ARRAY(BIGINT)", is_dttm=False))
        [{"name": "a", "type": "BIGINT", "is_dttm": False}]

    For rows, we return a list of the columns:

        >>> get_children(dict(name="a", type="ROW(BIGINT,FOO VARCHAR)",  is_dttm=False))
        [{'name': 'a._col0', 'type': 'BIGINT', 'is_dttm': False}, {'name': 'a.foo', 'type': 'VARCHAR', 'is_dttm': False}]
    :param column: dictionary representing a Presto column
    :return: list of dictionaries representing children columns
    """
    pattern: Pattern[str] = re.compile(r'(?P<type>\w+)\((?P<children>.*)\)')
    if not column['type']:
        raise ValueError
    match = pattern.match(cast(str, column['type']))
    if not match:
        raise Exception(f"Unable to parse column type {column['type']}")
    group = match.groupdict()
    type_: str = group['type'].upper()
    children_type: str = group['children']
    if type_ == 'ARRAY':
        return [{
            'column_name': column['column_name'],
            'name': column['column_name'],
            'type': children_type,
            'is_dttm': False,
        }]
    if type_ == 'ROW':
        nameless_columns = 0
        columns: List[Dict[str, Any]] = []
        for child in utils.split(children_type, ','):
            parts = list(utils.split(child.strip(), ' '))
            if len(parts) == 2:
                name, type_part = parts
                name = name.strip('"')
            else:
                name = f'_col{nameless_columns}'
                type_part = parts[0]
                nameless_columns += 1
            _column = {
                'column_name': f"{column['column_name']}.{name.lower()}",
                'name': f"{column['column_name']}.{name.lower()}",
                'type': type_part,
                'is_dttm': False,
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
    column_type_mappings: Tuple[
        Tuple[Pattern[str], Any, GenericDataType],
        ...,
    ] = (
        (re.compile('^boolean.*', re.IGNORECASE), types.BOOLEAN(), GenericDataType.BOOLEAN),
        (re.compile('^tinyint.*', re.IGNORECASE), TinyInteger(), GenericDataType.NUMERIC),
        (re.compile('^smallint.*', re.IGNORECASE), types.SmallInteger(), GenericDataType.NUMERIC),
        (re.compile('^integer.*', re.IGNORECASE), types.INTEGER(), GenericDataType.NUMERIC),
        (re.compile('^bigint.*', re.IGNORECASE), types.BigInteger(), GenericDataType.NUMERIC),
        (re.compile('^real.*', re.IGNORECASE), types.FLOAT(), GenericDataType.NUMERIC),
        (re.compile('^double.*', re.IGNORECASE), types.FLOAT(), GenericDataType.NUMERIC),
        (re.compile('^decimal.*', re.IGNORECASE), types.DECIMAL(), GenericDataType.NUMERIC),
        (re.compile('^varchar(\\((\\d+)\\))*$', re.IGNORECASE),
         lambda match: types.VARCHAR(int(match[2])) if match[2] else types.String(),
         GenericDataType.STRING),
        (re.compile('^char(\\((\\d+)\\))*$', re.IGNORECASE),
         lambda match: types.CHAR(int(match[2])) if match[2] else types.String(),
         GenericDataType.STRING),
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
    _time_grain_expressions: Dict[Optional[str], str] = {
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
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: Optional[Any] = None) -> Optional[str]:
        """
        Convert a Python `datetime` object to a SQL expression.
        :param target_type: The target type of expression
        :param dttm: The datetime object
        :param db_extra: The database extra object
        :return: The SQL expression
        Superset only defines time zone naive `datetime` objects, though this method
        handles both time zone naive and aware conversions.
        """
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
    def get_default_catalog(cls, database: Database) -> Optional[str]:
        """
        Return the default catalog.
        """
        if database.url_object.database is None:
            return None
        return database.url_object.database.split('/')[0]

    @classmethod
    def get_catalog_names(cls, database: Database, inspector: Inspector) -> Set[str]:
        """
        Get all catalogs.
        """
        return {catalog for catalog, in inspector.bind.execute('SHOW CATALOGS')}

    @classmethod
    def adjust_engine_params(
        cls,
        uri: URL,
        connect_args: Dict[str, Any],
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> Tuple[URL, Dict[str, Any]]:
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
        """
        Return the configured schema.
        For Presto the SQLAlchemy URI looks like this:
            presto://localhost:8080/hive[/default]
        """
        database = sqlalchemy_uri.database.strip('/')
        if '/' not in database:
            return None
        return parse.unquote(database.split('/')[1])

    @classmethod
    def estimate_statement_cost(cls, database: Database, statement: str, cursor: Any) -> Any:
        """
        Run a SQL query that estimates the cost of a given statement.
        :param database: A Database object
        :param statement: A single SQL statement
        :param cursor: Cursor instance
        :return: JSON response from Trino
        """
        sql = f'EXPLAIN (TYPE IO, FORMAT JSON) {statement}'
        cursor.execute(sql)
        result = json.loads(cursor.fetchone()[0])
        return result

    @classmethod
    def query_cost_formatter(cls, raw_cost: Any) -> List[Dict[str, str]]:
        """
        Format cost estimate.
        :param raw_cost: JSON estimate from Trino
        :return: Human readable cost estimate
        """

        def humanize(value: Any, suffix: str) -> str:
            try:
                value_int = int(value)
            except ValueError:
                return str(value)
            prefixes = ['K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
            prefix = ''
            to_next_prefix = 1000
            while value_int > to_next_prefix and prefixes:
                prefix = prefixes.pop(0)
                value_int //= to_next_prefix
            return f'{value_int} {prefix}{suffix}'

        cost: List[Dict[str, str]] = []
        columns: List[Tuple[str, str, str]] = [
            ('outputRowCount', 'Output count', ' rows'),
            ('outputSizeInBytes', 'Output size', 'B'),
            ('cpuCost', 'CPU cost', ''),
            ('maxMemory', 'Max memory', 'B'),
            ('networkCost', 'Network cost', ''),
        ]
        for row in raw_cost:
            estimate = row.get('estimate', {})
            statement_cost: Dict[str, str] = {}
            for key, label, suffix in columns:
                if key in estimate:
                    statement_cost[label] = humanize(estimate[key], suffix).strip()
            cost.append(statement_cost)
        return cost

    @classmethod
    @cache_manager.data_cache.memoize()
    def get_function_names(cls, database: Database) -> List[str]:
        """
        Get a list of function names that are able to be called on the database.
        Used for SQL Lab autocomplete.
        :param database: The database to get functions for
        :return: A list of function names useable in the database
        """
        return database.get_df('SHOW FUNCTIONS')['Function'].tolist()

    @classmethod
    def _partition_query(
        cls,
        table: Table,
        indexes: List[Dict[str, Any]],
        database: Database,
        limit: int = 0,
        order_by: Optional[List[Tuple[str, bool]]] = None,
        filters: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Return a partition query.
        Note the unused arguments are exposed for sub-classing purposes where custom
        integrations may require the schema, indexes, etc. to build the partition query.
        :param table: the table instance
        :param indexes: the indexes associated with the table
        :param database: the database the query will be run against
        :param limit: the number of partitions to be returned
        :param order_by: a list of tuples of field name and a boolean
            that determines if that field should be sorted in descending order
        :param filters: dict of field name and filter value combinations
        """
        limit_clause: str = f'LIMIT {limit}' if limit else ''
        order_by_clause: str = ''
        if order_by:
            l: List[str] = []
            for field, desc in order_by:
                l.append(field + ' DESC' if desc else '')
            order_by_clause = 'ORDER BY ' + ', '.join(l)
        where_clause: str = ''
        if filters:
            l: List[str] = []
            for field, value in filters.items():
                l.append(f"{field} = '{value}'")
            where_clause = 'WHERE ' + ' AND '.join(l)
        presto_version: Optional[str] = database.get_extra().get('version')
        if presto_version and Version(presto_version) < Version('0.199'):
            full_table_name: str = f'{table.schema}.{table.table}' if table.schema else table.table
            partition_select_clause: str = f'SHOW PARTITIONS FROM {full_table_name}'
        else:
            system_table_name: str = f'"{table.table}$partitions"'
            full_table_name = f'{table.schema}.{system_table_name}' if table.schema else system_table_name
            partition_select_clause = f'SELECT * FROM {full_table_name}'
        sql: str = dedent(f'''
            {partition_select_clause}
            {where_clause}
            {order_by_clause}
            {limit_clause}
        ''')
        return sql

    @classmethod
    def where_latest_partition(
        cls,
        database: Database,
        table: Table,
        query: Any,
        columns: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Any]:
        try:
            col_names, values = cls.latest_partition(database, table, show_first=True)
        except Exception:
            return None
        if values is None:
            return None
        column_type_by_name: Dict[Any, Any] = {column.get('column_name'): column.get('type') for column in columns or []}
        for col_name, value in zip(col_names, values):
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
    def _latest_partition_from_df(cls, df: pd.DataFrame) -> Optional[Any]:
        if not df.empty:
            return df.to_records(index=False)[0].item()
        return None

    @classmethod
    @cache_manager.data_cache.memoize(timeout=60)
    def latest_partition(
        cls,
        database: Database,
        table: Table,
        show_first: bool = False,
        indexes: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[str], Tuple[Any, ...]]:
        """Returns col name and the latest (max) partition value for a table
        :param table: the table instance
        :param database: database query will be run against
        :param show_first: displays the value for the first partitioning key
        :param indexes: indexes from the database
        >>> latest_partition('foo_table')
        (['ds'], ('2018-01-01',))
        """
        if indexes is None:
            indexes = database.get_indexes(table)
        if not indexes:
            raise SupersetTemplateException(f'Error getting partition for {table}. Verify that this table has a partition.')
        if len(indexes[0]['column_names']) < 1:
            raise SupersetTemplateException('The table should have one partitioned field')
        if not show_first and len(indexes[0]['column_names']) > 1:
            raise SupersetTemplateException('The table should have a single partitioned field to use this function. You may want to use `presto.latest_sub_partition`')
        column_names: List[str] = indexes[0]['column_names']
        partition_val = cls._latest_partition_from_df(
            df=database.get_df(
                sql=cls._partition_query(
                    table, indexes, database,
                    limit=1,
                    order_by=[(column_name, True) for column_name in column_names]
                ),
                catalog=table.catalog,
                schema=table.schema,
            )
        )
        return (column_names, partition_val if isinstance(partition_val, tuple) else (partition_val,))

    @classmethod
    def latest_sub_partition(cls, database: Database, table: Table, **kwargs: Any) -> str:
        """Returns the latest (max) partition value for a table
        A filtering criteria should be passed for all fields that are
        partitioned except for the field to be returned.
        :param database: database query will be run against
        :param table: the table instance
        :param kwargs: keyword arguments define the filtering criteria on the partition list.
        >>> latest_sub_partition('sub_partition_table', event_type='click')
        '2018-01-01'
        """
        indexes = database.get_indexes(table)
        part_fields: List[str] = indexes[0]['column_names']
        for k in kwargs.keys():
            if k not in part_fields:
                msg = f'Field [{k}] is not part of the portioning key'
                raise SupersetTemplateException(msg)
        if len(kwargs.keys()) != len(part_fields) - 1:
            msg = 'A filter needs to be specified for {} out of the {} fields.'.format(len(part_fields) - 1, len(part_fields))
            raise SupersetTemplateException(msg)
        field_to_return: str = ""
        for field in part_fields:
            if field not in kwargs:
                field_to_return = field
        sql = cls._partition_query(
            table,
            indexes,
            database,
            limit=1,
            order_by=[(field_to_return, True)],
            filters=kwargs,
        )
        df: pd.DataFrame = database.get_df(sql, table.catalog, table.schema)
        if df.empty:
            return ''
        return df.to_dict()[field_to_return][0]

    @classmethod
    def _show_columns(cls, inspector: Inspector, table: Table) -> List[Any]:
        """
        Show presto column names
        :param inspector: object that performs database schema inspection
        :param table: table instance
        :return: list of column objects
        """
        full_table_name: str = cls.quote_table(table, inspector.engine.dialect)
        return inspector.bind.execute(f'SHOW COLUMNS FROM {full_table_name}').fetchall()

    @classmethod
    def _create_column_info(cls, name: str, data_type: Any) -> Dict[str, Any]:
        """
        Create column info object
        :param name: column name
        :param data_type: column data type
        :return: column info object
        """
        return {'column_name': name, 'name': name, 'type': f'{data_type}', 'is_dttm': None, 'type_generic': None}

    @classmethod
    def get_columns(cls, inspector: Inspector, table: Table, options: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Get columns from a Presto data source. This includes handling row and array data types
        :param inspector: object that performs database schema inspection
        :param table: table instance
        :param options: Extra configuration options, not used by this backend
        :return: a list of results that contain column info
        """
        columns = cls._show_columns(inspector, table)
        result: List[Dict[str, Any]] = []
        for column in columns:
            if is_feature_enabled('PRESTO_EXPAND_DATA') and ('array' in column.Type or 'row' in column.Type):
                structural_column_index = len(result)
                cls._parse_structural_column(column.Column, column.Type, result)
                result[structural_column_index]['nullable'] = getattr(column, 'Null', True)
                result[structural_column_index]['default'] = None
                continue
            column_spec = cls.get_column_spec(column.Type)
            column_type = column_spec.sqla_type if column_spec else None
            if column_type is None:
                column_type = types.String()
                logger.info('Did not recognize type %s of column %s', str(column.Type), str(column.Column))
            column_info = cls._create_column_info(column.Column, column_type)
            column_info['nullable'] = getattr(column, 'Null', True)
            column_info['default'] = None
            column_info['column_name'] = column.Column
            result.append(column_info)
        return result

    @classmethod
    def _parse_structural_column(cls, parent_column_name: str, parent_data_type: str, result: List[Dict[str, Any]]) -> None:
        """
        Parse a row or array column
        :param result: list tracking the results
        """
        formatted_parent_column_name: str = parent_column_name
        if ' ' in parent_column_name:
            formatted_parent_column_name = f'"{parent_column_name}"'
        full_data_type: str = f'{formatted_parent_column_name} {parent_data_type}'
        original_result_len: int = len(result)
        data_types: List[str] = cls._split_data_type(full_data_type, '\\(')
        stack: List[Tuple[str, str]] = []
        for data_type in data_types:
            inner_types: List[str] = cls._split_data_type(data_type, '\\)')
            for inner_type in inner_types:
                if not inner_type and stack:
                    stack.pop()
                elif cls._has_nested_data_types(inner_type):
                    single_fields: List[str] = cls._split_data_type(inner_type, ',')
                    for single_field in single_fields:
                        single_field = single_field.strip()
                        if not single_field:
                            continue
                        field_info: List[str] = cls._split_data_type(single_field, '\\s')
                        column_spec = cls.get_column_spec(field_info[1])
                        column_type = column_spec.sqla_type if column_spec else None
                        if column_type is None:
                            column_type = types.String()
                            logger.info('Did not recognize type %s of column %s', field_info[1], field_info[0])
                        if field_info[1] == 'array' or field_info[1] == 'row':
                            stack.append((field_info[0], field_info[1]))
                            full_parent_path = cls._get_full_name(stack)
                            result.append(cls._create_column_info(full_parent_path, column_type))
                        else:
                            full_parent_path = cls._get_full_name(stack)
                            column_name = f'{full_parent_path}.{field_info[0]}'
                            result.append(cls._create_column_info(column_name, column_type))
                    if not (inner_type.endswith('array') or inner_type.endswith('row')):
                        if stack:
                            stack.pop()
                elif inner_type in ('array', 'row'):
                    stack.append(('', inner_type))
                elif stack:
                    stack.pop()
        if formatted_parent_column_name != parent_column_name:
            for index in range(original_result_len, len(result)):
                result[index]['column_name'] = result[index]['column_name'].replace(formatted_parent_column_name, parent_column_name)

    @classmethod
    def _split_data_type(cls, data_type: str, delimiter: str) -> List[str]:
        """
        Split data type based on given delimiter. Do not split the string if the
        delimiter is enclosed in quotes.
        :param data_type: data type
        :param delimiter: string separator (i.e. open parenthesis, closed parenthesis, comma, whitespace)
        :return: list of strings after breaking it by the delimiter
        """
        return re.split(f'{delimiter}(?=(?:[^\\"]*\\"[^\\"]*\\")*[^\\"]*$)', data_type)

    @classmethod
    def _has_nested_data_types(cls, component_type: str) -> bool:
        """
        Check if string contains a data type.
        :param component_type: data type
        :return: boolean
        """
        comma_regex: Pattern[str] = re.compile(r',(?=(?:[^\\"]*\\"[^\\"]*\\")*[^\\"]*$)')
        white_space_regex: Pattern[str] = re.compile(r'\s(?=(?:[^\\"]*\\"[^\\"]*\\")*[^\\"]*$)')
        return re.search(comma_regex, component_type) is not None or re.search(white_space_regex, component_type) is not None

    @classmethod
    def _get_full_name(cls, names: List[Tuple[str, str]]) -> str:
        """
        Get the full column name
        :param names: list of all individual column name tuples
        :return: full column name
        """
        return '.'.join((column[0] for column in names if column[0]))


class PrestoEngineSpec(PrestoBaseEngineSpec):
    engine: str = 'presto'
    engine_name: str = 'Presto'
    allows_alias_to_source_column: bool = False
    custom_errors: Dict[Pattern[str], Tuple[str, SupersetErrorType, Dict[str, Any]]] = {
        COLUMN_DOES_NOT_EXIST_REGEX: (__('We can\'t seem to resolve the column "%(column_name)s" at line %(location)s.'), SupersetErrorType.COLUMN_DOES_NOT_EXIST_ERROR, {}),
        TABLE_DOES_NOT_EXIST_REGEX: (__('The table "%(table_name)s" does not exist. A valid table must be used to run this query.'), SupersetErrorType.TABLE_DOES_NOT_EXIST_ERROR, {}),
        SCHEMA_DOES_NOT_EXIST_REGEX: (__('The schema "%(schema_name)s" does not exist. A valid schema must be used to run this query.'), SupersetErrorType.SCHEMA_DOES_NOT_EXIST_ERROR, {}),
        CONNECTION_ACCESS_DENIED_REGEX: (__('Either the username "%(username)s" or the password is incorrect.'), SupersetErrorType.CONNECTION_ACCESS_DENIED_ERROR, {}),
        CONNECTION_INVALID_HOSTNAME_REGEX: (__('The hostname "%(hostname)s" cannot be resolved.'), SupersetErrorType.CONNECTION_INVALID_HOSTNAME_ERROR, {}),
        CONNECTION_HOST_DOWN_REGEX: (__('The host "%(hostname)s" might be down, and can\'t be reached on port %(port)s.'), SupersetErrorType.CONNECTION_HOST_DOWN_ERROR, {}),
        CONNECTION_PORT_CLOSED_REGEX: (__('Port %(port)s on hostname "%(hostname)s" refused the connection.'), SupersetErrorType.CONNECTION_PORT_CLOSED_ERROR, {}),
        CONNECTION_UNKNOWN_DATABASE_ERROR: (__('Unable to connect to catalog named "%(catalog_name)s".'), SupersetErrorType.CONNECTION_UNKNOWN_DATABASE_ERROR, {}),
    }

    @classmethod
    def get_allow_cost_estimate(cls, extra: Dict[str, Any]) -> bool:
        version: Optional[str] = extra.get('version')
        return version is not None and Version(version) >= Version('0.319')

    @classmethod
    def update_impersonation_config(
        cls,
        database: Database,
        connect_args: Dict[str, Any],
        uri: URL,
        username: Optional[str],
        access_token: Optional[str],
    ) -> None:
        """
        Update a configuration dictionary to set properties for impersonating users.
        :param database: Database object
        :param connect_args: Configuration to be updated
        :param uri: URI string
        :param username: Effective username
        :param access_token: Personal access token for OAuth2
        """
        url = make_url_safe(uri)
        backend_name = url.get_backend_name()
        if backend_name == 'presto' and username is not None:
            connect_args['principal_username'] = username

    @classmethod
    def get_table_names(cls, database: Database, inspector: Inspector, schema: Optional[str]) -> Set[str]:
        """
        Get all the real table names within the specified schema.
        :param database: The database to inspect
        :param inspector: The SQLAlchemy inspector
        :param schema: The schema to inspect
        :returns: The physical table names
        """
        return super().get_table_names(database, inspector, schema) - cls.get_view_names(database, inspector, schema)

    @classmethod
    def get_view_names(cls, database: Database, inspector: Inspector, schema: Optional[str]) -> Set[str]:
        """
        Get all the view names within the specified schema.
        :param database: The database to inspect
        :param inspector: The SQLAlchemy inspector
        :param schema: The schema to inspect
        :returns: The view names
        """
        if schema:
            sql: str = dedent("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = %(schema)s
                AND table_type = 'VIEW'
            """).strip()
            params: Dict[str, Any] = {'schema': schema}
        else:
            sql = dedent("""
                SELECT table_name FROM information_schema.tables
                WHERE table_type = 'VIEW'
            """).strip()
            params = {}
        with database.get_raw_connection(schema=schema) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            results = cursor.fetchall()
            return {row[0] for row in results}

    @classmethod
    def _is_column_name_quoted(cls, column_name: str) -> bool:
        """
        Check if column name is in quotes
        :param column_name: column name
        :return: boolean
        """
        return column_name.startswith('"') and column_name.endswith('"')

    @classmethod
    def _get_fields(cls, cols: List[Dict[str, Any]]) -> List[ColumnClause]:
        """
        Format column clauses where names are in quotes and labels are specified
        :param cols: columns
        :return: column clauses
        """
        column_clauses: List[ColumnClause] = []
        dot_pattern: str = (
            r'\.                # split on period\n'
            r'(?=               # look ahead\n'
            r'(?:               # create non-capture group\n'
            r'[^\\"]*\\"[^\\"]*\\"  # two quotes\n'
            r')*[^\\"]*$)'        # end regex
        )
        dot_regex: Pattern[str] = re.compile(dot_pattern, re.VERBOSE)
        for col in cols:
            col_names = re.split(dot_regex, col['column_name'])
            for index, col_name in enumerate(col_names):
                if not cls._is_column_name_quoted(col_name):
                    col_names[index] = f'"{col_name}"'
            quoted_col_name = '.'.join(
                (col_name if cls._is_column_name_quoted(col_name) else f'"{col_name}"' for col_name in col_names)
            )
            column_clause = literal_column(quoted_col_name).label(col['column_name'])
            column_clauses.append(column_clause)
        return column_clauses

    @classmethod
    def select_star(
        cls,
        database: Database,
        table: Table,
        engine: Engine,
        limit: int = 100,
        show_cols: bool = False,
        indent: bool = True,
        latest_partition: bool = True,
        cols: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """
        Include selecting properties of row objects.
        """
        cols = cols or []
        presto_cols: List[Dict[str, Any]] = cols
        if is_feature_enabled('PRESTO_EXPAND_DATA') and show_cols:
            dot_regex: str = r'\.(?=(?:[^\\"]*\\"[^\\"]*\\")*[^\\"]*$)'
            presto_cols = [col for col in presto_cols if not re.search(dot_regex, col['column_name'])]
        return super().select_star(database, table, engine, limit, show_cols, indent, latest_partition, presto_cols)

    @classmethod
    def expand_data(
        cls,
        columns: List[Dict[str, Any]],
        data: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Expand nested row and array data for display.
        :param columns: columns selected in the query
        :param data: original data set
        :return: tuple of (all columns, expanded data set, expanded nested fields)
        """
        if not is_feature_enabled('PRESTO_EXPAND_DATA'):
            return (columns, data, [])
        to_process: Deque[Tuple[Dict[str, Any], int]] = deque(((column, 0) for column in columns))
        all_columns: List[Dict[str, Any]] = []
        expanded_columns: List[Dict[str, Any]] = []
        current_array_level: Optional[int] = None
        while to_process:
            column, level = to_process.popleft()
            if column['column_name'] not in [col['column_name'] for col in all_columns]:
                all_columns.append(column)
            if level != current_array_level:
                unnested_rows: Dict[int, int] = defaultdict(int)
                current_array_level = level
            name: str = column['column_name']
            if column['type'] and column['type'].startswith('ARRAY('):
                to_process.append((get_children(column)[0], level + 1))
                i: int = 0
                while i < len(data):
                    row = data[i]
                    values = row.get(name)
                    if isinstance(values, str):
                        row[name] = values = destringify(values)
                    if values:
                        extra_rows = len(values) - 1
                        current_unnested_rows = unnested_rows[i]
                        missing = extra_rows - current_unnested_rows
                        for _ in range(missing):
                            data.insert(i + current_unnested_rows + 1, {})
                            unnested_rows[i] += 1
                        for j, value in enumerate(values):
                            data[i + j][name] = value
                        i += unnested_rows[i]
                    i += 1
            if column['type'] and column['type'].startswith('ROW('):
                expanded = get_children(column)
                to_process.extendleft(((col, level) for col in expanded[::-1]))
                expanded_columns.extend(expanded)
                for row in data:
                    values = row.get(name) or []
                    if isinstance(values, str):
                        values = cast(Optional[List[Any]], destringify(values))
                        row[name] = values
                    for value, col in zip(values or [], expanded):
                        row[col['column_name']] = value
        new_data: List[Dict[str, Any]] = [
            {k['column_name']: row.get(k['column_name'], '') for k in all_columns} for row in data
        ]
        return (all_columns, new_data, expanded_columns)

    @classmethod
    def get_extra_table_metadata(cls, database: Database, table: Table) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if (indexes := database.get_indexes(table)):
            col_names, latest_parts = cls.latest_partition(database, table, show_first=True, indexes=indexes)
            if not latest_parts:
                latest_parts = tuple([None] * len(col_names))
            metadata['partitions'] = {
                'cols': sorted(indexes[0].get('column_names', [])),
                'latest': dict(zip(col_names, latest_parts)),
                'partitionQuery': cls._partition_query(table=table, indexes=indexes, database=database)
            }
        metadata['view'] = cast(Any, cls.get_create_view(database, table.schema, table.table))
        return metadata

    @classmethod
    def get_create_view(cls, database: Database, schema: str, table: str) -> Optional[str]:
        """
        Return a CREATE VIEW statement, or `None` if not a view.
        :param database: Database instance
        :param schema: Schema name
        :param table: Table (view) name
        """
        from pyhive.exc import DatabaseError
        with database.get_raw_connection(schema=schema) as conn:
            cursor = conn.cursor()
            sql = f'SHOW CREATE VIEW {schema}.{table}'
            try:
                cls.execute(cursor, sql, database)
                rows = cls.fetch_data(cursor, 1)
                return rows[0][0]
            except DatabaseError:
                return None

    @classmethod
    def get_tracking_url(cls, cursor: Any) -> Optional[str]:
        with contextlib.suppress(AttributeError):
            if cursor.last_query_id:
                return f'{cursor._protocol}://{cursor._host}:{cursor._port}/ui/query.html?{cursor.last_query_id}'
        return None

    @classmethod
    def handle_cursor(cls, cursor: Any, query: Query) -> None:
        """Updates progress information"""
        if (tracking_url := cls.get_tracking_url(cursor)):
            query.tracking_url = tracking_url
            db.session.commit()
        query_id: int = query.id
        poll_interval: float = query.database.connect_args.get('poll_interval', current_app.config['PRESTO_POLL_INTERVAL'])
        logger.info('Query %i: Polling the cursor for progress', query_id)
        polled: Any = cursor.poll()
        while polled:
            stats: Dict[str, Any] = polled.get('stats', {})
            query = db.session.query(type(query)).filter_by(id=query_id).one()
            if query.status in [QueryStatus.STOPPED, QueryStatus.TIMED_OUT]:
                cursor.cancel()
                break
            if stats:
                state: str = stats.get('state')
                if state == 'FINISHED':
                    break
                completed_splits: float = float(stats.get('completedSplits'))
                total_splits: float = float(stats.get('totalSplits'))
                if total_splits and completed_splits:
                    progress: float = 100 * (completed_splits / total_splits)
                    logger.info('Query %s progress: %s / %s splits', query_id, completed_splits, total_splits)
                    query.progress = max(query.progress, progress)
                    db.session.commit()
            time.sleep(poll_interval)
            logger.info('Query %i: Polling the cursor for progress', query_id)
            polled = cursor.poll()

    @classmethod
    def _extract_error_message(cls, ex: Exception) -> str:
        if hasattr(ex, 'orig') and type(ex.orig).__name__ == 'DatabaseError' and isinstance(ex.orig[0], dict):
            error_dict: Dict[str, Any] = ex.orig[0]
            return '{} at {}: {}'.format(error_dict.get('errorName'), error_dict.get('errorLocation'), error_dict.get('message'))
        if type(ex).__name__ == 'DatabaseError' and hasattr(ex, 'args') and ex.args:
            error_dict = ex.args[0]
            return error_dict.get('message', _('Unknown Presto Error'))
        return utils.error_msg_from_exception(ex)

    @classmethod
    def has_implicit_cancel(cls) -> bool:
        """
        Return True if the live cursor handles the implicit cancelation of the query,
        False otherwise.
        :return: Whether the live cursor implicitly cancels the query
        """
        return True