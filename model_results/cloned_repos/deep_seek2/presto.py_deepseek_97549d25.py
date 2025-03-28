from typing import Any, Dict, List, Optional, Set, Tuple, Union, Pattern, cast, Deque, DefaultDict
from datetime import datetime
from collections import defaultdict, deque
from re import Pattern as re_Pattern
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.result import Row as ResultRow
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.expression import ColumnClause, Select
from sqlalchemy import types
from flask import current_app
from flask_babel import gettext as __, lazy_gettext as _
from packaging.version import Version
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

COLUMN_DOES_NOT_EXIST_REGEX: re_Pattern[str] = re.compile(
    "line (?P<location>.+?): .*Column '(?P<column_name>.+?)' cannot be resolved"
)
TABLE_DOES_NOT_EXIST_REGEX: re_Pattern[str] = re.compile(".*Table (?P<table_name>.+?) does not exist")
SCHEMA_DOES_NOT_EXIST_REGEX: re_Pattern[str] = re.compile(
    "line (?P<location>.+?): .*Schema '(?P<schema_name>.+?)' does not exist"
)
CONNECTION_ACCESS_DENIED_REGEX: re_Pattern[str] = re.compile("Access Denied: Invalid credentials")
CONNECTION_INVALID_HOSTNAME_REGEX: re_Pattern[str] = re.compile(
    r"Failed to establish a new connection: \[Errno 8\] nodename nor servname "
    "provided, or not known"
)
CONNECTION_HOST_DOWN_REGEX: re_Pattern[str] = re.compile(
    r"Failed to establish a new connection: \[Errno 60\] Operation timed out"
)
CONNECTION_PORT_CLOSED_REGEX: re_Pattern[str] = re.compile(
    r"Failed to establish a new connection: \[Errno 61\] Connection refused"
)
CONNECTION_UNKNOWN_DATABASE_ERROR: re_Pattern[str] = re.compile(
    r"line (?P<location>.+?): Catalog '(?P<catalog_name>.+?)' does not exist"
)

logger: logging.Logger = logging.getLogger(__name__)

def get_children(column: ResultSetColumnType) -> List[ResultSetColumnType]:
    pattern: re_Pattern[str] = re.compile(r"(?P<type>\w+)\((?P<children>.*)\)")
    if not column["type"]:
        raise ValueError
    match: Optional[re.Match[str]] = pattern.match(cast(str, column["type"]))
    if not match:
        raise Exception(f"Unable to parse column type {column['type']}")

    group: Dict[str, str] = match.groupdict()
    type_: str = group["type"].upper()
    children_type: str = group["children"]
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
        nameless_columns: int = 0
        columns: List[ResultSetColumnType] = []
        for child in utils.split(children_type, ","):
            parts: List[str] = list(utils.split(child.strip(), " "))
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
    supports_dynamic_schema: bool = True
    supports_catalog: bool = supports_dynamic_catalog = True

    column_type_mappings: Tuple[Tuple[re_Pattern[str], Any, GenericDataType], ...] = (
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
        sqla_type: Optional[types.TypeEngine] = cls.get_sqla_column_type(target_type)

        if isinstance(sqla_type, types.Date):
            return f"DATE '{dttm.date().isoformat()}'"
        if isinstance(sqla_type, types.TIMESTAMP):
            return f"""TIMESTAMP '{dttm.isoformat(timespec="microseconds", sep=" ")}'"""

        return None

    @classmethod
    def epoch_to_dttm(cls) -> str:
        return "from_unixtime({col})"

    @classmethod
    def get_default_catalog(cls, database: Database) -> Optional[str]:
        if database.url_object.database is None:
            return None

        return database.url_object.database.split("/")[0]

    @classmethod
    def get_catalog_names(
        cls,
        database: Database,
        inspector: Inspector,
    ) -> Set[str]:
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

        adjusted_database: str = "/".join(
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
        database: str = sqlalchemy_uri.database.strip("/")

        if "/" not in database:
            return None

        return parse.unquote(database.split("/")[1])

    @classmethod
    def estimate_statement_cost(
        cls, database: Database, statement: str, cursor: Any
    ) -> Dict[str, Any]:
        sql: str = f"EXPLAIN (TYPE IO, FORMAT JSON) {statement}"
        cursor.execute(sql)

        result: Dict[str, Any] = json.loads(cursor.fetchone()[0])
        return result

    @classmethod
    def query_cost_formatter(
        cls, raw_cost: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        def humanize(value: Any, suffix: str) -> str:
            try:
                value = int(value)
            except ValueError:
                return str(value)

            prefixes: List[str] = ["K", "M", "G", "T", "P", "E", "Z", "Y"]
            prefix: str = ""
            to_next_prefix: int = 1000
            while value > to_next_prefix and prefixes:
                prefix = prefixes.pop(0)
                value //= to_next_prefix

            return f"{value} {prefix}{suffix}"

        cost: List[Dict[str, str]] = []
        columns: List[Tuple[str, str, str]] = [
            ("outputRowCount", "Output count", " rows"),
            ("outputSizeInBytes", "Output size", "B"),
            ("cpuCost", "CPU cost", ""),
            ("maxMemory", "Max memory", "B"),
            ("networkCost", "Network cost", ""),
        ]
        for row in raw_cost:
            estimate: Dict[str, float] = row.get("estimate", {})
            statement_cost: Dict[str, str] = {}
            for key, label, suffix in columns:
                if key in estimate:
                    statement_cost[label] = humanize(estimate[key], suffix).strip()
            cost.append(statement_cost)

        return cost

    @classmethod
    @cache_manager.data_cache.memoize()
    def get_function_names(cls, database: Database) -> List[str]:
        return database.get_df("SHOW FUNCTIONS")["Function"].tolist()

    @classmethod
    def _partition_query(
        cls,
        table: Table,
        indexes: List[Dict[str, Any]],
        database: Database,
        limit: int = 0,
        order_by: Optional[List[Tuple[str, bool]]] = None,
        filters: Optional[Dict[Any, Any]] = None,
    ) -> str:
        limit_clause: str = f"LIMIT {limit}" if limit else ""
        order_by_clause: str = ""
        if order_by:
            l: List[str] = []
            for field, desc in order_by:
                l.append(field + " DESC" if desc else "")
            order_by_clause = "ORDER BY " + ", ".join(l)

        where_clause: str = ""
        if filters:
            l: List[str] = []
            for field, value in filters.items():
                l.append(f"{field} = '{value}'")
            where_clause = "WHERE " + " AND ".join(l)

        presto_version: Optional[str] = database.get_extra().get("version")

        if presto_version and Version(presto_version) < Version("0.199"):
            full_table_name: str = (
                f"{table.schema}.{table.table}" if table.schema else table.table
            )
            partition_select_clause: str = f"SHOW PARTITIONS FROM {full_table_name}"
        else:
            system_table_name: str = f'"{table.table}$partitions"'
            full_table_name = (
                f"{table.schema}.{system_table_name}"
                if table.schema
                else system_table_name
            )
            partition_select_clause = f"SELECT * FROM {full_table_name}"

        sql: str = dedent(
            f"""\
            {partition_select_clause}
            {where_clause}
            {order_by_clause}
            {limit_clause}
        """
        )
        return sql

    @classmethod
    def where_latest_partition(
        cls,
        database: Database,
        table: Table,
        query: Select,
        columns: Optional[List[ResultSetColumnType]] = None,
    ) -> Optional[Select]:
        try:
            col_names, values = cls.latest_partition(database, table, show_first=True)
        except Exception:
            return None

        if values is None:
            return None

        column_type_by_name: Dict[str, Optional[str]] = {
            column.get("column_name"): column.get("type") for column in columns or []
        }

        for col_name, value in zip(col_names, values, strict=False):
            col_type: Optional[str] = column_type_by_name.get(col_name)

            if isinstance(col_type, str):
                col_type_class: Optional[types.TypeEngine] = getattr(types, col_type, None)
                col_type = col_type_class() if col_type_class else None

            if isinstance(col_type, types.DATE):
                col_type = Date()
            elif isinstance(col_type, types.TIMESTAMP):
                col_type = TimeStamp()

            query = query.where(Column(col_name, col_type) == value)

        return query

    @classmethod
    def _latest_partition_from_df(cls, df: pd.DataFrame) -> Optional[List[str]]:
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
    ) -> Tuple[List[str], Optional[List[str]]]:
        if indexes is None:
            indexes = database.get_indexes(table)

        if not indexes:
            raise SupersetTemplateException(
                f"Error getting partition for {table}. "
                "Verify that this table has a partition."
            )

        if len(indexes[0]["column_names"]) < 1:
            raise SupersetTemplateException(
                "The table should have one partitioned field"
            )

        if not show_first and len(indexes[0]["column_names"]) > 1:
            raise SupersetTemplateException(
                "The table should have a single partitioned field "
                "to use this function. You may want to use "
                "`presto.latest_sub_partition`"
            )

        column_names: List[str] = indexes[0]["column_names"]

        return column_names, cls._latest_partition_from_df(
            df=database.get_df(
                sql=cls._partition_query(
                    table,
                    indexes,
                    database,
                    limit=1,
                    order_by=[(column_name, True) for column_name in column_names],
                ),
                catalog=table.catalog,
                schema=table.schema,
            )
        )

    @classmethod
    def latest_sub_partition(
        cls,
        database: Database,
        table: Table,
        **kwargs: Any,
    ) -> Any:
        indexes: List[Dict[str, Any]] = database.get_indexes(table)
        part_fields: List[str] = indexes[0]["column_names"]
        for k in kwargs.keys():
            if k not in k in part_fields:
                msg: str = f"Field [{k}] is not part of the portioning key"
                raise SupersetTemplateException(msg)
        if len(kwargs.keys()) != len(part_fields) - 1:
            msg: str = (
                "A filter needs to be specified for {} out of the " "{} fields."
            ).format(len(part_fields) - 1, len(part_fields))
            raise SupersetTemplateException(msg)

        field_to_return: str = ""
        for field in part_fields:
            if field not in kwargs:
                field_to_return = field

        sql: str = cls._partition_query(
            table,
            indexes,
            database,
            limit=1,
            order_by=[(field_to_return, True)],
            filters=kwargs,
        )
        df: pd.DataFrame = database.get_df(sql, table.catalog, table.schema)
        if df.empty:
            return ""
        return df.to_dict()[field_to_return][0]

    @classmethod
    def _show_columns(
        cls,
        inspector: Inspector,
        table: Table,
    ) -> List[ResultRow]:
        full_table_name: str = cls.quote_table(table, inspector.engine.dialect)
        return inspector.bind.execute(f"SHOW COLUMNS FROM {full_table_name}").fetchall()

    @classmethod
    def _create_column_info(
        cls, name: str, data_type: types.TypeEngine
    ) -> ResultSetColumnType:
        return {
            "column_name": name,
            "name": name,
            "type": f"{data_type}",
            "is_dttm": None,
            "type_generic": None,
        }

    @classmethod
    def get_columns(
        cls,
        inspector: Inspector,
        table: Table,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[ResultSetColumnType]:
        columns: List[ResultRow] = cls._show_columns(inspector, table)
        result: List[ResultSetColumnType] = []
        for column in columns:
            if is_feature_enabled("PRESTO_EXPAND_DATA") and (
                "array" in column.Type or "row" in column.Type
            ):
                structural_column_index: int = len(result)
                cls._parse_structural_column(column.Column, column.Type, result)
                result[structural_column_index]["nullable"] = getattr(
                    column, "Null", True
                )
                result[structural_column_index]["default"] = None
                continue

            column_spec: Optional[BaseEngineSpec.ColumnSpec] = cls.get_column_spec(column.Type)
            column_type: Optional[types.TypeEngine] = column_spec.sqla_type if column_spec else None
            if column_type is None:
                column_type = types.String()
                logger.info(
                    "Did not recognize type %s of column %s",
                    str(column.Type),
                    str(column.Column),
                )
            column_info: ResultSetColumnType = cls._create_column_info(column.Column, column_type)
            column_info["nullable"] = getattr(column, "Null", True)
            column_info["default"] = None
            column_info["column_name"] = column.Column
            result.append(column_info)

        return result

    @classmethod
    def _parse_structural_column(
        cls,
        parent_column_name: str,
        parent_data_type: str,
        result: List[ResultSetColumnType],
    ) -> None:
        formatted_parent_column_name: str = parent_column_name
        if " " in parent_column_name:
            formatted_parent_column_name = f'"{parent_column_name}"'
        full_data_type: str = f"{formatted_parent_column_name} {parent_data_type}"
        original_result_len: int = len(result)
        data_types: List[str] = cls._split_data_type(full_data_type, r"\(")
        stack: List[Tuple[str, str]] = []
        for data_type in data_types:
            inner_types: List[str] = cls._split_data_type(data_type, r"\)")
            for inner_type in inner_types:
                if not inner_type and stack:
                    stack.pop()
                elif cls._has_nested_data_types(inner_type):
                    single_fields: List[str] = cls._split_data_type(inner_type, ",")
                    for single_field in single_fields:
                        single_field = single_field.strip()
                        if not single_field:
                            continue
                        field_info: List[str] = cls._split_data_type(single_field, r"\s")
                        column_spec: Optional[BaseEngineSpec.ColumnSpec] = cls.get_column_spec(field_info[1])
                        column_type: Optional[types.TypeEngine] = column_spec.sqla_type if column_spec else None
                        if column_type is None:
                            column_type = types.String()
                            logger.info(
                                "Did not recognize type %s of column %s",
                                field_info[1],
                                field_info[0],
                            )
                        if field_info[1] == "array" or field_info[1] == "row":
                            stack.append((field_info[0], field_info[1]))
                            full_parent_path: str = cls._get_full_name(stack)
                            result.append(
                                cls._create_column_info(full_parent_path, column_type)
                            )
                        else:
                            full_parent_path = cls._get_full_name(stack)
                            column_name: str = f"{full_parent_path}.{field_info[0]}"
                            result.append(
                                cls._create_column_info(column_name, column_type)
                            )
                    if not (inner_type.endswith("array") or inner_type.endswith("row")):
                        stack.pop()
                elif inner_type in ("array", "row"):
                    stack.append(("", inner_type))
                elif stack:
                    stack.pop()
        if formatted_parent_column_name != parent_column_name:
            for index in range(original_result_len, len(result)):
                result[index]["column_name"] = result[index]["column_name"].replace(
                    formatted_parent_column_name, parent_column_name
                )

    @classmethod
    def _split_data_type(cls, data_type: str, delimiter: str) -> List[str]:
        return re.split(rf"{delimiter}(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", data_type)

    @classmethod
    def _has_nested_data_types(cls, component_type: str) -> bool:
        comma_regex: re_Pattern[str] = r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)"
        white_space_regex: re_Pattern[str] = r"\s(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)"
        return (
            re.search(comma_regex, component_type) is not None
            or re.search(white_space_regex, component_type) is not None
        )

    @classmethod
    def _get_full_name(cls, names: List[Tuple[str, str]]) -> str:
        return ".".join(column[0] for column in names if column[0])

class PrestoEngineSpec(PrestoBaseEngineSpec):
    engine: str = "presto"
    engine_name: str = "Presto"
    allows_alias_to_source_column: bool = False

    custom_errors: Dict[re_Pattern[str], Tuple[str, SupersetErrorType, Dict[str, Any]]] = {
        COLUMN_DOES_NOT_EXIST_REGEX: (
            __(
                'We can\'t seem to resolve the column "%(column_name)s" at '
                "line %(location)s.",
            ),
            SupersetErrorType.COLUMN_DOES_NOT_EXIST_ERROR,
            {},
        ),
        TABLE_DOES_NOT_EXIST_REGEX: (
            __(
                'The table "%(table_name)s" does not exist. '
                "A valid table must be used to run this query.",
            ),
            SupersetErrorType.TABLE_DOES_NOT_EXIST_ERROR,
            {},
        ),
        SCHEMA_DOES_NOT_EXIST_REGEX: (
            __(
                'The schema "%(schema_name)s" does not exist. '
                "A valid schema must be used to run this query.",
            ),
            SupersetErrorType.SCHEMA_DOES_NOT_EXIST_ERROR,
            {},
        ),
        CONNECTION_ACCESS_DENIED_REGEX: (
            __('Either the username "%(username)s" or the password is incorrect.'),
            SupersetErrorType.CONNECTION_ACCESS_DENIED_ERROR,
            {},
        ),
        CONNECTION_INVALID_HOSTNAME_REGEX: (
            __('The hostname "%(hostname)s" cannot be resolved.'),
            SupersetErrorType.CONNECTION_INVALID_HOSTNAME_ERROR,
            {},
        ),
        CONNECTION_HOST_DOWN_REGEX: (
            __(
                'The host "%(hostname)s" might be down, and can\'t be '
                "reached on port %(port)s."
            ),
            SupersetErrorType.CONNECTION_HOST_DOWN_ERROR,
            {},
        ),
        CONNECTION_PORT_CLOSED_REGEX: (
            __('Port %(port)s on hostname "%(hostname)s" refused the connection.'),
            SupersetErrorType.CONNECTION_PORT_CLOSED_ERROR,
            {},
        ),
        CONNECTION_UNKNOWN_DATABASE_ERROR: (
            __('Unable to connect to catalog named "%(catalog_name)s".'),
            SupersetErrorType.CONNECTION_UNKNOWN_DATABASE_ERROR,
            {},
        ),
    }

    @classmethod
    def get_allow_cost_estimate(cls, extra: Dict[str, Any]) -> bool:
        version: Optional[str] = extra.get("version")
        return version is not None and Version(version) >= Version("0.319")

    @classmethod
    def update_impersonation_config(
        cls,
        database: Database,
        connect_args: Dict[str, Any],
        uri: str,
        username: Optional[str],
        access_token: Optional[str],
    ) -> None:
        url: URL = make_url_safe(uri)
        backend_name: str = url.get_backend_name()

        if backend_name == "presto" and username is not None:
            connect_args["principal_username"] = username

    @classmethod
    def get_table_names(
        cls,
        database: Database,
        inspector: Inspector,
        schema: Optional[str],
    ) -> Set[str]:
        return super().get_table_names(
            database, inspector, schema
        ) - cls.get_view_names(database, inspector, schema)

    @classmethod
    def get_view_names(
        cls,
        database: Database,
        inspector: Inspector,
        schema: Optional[str],
    ) -> Set[str]:
        if schema:
            sql: str = dedent(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = %(schema)s
                AND table_type = 'VIEW'
                """
            ).strip()
            params: Dict[str, str] = {"schema": schema}
        else:
            sql = dedent(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_type = 'VIEW'
                """
            ).strip()
            params = {}

        with database.get_raw_connection(schema=schema) as conn:
            cursor: Any = conn.cursor()
            cursor.execute(sql, params)
            results: List[Tuple[str]] = cursor.fetchall()
            return {row[0] for row in results}

    @classmethod
    def _is_column_name_quoted(cls, column_name: str) -> bool:
        return column_name.startswith('"') and column_name.endswith('"')

    @classmethod
    def _get_fields(cls, cols: List[ResultSetColumnType]) -> List[ColumnClause]:
        column_clauses: List[ColumnClause] = []
        dot_pattern: str = r"""\.                # split on period
                          (?=               # look ahead
                          (?:               # create non-capture group
                          [^\"]*\"[^\"]*\"  # two quotes
                          )*[^\"]*$)        # end regex"""
        dot_regex: re_Pattern[str] = re.compile(dot_pattern, re.VERBOSE)
        for col in cols:
            col_names: List[str] = re.split(dot_regex, col["column_name"])
            for index, col_name in enumerate(col_names):
                if not cls._is_column_name_quoted(col_name):
                    col_names[index] = f'"{col_name}"'
            quoted_col_name: str = ".".join(
                col_name if cls._is_column_name_quoted(col_name) else f'"{col_name}"'
                for col_name in col_names
            )
            column_clause: ColumnClause = literal_column(quoted_col_name).label(col["column_name"])
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
        cols: Optional[List[ResultSetColumnType]] = None,
    ) -> str:
        cols = cols or []
        presto_cols: List[ResultSetColumnType] = cols
        if is_feature_enabled("PRESTO_EXPAND_DATA") and show_cols:
            dot_regex: re_Pattern[str] = r"\.(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)"
            presto_cols = [
                col
                for col in presto_cols
                if not re.search(dot_regex, col["column_name"])
            ]
        return super().select_star(
            database,
            table,
            engine,
            limit,
            show_cols,
            indent,
            latest_partition,
            presto_cols,
        )

    @classmethod
    def expand_data(
        cls, columns: List[ResultSetColumnType], data: List[Dict[Any, Any]]
    ) -> Tuple[
        List[ResultSetColumnType], List[Dict[Any, Any]], List[ResultSetColumnType]
    ]:
        if not is_feature_enabled("PRESTO_EXPAND_DATA"):
            return columns, data, []

        to_process: Deque[Tuple[ResultSetColumnType, int]] = deque((column, 0) for column in columns)
        all_columns: List[ResultSetColumnType] = []
        expanded_columns: List[ResultSetColumnType] = []
        current_array_level: Optional[int] = None
        while to_process:
            column, level = to_process.popleft()
            if column["column_name"] not in [
                column["column_name"] for column in all_columns
            ]:
                all_columns.append(column)

            if level != current_array_level:
                unnested_rows: DefaultDict[int, int] = defaultdict(int)
                current_array_level = level

            name: str = column["column_name"]
            values: Union[str, List[Any], None]

            if column["type"] and column["type"].startswith("ARRAY("):
                to_process.append((get_children(column)[0], level + 1))

                i: int = 0
                while i < len(data):
                    row: Dict[Any, Any] = data[i]
                    values = row.get(name)
                    if isinstance(values, str):
                        row[name] = values = destringify(values)
                    if values:
                        extra_rows: int =