from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, ClassVar, Iterable, Mapping, Optional, Pattern

import pandas as pd
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.expression import ColumnClause, Select
from superset.constants import TimeGrain
from superset.db_engine_specs.presto import PrestoEngineSpec
from superset.models.core import Database
from superset.models.sql_lab import Query
from superset.sql_parse import Table
from superset.superset_typing import ResultSetColumnType

logger: logging.Logger

def upload_to_s3(filename: str, upload_prefix: str, table: Table) -> str: ...

class HiveEngineSpec(PrestoEngineSpec):
    engine: ClassVar[str]
    engine_name: ClassVar[str]
    max_column_name_length: ClassVar[int]
    allows_alias_to_source_column: ClassVar[bool]
    allows_hidden_orderby_agg: ClassVar[bool]
    supports_dynamic_schema: ClassVar[bool]
    _show_functions_column: ClassVar[str]
    _time_grain_expressions: ClassVar[dict[Optional[TimeGrain], str]]
    jobs_stats_r: ClassVar[Pattern[str]]
    launching_job_r: ClassVar[Pattern[str]]
    stage_progress_r: ClassVar[Pattern[str]]

    @classmethod
    def patch(cls) -> None: ...
    @classmethod
    def fetch_data(cls, cursor: Any, limit: Optional[int] = ...) -> list[tuple[Any, ...]]: ...
    @classmethod
    def df_to_sql(
        cls,
        database: Database,
        table: Table,
        df: pd.DataFrame,
        to_sql_kwargs: Mapping[str, Any],
    ) -> None: ...
    @classmethod
    def convert_dttm(
        cls, target_type: str, dttm: datetime, db_extra: Optional[dict[str, Any]] = ...
    ) -> Optional[str]: ...
    @classmethod
    def adjust_engine_params(
        cls,
        uri: URL,
        connect_args: dict[str, Any],
        catalog: Optional[str] = ...,
        schema: Optional[str] = ...,
    ) -> tuple[URL, dict[str, Any]]: ...
    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: dict[str, Any]) -> str: ...
    @classmethod
    def _extract_error_message(cls, ex: BaseException) -> str: ...
    @classmethod
    def progress(cls, log_lines: Iterable[str]) -> int: ...
    @classmethod
    def get_tracking_url_from_logs(cls, log_lines: Iterable[str]) -> Optional[str]: ...
    @classmethod
    def handle_cursor(cls, cursor: Any, query: Query) -> None: ...
    @classmethod
    def get_columns(
        cls, inspector: Inspector, table: Table, options: Optional[dict[str, Any]] = ...
    ) -> list[ResultSetColumnType]: ...
    @classmethod
    def where_latest_partition(
        cls,
        database: Database,
        table: Table,
        query: Select,
        columns: Optional[list[dict[str, Any]]] = ...,
    ) -> Optional[Select]: ...
    @classmethod
    def _get_fields(cls, cols: list[ResultSetColumnType]) -> list[ColumnClause]: ...
    @classmethod
    def latest_sub_partition(cls, database: Database, table: Table, **kwargs: Any) -> None: ...
    @classmethod
    def _latest_partition_from_df(cls, df: pd.DataFrame) -> Optional[list[str]]: ...
    @classmethod
    def _partition_query(
        cls,
        table: Table,
        indexes: Any,
        database: Database,
        limit: int = ...,
        order_by: Optional[str] = ...,
        filters: Optional[list[Any]] = ...,
    ) -> str: ...
    @classmethod
    def select_star(
        cls,
        database: Database,
        table: Table,
        engine: Engine,
        limit: int = ...,
        show_cols: bool = ...,
        indent: bool = ...,
        latest_partition: bool = ...,
        cols: Optional[list[ResultSetColumnType]] = ...,
    ) -> str: ...
    @classmethod
    def get_url_for_impersonation(
        cls, url: URL, impersonate_user: bool, username: Optional[str], access_token: Optional[str]
    ) -> URL: ...
    @classmethod
    def update_impersonation_config(
        cls,
        database: Database,
        connect_args: dict[str, Any],
        uri: str,
        username: Optional[str],
        access_token: Optional[str],
    ) -> None: ...
    @staticmethod
    def execute(cursor: Any, query: str, database: Database, async_: bool = ...) -> None: ...
    @classmethod
    def get_function_names(cls, database: Database) -> list[str]: ...
    @classmethod
    def has_implicit_cancel(cls) -> bool: ...
    @classmethod
    def get_view_names(
        cls, database: Database, inspector: Inspector, schema: Optional[str]
    ) -> set[str]: ...