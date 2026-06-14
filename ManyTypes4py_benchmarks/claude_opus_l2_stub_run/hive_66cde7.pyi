from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, TYPE_CHECKING

import pandas as pd
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.expression import ColumnClause, Select

from superset.db_engine_specs.presto import PrestoEngineSpec
from superset.models.sql_lab import Query
from superset.sql_parse import Table
from superset.superset_typing import ResultSetColumnType

if TYPE_CHECKING:
    from superset.models.core import Database

logger: logging.Logger

def upload_to_s3(filename: str, upload_prefix: str, table: Table) -> str: ...

class HiveEngineSpec(PrestoEngineSpec):
    engine: str
    engine_name: str
    max_column_name_length: int
    allows_alias_to_source_column: bool
    allows_hidden_orderby_agg: bool
    supports_dynamic_schema: bool
    _show_functions_column: str
    _time_grain_expressions: dict[str | None, str]
    jobs_stats_r: re.Pattern[str]
    launching_job_r: re.Pattern[str]
    stage_progress_r: re.Pattern[str]

    @classmethod
    def patch(cls) -> None: ...

    @classmethod
    def fetch_data(cls, cursor: Any, limit: int | None = None) -> list[Any]: ...

    @classmethod
    def df_to_sql(
        cls,
        database: Database,
        table: Table,
        df: pd.DataFrame,
        to_sql_kwargs: dict[str, Any],
    ) -> None: ...

    @classmethod
    def convert_dttm(
        cls,
        target_type: str,
        dttm: datetime,
        db_extra: dict[str, Any] | None = None,
    ) -> str | None: ...

    @classmethod
    def adjust_engine_params(
        cls,
        uri: URL,
        connect_args: dict[str, Any],
        catalog: str | None = None,
        schema: str | None = None,
    ) -> tuple[URL, dict[str, Any]]: ...

    @classmethod
    def get_schema_from_engine_params(
        cls,
        sqlalchemy_uri: URL,
        connect_args: dict[str, Any],
    ) -> str | None: ...

    @classmethod
    def _extract_error_message(cls, ex: Exception) -> str: ...

    @classmethod
    def progress(cls, log_lines: list[str]) -> int: ...

    @classmethod
    def get_tracking_url_from_logs(cls, log_lines: list[str]) -> str | None: ...

    @classmethod
    def handle_cursor(cls, cursor: Any, query: Query) -> None: ...

    @classmethod
    def get_columns(
        cls,
        inspector: Inspector,
        table: Table,
        options: dict[str, Any] | None = None,
    ) -> list[ResultSetColumnType]: ...

    @classmethod
    def where_latest_partition(
        cls,
        database: Database,
        table: Table,
        query: Select,
        columns: list[ResultSetColumnType] | None = None,
    ) -> Select | None: ...

    @classmethod
    def _get_fields(cls, cols: list[ResultSetColumnType]) -> list[ColumnClause]: ...

    @classmethod
    def latest_sub_partition(
        cls, database: Database, table: Table, **kwargs: Any
    ) -> None: ...

    @classmethod
    def _latest_partition_from_df(cls, df: pd.DataFrame) -> list[str] | None: ...

    @classmethod
    def _partition_query(
        cls,
        table: Table,
        indexes: list[dict[str, Any]],
        database: Database,
        limit: int = 0,
        order_by: list[tuple[str, bool]] | None = None,
        filters: dict[Any, Any] | None = None,
    ) -> str: ...

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
        cols: list[ResultSetColumnType] | None = None,
    ) -> str: ...

    @classmethod
    def get_url_for_impersonation(
        cls,
        url: URL,
        impersonate_user: bool,
        username: str | None,
        access_token: str | None,
    ) -> URL: ...

    @classmethod
    def update_impersonation_config(
        cls,
        database: Database,
        connect_args: dict[str, Any],
        uri: str,
        username: str | None,
        access_token: str | None,
    ) -> None: ...

    @staticmethod
    def execute(
        cursor: Any,
        query: str,
        database: Database,
        async_: bool = False,
    ) -> None: ...

    @classmethod
    def get_function_names(cls, database: Database) -> list[str]: ...

    @classmethod
    def has_implicit_cancel(cls) -> bool: ...

    @classmethod
    def get_view_names(
        cls,
        database: Database,
        inspector: Inspector,
        schema: str | None,
    ) -> set[str]: ...