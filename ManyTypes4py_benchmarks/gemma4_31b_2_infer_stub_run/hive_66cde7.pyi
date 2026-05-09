from __future__ import annotations
import re
from typing import Any, Optional, Union, Iterable, Sequence, Dict, List, Set
from datetime import datetime
import numpy as np
import pandas as pd
import pyarrow as pa
from sqlalchemy.engine import Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.expression import Select
from superset.constants import TimeGrain
from superset.sql_parse import Table
from superset.models.sql_lab import Query
from superset.db_engine_specs.presto import PrestoEngineSpec

def upload_to_s3(filename: str, upload_prefix: str, table: Table) -> str: ...

class HiveEngineSpec(PrestoEngineSpec):
    engine: str
    engine_name: str
    max_column_name_length: int
    allows_alias_to_source_column: bool
    allows_hidden_orderby_agg: bool
    supports_dynamic_schema: bool
    _show_functions_column: str
    _time_grain_expressions: Dict[Optional[TimeGrain], str]
    jobs_stats_r: re.Pattern[str]
    launching_job_r: re.Pattern[str]
    stage_progress_r: re.Pattern[str]

    @classmethod
    def patch(cls) -> None: ...

    @classmethod
    def fetch_data(cls, cursor: Any, limit: Optional[int] = ...) -> List[Any]: ...

    @classmethod
    def df_to_sql(cls, database: Any, table: Table, df: pd.DataFrame, to_sql_kwargs: Dict[str, Any]) -> None: ...

    @classmethod
    def convert_dttm(cls, target_type: Any, dttm: datetime, db_extra: Optional[Dict[str, Any]] = ...) -> Optional[str]: ...

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: Dict[str, Any], catalog: Optional[str] = ..., schema: Optional[str] = ...) -> tuple[URL, Dict[str, Any]]: ...

    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: Dict[str, Any]) -> str: ...

    @classmethod
    def _extract_error_message(cls, ex: Exception) -> str: ...

    @classmethod
    def progress(cls, log_lines: Iterable[str]) -> int: ...

    @classmethod
    def get_tracking_url_from_logs(cls, log_lines: Iterable[str]) -> Optional[str]: ...

    @classmethod
    def handle_cursor(cls, cursor: Any, query: Query) -> None: ...

    @classmethod
    def get_columns(cls, inspector: Inspector, table: str, options: Optional[Dict[str, Any]] = ...) -> List[Any]: ...

    @classmethod
    def where_latest_partition(cls, database: Any, table: Table, query: Select, columns: Optional[List[Dict[str, Any]]] = ...) -> Optional[Select]: ...

    @classmethod
    def _get_fields(cls, cols: List[Any]) -> List[Any]: ...

    @classmethod
    def latest_sub_partition(cls, database: Any, table: Table, **kwargs: Any) -> Any: ...

    @classmethod
    def _latest_partition_from_df(cls, df: pd.DataFrame) -> Optional[List[str]]: ...

    @classmethod
    def _partition_query(cls, table: Table, indexes: List[int], database: Any, limit: int = 0, order_by: Optional[str] = ..., filters: Optional[List[Any]] = ...) -> str: ...

    @classmethod
    def select_star(cls, database: Any, table: Table, engine: Engine, limit: int = 100, show_cols: bool = False, indent: bool = True, latest_partition: bool = True, cols: Optional[List[str]] = ...) -> str: ...

    @classmethod
    def get_url_for_impersonation(cls, url: URL, impersonate_user: bool, username: str, access_token: str) -> URL: ...

    @classmethod
    def update_impersonation_config(cls, database: Any, connect_args: Dict[str, Any], uri: str, username: Optional[str], access_token: str) -> None: ...

    @staticmethod
    def execute(cursor: Any, query: str, database: Any, async_: bool = False) -> None: ...

    @classmethod
    def get_function_names(cls, database: Any) -> List[str]: ...

    @classmethod
    def has_implicit_cancel(cls) -> bool: ...

    @classmethod
    def get_view_names(cls, database: Any, inspector: Inspector, schema: Optional[str]) -> Set[str]: ...