from __future__ import annotations
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.expression import ColumnClause, Select
from superset.models.core import Database
from superset.models.sql_lab import Query
from superset.superset_typing import ResultSetColumnType
import pyarrow
import pyhive
import pandas as pd

def upload_to_s3(filename: str, upload_prefix: str, table: Any) -> str: ...

class HiveEngineSpec(PrestoEngineSpec):
    engine: str = ...
    engine_name: str = ...
    max_column_name_length: int = ...
    allows_alias_to_source_column: bool = ...
    allows_hidden_orderby_agg: bool = ...
    supports_dynamic_schema: bool = ...
    _show_functions_column: str = ...
    _time_grain_expressions: Dict[str, str] = ...
    jobs_stats_r: re.Pattern = ...
    launching_job_r: re.Pattern = ...
    stage_progress_r: re.Pattern = ...

    @classmethod
    def patch(cls) -> None: ...

    @classmethod
    def fetch_data(cls, cursor: pyhive.hive.Cursor, limit: Optional[int] = None) -> list[tuple[Any, ...]]:
        ...

    @classmethod
    def df_to_sql(cls, database: Database, table: Table, df: pd.DataFrame, to_sql_kwargs: dict[str, Any]) -> None: ...

    @classmethod
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: Optional[dict[str, Any]] = None) -> Optional[str]:
        ...

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: dict[str, Any], catalog: Optional[str] = None, schema: Optional[str] = None) -> tuple[URL, dict[str, Any]]:
        ...

    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: dict[str, Any]) -> str: ...

    @classmethod
    def _extract_error_message(cls, ex: Exception) -> str: ...

    @classmethod
    def progress(cls, log_lines: List[str]) -> int: ...

    @classmethod
    def get_tracking_url_from_logs(cls, log_lines: List[str]) -> Optional[str]:
        ...

    @classmethod
    def handle_cursor(cls, cursor: pyhive.hive.Cursor, query: Query) -> None: ...

    @classmethod
    def get_columns(cls, inspector: Any, table: Any, options: Optional[dict[str, Any]] = None) -> List[ResultSetColumnType]:
        ...

    @classmethod
    def where_latest_partition(cls, database: Database, table: Table, query: Select, columns: Optional[List[dict[str, Any]]] = None) -> Optional[Select]:
        ...

    @classmethod
    def latest_sub_partition(cls, database: Database, table: Table, **kwargs: Any) -> None: ...

    @classmethod
    def _latest_partition_from_df(cls, df: pd.DataFrame) -> Optional[List[str]]:
        ...

    @classmethod
    def _partition_query(cls, table: Table, indexes: Any, database: Database, limit: int = 0, order_by: Optional[List[str]] = None, filters: Optional[List[str]] = None) -> str:
        ...

    @classmethod
    def select_star(cls, database: Database, table: Table, engine: Any, limit: int = 100, show_cols: bool = False, indent: bool = True, latest_partition: bool = True, cols: Optional[List[ColumnClause]] = None) -> str:
        ...

    @classmethod
    def get_url_for_impersonation(cls, url: URL, impersonate_user: bool, username: str, access_token: Optional[str] = None) -> URL:
        ...

    @classmethod
    def update_impersonation_config(cls, database: Database, connect_args: dict[str, Any], uri: str, username: str, access_token: Optional[str] = None) -> None:
        ...

    @staticmethod
    def execute(cursor: pyhive.hive.Cursor, query: str, database: Database, async_: bool = False) -> None:
        ...

    @classmethod
    @cache_manager.cache.memoize()
    def get_function_names(cls, database: Database) -> List[str]:
        ...

    @classmethod
    def has_implicit_cancel(cls) -> bool:
        ...

    @classmethod
    def get_view_names(cls, database: Database, inspector: Any, schema: Optional[str] = None) -> Set[str]:
        ...