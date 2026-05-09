from __future__ import annotations
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from flask import current_app
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.expression import ColumnClause
from superset.models.sql_lab import Query
from superset.superset_typing import ResultSetColumnType

logger: logging.Logger = ...

def upload_to_s3(filename: str, upload_prefix: str, table: Any) -> str:
    ...

class HiveEngineSpec(PrestoEngineSpec):
    engine: str = ...
    engine_name: str = ...
    max_column_name_length: int = ...
    allows_alias_to_source_column: bool = ...
    allows_hidden_orderby_agg: bool = ...
    supports_dynamic_schema: bool = ...
    _show_functions_column: str = ...
    _time_grain_expressions: Dict[Optional[TimeGrain], str] = ...
    jobs_stats_r: re.Pattern = ...
    launching_job_r: re.Pattern = ...
    stage_progress_r: re.Pattern = ...

    @classmethod
    def patch(cls) -> None:
        ...

    @classmethod
    def fetch_data(cls, cursor: Any, limit: Optional[int] = None) -> list[tuple[Any, ...]]:
        ...

    @classmethod
    def df_to_sql(cls, database: Any, table: Any, df: pd.DataFrame, to_sql_kwargs: Dict[str, Any]) -> None:
        ...

    @classmethod
    def convert_dttm(cls, target_type: Any, dttm: datetime, db_extra: Optional[Any] = None) -> Optional[str]:
        ...

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: Dict[str, Any], catalog: Optional[str] = None, schema: Optional[str] = None) -> Tuple[URL, Dict[str, Any]]:
        ...

    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: Dict[str, Any]) -> str:
        ...

    @classmethod
    def _extract_error_message(cls, ex: Exception) -> str:
        ...

    @classmethod
    def progress(cls, log_lines: List[str]) -> int:
        ...

    @classmethod
    def get_tracking_url_from_logs(cls, log_lines: List[str]) -> Optional[str]:
        ...

    @classmethod
    def handle_cursor(cls, cursor: Any, query: Query) -> None:
        ...

    @classmethod
    def get_columns(cls, inspector: Any, table: Any, options: Optional[Any] = None) -> List[Dict[str, Any]]:
        ...

    @classmethod
    def where_latest_partition(cls, database: Any, table: Any, query: Any, columns: Optional[List[Dict[str, Any]]] = None) -> Optional[Any]:
        ...

    @classmethod
    def _get_fields(cls, cols: Any) -> Any:
        ...

    @classmethod
    def latest_sub_partition(cls, database: Any, table: Any, **kwargs: Any) -> None:
        ...

    @classmethod
    def _latest_partition_from_df(cls, df: pd.DataFrame) -> Optional[List[str]]:
        ...

    @classmethod
    def _partition_query(cls, table: Any, indexes: Any, database: Any, limit: int = 0, order_by: Optional[Any] = None, filters: Optional[Any] = None) -> str:
        ...

    @classmethod
    def select_star(cls, database: Any, table: Any, engine: Any, limit: int = 100, show_cols: bool = False, indent: bool = True, latest_partition: bool = True, cols: Optional[Any] = None) -> str:
        ...

    @classmethod
    def get_url_for_impersonation(cls, url: URL, impersonate_user: bool, username: str, access_token: Optional[str] = None) -> URL:
        ...

    @classmethod
    def update_impersonation_config(cls, database: Any, connect_args: Dict[str, Any], uri: str, username: str, access_token: Optional[str] = None) -> None:
        ...

    @staticmethod
    def execute(cursor: Any, query: str, database: Any, async_: bool = False) -> None:
        ...

    @classmethod
    @cache_manager.cache.memoize()
    def get_function_names(cls, database: Any) -> List[str]:
        ...

    @classmethod
    def has_implicit_cancel(cls) -> bool:
        ...

    @classmethod
    def get_view_names(cls, database: Any, inspector: Any, schema: Optional[str] = None) -> Set[str]:
        ...