from __future__ import annotations
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Pattern
import numpy as np
import pandas as pd
from flask import Flask
from sqlalchemy import types
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.expression import ColumnClause, Select
from superset.common.db_query_status import QueryStatus
from superset.constants import TimeGrain
from superset.db_engine_specs.base import BaseEngineSpec
from superset.db_engine_specs.presto import PrestoEngineSpec
from superset.exceptions import SupersetException
from superset.extensions import cache_manager
from superset.models.sql_lab import Query
from superset.sql_parse import Table
from superset.superset_typing import ResultSetColumnType
from superset.models.core import Database

logger: logging.Logger

def upload_to_s3(filename: str, upload_prefix: str, table: Table) -> str: ...

class HiveEngineSpec(PrestoEngineSpec):
    engine: str = 'hive'
    engine_name: str = 'Apache Hive'
    max_column_name_length: int = 767
    allows_alias_to_source_column: bool = True
    allows_hidden_orderby_agg: bool = False
    supports_dynamic_schema: bool = True
    _show_functions_column: str = 'tab_name'
    _time_grain_expressions: Dict[Optional[TimeGrain], str]
    jobs_stats_r: Pattern[str]
    launching_job_r: Pattern[str]
    stage_progress_r: Pattern[str]

    @classmethod
    def patch(cls) -> None: ...

    @classmethod
    def fetch_data(cls, cursor: Any, limit: Optional[int] = ...) -> List[Tuple[Any, ...]]: ...

    @classmethod
    def df_to_sql(cls, database: Database, table: Table, df: pd.DataFrame, to_sql_kwargs: Dict[str, Any]) -> None: ...

    @classmethod
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: Optional[Dict[str, Any]] = ...) -> Optional[str]: ...

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: Dict[str, Any], catalog: Optional[str] = ..., schema: Optional[str] = ...) -> Tuple[URL, Dict[str, Any]]: ...

    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: Dict[str, Any]) -> str: ...

    @classmethod
    def _extract_error_message(cls, ex: Exception) -> str: ...

    @classmethod
    def progress(cls, log_lines: List[str]) -> int: ...

    @classmethod
    def get_tracking_url_from_logs(cls, log_lines: List[str]) -> Optional[str]: ...

    @classmethod
    def handle_cursor(cls, cursor: Any, query: Query) -> None: ...

    @classmethod
    def get_columns(cls, inspector: Inspector, table: Table, options: Optional[Dict[str, Any]] = ...) -> List[ResultSetColumnType]: ...

    @classmethod
    def where_latest_partition(cls, database: Database, table: Table, query: Select, columns: Optional[List[Dict[str, Any]]] = ...) -> Optional[Select]: ...

    @classmethod
    def _get_fields(cls, cols: List[ResultSetColumnType]) -> List[ColumnClause]: ...

    @classmethod
    def latest_sub_partition(cls, database: Database, table: Table, **kwargs: Any) -> None: ...

    @classmethod
    def _latest_partition_from_df(cls, df: pd.DataFrame) -> Optional[List[str]]: ...

    @classmethod
    def _partition_query(cls, table: Table, indexes: List[str], database: Database, limit: int = ..., order_by: Optional[List[Tuple[str, bool]]] = ..., filters: Optional[Dict[str, Any]] = ...) -> str: ...

    @classmethod
    def select_star(cls, database: Database, table: Table, engine: Engine, limit: int = ..., show_cols: bool = ..., indent: bool = ..., latest_partition: bool = ..., cols: Optional[List[ResultSetColumnType]] = ...) -> str: ...

    @classmethod
    def get_url_for_impersonation(cls, url: URL, impersonate_user: bool, username: Optional[str], access_token: Optional[str]) -> URL: ...

    @classmethod
    def update_impersonation_config(cls, database: Database, connect_args: Dict[str, Any], uri: str, username: Optional[str], access_token: Optional[str]) -> None: ...

    @staticmethod
    def execute(cursor: Any, query: str, database: Database, async_: bool = ...) -> None: ...

    @classmethod
    @cache_manager.cache.memoize()
    def get_function_names(cls, database: Database) -> List[str]: ...

    @classmethod
    def has_implicit_cancel(cls) -> bool: ...

    @classmethod
    def get_view_names(cls, database: Database, inspector: Inspector, schema: Optional[str]) -> List[str]: ...