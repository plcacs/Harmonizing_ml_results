from __future__ import annotations
import logging
import os
import re
import tempfile
import time
from datetime import datetime
from typing import Any, TYPE_CHECKING
from urllib import parse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from flask import current_app, g
from sqlalchemy import Column, text, types
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.expression import ColumnClause, Select
from superset import db
from superset.common.db_query_status import QueryStatus
from superset.constants import TimeGrain
from superset.databases.utils import make_url_safe
from superset.db_engine_specs.base import BaseEngineSpec
from superset.db_engine_specs.presto import PrestoEngineSpec
from superset.exceptions import SupersetException
from superset.extensions import cache_manager
from superset.models.sql_lab import Query
from superset.sql_parse import Table
from superset.superset_typing import ResultSetColumnType
if TYPE_CHECKING:
    from superset.models.core import Database
logger: logging.Logger = logging.getLogger(__name__)

def upload_to_s3(filename: str, upload_prefix: str, table: Table) -> str:
    ...

class HiveEngineSpec(PrestoEngineSpec):
    engine: str = 'hive'
    engine_name: str = 'Apache Hive'
    max_column_name_length: int = 767
    allows_alias_to_source_column: bool = True
    allows_hidden_orderby_agg: bool = False
    supports_dynamic_schema: bool = True
    _show_functions_column: str = 'tab_name'
    _time_grain_expressions: dict[TimeGrain, str] = {None: '{col}', TimeGrain.SECOND: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd HH:mm:ss')", ...}

    @classmethod
    def patch(cls) -> None:
        ...

    @classmethod
    def fetch_data(cls, cursor: Any, limit: int = None) -> list:
        ...

    @classmethod
    def df_to_sql(cls, database: Database, table: Table, df: pd.DataFrame, to_sql_kwargs: dict) -> None:
        ...

    @classmethod
    def convert_dttm(cls, target_type: Any, dttm: datetime, db_extra: Any = None) -> str:
        ...

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: dict, catalog: str = None, schema: str = None) -> tuple:
        ...

    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: dict) -> str:
        ...

    @classmethod
    def _extract_error_message(cls, ex: Exception) -> str:
        ...

    @classmethod
    def progress(cls, log_lines: list[str]) -> int:
        ...

    @classmethod
    def get_tracking_url_from_logs(cls, log_lines: list[str]) -> str:
        ...

    @classmethod
    def handle_cursor(cls, cursor: Any, query: Query) -> None:
        ...

    @classmethod
    def get_columns(cls, inspector: Inspector, table: Table, options: Any = None) -> list[Column]:
        ...

    @classmethod
    def where_latest_partition(cls, database: Database, table: Table, query: Select, columns: list[Column] = None) -> Select:
        ...

    @classmethod
    def _get_fields(cls, cols: list[Column]) -> list[str]:
        ...

    @classmethod
    def _latest_partition_from_df(cls, df: pd.DataFrame) -> list[str]:
        ...

    @classmethod
    def _partition_query(cls, table: Table, indexes: Any, database: Database, limit: int = 0, order_by: Any = None, filters: Any = None) -> str:
        ...

    @classmethod
    def select_star(cls, database: Database, table: Table, engine: Engine, limit: int = 100, show_cols: bool = False, indent: bool = True, latest_partition: bool = True, cols: Any = None) -> str:
        ...

    @classmethod
    def get_url_for_impersonation(cls, url: URL, impersonate_user: bool, username: str, access_token: str) -> URL:
        ...

    @classmethod
    def update_impersonation_config(cls, database: Database, connect_args: dict, uri: str, username: str, access_token: str) -> None:
        ...

    @staticmethod
    def execute(cursor: Any, query: str, database: Database, async_: bool = False) -> None:
        ...

    @classmethod
    def get_function_names(cls, database: Database) -> list[str]:
        ...

    @classmethod
    def has_implicit_cancel(cls) -> bool:
        ...

    @classmethod
    def get_view_names(cls, database: Database, inspector: Inspector, schema: str) -> set[str]:
        ...
