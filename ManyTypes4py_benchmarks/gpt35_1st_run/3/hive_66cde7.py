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
    """
    Upload the file to S3.

    :param filename: The file to upload
    :param upload_prefix: The S3 prefix
    :param table: The table that will be created
    :returns: The S3 location of the table
    """
    import boto3
    from boto3.s3.transfer import TransferConfig
    bucket_path: str = current_app.config['CSV_TO_HIVE_UPLOAD_S3_BUCKET']
    if not bucket_path:
        logger.info('No upload bucket specified')
        raise Exception('No upload bucket specified. You can specify one in the config file.')
    s3 = boto3.client('s3')
    location: str = os.path.join('s3a://', bucket_path, upload_prefix, table.table)
    s3.upload_file(filename, bucket_path, os.path.join(upload_prefix, table.table, os.path.basename(filename)), Config=TransferConfig(use_threads=False))
    return location

class HiveEngineSpec(PrestoEngineSpec):
    """Reuses PrestoEngineSpec functionality."""
    engine: str = 'hive'
    engine_name: str = 'Apache Hive'
    max_column_name_length: int = 767
    allows_alias_to_source_column: bool = True
    allows_hidden_orderby_agg: bool = False
    supports_dynamic_schema: bool = True
    _show_functions_column: str = 'tab_name'
    _time_grain_expressions: dict[TimeGrain, str] = {None: '{col}', TimeGrain.SECOND: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd HH:mm:ss')", TimeGrain.MINUTE: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd HH:mm:00')", TimeGrain.HOUR: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd HH:00:00')", TimeGrain.DAY: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd 00:00:00')", TimeGrain.WEEK: "date_format(date_sub({col}, CAST(7-from_unixtime(unix_timestamp({col}),'u') as int)), 'yyyy-MM-dd 00:00:00')", TimeGrain.MONTH: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-01 00:00:00')", TimeGrain.QUARTER: "date_format(add_months(trunc({col}, 'MM'), -(month({col})-1)%3), 'yyyy-MM-dd 00:00:00')", TimeGrain.YEAR: "from_unixtime(unix_timestamp({col}), 'yyyy-01-01 00:00:00')", TimeGrain.WEEK_ENDING_SATURDAY: "date_format(date_add({col}, INT(6-from_unixtime(unix_timestamp({col}), 'u'))), 'yyyy-MM-dd 00:00:00')", TimeGrain.WEEK_STARTING_SUNDAY: "date_format(date_add({col}, -INT(from_unixtime(unix_timestamp({col}), 'u'))), 'yyyy-MM-dd 00:00:00')"}
    jobs_stats_r: re.Pattern = re.compile('.*INFO.*Total jobs = (?P<max_jobs>[0-9]+)')
    launching_job_r: re.Pattern = re.compile('.*INFO.*Launching Job (?P<job_number>[0-9]+) out of (?P<max_jobs>[0-9]+)')
    stage_progress_r: re.Pattern = re.compile('.*INFO.*Stage-(?P<stage_number>[0-9]+).*map = (?P<map_progress>[0-9]+)%.*reduce = (?P<reduce_progress>[0-9]+)%.*')

    @classmethod
    def patch(cls) -> None:
        from pyhive import hive
        from TCLIService import constants as patched_constants, TCLIService as patched_TCLIService, ttypes as patched_ttypes
        hive.TCLIService = patched_TCLIService
        hive.constants = patched_constants
        hive.ttypes = patched_ttypes

    @classmethod
    def fetch_data(cls, cursor, limit: int = None) -> list:
        import pyhive
        from TCLIService import ttypes
        state = cursor.poll()
        if state.operationState == ttypes.TOperationState.ERROR_STATE:
            raise Exception('Query error', state.errorMessage)
        try:
            return super().fetch_data(cursor, limit)
        except pyhive.exc.ProgrammingError:
            return []

    @classmethod
    def df_to_sql(cls, database: Database, table: Table, df: pd.DataFrame, to_sql_kwargs: dict) -> None:
        ...

    @classmethod
    def convert_dttm(cls, target_type: Any, dttm: datetime, db_extra: Any = None) -> str:
        ...

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: dict, catalog: str = None, schema: str = None) -> tuple[URL, dict]:
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
    def get_columns(cls, inspector: Inspector, table: Table, options: Any = None) -> Any:
        ...

    @classmethod
    def where_latest_partition(cls, database: Database, table: Table, query: Select, columns: Any = None) -> Any:
        ...

    @classmethod
    def _get_fields(cls, cols: Any) -> Any:
        ...

    @classmethod
    def latest_sub_partition(cls, database: Database, table: Table, **kwargs: Any) -> None:
        ...

    @classmethod
    def _latest_partition_from_df(cls, df: pd.DataFrame) -> Any:
        ...

    @classmethod
    def _partition_query(cls, table: Table, indexes: Any, database: Database, limit: int = 0, order_by: Any = None, filters: Any = None) -> str:
        ...

    @classmethod
    def select_star(cls, database: Database, table: Table, engine: Engine, limit: int = 100, show_cols: bool = False, indent: bool = True, latest_partition: bool = True, cols: Any = None) -> Any:
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
    @cache_manager.cache.memoize()
    def get_function_names(cls, database: Database) -> list[str]:
        ...

    @classmethod
    def has_implicit_cancel(cls) -> bool:
        ...

    @classmethod
    def get_view_names(cls, database: Database, inspector: Inspector, schema: str) -> set[str]:
        ...
