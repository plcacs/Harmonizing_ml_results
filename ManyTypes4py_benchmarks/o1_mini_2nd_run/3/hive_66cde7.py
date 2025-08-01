from __future__ import annotations
import logging
import os
import re
import tempfile
import time
from datetime import datetime
from typing import Any, TYPE_CHECKING, Dict, Optional, Set, Tuple, List
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
logger = logging.getLogger(__name__)

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
    bucket_path = current_app.config['CSV_TO_HIVE_UPLOAD_S3_BUCKET']
    if not bucket_path:
        logger.info('No upload bucket specified')
        raise Exception('No upload bucket specified. You can specify one in the config file.')
    s3 = boto3.client('s3')
    location = os.path.join('s3a://', bucket_path, upload_prefix, table.table)
    s3.upload_file(
        filename, 
        bucket_path, 
        os.path.join(upload_prefix, table.table, os.path.basename(filename)),
        Config=TransferConfig(use_threads=False)
    )
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
    _time_grain_expressions: Dict[Optional[TimeGrain], str] = {
        None: '{col}',
        TimeGrain.SECOND: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd HH:mm:ss')",
        TimeGrain.MINUTE: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd HH:mm:00')",
        TimeGrain.HOUR: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd HH:00:00')",
        TimeGrain.DAY: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd 00:00:00')",
        TimeGrain.WEEK: "date_format(date_sub({col}, CAST(7-from_unixtime(unix_timestamp({col}),'u') as int)), 'yyyy-MM-dd 00:00:00')",
        TimeGrain.MONTH: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-01 00:00:00')",
        TimeGrain.QUARTER: "date_format(add_months(trunc({col}, 'MM'), -(month({col})-1)%3), 'yyyy-MM-dd 00:00:00')",
        TimeGrain.YEAR: "from_unixtime(unix_timestamp({col}), 'yyyy-01-01 00:00:00')",
        TimeGrain.WEEK_ENDING_SATURDAY: "date_format(date_add({col}, INT(6-from_unixtime(unix_timestamp({col}), 'u'))), 'yyyy-MM-dd 00:00:00')",
        TimeGrain.WEEK_STARTING_SUNDAY: "date_format(date_add({col}, -INT(from_unixtime(unix_timestamp({col}), 'u'))), 'yyyy-MM-dd 00:00:00')"
    }
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
    def fetch_data(cls, cursor: Any, limit: Optional[int] = None) -> Any:
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
    def df_to_sql(cls, database: Database, table: Table, df: pd.DataFrame, to_sql_kwargs: Dict[str, Any]) -> None:
        """
        Upload data from a Pandas DataFrame to a database.

        The data is stored via the binary Parquet format which is both less problematic
        and more performant than a text file. More specifically storing a table as a
        CSV text file has severe limitations including the fact that the Hive CSV SerDe
        does not support multiline fields.

        Note this method does not create metadata for the table.

        :param database: The database to upload the data to
        :param: table The table to upload the data to
        :param df: The dataframe with data to be uploaded
        :param to_sql_kwargs: The kwargs to be passed to pandas.DataFrame.to_sql` method
        """
        if to_sql_kwargs['if_exists'] == 'append':
            raise SupersetException('Append operation not currently supported')
        if to_sql_kwargs['if_exists'] == 'fail':
            if table.schema:
                table_exists = not database.get_df(f"SHOW TABLES IN {table.schema} LIKE '{table.table}'").empty
            else:
                table_exists = not database.get_df(f"SHOW TABLES LIKE '{table.table}'").empty
            if table_exists:
                raise SupersetException('Table already exists')
        elif to_sql_kwargs['if_exists'] == 'replace':
            with cls.get_engine(database, catalog=table.catalog, schema=table.schema) as engine:
                engine.execute(f'DROP TABLE IF EXISTS {str(table)}')

        def _get_hive_type(dtype: np.dtype) -> str:
            hive_type_by_dtype: Dict[np.dtype, str] = {
                np.dtype('bool'): 'BOOLEAN',
                np.dtype('float64'): 'DOUBLE',
                np.dtype('int64'): 'BIGINT',
                np.dtype('object'): 'STRING'
            }
            return hive_type_by_dtype.get(dtype, 'STRING')
        schema_definition: str = ', '.join((f'`{name}` {_get_hive_type(dtype)}' for name, dtype in df.dtypes.items()))
        with tempfile.NamedTemporaryFile(dir=current_app.config['UPLOAD_FOLDER'], suffix='.parquet') as file:
            pq.write_table(pa.Table.from_pandas(df), where=file.name)
            with cls.get_engine(database, catalog=table.catalog, schema=table.schema) as engine:
                engine.execute(
                    text(f'''
                        CREATE TABLE {str(table)} ({schema_definition})
                        STORED AS PARQUET
                        LOCATION :location
                    '''),
                    location=upload_to_s3(
                        filename=file.name,
                        upload_prefix=current_app.config['CSV_TO_HIVE_UPLOAD_DIRECTORY_FUNC'](database, g.user, table.schema),
                        table=table
                    )
                )

    @classmethod
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: Optional[Any] = None) -> Optional[str]:
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, types.Date):
            return f"CAST('{dttm.date().isoformat()}' AS DATE)"
        if isinstance(sqla_type, types.TIMESTAMP):
            return f"CAST('{dttm.isoformat(sep=' ', timespec='microseconds')}' AS TIMESTAMP)"
        return None

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: Dict[str, Any], catalog: Optional[str] = None, schema: Optional[str] = None) -> Tuple[URL, Dict[str, Any]]:
        if schema:
            uri = uri.set(database=parse.quote(schema, safe=''))
        return (uri, connect_args)

    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: Dict[str, Any]) -> str:
        """
        Return the configured schema.
        """
        return parse.unquote(sqlalchemy_uri.database)

    @classmethod
    def _extract_error_message(cls, ex: Exception) -> str:
        msg: str = str(ex)
        match: Optional[re.Match[str]] = re.search('errorMessage="(.*?)(?<!\\\\)"', msg)
        if match:
            msg = match.group(1)
        return msg

    @classmethod
    def progress(cls, log_lines: List[str]) -> int:
        total_jobs: int = 1
        current_job: int = 1
        stages: Dict[int, float] = {}
        for line in log_lines:
            match = cls.jobs_stats_r.match(line)
            if match:
                total_jobs = int(match.groupdict()['max_jobs']) or 1
            match = cls.launching_job_r.match(line)
            if match:
                current_job = int(match.groupdict()['job_number'])
                total_jobs = int(match.groupdict()['max_jobs']) or 1
                stages = {}
            match = cls.stage_progress_r.match(line)
            if match:
                stage_number: int = int(match.groupdict()['stage_number'])
                map_progress: int = int(match.groupdict()['map_progress'])
                reduce_progress: int = int(match.groupdict()['reduce_progress'])
                stages[stage_number] = (map_progress + reduce_progress) / 2
        logger.info('Progress detail: %s, current job %s, total jobs: %s', stages, current_job, total_jobs)
        stage_progress: float = sum(stages.values()) / len(stages.values()) if stages else 0
        progress: int = int(100 * (current_job - 1) / total_jobs + stage_progress / total_jobs)
        return progress

    @classmethod
    def get_tracking_url_from_logs(cls, log_lines: List[str]) -> Optional[str]:
        lkp: str = 'Tracking URL = '
        for line in log_lines:
            if lkp in line:
                return line.split(lkp)[1]
        return None

    @classmethod
    def handle_cursor(cls, cursor: Any, query: Query) -> None:
        """Updates progress information"""
        from pyhive import hive
        unfinished_states: Tuple[Any, ...] = (
            hive.ttypes.TOperationState.INITIALIZED_STATE,
            hive.ttypes.TOperationState.RUNNING_STATE
        )
        polled = cursor.poll()
        last_log_line: int = 0
        tracking_url: Optional[str] = None
        job_id: Optional[str] = None
        query_id: int = query.id
        while polled.operationState in unfinished_states:
            db.session.refresh(query)
            query = db.session.query(type(query)).filter_by(id=query_id).one()
            if query.status == QueryStatus.STOPPED:
                cursor.cancel()
                break
            try:
                logs: Optional[List[str]] = cursor.fetch_logs()
                log: str = '\n'.join(logs) if logs else ''
            except Exception:
                logger.warning('Call to GetLog() failed')
                log = ''
            if log:
                log_lines: List[str] = log.splitlines()
                progress: int = cls.progress(log_lines)
                logger.info('Query %s: Progress total: %s', str(query_id), str(progress))
                needs_commit: bool = False
                if progress > query.progress:
                    query.progress = progress
                    needs_commit = True
                if not tracking_url:
                    tracking_url = cls.get_tracking_url_from_logs(log_lines)
                    if tracking_url:
                        job_id = tracking_url.split('/')[-2]
                        logger.info('Query %s: Found the tracking url: %s', str(query_id), tracking_url)
                        query.tracking_url = tracking_url
                        logger.info('Query %s: Job id: %s', str(query_id), str(job_id))
                        needs_commit = True
                if job_id and len(log_lines) > last_log_line:
                    for l in log_lines[last_log_line:]:
                        logger.info('Query %s: [%s] %s', str(query_id), str(job_id), l)
                    last_log_line = len(log_lines)
                if needs_commit:
                    db.session.commit()
            if (sleep_interval := current_app.config.get('HIVE_POLL_INTERVAL')):
                logger.warning('HIVE_POLL_INTERVAL is deprecated and will be removed in 3.0. Please use DB_POLL_INTERVAL_SECONDS instead')
            else:
                sleep_interval = current_app.config['DB_POLL_INTERVAL_SECONDS'].get(cls.engine, 5)
            time.sleep(sleep_interval)
            polled = cursor.poll()

    @classmethod
    def get_columns(cls, inspector: Inspector, table: Table, options: Optional[Any] = None) -> List[ResultSetColumnType]:
        return BaseEngineSpec.get_columns(inspector, table, options)

    @classmethod
    def where_latest_partition(
        cls, 
        database: Database, 
        table: Table, 
        query: Select, 
        columns: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Select]:
        try:
            col_names, values = cls.latest_partition(database, table, show_first=True)
        except Exception:
            return None
        if values is not None and columns is not None:
            for col_name, value in zip(col_names, values, strict=False):
                for clm in columns:
                    if clm.get('name') == col_name:
                        query = query.where(Column(col_name) == value)
            return query
        return None

    @classmethod
    def _get_fields(cls, cols: Any) -> Any:
        return BaseEngineSpec._get_fields(cols)

    @classmethod
    def latest_sub_partition(cls, database: Database, table: Table, **kwargs: Any) -> None:
        pass

    @classmethod
    def _latest_partition_from_df(cls, df: pd.DataFrame) -> Optional[List[str]]:
        """Hive partitions look like ds={partition name}/ds={partition name}"""
        if not df.empty:
            return [partition_str.split('=')[1] for partition_str in df.iloc[:, 0].max().split('/')]
        return None

    @classmethod
    def _partition_query(cls, table: Table, indexes: Any, database: Database, limit: int = 0, order_by: Optional[Any] = None, filters: Optional[Any] = None) -> str:
        full_table_name: str = f'{table.schema}.{table.table}' if table.schema else table.table
        return f'SHOW PARTITIONS {full_table_name}'

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
        cols: Optional[Any] = None
    ) -> Select:
        return super(PrestoEngineSpec, cls).select_star(database, table, engine, limit, show_cols, indent, latest_partition, cols)

    @classmethod
    def get_url_for_impersonation(cls, url: URL, impersonate_user: bool, username: Optional[str], access_token: Optional[str]) -> URL:
        """
        Return a modified URL with the username set.

        :param url: SQLAlchemy URL object
        :param impersonate_user: Flag indicating if impersonation is enabled
        :param username: Effective username
        """
        return url

    @classmethod
    def update_impersonation_config(
        cls, 
        database: Database, 
        connect_args: Dict[str, Any], 
        uri: str, 
        username: Optional[str], 
        access_token: Optional[str]
    ) -> None:
        """
        Update a configuration dictionary
        that can set the correct properties for impersonating users
        :param database: the Database Object
        :param connect_args:
        :param uri: URI string
        :param impersonate_user: Flag indicating if impersonation is enabled
        :param username: Effective username
        :return: None
        """
        url: URL = make_url_safe(uri)
        backend_name: str = url.get_backend_name()
        if backend_name == 'hive' and username is not None:
            configuration: Dict[str, Any] = connect_args.get('configuration', {})
            configuration['hive.server2.proxy.user'] = username
            connect_args['configuration'] = configuration

    @staticmethod
    def execute(cursor: Any, query: str, database: Database, async_: bool = False) -> None:
        kwargs: Dict[str, Any] = {'async': async_}
        cursor.execute(query, **kwargs)

    @classmethod
    @cache_manager.cache.memoize()
    def get_function_names(cls, database: Database) -> List[str]:
        """
        Get a list of function names that are able to be called on the database.
        Used for SQL Lab autocomplete.

        :param database: The database to get functions for
        :return: A list of function names useable in the database
        """
        df: pd.DataFrame = database.get_df('SHOW FUNCTIONS')
        if cls._show_functions_column in df:
            return df[cls._show_functions_column].tolist()
        columns: List[str] = df.columns.values.tolist()
        logger.error(
            'Payload from `SHOW FUNCTIONS` has the incorrect format. Expected column `%s`, found: %s.',
            cls._show_functions_column,
            ', '.join(columns),
            exc_info=True
        )
        if len(columns) == 1:
            return df[columns[0]].tolist()
        return []

    @classmethod
    def has_implicit_cancel(cls) -> bool:
        """
        Return True if the live cursor handles the implicit cancelation of the query,
        False otherwise.

        :return: Whether the live cursor implicitly cancels the query
        :see: handle_cursor
        """
        return True

    @classmethod
    def get_view_names(cls, database: Database, inspector: Inspector, schema: Optional[str]) -> Set[str]:
        """
        Get all the view names within the specified schema.

        Per the SQLAlchemy definition if the schema is omitted the database’s default
        schema is used, however some dialects infer the request as schema agnostic.

        Note that PyHive's Hive SQLAlchemy dialect does not adhere to the specification
        where the `get_view_names` method returns both real tables and views. Furthermore
        the dialect wrongfully infers the request as schema agnostic when the schema is
        omitted.

        :param database: The database to inspect
        :param inspector: The SQLAlchemy inspector
        :param schema: The schema to inspect
        :returns: The view names
        """
        sql: str = 'SHOW VIEWS'
        if schema:
            sql += f' IN `{schema}`'
        with database.get_raw_connection(schema=schema) as conn:
            cursor: Any = conn.cursor()
            cursor.execute(sql)
            results: List[Tuple[str, ...]] = cursor.fetchall()
            return {row[0] for row in results}
