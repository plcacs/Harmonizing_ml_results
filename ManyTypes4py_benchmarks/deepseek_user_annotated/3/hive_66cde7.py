# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from datetime import datetime
from typing import Any, TYPE_CHECKING, Optional, List, Set, Dict, Tuple, Union
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
    import boto3
    from boto3.s3.transfer import TransferConfig

    bucket_path: Optional[str] = current_app.config["CSV_TO_HIVE_UPLOAD_S3_BUCKET"]

    if not bucket_path:
        logger.info("No upload bucket specified")
        raise Exception("No upload bucket specified. You can specify one in the config file.")

    s3 = boto3.client("s3")
    location: str = os.path.join("s3a://", bucket_path, upload_prefix, table.table)
    s3.upload_file(
        filename,
        bucket_path,
        os.path.join(upload_prefix, table.table, os.path.basename(filename)),
        Config=TransferConfig(use_threads=False),
    )
    return location


class HiveEngineSpec(PrestoEngineSpec):
    engine: str = "hive"
    engine_name: str = "Apache Hive"
    max_column_name_length: int = 767
    allows_alias_to_source_column: bool = True
    allows_hidden_orderby_agg: bool = False
    supports_dynamic_schema: bool = True
    _show_functions_column: str = "tab_name"

    _time_grain_expressions: Dict[Optional[TimeGrain], str] = {
        None: "{col}",
        TimeGrain.SECOND: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd HH:mm:ss')",
        TimeGrain.MINUTE: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd HH:mm:00')",
        TimeGrain.HOUR: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd HH:00:00')",
        TimeGrain.DAY: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd 00:00:00')",
        TimeGrain.WEEK: "date_format(date_sub({col}, CAST(7-from_unixtime(unix_timestamp({col}),'u') as int)), 'yyyy-MM-dd 00:00:00')",
        TimeGrain.MONTH: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-01 00:00:00')",
        TimeGrain.QUARTER: "date_format(add_months(trunc({col}, 'MM'), -(month({col})-1)%3), 'yyyy-MM-dd 00:00:00')",
        TimeGrain.YEAR: "from_unixtime(unix_timestamp({col}), 'yyyy-01-01 00:00:00')",
        TimeGrain.WEEK_ENDING_SATURDAY: "date_format(date_add({col}, INT(6-from_unixtime(unix_timestamp({col}), 'u'))), 'yyyy-MM-dd 00:00:00')",
        TimeGrain.WEEK_STARTING_SUNDAY: "date_format(date_add({col}, -INT(from_unixtime(unix_timestamp({col}), 'u'))), 'yyyy-MM-dd 00:00:00')",
    }

    jobs_stats_r: re.Pattern = re.compile(r".*INFO.*Total jobs = (?P<max_jobs>[0-9]+)")
    launching_job_r: re.Pattern = re.compile(
        ".*INFO.*Launching Job (?P<job_number>[0-9]+) out of (?P<max_jobs>[0-9]+)"
    )
    stage_progress_r: re.Pattern = re.compile(
        r".*INFO.*Stage-(?P<stage_number>[0-9]+).*"
        r"map = (?P<map_progress>[0-9]+)%.*"
        r"reduce = (?P<reduce_progress>[0-9]+)%.*"
    )

    @classmethod
    def patch(cls) -> None:
        from pyhive import hive
        from TCLIService import (
            constants as patched_constants,
            TCLIService as patched_TCLIService,
            ttypes as patched_ttypes,
        )

        hive.TCLIService = patched_TCLIService
        hive.constants = patched_constants
        hive.ttypes = patched_ttypes

    @classmethod
    def fetch_data(cls, cursor: Any, limit: Optional[int] = None) -> List[Tuple[Any, ...]]:
        import pyhive
        from TCLIService import ttypes

        state = cursor.poll()
        if state.operationState == ttypes.TOperationState.ERROR_STATE:
            raise Exception("Query error", state.errorMessage)
        try:
            return super().fetch_data(cursor, limit)
        except pyhive.exc.ProgrammingError:
            return []

    @classmethod
    def df_to_sql(
        cls,
        database: Database,
        table: Table,
        df: pd.DataFrame,
        to_sql_kwargs: Dict[str, Any],
    ) -> None:
        if to_sql_kwargs["if_exists"] == "append":
            raise SupersetException("Append operation not currently supported")

        if to_sql_kwargs["if_exists"] == "fail":
            if table.schema:
                table_exists = not database.get_df(
                    f"SHOW TABLES IN {table.schema} LIKE '{table.table}'"
                ).empty
            else:
                table_exists = not database.get_df(
                    f"SHOW TABLES LIKE '{table.table}'"
                ).empty

            if table_exists:
                raise SupersetException("Table already exists")
        elif to_sql_kwargs["if_exists"] == "replace":
            with cls.get_engine(
                database,
                catalog=table.catalog,
                schema=table.schema,
            ) as engine:
                engine.execute(f"DROP TABLE IF EXISTS {str(table)}")

        def _get_hive_type(dtype: np.dtype[Any]) -> str:
            hive_type_by_dtype: Dict[np.dtype[Any], str] = {
                np.dtype("bool"): "BOOLEAN",
                np.dtype("float64"): "DOUBLE",
                np.dtype("int64"): "BIGINT",
                np.dtype("object"): "STRING",
            }
            return hive_type_by_dtype.get(dtype, "STRING")

        schema_definition: str = ", ".join(
            f"`{name}` {_get_hive_type(dtype)}" for name, dtype in df.dtypes.items()
        )

        with tempfile.NamedTemporaryFile(
            dir=current_app.config["UPLOAD_FOLDER"], suffix=".parquet"
        ) as file:
            pq.write_table(pa.Table.from_pandas(df), where=file.name)

            with cls.get_engine(
                database,
                catalog=table.catalog,
                schema=table.schema,
            ) as engine:
                engine.execute(
                    text(
                        f"""
                        CREATE TABLE {str(table)} ({schema_definition})
                        STORED AS PARQUET
                        LOCATION :location
                        """
                    ),
                    location=upload_to_s3(
                        filename=file.name,
                        upload_prefix=current_app.config[
                            "CSV_TO_HIVE_UPLOAD_DIRECTORY_FUNC"
                        ](database, g.user, table.schema),
                        table=table,
                    ),
                )

    @classmethod
    def convert_dttm(
        cls, target_type: str, dttm: datetime, db_extra: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        sqla_type = cls.get_sqla_column_type(target_type)

        if isinstance(sqla_type, types.Date):
            return f"CAST('{dttm.date().isoformat()}' AS DATE)"
        if isinstance(sqla_type, types.TIMESTAMP):
            return f"""CAST('{dttm
                .isoformat(sep=" ", timespec="microseconds")}' AS TIMESTAMP)"""
        return None

    @classmethod
    def adjust_engine_params(
        cls,
        uri: URL,
        connect_args: Dict[str, Any],
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> Tuple[URL, Dict[str, Any]]:
        if schema:
            uri = uri.set(database=parse.quote(schema, safe=""))
        return uri, connect_args

    @classmethod
    def get_schema_from_engine_params(
        cls,
        sqlalchemy_uri: URL,
        connect_args: Dict[str, Any],
    ) -> Optional[str]:
        return parse.unquote(sqlalchemy_uri.database)

    @classmethod
    def _extract_error_message(cls, ex: Exception) -> str:
        msg = str(ex)
        match = re.search(r'errorMessage="(.*?)(?<!\\)"', msg)
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
                total_jobs = int(match.groupdict()["max_jobs"]) or 1
            match = cls.launching_job_r.match(line)
            if match:
                current_job = int(match.groupdict()["job_number"])
                total_jobs = int(match.groupdict()["max_jobs"]) or 1
                stages = {}
            match = cls.stage_progress_r.match(line)
            if match:
                stage_number = int(match.groupdict()["stage_number"])
                map_progress = int(match.groupdict()["map_progress"])
                reduce_progress = int(match.groupdict()["reduce_progress"])
                stages[stage_number] = (map_progress + reduce_progress) / 2

        stage_progress = sum(stages.values()) / len(stages.values()) if stages else 0
        progress = 100 * (current_job - 1) / total_jobs + stage_progress / total_jobs
        return int(progress)

    @classmethod
    def get_tracking_url_from_logs(cls, log_lines: List[str]) -> Optional[str]:
        lkp = "Tracking URL = "
        for line in log_lines:
            if lkp in line:
                return line.split(lkp)[1]
        return None

    @classmethod
    def handle_cursor(cls, cursor: Any, query: Query) -> None:
        from pyhive import hive
        from TCLIService import ttypes

        unfinished_states = (
            hive.ttypes.TOperationState.INITIALIZED_STATE,
            hive.ttypes.TOperationState.RUNNING_STATE,
        )
        polled = cursor.poll()
        last_log_line: int = 0
        tracking_url: Optional[str] = None
        job_id: Optional[str] = None
        query_id = query.id
        while polled.operationState in unfinished_states:
            db.session.refresh(query)
            query = db.session.query(type(query)).filter_by(id=query_id).one()
            if query.status == QueryStatus.STOPPED:
                cursor.cancel()
                break

            try:
                logs = cursor.fetch_logs()
                log = "\n".join(logs) if logs else ""
            except Exception:
                logger.warning("Call to GetLog() failed")
                log = ""

            if log:
                log_lines = log.splitlines()
                progress = cls.progress(log_lines)
                needs_commit = False
                if progress > query.progress:
                    query.progress = progress
                    needs_commit = True
                if not tracking_url:
                    tracking_url = cls.get_tracking_url_from_logs(log_lines)
                    if tracking_url:
                        job_id = tracking_url.split("/")[-2]
                        query.tracking_url = tracking_url
                        needs_commit = True
                if job_id and len(log_lines) > last_log_line:
                    for l in log_lines[last_log_line:]:
                        logger.info("Query %s: [%s] %s", str(query_id), str(job_id), l)
                    last_log_line = len(log_lines)
                if needs_commit:
                    db.session.commit()
            sleep_interval = current_app.config.get("HIVE_POLL_INTERVAL") or current_app.config["DB_POLL_INTERVAL_SECONDS"].get(
                cls.engine, 5
            )
            time.sleep(sleep_interval)
            polled = cursor.poll()

    @classmethod
    def get_columns(
        cls,
        inspector: Inspector,
        table: Table,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[ResultSetColumnType]:
        return BaseEngineSpec.get_columns(inspector, table, options)

    @classmethod
    def where_latest_partition(
        cls,
        database: Database,
        table: Table,
        query: Select,
        columns: Optional[List[ResultSetColumnType]] = None,
    ) -> Optional[Select]:
        try:
            col_names, values = cls.latest_partition(
                database,
                table,
                show_first=True,
            )
        except Exception:
            return None
        if values is not None and columns is not None:
            for col_name, value in zip(col_names, values, strict=False):
                for clm in columns:
                    if clm.get("name") == col_name:
                        query = query.where(Column(col_name) == value)
            return query
        return None

    @classmethod
    def _get_fields(cls, cols: List[ResultSetColumnType]) -> List[ColumnClause]:
        return BaseEngineSpec._get_fields(cols)

    @classmethod
    def latest_sub_partition(
        cls,
        database: Database,
        table: Table,
        **kwargs: Any,
    ) -> None:
        pass

    @classmethod
    def _latest_partition_from_df(cls, df: pd.DataFrame) -> Optional[List[str]]:
        if not df.empty:
            return [
                partition_str.split("=")[1]
                for partition_str in df.iloc[:, 0].max().split("/")
            ]
        return None

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
        full_table_name = (
            f"{table.schema}.{table.table}" if table.schema else table.table
        )
        return f"SHOW PARTITIONS {full_table_name}"

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
        return super(PrestoEngineSpec, cls).select_star(
            database,
            table,
            engine,
            limit,
            show_cols,
            indent,
            latest_partition,
            cols,
        )

    @classmethod
    def get_url_for_impersonation(
        cls,
        url: URL,
        impersonate_user: bool,
        username: Optional[str],
        access_token: Optional[str],
    ) -> URL:
        return url

    @classmethod
    def update_impersonation_config(
        cls,
        database: Database,
        connect_args: Dict[str, Any],
        uri: str,
        username: Optional[str],
        access_token: Optional[str],
    ) -> None:
        url = make_url_safe(uri)
        backend_name = url.get_backend_name()

        if backend_name == "hive" and username is not None:
            configuration = connect_args.get("configuration", {})
            configuration["hive.server2.proxy.user"] = username
            connect_args["configuration"] = configuration

    @staticmethod
    def execute(
        cursor: Any,
        query: str,
        database: Database,
        async_: bool = False,
    ) -> None:
        kwargs = {"async": async_}
        cursor.execute(query, **kwargs)

    @classmethod
    @cache_manager.c