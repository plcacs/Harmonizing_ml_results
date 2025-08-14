from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

import requests
from flask import current_app
from sqlalchemy import types
from sqlalchemy.engine.reflection import Inspector

from superset import db
from superset.constants import QUERY_EARLY_CANCEL_KEY, TimeGrain
from superset.db_engine_specs.base import BaseEngineSpec
from superset.models.sql_lab import Query

if TYPE_CHECKING:
    from superset.models.core import Database

logger = logging.getLogger(__name__)
# Query 5543ffdf692b7d02:f78a944000000000: 3% Complete (17 out of 547)
QUERY_PROGRESS_REGEX = re.compile(r"Query.*: (?P<query_progress>[0-9]+)%")

class ImpalaEngineSpec(BaseEngineSpec):
    """Engine spec for Cloudera's Impala"""

    engine: str = "impala"
    engine_name: str = "Apache Impala"

    _time_grain_expressions: dict[Optional[str], str] = {
        None: "{col}",
        TimeGrain.MINUTE: "TRUNC({col}, 'MI')",
        TimeGrain.HOUR: "TRUNC({col}, 'HH')",
        TimeGrain.DAY: "TRUNC({col}, 'DD')",
        TimeGrain.WEEK: "TRUNC({col}, 'WW')",
        TimeGrain.MONTH: "TRUNC({col}, 'MONTH')",
        TimeGrain.QUARTER: "TRUNC({col}, 'Q')",
        TimeGrain.YEAR: "TRUNC({col}, 'YYYY')",
    }

    has_query_id_before_execute: bool = False

    @classmethod
    def epoch_to_dttm(cls) -> str:
        return "from_unixtime({col})"

    @classmethod
    def convert_dttm(
        cls, target_type: str, dttm: datetime, db_extra: Optional[dict[str, Any]] = None
    ) -> Optional[str]:
        sqla_type = cls.get_sqla_column_type(target_type)

        if isinstance(sqla_type, types.Date):
            return f"CAST('{dttm.date().isoformat()}' AS DATE)"
        if isinstance(sqla_type, types.TIMESTAMP):
            return f"""CAST('{dttm.isoformat(timespec="microseconds")}' AS TIMESTAMP)"""
        return None

    @classmethod
    def get_schema_names(cls, inspector: Inspector) -> set[str]:
        return {
            row[0]
            for row in inspector.engine.execute("SHOW SCHEMAS")
            if not row[0].startswith("_")
        }

    @classmethod
    def has_implicit_cancel(cls) -> bool:
        """
        Return True if the live cursor handles the implicit cancelation of the query,
        False otherwise.

        :return: Whether the live cursor implicitly cancels the query
        :see: handle_cursor
        """
        return False

    @classmethod
    def execute(
        cls,
        cursor: Any,
        query: str,
        database: Database,
        **kwargs: Any,
    ) -> None:
        try:
            cursor.execute_async(query)
        except Exception as ex:
            raise cls.get_dbapi_mapped_exception(ex) from ex

    @classmethod
    def handle_cursor(cls, cursor: Any, query: Query) -> None:
        """Stop query and updates progress information"""

        query_id: int = query.id
        unfinished_states: tuple[str, str] = (
            "INITIALIZED_STATE",
            "RUNNING_STATE",
        )

        try:
            status: str = cursor.status()
            while status in unfinished_states:
                db.session.refresh(query)
                query = db.session.query(Query).filter_by(id=query_id).one()
                # if query cancelation was requested prior to the handle_cursor call, but
                # the query was still executed
                # modified in stop_query in views / core.py is reflected here.
                # stop query
                if query.extra.get(QUERY_EARLY_CANCEL_KEY):
                    cursor.cancel_operation()
                    cursor.close_operation()
                    cursor.close()
                    break

                # updates progress info by log
                try:
                    log: str = cursor.get_log() or ""
                except Exception:  # pylint: disable=broad-except
                    logger.warning("Call to GetLog() failed")
                    log = ""

                if log:
                    match: Optional[re.Match[str]] = QUERY_PROGRESS_REGEX.match(log)
                    if match:
                        progress: int = int(match.groupdict()["query_progress"])
                        logger.debug(
                            "Query %s: Progress total: %s", str(query_id), str(progress)
                        )
                        needs_commit: bool = False
                        if progress > query.progress:
                            query.progress = progress
                            needs_commit = True

                        if needs_commit:
                            db.session.commit()  # pylint: disable=consider-using-transaction
                sleep_interval: int = current_app.config["DB_POLL_INTERVAL_SECONDS"].get(
                    cls.engine, 5
                )
                time.sleep(sleep_interval)
                status = cursor.status()
        except Exception:  # pylint: disable=broad-except
            logger.debug("Call to status() failed ")
            return

    @classmethod
    def get_cancel_query_id(cls, cursor: Any, query: Query) -> Optional[str]:
        """
        Get Impala Query ID that will be used to cancel the running
        queries to release impala resources.

        :param cursor: Cursor instance in which the query will be executed
        :param query: Query instance
        :return: Impala Query ID
        """
        last_operation: Any = getattr(cursor, "_last_operation", None)
        if not last_operation:
            return None
        guid: str = last_operation.handle.operationId.guid[::-1].hex()
        return f"{guid[-16:]}:{guid[:16]}"

    @classmethod
    def cancel_query(cls, cursor: Any, query: Query, cancel_query_id: str) -> bool:
        """
        Cancel query in the underlying database.

        :param cursor: New cursor instance to the db of the query
        :param query: Query instance
        :param cancel_query_id: impala db not need
        :return: True if query cancelled successfully, False otherwise
        """
        try:
            impala_host: str = query.database.url_object.host
            url: str = f"http://{impala_host}:25000/cancel_query?query_id={cancel_query_id}"
            response: Any = requests.post(url, timeout=3)
        except Exception:  # pylint: disable=broad-except
            return False

        return bool(response and response.status_code == 200)