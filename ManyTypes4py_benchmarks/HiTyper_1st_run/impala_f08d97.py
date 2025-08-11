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
QUERY_PROGRESS_REGEX = re.compile('Query.*: (?P<query_progress>[0-9]+)%')

class ImpalaEngineSpec(BaseEngineSpec):
    """Engine spec for Cloudera's Impala"""
    engine = 'impala'
    engine_name = 'Apache Impala'
    _time_grain_expressions = {None: '{col}', TimeGrain.MINUTE: "TRUNC({col}, 'MI')", TimeGrain.HOUR: "TRUNC({col}, 'HH')", TimeGrain.DAY: "TRUNC({col}, 'DD')", TimeGrain.WEEK: "TRUNC({col}, 'WW')", TimeGrain.MONTH: "TRUNC({col}, 'MONTH')", TimeGrain.QUARTER: "TRUNC({col}, 'Q')", TimeGrain.YEAR: "TRUNC({col}, 'YYYY')"}
    has_query_id_before_execute = False

    @classmethod
    def epoch_to_dttm(cls: Union[str, int, typing.Type]) -> typing.Text:
        return 'from_unixtime({col})'

    @classmethod
    def convert_dttm(cls: Union[str, None, T], target_type: Union[str, None, T], dttm: Union[datetime.datetime.datetime, None], db_extra: Union[None, str, bool]=None) -> Union[typing.Text, None]:
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, types.Date):
            return f"CAST('{dttm.date().isoformat()}' AS DATE)"
        if isinstance(sqla_type, types.TIMESTAMP):
            return f"CAST('{dttm.isoformat(timespec='microseconds')}' AS TIMESTAMP)"
        return None

    @classmethod
    def get_schema_names(cls: Union[str, list[dict[str, typing.Any]], typing.Type, None], inspector: Union[str, django.db.models.Model]) -> set:
        return {row[0] for row in inspector.engine.execute('SHOW SCHEMAS') if not row[0].startswith('_')}

    @classmethod
    def has_implicit_cancel(cls: Union[typing.Type, typing.Callable[typing.Any, T]]) -> bool:
        """
        Return True if the live cursor handles the implicit cancelation of the query,
        False otherwise.

        :return: Whether the live cursor implicitly cancels the query
        :see: handle_cursor
        """
        return False

    @classmethod
    def execute(cls: Union[str, list[str], sqlalchemy.engine.url.URL], cursor: Union[sqlalchemy.orm.scoped_session, str, sqlalchemy.engine.Engine], query: Union[sqlalchemy.orm.scoped_session, str, sqlalchemy.engine.Engine], database: Union[str, bool, typing.Sequence[str]], **kwargs) -> None:
        try:
            cursor.execute_async(query)
        except Exception as ex:
            raise cls.get_dbapi_mapped_exception(ex) from ex

    @classmethod
    def handle_cursor(cls: Union[str, dict, set[str]], cursor: zerver.models.Realm, query: Union[str, supersemodels.sql_lab.Query, zerver.models.Realm]) -> None:
        """Stop query and updates progress information"""
        query_id = query.id
        unfinished_states = ('INITIALIZED_STATE', 'RUNNING_STATE')
        try:
            status = cursor.status()
            while status in unfinished_states:
                db.session.refresh(query)
                query = db.session.query(Query).filter_by(id=query_id).one()
                if query.extra.get(QUERY_EARLY_CANCEL_KEY):
                    cursor.cancel_operation()
                    cursor.close_operation()
                    cursor.close()
                    break
                try:
                    log = cursor.get_log() or ''
                except Exception:
                    logger.warning('Call to GetLog() failed')
                    log = ''
                if log:
                    match = QUERY_PROGRESS_REGEX.match(log)
                    if match:
                        progress = int(match.groupdict()['query_progress'])
                    logger.debug('Query %s: Progress total: %s', str(query_id), str(progress))
                    needs_commit = False
                    if progress > query.progress:
                        query.progress = progress
                        needs_commit = True
                    if needs_commit:
                        db.session.commit()
                sleep_interval = current_app.config['DB_POLL_INTERVAL_SECONDS'].get(cls.engine, 5)
                time.sleep(sleep_interval)
                status = cursor.status()
        except Exception:
            logger.debug('Call to status() failed ')
            return

    @classmethod
    def get_cancel_query_id(cls: Union[str, dict], cursor: Union[str, dict, None], query: Union[str, dict]) -> Union[None, typing.Text]:
        """
        Get Impala Query ID that will be used to cancel the running
        queries to release impala resources.

        :param cursor: Cursor instance in which the query will be executed
        :param query: Query instance
        :return: Impala Query ID
        """
        last_operation = getattr(cursor, '_last_operation', None)
        if not last_operation:
            return None
        guid = last_operation.handle.operationId.guid[::-1].hex()
        return f'{guid[-16:]}:{guid[:16]}'

    @classmethod
    def cancel_query(cls: Union[int, list[dict], list], cursor: Union[int, list[dict], list], query: Union[int, str, None], cancel_query_id: Union[str, int, None, dict]) -> bool:
        """
        Cancel query in the underlying database.

        :param cursor: New cursor instance to the db of the query
        :param query: Query instance
        :param cancel_query_id: impala db not need
        :return: True if query cancelled successfully, False otherwise
        """
        try:
            impala_host = query.database.url_object.host
            url = f'http://{impala_host}:25000/cancel_query?query_id={cancel_query_id}'
            response = requests.post(url, timeout=3)
        except Exception:
            return False
        return bool(response and response.status_code == 200)