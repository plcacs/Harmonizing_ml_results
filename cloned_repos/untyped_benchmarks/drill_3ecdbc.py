from __future__ import annotations
from datetime import datetime
from typing import Any
from urllib import parse
from sqlalchemy import types
from sqlalchemy.engine.url import URL
from superset.constants import TimeGrain
from superset.db_engine_specs.base import BaseEngineSpec
from superset.db_engine_specs.exceptions import SupersetDBAPIProgrammingError

class DrillEngineSpec(BaseEngineSpec):
    """Engine spec for Apache Drill"""
    engine = 'drill'
    engine_name = 'Apache Drill'
    default_driver = 'sadrill'
    supports_dynamic_schema = True
    _time_grain_expressions = {None: '{col}', TimeGrain.SECOND: "NEARESTDATE({col}, 'SECOND')", TimeGrain.MINUTE: "NEARESTDATE({col}, 'MINUTE')", TimeGrain.FIFTEEN_MINUTES: "NEARESTDATE({col}, 'QUARTER_HOUR')", TimeGrain.THIRTY_MINUTES: "NEARESTDATE({col}, 'HALF_HOUR')", TimeGrain.HOUR: "NEARESTDATE({col}, 'HOUR')", TimeGrain.DAY: "NEARESTDATE({col}, 'DAY')", TimeGrain.WEEK: "NEARESTDATE({col}, 'WEEK_SUNDAY')", TimeGrain.MONTH: "NEARESTDATE({col}, 'MONTH')", TimeGrain.QUARTER: "NEARESTDATE({col}, 'QUARTER')", TimeGrain.YEAR: "NEARESTDATE({col}, 'YEAR')"}

    @classmethod
    def epoch_to_dttm(cls):
        return cls.epoch_ms_to_dttm().replace('{col}', '({col}*1000)')

    @classmethod
    def epoch_ms_to_dttm(cls):
        return 'TO_DATE({col})'

    @classmethod
    def convert_dttm(cls, target_type, dttm, db_extra=None):
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, types.Date):
            return f"TO_DATE('{dttm.date().isoformat()}', 'yyyy-MM-dd')"
        if isinstance(sqla_type, types.TIMESTAMP):
            datetime_formatted = dttm.isoformat(sep=' ', timespec='seconds')
            return f"TO_TIMESTAMP('{datetime_formatted}', 'yyyy-MM-dd HH:mm:ss')"
        return None

    @classmethod
    def adjust_engine_params(cls, uri, connect_args, catalog=None, schema=None):
        if schema:
            uri = uri.set(database=parse.quote(schema.replace('.', '/'), safe=''))
        return (uri, connect_args)

    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri, connect_args):
        """
        Return the configured schema.
        """
        return parse.unquote(sqlalchemy_uri.database).replace('/', '.')

    @classmethod
    def get_url_for_impersonation(cls, url, impersonate_user, username, access_token):
        """
        Return a modified URL with the username set.

        :param url: SQLAlchemy URL object
        :param impersonate_user: Flag indicating if impersonation is enabled
        :param username: Effective username
        """
        if impersonate_user and username is not None:
            if url.drivername == 'drill+odbc':
                url = url.update_query_dict({'DelegationUID': username})
            elif url.drivername in ['drill+sadrill', 'drill+jdbc']:
                url = url.update_query_dict({'impersonation_target': username})
            else:
                raise SupersetDBAPIProgrammingError(f'impersonation is not supported for {url.drivername}')
        return url

    @classmethod
    def fetch_data(cls, cursor, limit=None):
        """
        Custom `fetch_data` for Drill.

        When no rows are returned, Drill raises a `RuntimeError` with the message
        "generator raised StopIteration". This method catches the exception and
        returns an empty list instead.
        """
        try:
            return super().fetch_data(cursor, limit)
        except RuntimeError as ex:
            if str(ex) == 'generator raised StopIteration':
                return []
            raise