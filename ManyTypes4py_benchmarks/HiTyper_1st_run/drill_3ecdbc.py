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
    def epoch_to_dttm(cls: Union[str, T]) -> Union[int, datetime.datetime.datetime, None, datetime.date]:
        return cls.epoch_ms_to_dttm().replace('{col}', '({col}*1000)')

    @classmethod
    def epoch_ms_to_dttm(cls: Union[str, int]) -> typing.Text:
        return 'TO_DATE({col})'

    @classmethod
    def convert_dttm(cls: Union[str, None, T], target_type: Union[str, None, T], dttm: Union[datetime.datetime.datetime, None], db_extra: Union[None, str, bool]=None) -> Union[typing.Text, None]:
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, types.Date):
            return f"TO_DATE('{dttm.date().isoformat()}', 'yyyy-MM-dd')"
        if isinstance(sqla_type, types.TIMESTAMP):
            datetime_formatted = dttm.isoformat(sep=' ', timespec='seconds')
            return f"TO_TIMESTAMP('{datetime_formatted}', 'yyyy-MM-dd HH:mm:ss')"
        return None

    @classmethod
    def adjust_engine_params(cls: Union[bool, None, sqlalchemy.Table], uri: Union[str, yarl.URL, typing.Type], connect_args: Union[str, typing.Callable], catalog: Union[None, bool, sqlalchemy.Table]=None, schema: Union[str, cmk.base.api.agent_based.type_defs.InventoryPlugin]=None) -> tuple[typing.Union[str,yarl.URL,typing.Type,typing.Callable]]:
        if schema:
            uri = uri.set(database=parse.quote(schema.replace('.', '/'), safe=''))
        return (uri, connect_args)

    @classmethod
    def get_schema_from_engine_params(cls: Union[str, list[str]], sqlalchemy_uri: str, connect_args: Union[str, list[str]]) -> Union[str, typing.BinaryIO]:
        """
        Return the configured schema.
        """
        return parse.unquote(sqlalchemy_uri.database).replace('/', '.')

    @classmethod
    def get_url_for_impersonation(cls: Union[str, None, bool], url: Union[str, None, yarl.URL], impersonate_user: Union[str, None, bool], username: Union[str, None, bool], access_token: Union[str, None, bool]) -> Union[str, None, yarl.URL, tuple[typing.Union[str,int]]]:
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
    def fetch_data(cls: Union[int, str], cursor: Union[int, None], limit: Union[None, int]=None) -> Union[str, dict, list[str], list]:
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