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
    engine: str = 'drill'
    engine_name: str = 'Apache Drill'
    default_driver: str = 'sadrill'
    supports_dynamic_schema: bool = True
    _time_grain_expressions: dict[TimeGrain, str] = {None: '{col}', TimeGrain.SECOND: "NEARESTDATE({col}, 'SECOND')", TimeGrain.MINUTE: "NEARESTDATE({col}, 'MINUTE')", TimeGrain.FIFTEEN_MINUTES: "NEARESTDATE({col}, 'QUARTER_HOUR')", TimeGrain.THIRTY_MINUTES: "NEARESTDATE({col}, 'HALF_HOUR')", TimeGrain.HOUR: "NEARESTDATE({col}, 'HOUR')", TimeGrain.DAY: "NEARESTDATE({col}, 'DAY')", TimeGrain.WEEK: "NEARESTDATE({col}, 'WEEK_SUNDAY')", TimeGrain.MONTH: "NEARESTDATE({col}, 'MONTH')", TimeGrain.QUARTER: "NEARESTDATE({col}, 'QUARTER')", TimeGrain.YEAR: "NEARESTDATE({col}, 'YEAR')"}

    @classmethod
    def epoch_to_dttm(cls) -> str:
        return cls.epoch_ms_to_dttm().replace('{col}', '({col}*1000)')

    @classmethod
    def epoch_ms_to_dttm(cls) -> str:
        return 'TO_DATE({col})'

    @classmethod
    def convert_dttm(cls, target_type: Any, dttm: datetime, db_extra: Any = None) -> str:
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, types.Date):
            return f"TO_DATE('{dttm.date().isoformat()}', 'yyyy-MM-dd')"
        if isinstance(sqla_type, types.TIMESTAMP):
            datetime_formatted = dttm.isoformat(sep=' ', timespec='seconds')
            return f"TO_TIMESTAMP('{datetime_formatted}', 'yyyy-MM-dd HH:mm:ss')"
        return None

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: dict, catalog: str = None, schema: str = None) -> tuple[URL, dict]:
        if schema:
            uri = uri.set(database=parse.quote(schema.replace('.', '/'), safe=''))
        return (uri, connect_args)

    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: dict) -> str:
        return parse.unquote(sqlalchemy_uri.database).replace('/', '.')

    @classmethod
    def get_url_for_impersonation(cls, url: URL, impersonate_user: bool, username: str, access_token: str) -> URL:
        if impersonate_user and username is not None:
            if url.drivername == 'drill+odbc':
                url = url.update_query_dict({'DelegationUID': username})
            elif url.drivername in ['drill+sadrill', 'drill+jdbc']:
                url = url.update_query_dict({'impersonation_target': username})
            else:
                raise SupersetDBAPIProgrammingError(f'impersonation is not supported for {url.drivername}')
        return url

    @classmethod
    def fetch_data(cls, cursor: Any, limit: int = None) -> list:
        try:
            return super().fetch_data(cursor, limit)
        except RuntimeError as ex:
            if str(ex) == 'generator raised StopIteration':
                return []
            raise
