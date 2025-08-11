import contextlib
import re
from datetime import datetime
from decimal import Decimal
from re import Pattern
from typing import Any, Callable, Optional
from urllib import parse
from flask_babel import gettext as __
from sqlalchemy import types
from sqlalchemy.dialects.mysql import BIT, DECIMAL, DOUBLE, FLOAT, INTEGER, LONGTEXT, MEDIUMINT, MEDIUMTEXT, TINYINT, TINYTEXT
from sqlalchemy.engine.url import URL
from superset.constants import TimeGrain
from superset.db_engine_specs.base import BaseEngineSpec, BasicParametersMixin
from superset.errors import SupersetErrorType
from superset.models.sql_lab import Query
from superset.utils.core import GenericDataType
CONNECTION_ACCESS_DENIED_REGEX = re.compile("Access denied for user '(?P<username>.*?)'@'(?P<hostname>.*?)'")
CONNECTION_INVALID_HOSTNAME_REGEX = re.compile("Unknown MySQL server host '(?P<hostname>.*?)'")
CONNECTION_HOST_DOWN_REGEX = re.compile("Can't connect to MySQL server on '(?P<hostname>.*?)'")
CONNECTION_UNKNOWN_DATABASE_REGEX = re.compile("Unknown database '(?P<database>.*?)'")
SYNTAX_ERROR_REGEX = re.compile("check the manual that corresponds to your MySQL server version for the right syntax to use near '(?P<server_error>.*)")

class MySQLEngineSpec(BasicParametersMixin, BaseEngineSpec):
    engine = 'mysql'
    engine_name = 'MySQL'
    max_column_name_length = 64
    default_driver = 'mysqldb'
    sqlalchemy_uri_placeholder = 'mysql://user:password@host:port/dbname[?key=value&key=value...]'
    encryption_parameters = {'ssl': '1'}
    supports_dynamic_schema = True
    column_type_mappings = ((re.compile('^int.*', re.IGNORECASE), INTEGER(), GenericDataType.NUMERIC), (re.compile('^tinyint', re.IGNORECASE), TINYINT(), GenericDataType.NUMERIC), (re.compile('^mediumint', re.IGNORECASE), MEDIUMINT(), GenericDataType.NUMERIC), (re.compile('^decimal', re.IGNORECASE), DECIMAL(), GenericDataType.NUMERIC), (re.compile('^float', re.IGNORECASE), FLOAT(), GenericDataType.NUMERIC), (re.compile('^double', re.IGNORECASE), DOUBLE(), GenericDataType.NUMERIC), (re.compile('^bit', re.IGNORECASE), BIT(), GenericDataType.NUMERIC), (re.compile('^tinytext', re.IGNORECASE), TINYTEXT(), GenericDataType.STRING), (re.compile('^mediumtext', re.IGNORECASE), MEDIUMTEXT(), GenericDataType.STRING), (re.compile('^longtext', re.IGNORECASE), LONGTEXT(), GenericDataType.STRING))
    column_type_mutators = {DECIMAL: lambda val: Decimal(val) if isinstance(val, str) else val}
    _time_grain_expressions = {None: '{col}', TimeGrain.SECOND: 'DATE_ADD(DATE({col}), INTERVAL (HOUR({col})*60*60 + MINUTE({col})*60 + SECOND({col})) SECOND)', TimeGrain.MINUTE: 'DATE_ADD(DATE({col}), INTERVAL (HOUR({col})*60 + MINUTE({col})) MINUTE)', TimeGrain.HOUR: 'DATE_ADD(DATE({col}), INTERVAL HOUR({col}) HOUR)', TimeGrain.DAY: 'DATE({col})', TimeGrain.WEEK: 'DATE(DATE_SUB({col}, INTERVAL DAYOFWEEK({col}) - 1 DAY))', TimeGrain.MONTH: 'DATE(DATE_SUB({col}, INTERVAL DAYOFMONTH({col}) - 1 DAY))', TimeGrain.QUARTER: 'MAKEDATE(YEAR({col}), 1) + INTERVAL QUARTER({col}) QUARTER - INTERVAL 1 QUARTER', TimeGrain.YEAR: 'DATE(DATE_SUB({col}, INTERVAL DAYOFYEAR({col}) - 1 DAY))', TimeGrain.WEEK_STARTING_MONDAY: 'DATE(DATE_SUB({col}, INTERVAL DAYOFWEEK(DATE_SUB({col}, INTERVAL 1 DAY)) - 1 DAY))'}
    type_code_map = {}
    custom_errors = {CONNECTION_ACCESS_DENIED_REGEX: (__('Either the username "%(username)s" or the password is incorrect.'), SupersetErrorType.CONNECTION_ACCESS_DENIED_ERROR, {'invalid': ['username', 'password']}), CONNECTION_INVALID_HOSTNAME_REGEX: (__('Unknown MySQL server host "%(hostname)s".'), SupersetErrorType.CONNECTION_INVALID_HOSTNAME_ERROR, {'invalid': ['host']}), CONNECTION_HOST_DOWN_REGEX: (__('The host "%(hostname)s" might be down and can\'t be reached.'), SupersetErrorType.CONNECTION_HOST_DOWN_ERROR, {'invalid': ['host', 'port']}), CONNECTION_UNKNOWN_DATABASE_REGEX: (__('Unable to connect to database "%(database)s".'), SupersetErrorType.CONNECTION_UNKNOWN_DATABASE_ERROR, {'invalid': ['database']}), SYNTAX_ERROR_REGEX: (__('Please check your query for syntax errors near "%(server_error)s". Then, try running your query again.'), SupersetErrorType.SYNTAX_ERROR, {})}
    disallow_uri_query_params = {'mysqldb': {'local_infile'}, 'mysqlconnector': {'allow_local_infile'}}
    enforce_uri_query_params = {'mysqldb': {'local_infile': 0}, 'mysqlconnector': {'allow_local_infile': 0}}

    @classmethod
    def convert_dttm(cls: Union[str, None, T], target_type: Union[str, None, T], dttm: Union[datetime.datetime.datetime, None], db_extra: Union[None, str, bool]=None) -> Union[typing.Text, None]:
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, types.Date):
            return f"STR_TO_DATE('{dttm.date().isoformat()}', '%Y-%m-%d')"
        if isinstance(sqla_type, types.DateTime):
            datetime_formatted = dttm.isoformat(sep=' ', timespec='microseconds')
            return f"STR_TO_DATE('{datetime_formatted}', '%Y-%m-%d %H:%i:%s.%f')"
        return None

    @classmethod
    def adjust_engine_params(cls: Union[bool, None, sqlalchemy.Table], uri: Union[str, bool, typing.Callable], connect_args: Union[str, typing.Callable, None], catalog: Union[None, str, typing.Callable]=None, schema: Union[None, str, sqlalchemy.engine.Engine]=None) -> tuple[typing.Union[str,caches.core.CacheURL,dict,set[str],list[str]]]:
        uri, new_connect_args = super().adjust_engine_params(uri, connect_args, catalog, schema)
        if schema:
            uri = uri.set(database=parse.quote(schema, safe=''))
        return (uri, new_connect_args)

    @classmethod
    def get_schema_from_engine_params(cls: Union[str, list[str]], sqlalchemy_uri: Union[str, None], connect_args: Union[str, list[str]]):
        """
        Return the configured schema.

        A MySQL database is a SQLAlchemy schema.
        """
        return parse.unquote(sqlalchemy_uri.database)

    @classmethod
    def get_datatype(cls: Union[str, typing.Type], type_code: Union[int, str]) -> Union[int, str, None, list[str]]:
        if not cls.type_code_map:
            import MySQLdb
            ft = MySQLdb.constants.FIELD_TYPE
            cls.type_code_map = {getattr(ft, k): k for k in dir(ft) if not k.startswith('_')}
        datatype = type_code
        if isinstance(type_code, int):
            datatype = cls.type_code_map.get(type_code)
        if datatype and isinstance(datatype, str) and datatype:
            return datatype
        return None

    @classmethod
    def epoch_to_dttm(cls: Union[str, int, typing.Type]) -> typing.Text:
        return 'from_unixtime({col})'

    @classmethod
    def _extract_error_message(cls: Union[str, dict, django.db.models.Model], ex: Union[str, None, ValueError]) -> str:
        """Extract error message for queries"""
        message = str(ex)
        with contextlib.suppress(AttributeError, KeyError):
            if isinstance(ex.args, tuple) and len(ex.args) > 1:
                message = ex.args[1]
        return message

    @classmethod
    def get_cancel_query_id(cls: Union[str, dict], cursor: str, query: Union[str, dict]) -> str:
        """
        Get MySQL connection ID that will be used to cancel all other running
        queries in the same connection.

        :param cursor: Cursor instance in which the query will be executed
        :param query: Query instance
        :return: MySQL Connection ID
        """
        cursor.execute('SELECT CONNECTION_ID()')
        row = cursor.fetchone()
        return row[0]

    @classmethod
    def cancel_query(cls: Union[int, list[dict], list], cursor: Union[int, str], query: Union[int, list[dict], list], cancel_query_id: Union[int, str]) -> bool:
        """
        Cancel query in the underlying database.

        :param cursor: New cursor instance to the db of the query
        :param query: Query instance
        :param cancel_query_id: MySQL Connection ID
        :return: True if query cancelled successfully, False otherwise
        """
        try:
            cursor.execute(f'KILL CONNECTION {cancel_query_id}')
        except Exception:
            return False
        return True