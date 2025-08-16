from datetime import datetime
from decimal import Decimal
from typing import Any, Optional
from unittest.mock import Mock, patch
import pytest
from sqlalchemy import types
from sqlalchemy.dialects.mysql import BIT, DECIMAL, DOUBLE, FLOAT, INTEGER, LONGTEXT, MEDIUMINT, MEDIUMTEXT, TINYINT, TINYTEXT
from sqlalchemy.engine.url import make_url, URL
from superset.utils.core import GenericDataType
from tests.unit_tests.db_engine_specs.utils import assert_column_spec, assert_convert_dttm
from tests.unit_tests.fixtures.common import dttm

@pytest.mark.parametrize('native_type: str, sqla_type: Any, attrs: Optional[Any], generic_type: GenericDataType, is_dttm: bool', [('TINYINT', TINYINT, None, GenericDataType.NUMERIC, False), ('SMALLINT', types.SmallInteger, None, GenericDataType.NUMERIC, False), ('MEDIUMINT', MEDIUMINT, None, GenericDataType.NUMERIC, False), ('INT', INTEGER, None, GenericDataType.NUMERIC, False), ('BIGINT', types.BigInteger, None, GenericDataType.NUMERIC, False), ('DECIMAL', DECIMAL, None, GenericDataType.NUMERIC, False), ('FLOAT', FLOAT, None, GenericDataType.NUMERIC, False), ('DOUBLE', DOUBLE, None, GenericDataType.NUMERIC, False), ('BIT', BIT, None, GenericDataType.NUMERIC, False), ('CHAR', types.String, None, GenericDataType.STRING, False), ('VARCHAR', types.String, None, GenericDataType.STRING, False), ('TINYTEXT', TINYTEXT, None, GenericDataType.STRING, False), ('MEDIUMTEXT', MEDIUMTEXT, None, GenericDataType.STRING, False), ('LONGTEXT', LONGTEXT, None, GenericDataType.STRING, False), ('DATE', types.Date, None, GenericDataType.TEMPORAL, True), ('DATETIME', types.DateTime, None, GenericDataType.TEMPORAL, True), ('TIMESTAMP', types.TIMESTAMP, None, GenericDataType.TEMPORAL, True), ('TIME', types.Time, None, GenericDataType.TEMPORAL, True)])
def test_get_column_spec(native_type, sqla_type, attrs, generic_type, is_dttm):
    from superset.db_engine_specs.mysql import MySQLEngineSpec as spec
    assert_column_spec(spec, native_type, sqla_type, attrs, generic_type, is_dttm)

@pytest.mark.parametrize('target_type: str, expected_result: str', [('Date', "STR_TO_DATE('2019-01-02', '%Y-%m-%d')"), ('DateTime', "STR_TO_DATE('2019-01-02 03:04:05.678900', '%Y-%m-%d %H:%i:%s.%f')"), ('UnknownType', None)])
def test_convert_dttm(target_type, expected_result, dttm):
    from superset.db_engine_specs.mysql import MySQLEngineSpec as spec
    assert_convert_dttm(spec, target_type, expected_result, dttm)

@pytest.mark.parametrize('sqlalchemy_uri: str, error: bool', [('mysql://user:password@host/db1?local_infile=1', True), ('mysql+mysqlconnector://user:password@host/db1?allow_local_infile=1', True), ('mysql://user:password@host/db1?local_infile=0', True), ('mysql+mysqlconnector://user:password@host/db1?allow_local_infile=0', True), ('mysql://user:password@host/db1', False), ('mysql+mysqlconnector://user:password@host/db1', False)])
def test_validate_database_uri(sqlalchemy_uri, error):
    from superset.db_engine_specs.mysql import MySQLEngineSpec
    url = make_url(sqlalchemy_uri)
    if error:
        with pytest.raises(ValueError):
            MySQLEngineSpec.validate_database_uri(url)
        return
    MySQLEngineSpec.validate_database_uri(url)

@pytest.mark.parametrize('sqlalchemy_uri: str, connect_args: dict, returns: dict', [('mysql://user:password@host/db1', {'local_infile': 1}, {'local_infile': 0}), ('mysql+mysqlconnector://user:password@host/db1', {'allow_local_infile': 1}, {'allow_local_infile': 0}), ('mysql://user:password@host/db1', {'local_infile': -1}, {'local_infile': 0}), ('mysql+mysqlconnector://user:password@host/db1', {'allow_local_infile': -1}, {'allow_local_infile': 0}), ('mysql://user:password@host/db1', {'local_infile': 0}, {'local_infile': 0}), ('mysql+mysqlconnector://user:password@host/db1', {'allow_local_infile': 0}, {'allow_local_infile': 0}), ('mysql://user:password@host/db1', {'param1': 'some_value'}, {'local_infile': 0, 'param1': 'some_value'}), ('mysql+mysqlconnector://user:password@host/db1', {'param1': 'some_value'}, {'allow_local_infile': 0, 'param1': 'some_value'}), ('mysql://user:password@host/db1', {'local_infile': 1, 'param1': 'some_value'}, {'local_infile': 0, 'param1': 'some_value'}), ('mysql+mysqlconnector://user:password@host/db1', {'allow_local_infile': 1, 'param1': 'some_value'}, {'allow_local_infile': 0, 'param1': 'some_value'})])
def test_adjust_engine_params(sqlalchemy_uri, connect_args, returns):
    from superset.db_engine_specs.mysql import MySQLEngineSpec
    url = make_url(sqlalchemy_uri)
    returned_url, returned_connect_args = MySQLEngineSpec.adjust_engine_params(url, connect_args)
    assert returned_connect_args == returns

@patch('sqlalchemy.engine.Engine.connect')
def test_get_cancel_query_id(engine_mock):
    from superset.db_engine_specs.mysql import MySQLEngineSpec
    from superset.models.sql_lab import Query
    query = Query()
    cursor_mock = engine_mock.return_value.__enter__.return_value
    cursor_mock.fetchone.return_value = ['123']
    assert MySQLEngineSpec.get_cancel_query_id(cursor_mock, query) == '123'

@patch('sqlalchemy.engine.Engine.connect')
def test_cancel_query(engine_mock):
    from superset.db_engine_specs.mysql import MySQLEngineSpec
    from superset.models.sql_lab import Query
    query = Query()
    cursor_mock = engine_mock.return_value.__enter__.return_value
    assert MySQLEngineSpec.cancel_query(cursor_mock, query, '123') is True

@patch('sqlalchemy.engine.Engine.connect')
def test_cancel_query_failed(engine_mock):
    from superset.db_engine_specs.mysql import MySQLEngineSpec
    from superset.models.sql_lab import Query
    query = Query()
    cursor_mock = engine_mock.raiseError.side_effect = Exception()
    assert MySQLEngineSpec.cancel_query(cursor_mock, query, '123') is False

def test_get_schema_from_engine_params():
    """
    Test the ``get_schema_from_engine_params`` method.
    """
    from superset.db_engine_specs.mysql import MySQLEngineSpec
    assert MySQLEngineSpec.get_schema_from_engine_params(make_url('mysql://user:password@host/db1'), {}) == 'db1'

@pytest.mark.parametrize('data: list, description: list, expected_result: list', [([('1.23456', 'abc')], [('dec', 'decimal(12,6)'), ('str', 'varchar(3)')], [(Decimal('1.23456'), 'abc')]), ([(Decimal('1.23456'), 'abc')], [('dec', 'decimal(12,6)'), ('str', 'varchar(3)')], [(Decimal('1.23456'), 'abc')]), ([(None, 'abc')], [('dec', 'decimal(12,6)'), ('str', 'varchar(3)')], [(None, 'abc')]), ([('1.23456', 'abc')], [('dec', 'varchar(255)'), ('str', 'varchar(3)')], [('1.23456', 'abc')])])
def test_column_type_mutator(data, description, expected_result):
    from superset.db_engine_specs.mysql import MySQLEngineSpec as spec
    mock_cursor = Mock()
    mock_cursor.fetchall.return_value = data
    mock_cursor.description = description
    assert spec.fetch_data(mock_cursor) == expected_result
