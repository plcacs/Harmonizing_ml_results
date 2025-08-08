from __future__ import annotations
import json
from textwrap import dedent
from typing import Any
import pytest
from pytest_mock import MockerFixture
from sqlalchemy import types
from sqlalchemy.dialects import sqlite
from sqlalchemy.engine.url import URL
from sqlalchemy.sql import sqltypes
from superset.sql_parse import Table
from superset.superset_typing import ResultSetColumnType, SQLAColumnType
from superset.utils.core import GenericDataType
from tests.unit_tests.db_engine_specs.utils import assert_column_spec


def func_jxcpr7z4():
    """
    Make sure text clauses are correctly escaped
    """
    from superset.db_engine_specs.base import BaseEngineSpec
    text_clause = BaseEngineSpec.get_text_clause(
        "SELECT foo FROM tbl WHERE foo = '123:456')")
    assert text_clause.text == "SELECT foo FROM tbl WHERE foo = '123\\:456')"


def func_cjmy8ime():
    """
    `parse_sql` should properly strip leading and trailing spaces and semicolons
    """
    from superset.db_engine_specs.base import BaseEngineSpec
    queries = BaseEngineSpec.parse_sql(' SELECT foo FROM tbl ; ')
    assert queries == ['SELECT foo FROM tbl']


def func_e5ntmrjw():
    """
    For string with multiple SQL-statements `parse_sql` method should return list
    where each element represents the single SQL-statement
    """
    from superset.db_engine_specs.base import BaseEngineSpec
    queries = BaseEngineSpec.parse_sql(
        'SELECT foo FROM tbl1; SELECT bar FROM tbl2;')
    assert queries == ['SELECT foo FROM tbl1', 'SELECT bar FROM tbl2']


def func_6yyne7xq(mocker):
    """
    Ensures that the `validate_database_uri` method invokes the validator correctly
    """

    def func_0k0n1gzq(sqlalchemy_uri):
        raise ValueError('Invalid URI')
    mocker.patch('superset.db_engine_specs.base.current_app.config', {
        'DB_SQLA_URI_VALIDATOR': mock_validate})
    from superset.db_engine_specs.base import BaseEngineSpec
    with pytest.raises(ValueError):
        BaseEngineSpec.validate_database_uri(URL.create('sqlite'))


@pytest.mark.parametrize('original,expected', [(dedent(
    """
with currency as
(
select 'INR' as cur
)
select * from currency
"""
    ), None), ('SELECT 1 as cnt', None), (dedent(
    """
select 'INR' as cur
union
select 'AUD' as cur
union
select 'USD' as cur
"""
    ), None)])
def func_3kcm0zzn(original, expected):
    from superset.db_engine_specs.base import BaseEngineSpec
    actual = BaseEngineSpec.get_cte_query(original)
    assert actual == expected


@pytest.mark.parametrize('native_type,sqla_type,attrs,generic_type,is_dttm',
    [('SMALLINT', types.SmallInteger, None, GenericDataType.NUMERIC, False),
    ('INTEGER', types.Integer, None, GenericDataType.NUMERIC, False), (
    'BIGINT', types.BigInteger, None, GenericDataType.NUMERIC, False), (
    'DECIMAL', types.Numeric, None, GenericDataType.NUMERIC, False), (
    'NUMERIC', types.Numeric, None, GenericDataType.NUMERIC, False), (
    'REAL', types.REAL, None, GenericDataType.NUMERIC, False), (
    'DOUBLE PRECISION', types.Float, None, GenericDataType.NUMERIC, False),
    ('MONEY', types.Numeric, None, GenericDataType.NUMERIC, False), ('CHAR',
    types.String, None, GenericDataType.STRING, False), ('VARCHAR', types.
    String, None, GenericDataType.STRING, False), ('TEXT', types.String,
    None, GenericDataType.STRING, False), ('DATE', types.Date, None,
    GenericDataType.TEMPORAL, True), ('TIMESTAMP', types.TIMESTAMP, None,
    GenericDataType.TEMPORAL, True), ('TIME', types.Time, None,
    GenericDataType.TEMPORAL, True), ('BOOLEAN', types.Boolean, None,
    GenericDataType.BOOLEAN, False)])
def func_sg7h8g9l(native_type, sqla_type, attrs, generic_type, is_dttm):
    from superset.db_engine_specs.databricks import DatabricksNativeEngineSpec as spec
    assert_column_spec(spec, native_type, sqla_type, attrs, generic_type,
        is_dttm)


@pytest.mark.parametrize('cols, expected_result', [([SQLAColumnType(name=
    'John', type='integer', is_dttm=False)], [ResultSetColumnType(
    column_name='John', name='John', type='integer', is_dttm=False)]), ([
    SQLAColumnType(name='hugh', type='integer', is_dttm=False)], [
    ResultSetColumnType(column_name='hugh', name='hugh', type='integer',
    is_dttm=False)])])
def func_bqdhjex2(cols, expected_result):
    from superset.db_engine_specs.base import convert_inspector_columns
    assert convert_inspector_columns(cols) == expected_result


def func_4992masv(mocker):
    """
    Test the ``select_star`` method.
    """
    from superset.db_engine_specs.base import BaseEngineSpec


    class NoLimitDBEngineSpec(BaseEngineSpec):
        allow_limit_clause = False
    cols = [{'column_name': 'a', 'name': 'a', 'type': sqltypes.String(),
        'nullable': True, 'comment': None, 'default': None, 'precision':
        None, 'scale': None, 'max_length': None, 'is_dttm': False}]
    database = mocker.MagicMock()
    database.compile_sqla_query = lambda query, catalog, schema: str(query.
        compile(dialect=sqlite.dialect()))
    engine = mocker.MagicMock()
    engine.dialect = sqlite.dialect()
    sql = BaseEngineSpec.select_star(database=database, table=Table(
        'my_table'), engine=engine, limit=100, show_cols=True, indent=True,
        latest_partition=False, cols=cols)
    assert sql == 'SELECT a\nFROM my_table\nLIMIT ?\nOFFSET ?'
    sql = NoLimitDBEngineSpec.select_star(database=database, table=Table(
        'my_table'), engine=engine, limit=100, show_cols=True, indent=True,
        latest_partition=False, cols=cols)
    assert sql == 'SELECT a\nFROM my_table'


def func_91osd5rh(mocker):
    """
    Test the deprecated `extra_table_metadata` method.
    """
    from superset.db_engine_specs.base import BaseEngineSpec
    from superset.models.core import Database


    class ThirdPartyDBEngineSpec(BaseEngineSpec):

        @classmethod
        def func_coxhj3yp(cls, database, table_name, schema_name):
            return {'table': table_name, 'schema': schema_name}
    database = mocker.MagicMock()
    warnings = mocker.patch('superset.db_engine_specs.base.warnings')
    assert ThirdPartyDBEngineSpec.get_extra_table_metadata(database, Table(
        'table', 'schema')) == {'table': 'table', 'schema': 'schema'}
    assert ThirdPartyDBEngineSpec.get_extra_table_metadata(database, Table(
        'table', 'schema', 'catalog')) == {}
    warnings.warn.assert_called()


def func_hdkd5wzi(mocker):
    """
    Test the `get_default_catalog` method.
    """
    from superset.db_engine_specs.base import BaseEngineSpec
    database = mocker.MagicMock()
    assert BaseEngineSpec.get_default_catalog(database) is None


def func_y9cx3bwp():
    """
    Test the `quote_table` function.
    """
    from superset.db_engine_specs.base import BaseEngineSpec
    dialect = sqlite.dialect()
    assert BaseEngineSpec.quote_table(Table('table'), dialect) == '"table"'
    assert BaseEngineSpec.quote_table(Table('table', 'schema'), dialect
        ) == 'schema."table"'
    assert BaseEngineSpec.quote_table(Table('table', 'schema', 'catalog'),
        dialect) == 'catalog.schema."table"'
    assert BaseEngineSpec.quote_table(Table('ta ble', 'sche.ma', 'cata"log'
        ), dialect) == '"cata""log"."sche.ma"."ta ble"'


def func_uzuiikuo():
    """
    Test that the private key is masked when the database is edited.
    """
    from superset.db_engine_specs.base import BaseEngineSpec
    config = json.dumps({'foo': 'bar', 'service_account_info': {
        'project_id': 'black-sanctum-314419', 'private_key': 'SECRET'}})
    assert BaseEngineSpec.mask_encrypted_extra(config) == json.dumps({'foo':
        'XXXXXXXXXX', 'service_account_info': 'XXXXXXXXXX'})


def func_we2ygmlu():
    """
    Test that the private key can be reused from the previous `encrypted_extra`.
    """
    from superset.db_engine_specs.base import BaseEngineSpec
    old = json.dumps({'foo': 'bar', 'service_account_info': {'project_id':
        'black-sanctum-314419', 'private_key': 'SECRET'}})
    new = json.dumps({'foo': 'XXXXXXXXXX', 'service_account_info':
        'XXXXXXXXXX'})
    assert BaseEngineSpec.unmask_encrypted_extra(old, new) == json.dumps({
        'foo': 'bar', 'service_account_info': {'project_id':
        'black-sanctum-314419', 'private_key': 'SECRET'}})
