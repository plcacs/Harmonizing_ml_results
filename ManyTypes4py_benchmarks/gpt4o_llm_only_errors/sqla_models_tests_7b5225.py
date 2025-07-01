from __future__ import annotations
import re
from datetime import datetime
from typing import Any, Literal, NamedTuple, Optional, Union, List, Dict
from re import Pattern
from unittest.mock import Mock, patch
import pytest
import numpy as np
import pandas as pd
from flask.ctx import AppContext
from pytest_mock import MockerFixture
from sqlalchemy.sql import text
from sqlalchemy.sql.elements import TextClause
from superset import db
from superset.connectors.sqla.models import SqlaTable, TableColumn, SqlMetric
from superset.constants import EMPTY_STRING, NULL_STRING
from superset.db_engine_specs.bigquery import BigQueryEngineSpec
from superset.db_engine_specs.druid import DruidEngineSpec
from superset.exceptions import QueryObjectValidationError
from superset.models.core import Database
from superset.utils.core import AdhocMetricExpressionType, FilterOperator, GenericDataType
from superset.utils.database import get_example_database
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from .base_tests import SupersetTestCase
from .conftest import only_postgresql

VIRTUAL_TABLE_INT_TYPES: Dict[str, Pattern] = {
    'hive': re.compile('^INT_TYPE$'),
    'mysql': re.compile('^LONGLONG$'),
    'postgresql': re.compile('^INTEGER$'),
    'presto': re.compile('^INTEGER$'),
    'sqlite': re.compile('^INT$')
}

VIRTUAL_TABLE_STRING_TYPES: Dict[str, Pattern] = {
    'hive': re.compile('^STRING_TYPE$'),
    'mysql': re.compile('^VAR_STRING$'),
    'postgresql': re.compile('^STRING$'),
    'presto': re.compile('^VARCHAR*'),
    'sqlite': re.compile('^STRING$')
}

class FilterTestCase(NamedTuple):
    column: str
    operator: FilterOperator
    value: Any
    expected: Union[str, List[str]]

class TestDatabaseModel(SupersetTestCase):

    def test_is_time_druid_time_col(self) -> None:
        """Druid has a special __time column"""
        database = Database(database_name='druid_db', sqlalchemy_uri='druid://db')
        tbl = SqlaTable(table_name='druid_tbl', database=database)
        col = TableColumn(column_name='__time', type='INTEGER', table=tbl)
        assert col.is_dttm is None
        DruidEngineSpec.alter_new_orm_column(col)
        assert col.is_dttm is True
        col = TableColumn(column_name='__not_time', type='INTEGER', table=tbl)
        assert col.is_temporal is False

    def test_temporal_varchar(self) -> None:
        """Ensure a column with is_dttm set to true evaluates to is_temporal == True"""
        database = get_example_database()
        tbl = SqlaTable(table_name='test_tbl', database=database)
        col = TableColumn(column_name='ds', type='VARCHAR', table=tbl)
        assert col.is_temporal is False
        col.is_dttm = True
        assert col.is_temporal is True

    def test_db_column_types(self) -> None:
        test_cases: Dict[str, GenericDataType] = {
            'CHAR': GenericDataType.STRING,
            'VARCHAR': GenericDataType.STRING,
            'NVARCHAR': GenericDataType.STRING,
            'STRING': GenericDataType.STRING,
            'TEXT': GenericDataType.STRING,
            'NTEXT': GenericDataType.STRING,
            'INTEGER': GenericDataType.NUMERIC,
            'BIGINT': GenericDataType.NUMERIC,
            'DECIMAL': GenericDataType.NUMERIC,
            'DATE': GenericDataType.TEMPORAL,
            'DATETIME': GenericDataType.TEMPORAL,
            'TIME': GenericDataType.TEMPORAL,
            'TIMESTAMP': GenericDataType.TEMPORAL
        }
        tbl = SqlaTable(table_name='col_type_test_tbl', database=get_example_database())
        for str_type, db_col_type in test_cases.items():
            col = TableColumn(column_name='foo', type=str_type, table=tbl)
            assert col.is_temporal == (db_col_type == GenericDataType.TEMPORAL)
            assert col.is_numeric == (db_col_type == GenericDataType.NUMERIC)
            assert col.is_string == (db_col_type == GenericDataType.STRING)
        for str_type, db_col_type in test_cases.items():
            col = TableColumn(column_name='foo', type=str_type, table=tbl, is_dttm=True)
            assert col.is_temporal

    @patch('superset.jinja_context.get_username', return_value='abc')
    def test_jinja_metrics_and_calc_columns(self, mock_username: Mock) -> None:
        base_query_obj: Dict[str, Any] = {
            'granularity': None,
            'from_dttm': None,
            'to_dttm': None,
            'columns': ['user', 'expr', {'hasCustomLabel': True, 'label': 'adhoc_column', 'sqlExpression': "'{{ 'foo_' + time_grain }}'"}],
            'metrics': [{'hasCustomLabel': True, 'label': 'adhoc_metric', 'expressionType': AdhocMetricExpressionType.SQL, 'sqlExpression': "SUM(case when user = '{{ 'user_' + current_username() }}' then 1 else 0 end)"}, 'count_timegrain'],
            'is_timeseries': False,
            'filter': [],
            'extras': {'time_grain_sqla': 'P1D'}
        }
        table = SqlaTable(table_name='test_has_jinja_metric_and_expr', sql="SELECT '{{ 'user_' + current_username() }}' as user, '{{ 'xyz_' + time_grain }}' as time_grain", database=get_example_database())
        TableColumn(column_name='expr', expression="case when '{{ current_username() }}' = 'abc' then 'yes' else 'no' end", type='VARCHAR(100)', table=table)
        SqlMetric(metric_name='count_timegrain', expression="count('{{ 'bar_' + time_grain }}')", table=table)
        db.session.commit()
        sqla_query = table.get_sqla_query(**base_query_obj)
        query = table.database.compile_sqla_query(sqla_query.sqla_query)
        assert "SELECT 'user_abc' as user, 'xyz_P1D' as time_grain" in query
        assert "case when 'abc' = 'abc' then 'yes' else 'no' end" in query
        assert "'foo_P1D'" in query
        assert "count('bar_P1D')" in query
        assert "SUM(case when user = 'user_abc' then 1 else 0 end)" in query
        db.session.delete(table)
        db.session.commit()

    @patch('superset.jinja_context.get_dataset_id_from_context')
    def test_jinja_metric_macro(self, mock_dataset_id_from_context: Mock) -> None:
        self.login(username='admin')
        table = self.get_table(name='birth_names')
        metric = SqlMetric(metric_name='count_jinja_metric', expression='count(*)', table=table)
        db.session.commit()
        base_query_obj: Dict[str, Any] = {
            'granularity': None,
            'from_dttm': None,
            'to_dttm': None,
            'columns': [],
            'metrics': [{'hasCustomLabel': True, 'label': 'Metric using Jinja macro', 'expressionType': AdhocMetricExpressionType.SQL, 'sqlExpression': "{{ metric('count_jinja_metric') }}"}, {'hasCustomLabel': True, 'label': 'Same but different', 'expressionType': AdhocMetricExpressionType.SQL, 'sqlExpression': "{{ metric('count_jinja_metric', " + str(table.id) + ') }}'}],
            'is_timeseries': False,
            'filter': [],
            'extras': {'time_grain_sqla': 'P1D'}
        }
        mock_dataset_id_from_context.return_value = table.id
        sqla_query = table.get_sqla_query(**base_query_obj)
        query = table.database.compile_sqla_query(sqla_query.sqla_query)
        database = table.database
        with database.get_sqla_engine() as engine:
            quote = engine.dialect.identifier_preparer.quote_identifier
        for metric_label in {'metric using jinja macro', 'same but different'}:
            assert f'count(*) as {quote(metric_label)}' in query.lower()
        db.session.delete(metric)
        db.session.commit()

    def test_adhoc_metrics_and_calc_columns(self) -> None:
        base_query_obj: Dict[str, Any] = {
            'granularity': None,
            'from_dttm': None,
            'to_dttm': None,
            'groupby': ['user', 'expr'],
            'metrics': [{'expressionType': AdhocMetricExpressionType.SQL, 'sqlExpression': '(SELECT (SELECT * from birth_names) from test_validate_adhoc_sql)', 'label': 'adhoc_metrics'}],
            'is_timeseries': False,
            'filter': []
        }
        table = SqlaTable(table_name='test_validate_adhoc_sql', database=get_example_database())
        db.session.commit()
        with pytest.raises(QueryObjectValidationError):
            table.get_sqla_query(**base_query_obj)
        db.session.delete(table)
        db.session.commit()

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_where_operators(self) -> None:
        filters: List[FilterTestCase] = [
            FilterTestCase('num', FilterOperator.IS_NULL, '', 'IS NULL'),
            FilterTestCase('num', FilterOperator.IS_NOT_NULL, '', 'IS NOT NULL'),
            FilterTestCase('num', FilterOperator.IS_TRUE, '', ['IS 1', 'IS true']),
            FilterTestCase('num', FilterOperator.IS_FALSE, '', ['IS 0', 'IS false']),
            FilterTestCase('num', FilterOperator.GREATER_THAN, 0, '> 0'),
            FilterTestCase('num', FilterOperator.GREATER_THAN_OR_EQUALS, 0, '>= 0'),
            FilterTestCase('num', FilterOperator.LESS_THAN, 0, '< 0'),
            FilterTestCase('num', FilterOperator.LESS_THAN_OR_EQUALS, 0, '<= 0'),
            FilterTestCase('num', FilterOperator.EQUALS, 0, '= 0'),
            FilterTestCase('num', FilterOperator.NOT_EQUALS, 0, '!= 0'),
            FilterTestCase('num', FilterOperator.IN, ['1', '2'], 'IN (1, 2)'),
            FilterTestCase('num', FilterOperator.NOT_IN, ['1', '2'], 'NOT IN (1, 2)'),
            FilterTestCase('ds', FilterOperator.TEMPORAL_RANGE, '2020 : 2021', '2020-01-01')
        ]
        table = self.get_table(name='birth_names')
        for filter_ in filters:
            query_obj: Dict[str, Any] = {
                'granularity': None,
                'from_dttm': None,
                'to_dttm': None,
                'groupby': ['gender'],
                'metrics': ['count'],
                'is_timeseries': False,
                'filter': [{'col': filter_.column, 'op': filter_.operator, 'val': filter_.value}],
                'extras': {}
            }
            sqla_query = table.get_sqla_query(**query_obj)
            sql = table.database.compile_sqla_query(sqla_query.sqla_query)
            if isinstance(filter_.expected, list):
                assert any([candidate in sql for candidate in filter_.expected])
            else:
                assert filter_.expected in sql

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_boolean_type_where_operators(self) -> None:
        table = self.get_table(name='birth_names')
        db.session.add(TableColumn(column_name='boolean_gender', expression="case when gender = 'boy' then True else False end", type='BOOLEAN', table=table))
        query_obj: Dict[str, Any] = {
            'granularity': None,
            'from_dttm': None,
            'to_dttm': None,
            'groupby': ['boolean_gender'],
            'metrics': ['count'],
            'is_timeseries': False,
            'filter': [{'col': 'boolean_gender', 'op': FilterOperator.IN, 'val': ['true', 'false']}],
            'extras': {}
        }
        sqla_query = table.get_sqla_query(**query_obj)
        sql = table.database.compile_sqla_query(sqla_query.sqla_query)
        dialect = table.database.get_dialect()
        operand = '(true, false)'
        if not dialect.supports_native_boolean and dialect.name != 'mysql':
            operand = '(1, 0)'
        assert f'IN {operand}' in sql

    def test_incorrect_jinja_syntax_raises_correct_exception(self) -> None:
        query_obj: Dict[str, Any] = {
            'granularity': None,
            'from_dttm': None,
            'to_dttm': None,
            'groupby': ['user'],
            'metrics': [],
            'is_timeseries': False,
            'filter': [],
            'extras': {}
        }
        table = SqlaTable(table_name='test_table', sql="SELECT '{{ abcd xyz + 1 ASDF }}' as user", database=get_example_database())
        if get_example_database().backend != 'presto':
            with pytest.raises(QueryObjectValidationError):
                table.get_sqla_query(**query_obj)

    def test_query_format_strip_trailing_semicolon(self) -> None:
        query_obj: Dict[str, Any] = {
            'granularity': None,
            'from_dttm': None,
            'to_dttm': None,
            'groupby': ['user'],
            'metrics': [],
            'is_timeseries': False,
            'filter': [],
            'extras': {}
        }
        table = SqlaTable(table_name='another_test_table', sql='SELECT * from test_table;', database=get_example_database())
        sqlaq = table.get_sqla_query(**query_obj)
        sql = table.database.compile_sqla_query(sqlaq.sqla_query)
        assert sql[-1] != ';'

    def test_multiple_sql_statements_raises_exception(self) -> None:
        base_query_obj: Dict[str, Any] = {
            'granularity': None,
            'from_dttm': None,
            'to_dttm': None,
            'groupby': ['grp'],
            'metrics': [],
            'is_timeseries': False,
            'filter': []
        }
        table = SqlaTable(table_name='test_multiple_sql_statements', sql="SELECT 'foo' as grp, 1 as num; SELECT 'bar' as grp, 2 as num", database=get_example_database())
        query_obj = dict(**base_query_obj, extras={})
        with pytest.raises(QueryObjectValidationError):
            table.get_sqla_query(**query_obj)

    def test_dml_statement_raises_exception(self) -> None:
        base_query_obj: Dict[str, Any] = {
            'granularity': None,
            'from_dttm': None,
            'to_dttm': None,
            'groupby': ['grp'],
            'metrics': [],
            'is_timeseries': False,
            'filter': []
        }
        table = SqlaTable(table_name='test_dml_statement', sql='DELETE FROM foo', database=get_example_database())
        query_obj = dict(**base_query_obj, extras={})
        with pytest.raises(QueryObjectValidationError):
            table.get_sqla_query(**query_obj)

    def test_fetch_metadata_for_updated_virtual_table(self) -> None:
        table = SqlaTable(table_name='updated_sql_table', database=get_example_database(), sql="select 123 as intcol, 'abc' as strcol, 'abc' as mycase")
        TableColumn(column_name='intcol', type='FLOAT', table=table)
        TableColumn(column_name='oldcol', type='INT', table=table)
        TableColumn(column_name='expr', expression='case when 1 then 1 else 0 end', type='INT', table=table)
        TableColumn(column_name='mycase', expression='case when 1 then 1 else 0 end', type='INT', table=table)
        assert len(table.columns) == 4
        table.fetch_metadata()
        assert {col.column_name for col in table.columns} == {'intcol', 'strcol', 'mycase', 'expr'}
        cols = {col.column_name: col for col in table.columns}
        backend = table.database.backend
        assert VIRTUAL_TABLE_INT_TYPES[backend].match(cols['intcol'].type)
        assert cols['mycase'].expression == ''
        assert VIRTUAL_TABLE_STRING_TYPES[backend].match(cols['mycase'].type)
        assert cols['expr'].expression == 'case when 1 then 1 else 0 end'

    @patch('superset.models.core.Database.db_engine_spec', BigQueryEngineSpec)
    def test_labels_expected_on_mutated_query(self) -> None:
        query_obj: Dict[str, Any] = {
            'granularity': None,
            'from_dttm': None,
            'to_dttm': None,
            'groupby': ['user'],
            'metrics': [{'expressionType': 'SIMPLE', 'column': {'column_name': 'user'}, 'aggregate': 'COUNT_DISTINCT', 'label': 'COUNT_DISTINCT(user)'}],
            'is_timeseries': False,
            'filter': [],
            'extras': {}
        }
        database = Database(database_name='testdb', sqlalchemy_uri='sqlite://')
        table = SqlaTable(table_name='bq_table', database=database)
        db.session.add(database)
        db.session.add(table)
        db.session.commit()
        sqlaq = table.get_sqla_query(**query_obj)
        assert sqlaq.labels_expected == ['user', 'COUNT_DISTINCT(user)']
        sql = table.database.compile_sqla_query(sqlaq.sqla_query)
        assert 'COUNT_DISTINCT_user__00db1' in sql
        db.session.delete(table)
        db.session.delete(database)
        db.session.commit()

@pytest.fixture
def text_column_table(app_context: AppContext) -> SqlaTable:
    table = SqlaTable(
        table_name='text_column_table',
        sql=(
            "SELECT 'foo' as foo UNION SELECT '' UNION SELECT NULL UNION SELECT 'null' "
            "UNION SELECT '\"text in double quotes\"' UNION SELECT '''text in single quotes''' "
            "UNION SELECT 'double quotes \" in text' UNION SELECT 'single quotes '' in text' "
        ),
        database=get_example_database()
    )
    TableColumn(column_name='foo', type='VARCHAR(255)', table=table)
    SqlMetric(metric_name='count', expression='count(*)', table=table)
    return table

def test_values_for_column_on_text_column(text_column_table: SqlaTable) -> None:
    with_null = text_column_table.values_for_column(column_name='foo', limit=10000)
    assert None in with_null
    assert len(with_null) == 8

def test_values_for_column_on_text_column_with_rls(text_column_table: SqlaTable) -> None:
    with patch.object(text_column_table, 'get_sqla_row_level_filters', return_value=[TextClause("foo = 'foo'")]):
        with_rls = text_column_table.values_for_column(column_name='foo', limit=10000)
        assert with_rls == ['foo']
        assert len(with_rls) == 1

def test_values_for_column_on_text_column_with_rls_no_values(text_column_table: SqlaTable) -> None:
    with patch.object(text_column_table, 'get_sqla_row_level_filters', return_value=[TextClause("foo = 'bar'")]):
        with_rls = text_column_table.values_for_column(column_name='foo', limit=10000)
        assert with_rls == []
        assert len(with_rls) == 0

def test_filter_on_text_column(text_column_table: SqlaTable) -> None:
    table = text_column_table
    result_object = table.query({'metrics': ['count'], 'filter': [{'col': 'foo', 'val': [NULL_STRING], 'op': 'IN'}], 'is_timeseries': False})
    assert result_object.df['count'][0] == 1
    result_object = table.query({'metrics': ['count'], 'filter': [{'col': 'foo', 'val': [None], 'op': 'IN'}], 'is_timeseries': False})
    assert result_object.df['count'][0] == 1
    result_object = table.query({'metrics': ['count'], 'filter': [{'col': 'foo', 'val': [EMPTY_STRING], 'op': 'IN'}], 'is_timeseries': False})
    assert result_object.df['count'][0] == 1
    result_object = table.query({'metrics': ['count'], 'filter': [{'col': 'foo', 'val': [''], 'op': 'IN'}], 'is_timeseries': False})
    assert result_object.df['count'][0] == 1
    result_object = table.query({'metrics': ['count'], 'filter': [{'col': 'foo', 'val': [EMPTY_STRING, NULL_STRING, 'null', 'foo'], 'op': 'IN'}], 'is_timeseries': False})
    assert result_object.df['count'][0] == 4
    result_object = table.query({'metrics': ['count'], 'filter': [{'col': 'foo', 'val': ['"text in double quotes"'], 'op': 'IN'}], 'is_timeseries': False})
    assert result_object.df['count'][0] == 1
    result_object = table.query({'metrics': ['count'], 'filter': [{'col': 'foo', 'val': ["'text in single quotes'"], 'op': 'IN'}], 'is_timeseries': False})
    assert result_object.df['count'][0] == 1
    result_object = table.query({'metrics': ['count'], 'filter': [{'col': 'foo', 'val': ['double quotes " in text'], 'op': 'IN'}], 'is_timeseries': False})
    assert result_object.df['count'][0] == 1
    result_object = table.query({'metrics': ['count'], 'filter': [{'col': 'foo', 'val': ["single quotes ' in text"], 'op': 'IN'}], 'is_timeseries': False})
    assert result_object.df['count'][0] == 1

@only_postgresql
def test_should_generate_closed_and_open_time_filter_range(login_as_admin: None) -> None:
    table = SqlaTable(
        table_name='temporal_column_table',
        sql=(
            "SELECT '2021-12-31'::timestamp as datetime_col UNION SELECT '2022-01-01'::timestamp "
            "UNION SELECT '2022-03-10'::timestamp UNION SELECT '2023-01-01'::timestamp "
            "UNION SELECT '2023-03-10'::timestamp "
        ),
        database=get_example_database()
    )
    TableColumn(column_name='datetime_col', type='TIMESTAMP', table=table, is_dttm=True)
    SqlMetric(metric_name='count', expression='count(*)', table=table)
    result_object = table.query({
        'metrics': ['count'],
        'is_timeseries': False,
        'filter': [],
        'from_dttm': datetime(2022, 1, 1),
        'to_dttm': datetime(2023, 1, 1),
        'granularity': 'datetime_col'
    })
    assert result_object.df.iloc[0]['count'] == 2

def test_none_operand_in_filter(login_as_admin: None, physical_dataset: SqlaTable) -> None:
    expected_results = [
        {'operator': FilterOperator.EQUALS.value, 'count': 10, 'sql_should_contain': 'COL4 IS NULL'},
        {'operator': FilterOperator.NOT_EQUALS.value, 'count': 0, 'sql_should_contain': 'COL4 IS NOT NULL'}
    ]
    for expected in expected_results:
        result = physical_dataset.query({
            'metrics': ['count'],
            'filter': [{'col': 'col4', 'val': None, 'op': expected['operator']}],
            'is_timeseries': False
        })
        assert result.df['count'][0] == expected['count']
        assert expected['sql_should_contain'] in result.query.upper()
    with pytest.raises(QueryObjectValidationError):
        for flt in [FilterOperator.GREATER_THAN, FilterOperator.LESS_THAN, FilterOperator.GREATER_THAN_OR_EQUALS, FilterOperator.LESS_THAN_OR_EQUALS, FilterOperator.LIKE, FilterOperator.ILIKE]:
            physical_dataset.query({
                'metrics': ['count'],
                'filter': [{'col': 'col4', 'val': None, 'op': flt.value}],
                'is_timeseries': False
            })

@pytest.mark.usefixtures('app_context')
@pytest.mark.parametrize('table_name,sql,expected_cache_keys,has_extra_cache_keys', [
    ('test_has_extra_cache_keys_table', "\n            SELECT\n            '{{ current_user_id() }}' as id,\n            '{{ current_username() }}' as username,\n            '{{ current_user_email() }}' as email\n            ", {1, 'abc', 'abc@test.com'}, True),
    ('test_has_extra_cache_keys_table_with_set', "\n            {% set user_email = current_user_email() %}\n            SELECT\n            '{{ current_user_id() }}' as id,\n            '{{ current_username() }}' as username,\n            '{{ user_email }}' as email\n            ", {1, 'abc', 'abc@test.com'}, True),
    ('test_has_extra_cache_keys_table_with_se_multiple', "\n            {% set user_conditional_id = current_user_email() and current_user_id() %}\n            SELECT\n            '{{ user_conditional_id }}' as conditional\n            ", {1, 'abc@test.com'}, True),
    ('test_has_extra_cache_keys_disabled_table', "\n            SELECT\n            '{{ current_user_id(False) }}' as id,\n            '{{ current_username(False) }}' as username,\n            '{{ current_user_email(False) }}' as email\n            ", [], True),
    ('test_has_no_extra_cache_keys_table', "SELECT 'abc' as user", [], False)
])
@patch('superset.jinja_context.get_user_id', return_value=1)
@patch('superset.jinja_context.get_username', return_value='abc')
@patch('superset.jinja_context.get_user_email', return_value='abc@test.com')
def test_extra_cache_keys(mock_user_email: Mock, mock_username: Mock, mock_user_id: Mock, table_name: str, sql: str, expected_cache_keys: List[Any], has_extra_cache_keys: bool) -> None:
    table = SqlaTable(table_name=table_name, sql=sql, database=get_example_database())
    base_query_obj: Dict[str, Any] = {
        'granularity': None,
        'from_dttm': None,
        'to_dttm': None,
        'groupby': ['id', 'username', 'email'],
        'metrics': [],
        'is_timeseries': False,
        'filter': []
    }
    query_obj = dict(**base_query_obj, extras={})
    extra_cache_keys = table.get_extra_cache_keys(query_obj)
    assert table.has_extra_cache_key_calls(query_obj) == has_extra_cache_keys
    assert set(extra_cache_keys) == set(expected_cache_keys)

@pytest.mark.usefixtures('app_context')
@pytest.mark.parametrize('sql_expression,expected_cache_keys,has_extra_cache_keys', [
    ("(user != '{{ current_username() }}')", ['abc'], True),
    ("(user != 'abc')", [], False)
])
@patch('superset.jinja_context.get_user_id', return_value=1)
@patch('superset.jinja_context.get_username', return_value='abc')
@patch('superset.jinja_context.get_user_email', return_value='abc@test.com')
def test_extra_cache_keys_in_sql_expression(mock_user_email: Mock, mock_username: Mock, mock_user_id: Mock, sql_expression: str, expected_cache_keys: List[Any], has_extra_cache_keys: bool) -> None:
    table = SqlaTable(table_name='test_has_no_extra_cache_keys_table', sql="SELECT 'abc' as user", database=get_example_database())
    base_query_obj: Dict[str, Any] = {
        'granularity': None,
        'from_dttm': None,
        'to_dttm': None,
        'groupby': ['id', 'username', 'email'],
        'metrics': [],
        'is_timeseries': False,
        'filter': []
    }
    query_obj = dict(**base_query_obj, extras={'where': sql_expression})
    extra_cache_keys = table.get_extra_cache_keys(query_obj)
    assert table.has_extra_cache_key_calls(query_obj) == has_extra_cache_keys
    assert extra_cache_keys == expected_cache_keys

@pytest.mark.usefixtures('app_context')
@pytest.mark.parametrize('sql_expression,expected_cache_keys,has_extra_cache_keys,item_type', [
    ("'{{ current_username() }}'", ['abc'], True, 'columns'),
    ("(user != 'abc')", [], False, 'columns'),
    ('{{ current_user_id() }}', [1], True, 'metrics'),
    ('COUNT(*)', [], False, 'metrics')
])
@patch('superset.jinja_context.get_user_id', return_value=1)
@patch('superset.jinja_context.get_username', return_value='abc')
def test_extra_cache_keys_in_adhoc_metrics_and_columns(mock_username: Mock, mock_user_id: Mock, sql_expression: str, expected_cache_keys: List[Any], has_extra_cache_keys: bool, item_type: str) -> None:
    table = SqlaTable(table_name='test_has_no_extra_cache_keys_table', sql="SELECT 'abc' as user", database=get_example_database())
    base_query_obj: Dict[str, Any] = {
        'granularity': None,
        'from_dttm': None,
        'to_dttm': None,
        'groupby': [],
        'metrics': [],
        'columns': [],
        'is_timeseries': False,
        'filter': []
    }
    items = {item_type: [{'label': None, 'expressionType': 'SQL', 'sqlExpression': sql_expression}]}
    query_obj = {**base_query_obj, **items}
    extra_cache_keys = table.get_extra_cache_keys(query_obj)
    assert table.has_extra_cache_key_calls(query_obj) == has_extra_cache_keys
    assert extra_cache_keys == expected_cache_keys

@pytest.mark.usefixtures('app_context')
@patch('superset.jinja_context.get_user_id', return_value=1)
@patch('superset.jinja_context.get_username', return_value='abc')
def test_extra_cache_keys_in_dataset_metrics_and_columns(mock_username: Mock, mock_user_id: Mock) -> None:
    table = SqlaTable(
        table_name='test_has_no_extra_cache_keys_table',
        sql="SELECT 'abc' as user",
        database=get_example_database(),
        columns=[
            TableColumn(column_name='user', type='VARCHAR(255)'),
            TableColumn(column_name='username', type='VARCHAR(255)', expression='{{ current_username() }}')
        ],
        metrics=[
            SqlMetric(metric_name='variable_profit', expression="SUM(price) * {{ url_param('multiplier') }}")
        ]
    )
    query_obj = {
        'granularity': None,
        'from_dttm': None,
        'to_dttm': None,
        'groupby': [],
        'columns': ['username'],
        'metrics': ['variable_profit'],
        'is_timeseries': False,
        'filter': []
    }
    extra_cache_keys = table.get_extra_cache_keys(query_obj)
    assert table.has_extra_cache_key_calls(query_obj) is True
    assert set(extra_cache_keys) == {'abc', None}

@pytest.mark.usefixtures('app_context')
@pytest.mark.parametrize('row,dimension,result', [
    (pd.Series({'foo': 'abc'}), 'foo', 'abc'),
    (pd.Series({'bar': True}), 'bar', True),
    (pd.Series({'baz': 123}), 'baz', 123),
    (pd.Series({'baz': np.int16(123)}), 'baz', 123),
    (pd.Series({'baz': np.uint32(123)}), 'baz', 123),
    (pd.Series({'baz': np.int64(123)}), 'baz', 123),
    (pd.Series({'qux': 123.456}), 'qux', 123.456),
    (pd.Series({'qux': np.float32(123.456)}), 'qux', 123.45600128173828),
    (pd.Series({'qux': np.float64(123.456)}), 'qux', 123.456),
    (pd.Series({'quux': '2021-01-01'}), 'quux', '2021-01-01'),
    (pd.Series({'quuz': '2021-01-01T00:00:00'}), 'quuz', text("TIME_PARSE('2021-01-01T00:00:00')"))
])
def test__normalize_prequery_result_type(mocker: MockerFixture, row: pd.Series, dimension: str, result: Any) -> None:

    def _convert_dttm(target_type: str, dttm: datetime, db_extra: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if target_type.upper() == 'TIMESTAMP':
            return f"TIME_PARSE('{dttm.isoformat(timespec='seconds')}')"
        return None

    table = SqlaTable(table_name='foobar', database=get_example_database())
    mocker.patch.object(table.db_engine_spec, 'convert_dttm', new=_convert_dttm)
    columns_by_name: Dict[str, TableColumn] = {
        'foo': TableColumn(column_name='foo', is_dttm=False, table=table, type='STRING'),
        'bar': TableColumn(column_name='bar', is_dttm=False, table=table, type='BOOLEAN'),
        'baz': TableColumn(column_name='baz', is_dttm=False, table=table, type='INTEGER'),
        'qux': TableColumn(column_name='qux', is_dttm=False, table=table, type='FLOAT'),
        'quux': TableColumn(column_name='quuz', is_dttm=True, table=table, type='STRING'),
        'quuz': TableColumn(column_name='quux', is_dttm=True, table=table, type='TIMESTAMP')
    }
    normalized = table._normalize_prequery_result_type(row, dimension, columns_by_name)
    assert isinstance(normalized, type(result))
    if isinstance(normalized, TextClause):
        assert str(normalized) == str(result)
    else:
        assert normalized == result

@pytest.mark.usefixtures('app_context')
def test__temporal_range_operator_in_adhoc_filter(physical_dataset: SqlaTable) -> None:
    result = physical_dataset.query({
        'columns': ['col1', 'col2'],
        'filter': [
            {'col': 'col5', 'val': '2000-01-05 : 2000-01-06', 'op': FilterOperator.TEMPORAL_RANGE.value},
            {'col': 'col6', 'val': '2002-05-11 : 2002-05-12', 'op': FilterOperator.TEMPORAL_RANGE.value}
        ],
        'is_timeseries': False
    })
    df = pd.DataFrame(index=[0], data={'col1': 4, 'col2': 'e'})
    assert df.equals(result.df)
