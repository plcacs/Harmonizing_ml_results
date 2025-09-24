from __future__ import annotations
import re
from datetime import datetime
from typing import Any, Literal, NamedTuple, Optional, Union, Dict, List, Tuple, Set, cast
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

VIRTUAL_TABLE_INT_TYPES: Dict[str, Pattern[str]] = {
    'hive': re.compile('^INT_TYPE$'), 
    'mysql': re.compile('^LONGLONG$'), 
    'postgresql': re.compile('^INTEGER$'), 
    'presto': re.compile('^INTEGER$'), 
    'sqlite': re.compile('^INT$')
}
VIRTUAL_TABLE_STRING_TYPES: Dict[str, Pattern[str]] = {
    'hive': re.compile('^STRING_TYPE$'), 
    'mysql': re.compile('^VAR_STRING$'), 
    'postgresql': re.compile('^STRING$'), 
    'presto': re.compile('^VARCHAR*'), 
    'sqlite': re.compile('^STRING$')
}

class FilterTestCase(NamedTuple):
    column: str
    operator: str
    value: Union[float, int, List[Any], str]
    expected: Union[str, List[str]]

class TestDatabaseModel(SupersetTestCase):

    def test_is_time_druid_time_col(self) -> None:
        """Druid has a special __time column"""
        database: Database = Database(database_name='druid_db', sqlalchemy_uri='druid://db')
        tbl: SqlaTable = SqlaTable(table_name='druid_tbl', database=database)
        col: TableColumn = TableColumn(column_name='__time', type='INTEGER', table=tbl)
        assert col.is_dttm is None
        DruidEngineSpec.alter_new_orm_column(col)
        assert col.is_dttm is True
        col = TableColumn(column_name='__not_time', type='INTEGER', table=tbl)
        assert col.is_temporal is False

    def test_temporal_varchar(self) -> None:
        """Ensure a column with is_dttm set to true evaluates to is_temporal == True"""
        database: Database = get_example_database()
        tbl: SqlaTable = SqlaTable(table_name='test_tbl', database=database)
        col: TableColumn = TableColumn(column_name='ds', type='VARCHAR', table=tbl)
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
        tbl: SqlaTable = SqlaTable(table_name='col_type_test_tbl', database=get_example_database())
        for (str_type, db_col_type) in test_cases.items():
            col: TableColumn = TableColumn(column_name='foo', type=str_type, table=tbl)
            assert col.is_temporal == (db_col_type == GenericDataType.TEMPORAL)
            assert col.is_numeric == (db_col_type == GenericDataType.NUMERIC)
            assert col.is_string == (db_col_type == GenericDataType.STRING)
        for (str_type, db_col_type) in test_cases.items():
            col = TableColumn(column_name='foo', type=str_type, table=tbl, is_dttm=True)
            assert col.is_temporal

    @patch('superset.jinja_context.get_username', return_value='abc')
    def test_jinja_metrics_and_calc_columns(self, mock_username: Mock) -> None:
        base_query_obj: Dict[str, Any] = {
            'granularity': None, 
            'from_dttm': None, 
            'to_dttm': None, 
            'columns': ['user', 'expr', {
                'hasCustomLabel': True, 
                'label': 'adhoc_column', 
                'sqlExpression': "'{{ 'foo_' + time_grain }}'"
            }], 
            'metrics': [{
                'hasCustomLabel': True, 
                'label': 'adhoc_metric', 
                'expressionType': AdhocMetricExpressionType.SQL, 
                'sqlExpression': "SUM(case when user = '{{ 'user_' + current_username() }}' then 1 else 0 end)"
            }, 'count_timegrain'], 
            'is_timeseries': False, 
            'filter': [], 
            'extras': {'time_grain_sqla': 'P1D'}
        }
        table: SqlaTable = SqlaTable(
            table_name='test_has_jinja_metric_and_expr', 
            sql="SELECT '{{ 'user_' + current_username() }}' as user, '{{ 'xyz_' + time_grain }}' as time_grain", 
            database=get_example_database()
        )
        TableColumn(
            column_name='expr', 
            expression="case when '{{ current_username() }}' = 'abc' then 'yes' else 'no' end", 
            type='VARCHAR(100)', 
            table=table
        )
        SqlMetric(
            metric_name='count_timegrain', 
            expression="count('{{ 'bar_' + time_grain }}')", 
            table=table
        )
        db.session.commit()
        sqla_query: Any = table.get_sqla_query(**base_query_obj)
        query: str = table.database.compile_sqla_query(sqla_query.sqla_query)
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
        table: SqlaTable = self.get_table(name='birth_names')
        metric: SqlMetric = SqlMetric(metric_name='count_jinja_metric', expression='count(*)', table=table)
        db.session.commit()
        base_query_obj: Dict[str, Any] = {
            'granularity': None, 
            'from_dttm': None, 
            'to_dttm': None, 
            'columns': [], 
            'metrics': [{
                'hasCustomLabel': True, 
                'label': 'Metric using Jinja macro', 
                'expressionType': AdhocMetricExpressionType.SQL, 
                'sqlExpression': "{{ metric('count_jinja_metric') }}"
            }, {
                'hasCustomLabel': True, 
                'label': 'Same but different', 
                'expressionType': AdhocMetricExpressionType.SQL, 
                'sqlExpression': "{{ metric('count_jinja_metric', " + str(table.id) + ') }}'
            }], 
            'is_timeseries': False, 
            'filter': [], 
            'extras': {'time_grain_sqla': 'P1D'}
        }
        mock_dataset_id_from_context.return_value = table.id
        sqla_query: Any = table.get_sqla_query(**base_query_obj)
        query: str = table.database.compile_sqla_query(sqla_query.sqla_query)
        database: Database = table.database
        with database.get_sqla_engine() as engine:
            quote: Any = engine.dialect.identifier_preparer.quote_identifier
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
            'metrics': [{
                'expressionType': AdhocMetricExpressionType.SQL, 
                'sqlExpression': '(SELECT (SELECT * from birth_names) from test_validate_adhoc_sql)', 
                'label': 'adhoc_metrics'
            }], 
            'is_timeseries': False, 
            'filter': []
        }
        table: SqlaTable = SqlaTable(table_name='test_validate_adhoc_sql', database=get_example_database())
        db.session.commit()
        with pytest.raises(QueryObjectValidationError):
            table.get_sqla_query(**base_query_obj)
        db.session.delete(table)
        db.session.commit()

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_where_operators(self) -> None:
        filters: Tuple[FilterTestCase, ...] = (
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
        )
        table: SqlaTable = self.get_table(name='birth_names')
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
            sqla_query: Any = table.get_sqla_query(**query_obj)
            sql: str = table.database.compile_sqla_query(sqla_query.sqla_query)
            if isinstance(filter_.expected, list):
                assert any([candidate in sql for candidate in filter_.expected])
            else:
                assert filter_.expected in sql

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_boolean_type_where_operators(self) -> None:
        table: SqlaTable = self.get_table(name='birth_names')
        db.session.add(TableColumn(
            column_name='boolean_gender', 
            expression="case when gender = 'boy' then True else False end", 
            type='BOOLEAN', 
            table=table
        ))
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
        sqla_query: Any = table.get_sqla_query(**query_obj)
        sql: str = table.database.compile_sqla_query(sqla_query.sqla_query)
        dialect: Any = table.database.get_dialect()
        operand: str = '(true, false)'
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
        table: SqlaTable = SqlaTable(
            table_name='test_table', 
            sql="SELECT '{{ abcd xyz + 1 ASDF }}' as user", 
            database=get_example_database()
        )
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
        table: SqlaTable = SqlaTable(
            table_name='another_test_table', 
            sql='SELECT * from test_table;', 
            database=get_example_database()
        )
        sqlaq: Any = table.get_sqla_query(**query_obj)
        sql: str = table.database.compile_sqla_query(sqlaq.sqla_query)
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
        table: SqlaTable = SqlaTable(
            table_name='test_multiple_sql_statements', 
            sql="SELECT 'foo' as grp, 1 as num; SELECT 'bar' as grp, 2 as num", 
            database=get_example_database()
        )
        query_obj: Dict[str, Any] = dict(**base_query_obj, extras={})
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
        table: SqlaTable = SqlaTable(
            table_name='test_dml_statement', 
            sql='DELETE FROM foo', 
            database=get_example_database()
        )
        query_obj: Dict[str, Any] = dict(**base_query_obj, extras={})
        with pytest.raises(QueryObjectValidationError):
            table.get_sqla_query(**query_obj)

    def test_fetch_metadata_for_updated_virtual_table(self) -> None:
        table: SqlaTable = SqlaTable(
            table_name='updated_sql_table', 
            database=get_example_database(), 
            sql="select 123 as intcol, 'abc' as strcol, 'abc' as mycase"
        )
        TableColumn(column_name='intcol', type='FLOAT',