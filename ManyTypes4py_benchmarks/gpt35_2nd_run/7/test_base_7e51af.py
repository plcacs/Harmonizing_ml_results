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

def test_get_text_clause_with_colon() -> None:
    ...

def test_parse_sql_single_statement() -> None:
    ...

def test_parse_sql_multi_statement() -> None:
    ...

def test_validate_db_uri(mocker: MockerFixture) -> None:
    ...

@pytest.mark.parametrize('original,expected', [(dedent("\nwith currency as\n(\nselect 'INR' as cur\n)\nselect * from currency\n"), None), ('SELECT 1 as cnt', None), (dedent("\nselect 'INR' as cur\nunion\nselect 'AUD' as cur\nunion\nselect 'USD' as cur\n"), None)])
def test_cte_query_parsing(original: str, expected: Any) -> None:
    ...

@pytest.mark.parametrize('native_type,sqla_type,attrs,generic_type,is_dttm', [('SMALLINT', types.SmallInteger, None, GenericDataType.NUMERIC, False), ('INTEGER', types.Integer, None, GenericDataType.NUMERIC, False), ('BIGINT', types.BigInteger, None, GenericDataType.NUMERIC, False), ('DECIMAL', types.Numeric, None, GenericDataType.NUMERIC, False), ('NUMERIC', types.Numeric, None, GenericDataType.NUMERIC, False), ('REAL', types.REAL, None, GenericDataType.NUMERIC, False), ('DOUBLE PRECISION', types.Float, None, GenericDataType.NUMERIC, False), ('MONEY', types.Numeric, None, GenericDataType.NUMERIC, False), ('CHAR', types.String, None, GenericDataType.STRING, False), ('VARCHAR', types.String, None, GenericDataType.STRING, False), ('TEXT', types.String, None, GenericDataType.STRING, False), ('DATE', types.Date, None, GenericDataType.TEMPORAL, True), ('TIMESTAMP', types.TIMESTAMP, None, GenericDataType.TEMPORAL, True), ('TIME', types.Time, None, GenericDataType.TEMPORAL, True), ('BOOLEAN', types.Boolean, None, GenericDataType.BOOLEAN, False)])
def test_get_column_spec(native_type: str, sqla_type: types.TypeEngine, attrs: Any, generic_type: GenericDataType, is_dttm: bool) -> None:
    ...

@pytest.mark.parametrize('cols, expected_result', [([SQLAColumnType(name='John', type='integer', is_dttm=False)], [ResultSetColumnType(column_name='John', name='John', type='integer', is_dttm=False)]), ([SQLAColumnType(name='hugh', type='integer', is_dttm=False)], [ResultSetColumnType(column_name='hugh', name='hugh', type='integer', is_dttm=False)])])
def test_convert_inspector_columns(cols: List[SQLAColumnType], expected_result: List[ResultSetColumnType]) -> None:
    ...

def test_select_star(mocker: MockerFixture) -> None:
    ...

def test_extra_table_metadata(mocker: MockerFixture) -> None:
    ...

def test_get_default_catalog(mocker: MockerFixture) -> None:
    ...

def test_quote_table() -> None:
    ...

def test_mask_encrypted_extra() -> None:
    ...

def test_unmask_encrypted_extra() -> None:
    ...
