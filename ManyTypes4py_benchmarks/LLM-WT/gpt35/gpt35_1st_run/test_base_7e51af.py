from __future__ import annotations
import json
from textwrap import dedent
from typing import Any, List, Tuple
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

def test_cte_query_parsing(original: str, expected: Any) -> None:
    ...

def test_get_column_spec(native_type: str, sqla_type: types.TypeEngine, attrs: Any, generic_type: GenericDataType, is_dttm: bool) -> None:
    ...

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
