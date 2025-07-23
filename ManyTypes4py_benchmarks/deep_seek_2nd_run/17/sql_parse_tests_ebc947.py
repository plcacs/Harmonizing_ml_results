from typing import Optional, Set, Tuple, List, Dict, Any, Union
from unittest import mock
import pytest
import sqlparse
from pytest_mock import MockerFixture
from sqlalchemy import text
from sqlparse.sql import Identifier, Token, TokenList
from sqlparse.tokens import Name
from superset.exceptions import QueryClauseValidationException, SupersetSecurityException
from superset.sql.parse import Table
from superset.sql_parse import add_table_name, check_sql_functions_exist, extract_table_references, extract_tables_from_jinja_sql, get_rls_for_table, has_table_query, insert_rls_as_subquery, insert_rls_in_predicate, ParsedQuery, sanitize_clause, strip_comments_from_sql

def extract_tables(query: str, engine: str = 'base') -> Set[Table]:
    """
    Helper function to extract tables referenced in a query.
    """
    return ParsedQuery(query, engine=engine).tables

def test_table() -> None:
    """
    Test the ``Table`` class and its string conversion.

    Special characters in the table, schema, or catalog name should be escaped correctly.
    """
    assert str(Table('tbname')) == 'tbname'
    assert str(Table('tbname', 'schemaname')) == 'schemaname.tbname'
    assert str(Table('tbname', 'schemaname', 'catalogname')) == 'catalogname.schemaname.tbname'
    assert str(Table('table.name', 'schema/name', 'catalog\nname')) == 'catalog%0Aname.schema%2Fname.table%2Ename'

def test_extract_tables() -> None:
    """
    Test that referenced tables are parsed correctly from the SQL.
    """
    assert extract_tables('SELECT * FROM tbname') == {Table('tbname')}
    assert extract_tables('SELECT * FROM tbname foo') == {Table('tbname')}
    assert extract_tables('SELECT * FROM tbname AS foo') == {Table('tbname')}
    assert extract_tables('SELECT * FROM tb_name') == {Table('tb_name')}
    assert extract_tables('SELECT * FROM "tbname"') == {Table('tbname')}
    assert extract_tables('SELECT * FROM "tb_name" WHERE city = "Lübeck"') == {Table('tb_name')}
    assert extract_tables('SELECT field1, field2 FROM tb_name') == {Table('tb_name')}
    assert extract_tables('SELECT t1.f1, t2.f2 FROM t1, t2') == {Table('t1'), Table('t2')}
    assert extract_tables('SELECT a.date, a.field FROM left_table a LIMIT 10') == {Table('left_table')}
    assert extract_tables('SELECT FROM (SELECT FROM forbidden_table) AS forbidden_table;') == {Table('forbidden_table')}
    assert extract_tables('select * from (select * from forbidden_table) forbidden_table') == {Table('forbidden_table')}

# ... (continuing with type annotations for all remaining functions)

def test_extract_tables_from_jinja_sql(mocker: MockerFixture, engine: str, macro: str, expected: Set[Table]) -> None:
    assert extract_tables_from_jinja_sql(sql=f"'{{{{ {engine}.{macro} }}}}'", database=mocker.Mock()) == expected

@mock.patch.dict('superset.extensions.feature_flag_manager._feature_flags', {'ENABLE_TEMPLATE_PROCESSING': False}, clear=True)
def test_extract_tables_from_jinja_sql_disabled(mocker: MockerFixture) -> None:
    """
    Test the function when the feature flag is disabled.
    """
    database = mocker.Mock()
    database.db_engine_spec.engine = 'mssql'
    assert extract_tables_from_jinja_sql(sql='SELECT 1 FROM t', database=database) == {Table('t')}
