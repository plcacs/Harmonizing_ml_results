from typing import Optional, Set, Tuple, List, Any, Dict
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

def test_extract_tables_subselect() -> None:
    """
    Test that tables inside subselects are parsed correctly.
    """
    assert extract_tables("\nSELECT sub.*\nFROM (\n    SELECT *\n        FROM s1.t1\n        WHERE day_of_week = 'Friday'\n    ) sub, s2.t2\nWHERE sub.resolution = 'NONE'\n") == {Table('t1', 's1'), Table('t2', 's2')}
    assert extract_tables("\nSELECT sub.*\nFROM (\n    SELECT *\n    FROM s1.t1\n    WHERE day_of_week = 'Friday'\n) sub\nWHERE sub.resolution = 'NONE'\n") == {Table('t1', 's1')}
    assert extract_tables('\nSELECT * FROM t1\nWHERE s11 > ANY (\n    SELECT COUNT(*) /* no hint */ FROM t2\n    WHERE NOT EXISTS (\n        SELECT * FROM t3\n        WHERE ROW(5*t2.s1,77)=(\n            SELECT 50,11*s1 FROM t4\n        )\n    )\n)\n') == {Table('t1'), Table('t2'), Table('t3'), Table('t4')}

def test_extract_tables_select_in_expression() -> None:
    """
    Test that parser works with ``SELECT``s used as expressions.
    """
    assert extract_tables('SELECT f1, (SELECT count(1) FROM t2) FROM t1') == {Table('t1'), Table('t2')}
    assert extract_tables('SELECT f1, (SELECT count(1) FROM t2) as f2 FROM t1') == {Table('t1'), Table('t2')}

def test_extract_tables_parenthesis() -> None:
    """
    Test that parenthesis are parsed correctly.
    """
    assert extract_tables('SELECT f1, (x + y) AS f2 FROM t1') == {Table('t1')}

def test_extract_tables_with_schema() -> None:
    """
    Test that schemas are parsed correctly.
    """
    assert extract_tables('SELECT * FROM schemaname.tbname') == {Table('tbname', 'schemaname')}
    assert extract_tables('SELECT * FROM "schemaname"."tbname"') == {Table('tbname', 'schemaname')}
    assert extract_tables('SELECT * FROM "schemaname"."tbname" foo') == {Table('tbname', 'schemaname')}
    assert extract_tables('SELECT * FROM "schemaname"."tbname" AS foo') == {Table('tbname', 'schemaname')}

def test_extract_tables_union() -> None:
    """
    Test that ``UNION`` queries work as expected.
    """
    assert extract_tables('SELECT * FROM t1 UNION SELECT * FROM t2') == {Table('t1'), Table('t2')}
    assert extract_tables('SELECT * FROM t1 UNION ALL SELECT * FROM t2') == {Table('t1'), Table('t2')}
    assert extract_tables('SELECT * FROM t1 INTERSECT ALL SELECT * FROM t2') == {Table('t1'), Table('t2')}

def test_extract_tables_select_from_values() -> None:
    """
    Test that selecting from values returns no tables.
    """
    assert extract_tables('SELECT * FROM VALUES (13, 42)') == set()

def test_extract_tables_select_array() -> None:
    """
    Test that queries selecting arrays work as expected.
    """
    assert extract_tables('\nSELECT ARRAY[1, 2, 3] AS my_array\nFROM t1 LIMIT 10\n') == {Table('t1')}

def test_extract_tables_select_if() -> None:
    """
    Test that queries with an ``IF`` work as expected.
    """
    assert extract_tables('\nSELECT IF(CARDINALITY(my_array) >= 3, my_array[3], NULL)\nFROM t1 LIMIT 10\n') == {Table('t1')}

def test_extract_tables_with_catalog() -> None:
    """
    Test that catalogs are parsed correctly.
    """
    assert extract_tables('SELECT * FROM catalogname.schemaname.tbname') == {Table('tbname', 'schemaname', 'catalogname')}

def test_extract_tables_illdefined() -> None:
    """
    Test that ill-defined tables return an empty set.
    """
    with pytest.raises(SupersetSecurityException) as excinfo:
        extract_tables('SELECT * FROM schemaname.')
    assert str(excinfo.value) == "You may have an error in your SQL statement. Error parsing near '.' at line 1:25"
    with pytest.raises(SupersetSecurityException) as excinfo:
        extract_tables('SELECT * FROM catalogname.schemaname.')
    assert str(excinfo.value) == "You may have an error in your SQL statement. Error parsing near '.' at line 1:37"
    with pytest.raises(SupersetSecurityException) as excinfo:
        extract_tables('SELECT * FROM catalogname..')
    assert str(excinfo.value) == "You may have an error in your SQL statement. Error parsing near '.' at line 1:27"
    with pytest.raises(SupersetSecurityException) as excinfo:
        extract_tables('SELECT * FROM "tbname')
    assert str(excinfo.value) == 'You may have an error in your SQL statement. Unable to tokenize script'
    assert extract_tables('SELECT * FROM catalogname..tbname') == {Table(table='tbname', schema=None, catalog='catalogname')}

def test_extract_tables_show_tables_from() -> None:
    """
    Test ``SHOW TABLES FROM``.
    """
    assert extract_tables("SHOW TABLES FROM s1 like '%order%'", 'mysql') == set()

def test_extract_tables_show_columns_from() -> None:
    """
    Test ``SHOW COLUMNS FROM``.
    """
    assert extract_tables('SHOW COLUMNS FROM t1') == {Table('t1')}

def test_extract_tables_where_subquery() -> None:
    """
    Test that tables in a ``WHERE`` subquery are parsed correctly.
    """
    assert extract_tables('\nSELECT name\nFROM t1\nWHERE regionkey = (SELECT max(regionkey) FROM t2)\n') == {Table('t1'), Table('t2')}
    assert extract_tables('\nSELECT name\nFROM t1\nWHERE regionkey IN (SELECT regionkey FROM t2)\n') == {Table('t1'), Table('t2')}
    assert extract_tables('\nSELECT name\nFROM t1\nWHERE EXISTS (SELECT 1 FROM t2 WHERE t1.regionkey = t2.regionkey);\n') == {Table('t1'), Table('t2')}

def test_extract_tables_describe() -> None:
    """
    Test ``DESCRIBE``.
    """
    assert extract_tables('DESCRIBE t1') == {Table('t1')}

def test_extract_tables_show_partitions() -> None:
    """
    Test ``SHOW PARTITIONS``.
    """
    assert extract_tables("\nSHOW PARTITIONS FROM orders\nWHERE ds >= '2013-01-01' ORDER BY ds DESC\n") == {Table('orders')}

def test_extract_tables_join() -> None:
    """
    Test joins.
    """
    assert extract_tables('SELECT t1.*, t2.* FROM t1 JOIN t2 ON t1.a = t2.a;') == {Table('t1'), Table('t2')}
    assert extract_tables('\nSELECT a.date, b.name\nFROM left_table a\nJOIN (\n    SELECT\n        CAST((b.year) as VARCHAR) date,\n        name\n    FROM right_table\n) b\nON a.date = b.date\n') == {Table('left_table'), Table('right_table')}
    assert extract_tables('\nSELECT a.date, b.name\nFROM left_table a\nLEFT INNER JOIN (\n    SELECT\n        CAST((b.year) as VARCHAR) date,\n        name\n    FROM right_table\n) b\nON a.date = b.date\n') == {Table('left_table'), Table('right_table')}
    assert extract_tables('\nSELECT a.date, b.name\nFROM left_table a\nRIGHT OUTER JOIN (\n    SELECT\n        CAST((b.year) as VARCHAR) date,\n        name\n    FROM right_table\n) b\nON a.date = b.date\n') == {Table('left_table'), Table('right_table')}
    assert extract_tables('\nSELECT a.date, b.name\nFROM left_table a\nFULL OUTER JOIN (\n    SELECT\n        CAST((b.year) as VARCHAR) date,\n        name\n        FROM right_table\n) b\nON a.date = b.date\n') == {Table('left_table'), Table('right_table')}

def test_extract_tables_semi_join() -> None:
    """
    Test ``LEFT SEMI JOIN``.
    """
    assert extract_tables('\nSELECT a.date, b.name\nFROM left_table a\nLEFT SEMI JOIN (\n    SELECT\n        CAST((b.year) as VARCHAR) date,\n        name\n    FROM right_table\n) b\nON a.data = b.date\n') == {Table('left_table'), Table('right_table')}

def test_extract_tables_combinations() -> None:
    """
    Test a complex case with nested queries.
    """
    assert extract_tables('\nSELECT * FROM t1\nWHERE s11 > ANY (\n    SELECT * FROM t1 UNION ALL SELECT * FROM (\n        SELECT t6.*, t3.* FROM t6 JOIN t3 ON t6.a = t3.a\n    ) tmp_join\n    WHERE NOT EXISTS (\n        SELECT * FROM t3\n        WHERE ROW(5*t3.s1,77)=(\n            SELECT 50,11*s1 FROM t4\n        )\n    )\n)\n') == {Table('t1'), Table('t3'), Table('t4'), Table('t6')}
    assert extract_tables('\nSELECT * FROM (\n    SELECT * FROM (\n        SELECT * FROM (\n            SELECT * FROM EmployeeS\n        ) AS S1\n    ) AS S2\n) AS S3\n') == {Table('EmployeeS')}

def test_extract_tables_with() -> None:
    """
    Test ``WITH``.
    """
    assert extract_tables('\nWITH\n    x AS (SELECT a FROM t1),\n    y AS (SELECT a AS b FROM t2),\n    z AS (SELECT b AS c FROM t3)\nSELECT c FROM z\n') == {Table('t1'), Table('t2'), Table('t3')}
    assert extract_tables('\nWITH\n    x AS (SELECT a FROM t1),\n    y AS (SELECT a AS b FROM x),\n    z AS (SELECT b AS c FROM y)\nSELECT c FROM z\n') == {Table('t1')}

def test_extract_tables_reusing_aliases() -> None:
    """
    Test that the parser follows aliases.
    """
    assert extract_tables("\nwith q1 as ( select key from q2 where key = '5'),\nq2 as ( select key from src where key = '5')\nselect * from (select key from q1) a\n") == {Table('src')}
    assert extract_tables("\nwith src as ( select key from q2 where key = '5'),\nq2 as ( select key from src where key = '5')\nselect * from (select key from src) a\n") == set()

def test_extract_tables_multistatement() -> None:
    """
    Test that the parser works with multiple statements.
    """
    assert extract_tables('SELECT * FROM t1; SELECT * FROM t2') == {Table('t1'), Table('t2')}
    assert extract_tables('SELECT * FROM t1; SELECT * FROM t2;') == {Table('t1'), Table('t2')}
    assert extract_tables('ADD JAR file:///hive.jar; SELECT * FROM t1;', engine='hive') == {Table('t1')}

def test_extract_tables_complex() -> None:
    """
    Test a few complex queries.
    """
    assert extract_tables('\nSELECT sum(m_examples) AS "sum__m_example"\nFROM (\n    SELECT\n        COUNT(DISTINCT id_userid) AS m_examples,\n        some_more_info\n    FROM my_b_table b\n    JOIN my_t_table t ON b.ds=t.ds\n    JOIN my_l_table l ON b.uid=l.uid\n    WHERE\n        b.rid IN (\n            SELECT other_col\n            FROM inner_table\n        )\n        AND l.bla IN (\'x\', \'y\')\n    GROUP BY 2\n    ORDER BY 2 ASC\n) AS "meh"\nORDER BY "sum__m_example" DESC\nLIMIT 10;\n') == {Table('my_l_table'), Table('my_b_table'), Table('my_t_table'), Table('inner_table')}
    assert extract_tables('\nSELECT *\nFROM table_a AS a, table_b AS b, table_c as c\nWHERE a.id = b.id and b.id = c.id\n') == {Table('table_a'), Table('table_b'), Table('table_c')}
    assert extract_tables('\nSELECT somecol AS somecol\nFROM (\n    WITH bla AS (\n        SELECT col_a\n        FROM a\n        WHERE\n            1=1\n            AND column_of_choice NOT IN (\n                SELECT interesting_col\n                FROM b\n            )\n    ),\n    rb AS (\n        SELECT yet_another_column\n        FROM (\n            SELECT a\n            FROM c\n            GROUP BY the_other_col\n        ) not_table\n        LEFT JOIN bla foo\n        ON foo.prop = not_table.bad_col0\n        WHERE 1=1\n        GROUP BY\n            not_table.bad_col1 ,\n            not_table.bad_col2 ,\n        ORDER BY not_table.bad_col_3 DESC ,\n            not_table.bad_col4 ,\n            not_table.bad_col5\n    )\n    SELECT random_col\n    FROM d\n    WHERE 1=1\n    UNION ALL SELECT even_more_cols\n    FROM e\n    WHERE 1=1\n    UNION ALL SELECT lets_go_deeper\n    FROM f\n    WHERE 1=1\n    WHERE 2=2\n    GROUP BY last_col\n    LIMIT 50000\n)\n') == {Table('a'), Table('b'), Table('c'), Table('d'), Table('e'), Table('f')}

def test_extract_tables_mixed_from_clause() -> None:
    """
    Test that the parser handles a ``FROM`` clause with table and subselect.
    """
    assert extract_tables('\nSELECT *\nFROM table_a AS a, (select * from table_b) AS b, table_c as c\nWHERE a.id = b.id and b.id = c.id\n') == {Table('table_a'), Table('table_b'), Table('table_c')}

def test_extract_tables_nested_select() -> None:
    """
    Test that the parser