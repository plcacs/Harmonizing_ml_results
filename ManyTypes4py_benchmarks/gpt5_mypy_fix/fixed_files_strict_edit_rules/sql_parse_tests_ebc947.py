from typing import Optional, Union
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


def extract_tables(query: str, engine: str = 'base') -> set[Table]:
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
    Test that the parser handles selects inside functions.
    """
    assert extract_tables('\nselect (extractvalue(1,concat(0x7e,(select GROUP_CONCAT(TABLE_NAME)\nfrom INFORMATION_SCHEMA.COLUMNS\nWHERE TABLE_SCHEMA like "%bi%"),0x7e)));\n', 'mysql') == {Table('COLUMNS', 'INFORMATION_SCHEMA')}
    assert extract_tables('\nselect (extractvalue(1,concat(0x7e,(select GROUP_CONCAT(COLUMN_NAME)\nfrom INFORMATION_SCHEMA.COLUMNS\nWHERE TABLE_NAME="bi_achievement_daily"),0x7e)));\n', 'mysql') == {Table('COLUMNS', 'INFORMATION_SCHEMA')}


def test_extract_tables_complex_cte_with_prefix() -> None:
    """
    Test that the parser handles CTEs with prefixes.
    """
    assert extract_tables('\nWITH CTE__test (SalesPersonID, SalesOrderID, SalesYear)\nAS (\n    SELECT SalesPersonID, SalesOrderID, YEAR(OrderDate) AS SalesYear\n    FROM SalesOrderHeader\n    WHERE SalesPersonID IS NOT NULL\n)\nSELECT SalesPersonID, COUNT(SalesOrderID) AS TotalSales, SalesYear\nFROM CTE__test\nGROUP BY SalesYear, SalesPersonID\nORDER BY SalesPersonID, SalesYear;\n') == {Table('SalesOrderHeader')}


def test_extract_tables_identifier_list_with_keyword_as_alias() -> None:
    """
    Test that aliases that are keywords are parsed correctly.
    """
    assert extract_tables('\nWITH\n    f AS (SELECT * FROM foo),\n    match AS (SELECT * FROM f)\nSELECT * FROM match\n') == {Table('foo')}


def test_update() -> None:
    """
    Test that ``UPDATE`` is not detected as ``SELECT``.
    """
    assert ParsedQuery('UPDATE t1 SET col1 = NULL').is_select() is False


def test_set() -> None:
    """
    Test that ``SET`` is detected correctly.
    """
    query = ParsedQuery("\n-- comment\nSET hivevar:desc='Legislators';\n")
    assert query.is_set() is True
    assert query.is_select() is False
    assert ParsedQuery("set hivevar:desc='bla'").is_set() is True
    assert ParsedQuery('SELECT 1').is_set() is False


def test_show() -> None:
    """
    Test that ``SHOW`` is detected correctly.
    """
    query = ParsedQuery('\n-- comment\nSHOW LOCKS test EXTENDED;\n-- comment\n')
    assert query.is_show() is True
    assert query.is_select() is False
    assert ParsedQuery('SHOW TABLES').is_show() is True
    assert ParsedQuery('shOw TABLES').is_show() is True
    assert ParsedQuery('show TABLES').is_show() is True
    assert ParsedQuery('SELECT 1').is_show() is False


def test_is_explain() -> None:
    """
    Test that ``EXPLAIN`` is detected correctly.
    """
    assert ParsedQuery('EXPLAIN SELECT 1').is_explain() is True
    assert ParsedQuery('EXPLAIN SELECT 1').is_select() is False
    assert ParsedQuery('\n-- comment\nEXPLAIN select * from table\n-- comment 2\n').is_explain() is True
    assert ParsedQuery("\n-- comment\nEXPLAIN select * from table\nwhere col1 = 'something'\n-- comment 2\n\n-- comment 3\nEXPLAIN select * from table\nwhere col1 = 'something'\n-- comment 4\n").is_explain() is True
    assert ParsedQuery('\n-- This is a comment\n    -- this is another comment but with a space in the front\nEXPLAIN SELECT * FROM TABLE\n').is_explain() is True
    assert ParsedQuery('\n/* This is a comment\n     with stars instead */\nEXPLAIN SELECT * FROM TABLE\n').is_explain() is True
    assert ParsedQuery("\n-- comment\nselect * from table\nwhere col1 = 'something'\n-- comment 2\n").is_explain() is False


def test_is_valid_ctas() -> None:
    """
    Test if a query is a valid CTAS.

    A valid CTAS has a ``SELECT`` as its last statement.
    """
    assert ParsedQuery('SELECT * FROM table', strip_comments=True).is_valid_ctas() is True
    assert ParsedQuery('\n-- comment\nSELECT * FROM table\n-- comment 2\n', strip_comments=True).is_valid_ctas() is True
    assert ParsedQuery('\n-- comment\nSET @value = 42;\nSELECT @value as foo;\n-- comment 2\n', strip_comments=True).is_valid_ctas() is True
    assert ParsedQuery('\n-- comment\nEXPLAIN SELECT * FROM table\n-- comment 2\n', strip_comments=True).is_valid_ctas() is False
    assert ParsedQuery('\nSELECT * FROM table;\nINSERT INTO TABLE (foo) VALUES (42);\n', strip_comments=True).is_valid_ctas() is False


def test_is_valid_cvas() -> None:
    """
    Test if a query is a valid CVAS.

    A valid CVAS has a single ``SELECT`` statement.
    """
    assert ParsedQuery('SELECT * FROM table', strip_comments=True).is_valid_cvas() is True
    assert ParsedQuery('\n-- comment\nSELECT * FROM table\n-- comment 2\n', strip_comments=True).is_valid_cvas() is True
    assert ParsedQuery('\n-- comment\nSET @value = 42;\nSELECT @value as foo;\n-- comment 2\n', strip_comments=True).is_valid_cvas() is False
    assert ParsedQuery('\n-- comment\nEXPLAIN SELECT * FROM table\n-- comment 2\n', strip_comments=True).is_valid_cvas() is False
    assert ParsedQuery('\nSELECT * FROM table;\nINSERT INTO TABLE (foo) VALUES (42);\n', strip_comments=True).is_valid_cvas() is False


def test_is_select_cte_with_comments() -> None:
    """
    Some CTES with comments are not correctly identified as SELECTS.
    """
    sql = ParsedQuery('WITH blah AS\n  (SELECT * FROM core_dev.manager_team),\n\nblah2 AS\n  (SELECT * FROM core_dev.manager_workspace)\n\nSELECT * FROM blah\nINNER JOIN blah2 ON blah2.team_id = blah.team_id')
    assert sql.is_select()
    sql = ParsedQuery('WITH blah AS\n/*blahblahbalh*/\n  (SELECT * FROM core_dev.manager_team),\n--blahblahbalh\n\nblah2 AS\n  (SELECT * FROM core_dev.manager_workspace)\n\nSELECT * FROM blah\nINNER JOIN blah2 ON blah2.team_id = blah.team_id')
    assert sql.is_select()


def test_cte_is_select() -> None:
    """
    Some CTEs are not correctly identified as SELECTS.
    """
    sql = ParsedQuery('WITH foo AS(\nSELECT\n  FLOOR(__time TO WEEK) AS "week",\n  name,\n  COUNT(DISTINCT user_id) AS "unique_users"\nFROM "druid"."my_table"\nGROUP BY 1,2\n)\nSELECT\n  f.week,\n  f.name,\n  f.unique_users\nFROM foo f')
    assert sql.is_select()


def test_cte_is_select_lowercase() -> None:
    """
    Some CTEs with lowercase select are not correctly identified as SELECTS.
    """
    sql = ParsedQuery('WITH foo AS(\nselect\n  FLOOR(__time TO WEEK) AS "week",\n  name,\n  COUNT(DISTINCT user_id) AS "unique_users"\nFROM "druid"."my_table"\nGROUP BY 1,2\n)\nselect\n  f.week,\n  f.name,\n  f.unique_users\nFROM foo f')
    assert sql.is_select()


def test_cte_insert_is_not_select() -> None:
    """
    Some CTEs with lowercase select are not correctly identified as SELECTS.
    """
    sql = ParsedQuery('WITH foo AS(\n        INSERT INTO foo (id) VALUES (1) RETURNING 1\n        ) select * FROM foo f')
    assert sql.is_select() is False


def test_cte_delete_is_not_select() -> None:
    """
    Some CTEs with lowercase select are not correctly identified as SELECTS.
    """
    sql = ParsedQuery('WITH foo AS(\n        DELETE FROM foo RETURNING *\n        ) select * FROM foo f')
    assert sql.is_select() is False


def test_cte_is_not_select_lowercase() -> None:
    """
    Some CTEs with lowercase select are not correctly identified as SELECTS.
    """
    sql = ParsedQuery('WITH foo AS(\n        insert into foo (id) values (1) RETURNING 1\n        ) select * FROM foo f')
    assert sql.is_select() is False


def test_cte_with_multiple_selects() -> None:
    sql = ParsedQuery('WITH a AS ( select * from foo1 ), b as (select * from foo2) SELECT * FROM a;')
    assert sql.is_select()


def test_cte_with_multiple_with_non_select() -> None:
    sql = ParsedQuery('WITH a AS (\n        select * from foo1\n        ), b as (\n        update foo2 set id=2\n        ) SELECT * FROM a')
    assert sql.is_select() is False
    sql = ParsedQuery('WITH a AS (\n         update foo2 set name=2\n         ),\n        b as (\n        select * from foo1\n        ) SELECT * FROM a')
    assert sql.is_select() is False
    sql = ParsedQuery('WITH a AS (\n         update foo2 set name=2\n         ),\n        b as (\n        update foo1 set name=2\n        ) SELECT * FROM a')
    assert sql.is_select() is False
    sql = ParsedQuery('WITH a AS (\n        INSERT INTO foo (id) VALUES (1)\n        ),\n        b as (\n        select 1\n        ) SELECT * FROM a')
    assert sql.is_select() is False


def test_unknown_select() -> None:
    """
    Test that `is_select` works when sqlparse fails to identify the type.
    """
    sql = 'WITH foo AS(SELECT 1) SELECT 1'
    assert sqlparse.parse(sql)[0].get_type() == 'SELECT'
    assert ParsedQuery(sql).is_select()
    sql = 'WITH foo AS(SELECT 1) INSERT INTO my_table (a) VALUES (1)'
    assert sqlparse.parse(sql)[0].get_type() == 'INSERT'
    assert not ParsedQuery(sql).is_select()
    sql = 'WITH foo AS(SELECT 1) DELETE FROM my_table'
    assert sqlparse.parse(sql)[0].get_type() == 'DELETE'
    assert not ParsedQuery(sql).is_select()


def test_get_query_with_new_limit_comment() -> None:
    """
    Test that limit is applied correctly.
    """
    query = ParsedQuery('SELECT * FROM birth_names -- SOME COMMENT')
    assert query.set_or_update_query_limit(1000) == 'SELECT * FROM birth_names -- SOME COMMENT\nLIMIT 1000'


def test_get_query_with_new_limit_comment_with_limit() -> None:
    """
    Test that limits in comments are ignored.
    """
    query = ParsedQuery('SELECT * FROM birth_names -- SOME COMMENT WITH LIMIT 555')
    assert query.set_or_update_query_limit(1000) == 'SELECT * FROM birth_names -- SOME COMMENT WITH LIMIT 555\nLIMIT 1000'


def test_get_query_with_new_limit_lower() -> None:
    """
    Test that lower limits are not replaced.
    """
    query = ParsedQuery('SELECT * FROM birth_names LIMIT 555')
    assert query.set_or_update_query_limit(1000) == 'SELECT * FROM birth_names LIMIT 555'


def test_get_query_with_new_limit_upper() -> None:
    """
    Test that higher limits are replaced.
    """
    query = ParsedQuery('SELECT * FROM birth_names LIMIT 2000')
    assert query.set_or_update_query_limit(1000) == 'SELECT * FROM birth_names LIMIT 1000'


def test_basic_breakdown_statements() -> None:
    """
    Test that multiple statements are parsed correctly.
    """
    query = ParsedQuery('\nSELECT * FROM birth_names;\nSELECT * FROM birth_names LIMIT 1;\n')
    assert query.get_statements() == ['SELECT * FROM birth_names', 'SELECT * FROM birth_names LIMIT 1']


def test_messy_breakdown_statements() -> None:
    """
    Test the messy multiple statements are parsed correctly.
    """
    query = ParsedQuery('\nSELECT 1;\t\n\n\n  \t\n\t\nSELECT 2;\nSELECT * FROM birth_names;;;\nSELECT * FROM birth_names LIMIT 1\n')
    assert query.get_statements() == ['SELECT 1', 'SELECT 2', 'SELECT * FROM birth_names', 'SELECT * FROM birth_names LIMIT 1']


def test_sqlparse_formatting() -> None:
    """
    Test that ``from_unixtime`` is formatted correctly.
    """
    assert sqlparse.format("SELECT extract(HOUR from from_unixtime(hour_ts) AT TIME ZONE 'America/Los_Angeles') from table", reindent=True) == "SELECT extract(HOUR\n               from from_unixtime(hour_ts) AT TIME ZONE 'America/Los_Angeles')\nfrom table"


def test_strip_comments_from_sql() -> None:
    """
    Test that comments are stripped out correctly.
    """
    assert strip_comments_from_sql('SELECT col1, col2 FROM table1') == 'SELECT col1, col2 FROM table1'
    assert strip_comments_from_sql('SELECT col1, col2 FROM table1\n-- comment') == 'SELECT col1, col2 FROM table1\n'
    assert strip_comments_from_sql("SELECT '--abc' as abc, col2 FROM table1\n") == "SELECT '--abc' as abc, col2 FROM table1"


def test_check_sql_functions_exist() -> None:
    """
    Test that comments are stripped out correctly.
    """
    assert not check_sql_functions_exist('select a, b from version', {'version'}, 'postgresql')
    assert check_sql_functions_exist('select version()', {'version'}, 'postgresql')
    assert check_sql_functions_exist('select version from version()', {'version'}, 'postgresql')
    assert check_sql_functions_exist('select 1, a.version from (select version from version()) as a', {'version'}, 'postgresql')
    assert check_sql_functions_exist('select 1, a.version from (select version()) as a', {'version'}, 'postgresql')


def test_sanitize_clause_valid() -> None:
    assert sanitize_clause('col = 1') == 'col = 1'
    assert sanitize_clause('1=\t\n1') == '1=\t\n1'
    assert sanitize_clause('(col = 1)') == '(col = 1)'
    assert sanitize_clause('(col1 = 1) AND (col2 = 2)') == '(col1 = 1) AND (col2 = 2)'
    assert sanitize_clause("col = 'abc' -- comment") == "col = 'abc' -- comment\n"
    assert sanitize_clause("col = 'col1 = 1) AND (col2 = 2'") == "col = 'col1 = 1) AND (col2 = 2'"
    assert sanitize_clause("col = 'select 1; select 2'") == "col = 'select 1; select 2'"
    assert sanitize_clause("col = 'abc -- comment'") == "col = 'abc -- comment'"


def test_sanitize_clause_closing_unclosed() -> None:
    with pytest.raises(QueryClauseValidationException):
        sanitize_clause('col1 = 1) AND (col2 = 2)')


def test_sanitize_clause_unclosed() -> None:
    with pytest.raises(QueryClauseValidationException):
        sanitize_clause('(col1 = 1) AND (col2 = 2')


def test_sanitize_clause_closing_and_unclosed() -> None:
    with pytest.raises(QueryClauseValidationException):
        sanitize_clause('col1 = 1) AND (col2 = 2')


def test_sanitize_clause_closing_and_unclosed_nested() -> None:
    with pytest.raises(QueryClauseValidationException):
        sanitize_clause('(col1 = 1)) AND ((col2 = 2)')


def test_sanitize_clause_multiple() -> None:
    with pytest.raises(QueryClauseValidationException):
        sanitize_clause('TRUE; SELECT 1')


def test_sqlparse_issue_652() -> None:
    stmt = sqlparse.parse("foo = '\\' AND bar = 'baz'")[0]
    assert len(stmt.tokens) == 5
    assert str(stmt.tokens[0]) == "foo = '\\'"


@pytest.mark.parametrize(
    ('engine', 'sql', 'expected'),
    [
        ('postgresql', 'extract(HOUR from from_unixtime(hour_ts))', False),
        ('postgresql', 'SELECT * FROM table', True),
        ('postgresql', '(SELECT * FROM table)', True),
        ('postgresql', 'SELECT a FROM (SELECT 1 AS a) JOIN (SELECT * FROM table)', True),
        ('postgresql', '(SELECT COUNT(DISTINCT name) AS foo FROM    birth_names)', True),
        ('postgresql', 'COUNT(*)', False),
        ('postgresql', 'SELECT a FROM (SELECT 1 AS a)', False),
        ('postgresql', 'SELECT a FROM (SELECT 1 AS a) JOIN table', True),
        ('postgresql', 'SELECT * FROM (SELECT 1 AS foo, 2 AS bar) ORDER BY foo ASC, bar', False),
        ('postgresql', 'SELECT * FROM other_table', True),
        ('postgresql', '(SELECT COUNT(DISTINCT name) from birth_names)', True),
        ('postgresql', "(SELECT table_name FROM information_schema.tables WHERE table_name LIKE '%user%' LIMIT 1)", True),
        ('postgresql', "(SELECT table_name FROM /**/ information_schema.tables WHERE table_name LIKE '%user%' LIMIT 1)", True),
        ('postgresql', 'SELECT FROM (SELECT FROM forbidden_table) AS forbidden_table;', True),
        ('postgresql', 'SELECT * FROM (SELECT * FROM forbidden_table) forbidden_table', True),
        ('postgresql', "((select users.id from (select 'majorie' as a) b, users where b.a = users.name and users.name in ('majorie') limit 1) like 'U%')", True),
    ],
)
def test_has_table_query(engine: str, sql: str, expected: bool) -> None:
    """
    Test if a given statement queries a table.

    This is used to prevent ad-hoc metrics from querying unauthorized tables, bypassing
    row-level security.
    """
    assert has_table_query(sql, engine) == expected


@pytest.mark.parametrize(
    'sql,table,rls,expected',
    [
        (
            'SELECT * FROM some_table WHERE 1=1',
            'some_table',
            'id=42',
            'SELECT * FROM (SELECT * FROM some_table WHERE some_table.id=42) AS some_table WHERE 1=1',
        ),
        ('SELECT * FROM table WHERE 1=1', 'table', 'id=42', 'SELECT * FROM (SELECT * FROM table WHERE table.id=42) AS table WHERE 1=1'),
        ('SELECT * FROM table WHERE 1=1', 'other_table', 'id=42', 'SELECT * FROM table WHERE 1=1'),
        ('SELECT * FROM other_table WHERE 1=1', 'table', 'id=42', 'SELECT * FROM other_table WHERE 1=1'),
        ('SELECT * FROM table JOIN other_table ON table.id = other_table.id', 'other_table', 'id=42', 'SELECT * FROM table JOIN (SELECT * FROM other_table WHERE other_table.id=42) AS other_table ON table.id = other_table.id'),
        ('SELECT * FROM (SELECT * FROM other_table)', 'other_table', 'id=42', 'SELECT * FROM (SELECT * FROM (SELECT * FROM other_table WHERE other_table.id=42) AS other_table)'),
        ('SELECT * FROM table UNION ALL SELECT * FROM other_table', 'table', 'id=42', 'SELECT * FROM (SELECT * FROM table WHERE table.id=42) AS table UNION ALL SELECT * FROM other_table'),
        ('SELECT * FROM table UNION ALL SELECT * FROM other_table', 'other_table', 'id=42', 'SELECT * FROM table UNION ALL SELECT * FROM (SELECT * FROM other_table WHERE other_table.id=42) AS other_table'),
        ('SELECT * FROM schema.table_name', 'table_name', 'id=42', 'SELECT * FROM (SELECT * FROM schema.table_name WHERE table_name.id=42) AS table_name'),
        ('SELECT * FROM schema.table_name', 'schema.table_name', 'id=42', 'SELECT * FROM (SELECT * FROM schema.table_name WHERE schema.table_name.id=42) AS table_name'),
        ('SELECT * FROM table_name', 'schema.table_name', 'id=42', 'SELECT * FROM (SELECT * FROM table_name WHERE schema.table_name.id=42) AS table_name'),
        ('SELECT a.*, b.* FROM tbl_a AS a INNER JOIN tbl_b AS b ON a.col = b.col', 'tbl_a', 'id=42', 'SELECT a.*, b.* FROM (SELECT * FROM tbl_a WHERE tbl_a.id=42) AS a INNER JOIN tbl_b AS b ON a.col = b.col'),
        ('SELECT a.*, b.* FROM tbl_a a INNER JOIN tbl_b b ON a.col = b.col', 'tbl_a', 'id=42', 'SELECT a.*, b.* FROM (SELECT * FROM tbl_a WHERE tbl_a.id=42) AS a INNER JOIN tbl_b b ON a.col = b.col'),
    ],
)
def test_insert_rls_as_subquery(mocker: MockerFixture, sql: str, table: str, rls: str, expected: str) -> None:
    """
    Insert into a statement a given RLS condition associated with a table.
    """
    condition = sqlparse.parse(rls)[0]
    add_table_name(condition, table)

    def get_rls_for_table(candidate: Union[Identifier, TokenList], database_id: int, default_schema: str) -> Optional[TokenList]:
        """
        Return the RLS ``condition`` if ``candidate`` matches ``table``.
        """
        if not isinstance(candidate, Identifier):
            candidate = Identifier([Token(Name, candidate.value)])
        candidate_table = ParsedQuery.get_table(candidate)
        if not candidate_table:
            return None
        candidate_table_name = f'{candidate_table.schema}.{candidate_table.table}' if candidate_table.schema else candidate_table.table
        for left, right in zip(candidate_table_name.split('.')[::-1], table.split('.')[::-1], strict=False):
            if left != right:
                return None
        return condition

    mocker.patch('superset.sql_parse.get_rls_for_table', new=get_rls_for_table)
    statement = sqlparse.parse(sql)[0]
    assert str(insert_rls_as_subquery(token_list=statement, database_id=1, default_schema='my_schema')).strip() == expected.strip()


@pytest.mark.parametrize(
    'sql,table,rls,expected',
    [
        ('SELECT * FROM some_table WHERE 1=1', 'some_table', 'id=42', 'SELECT * FROM some_table WHERE ( 1=1) AND some_table.id=42'),
        ('SELECT * FROM some_table WHERE TRUE OR FALSE', 'some_table', '1=0', 'SELECT * FROM some_table WHERE ( TRUE OR FALSE) AND 1=0'),
        ('SELECT * FROM table WHERE 1=1', 'table', 'id=42', 'SELECT * FROM table WHERE ( 1=1) AND table.id=42'),
        ('SELECT * FROM table WHERE 1=1', 'other_table', 'id=42', 'SELECT * FROM table WHERE 1=1'),
        ('SELECT * FROM other_table WHERE 1=1', 'table', 'id=42', 'SELECT * FROM other_table WHERE 1=1'),
        ('SELECT * FROM table', 'table', 'id=42', 'SELECT * FROM table WHERE table.id=42'),
        ('SELECT * FROM some_table', 'some_table', 'id=42', 'SELECT * FROM some_table WHERE some_table.id=42'),
        ('SELECT * FROM table ORDER BY id', 'table', 'id=42', 'SELECT * FROM table  WHERE table.id=42 ORDER BY id'),
        ('SELECT * FROM some_table;', 'some_table', 'id=42', 'SELECT * FROM some_table WHERE some_table.id=42 ;'),
        ('SELECT * FROM some_table       ;', 'some_table', 'id=42', 'SELECT * FROM some_table        WHERE some_table.id=42 ;'),
        ('SELECT * FROM some_table       ', 'some_table', 'id=42', 'SELECT * FROM some_table        WHERE some_table.id=42'),
        ('SELECT * FROM table WHERE 1=1 AND table.id=42', 'table', 'id=42', 'SELECT * FROM table WHERE ( 1=1 AND table.id=42) AND table.id=42'),
        ('SELECT * FROM table JOIN other_table ON table.id = other_table.id AND other_table.id=42', 'other_table', 'id=42', 'SELECT * FROM table JOIN other_table ON other_table.id=42 AND ( table.id = other_table.id AND other_table.id=42 )'),
        ('SELECT * FROM table WHERE 1=1 AND id=42', 'table', 'id=42', 'SELECT * FROM table WHERE ( 1=1 AND id=42) AND table.id=42'),
        ('SELECT * FROM table JOIN other_table ON table.id = other_table.id', 'other_table', 'id=42', 'SELECT * FROM table JOIN other_table ON other_table.id=42 AND ( table.id = other_table.id )'),
        ('SELECT * FROM table JOIN other_table ON table.id = other_table.id WHERE 1=1', 'other_table', 'id=42', 'SELECT * FROM table JOIN other_table ON other_table.id=42 AND ( table.id = other_table.id  ) WHERE 1=1'),
        ('SELECT * FROM (SELECT * FROM other_table)', 'other_table', 'id=42', 'SELECT * FROM (SELECT * FROM other_table WHERE other_table.id=42 )'),
        ('SELECT * FROM table UNION ALL SELECT * FROM other_table', 'table', 'id=42', 'SELECT * FROM table  WHERE table.id=42 UNION ALL SELECT * FROM other_table'),
        ('SELECT * FROM table UNION ALL SELECT * FROM other_table', 'other_table', 'id=42', 'SELECT * FROM table UNION ALL SELECT * FROM other_table WHERE other_table.id=42'),
        ('SELECT * FROM schema.table_name', 'table_name', 'id=42', 'SELECT * FROM schema.table_name WHERE table_name.id=42'),
        ('SELECT * FROM schema.table_name', 'schema.table_name', 'id=42', 'SELECT * FROM schema.table_name WHERE schema.table_name.id=42'),
        ('SELECT * FROM table_name', 'schema.table_name', 'id=42', 'SELECT * FROM table_name WHERE schema.table_name.id=42'),
    ],
)
def test_insert_rls_in_predicate(mocker: MockerFixture, sql: str, table: str, rls: str, expected: str) -> None:
    """
    Insert into a statement a given RLS condition associated with a table.
    """
    condition = sqlparse.parse(rls)[0]
    add_table_name(condition, table)

    def get_rls_for_table(candidate: Union[Identifier, TokenList], database_id: int, default_schema: str) -> Optional[TokenList]:
        """
        Return the RLS ``condition`` if ``candidate`` matches ``table``.
        """
        for left, right in zip(str(candidate).split('.')[::-1], table.split('.')[::-1], strict=False):
            if left != right:
                return None
        return condition

    mocker.patch('superset.sql_parse.get_rls_for_table', new=get_rls_for_table)
    statement = sqlparse.parse(sql)[0]
    assert str(insert_rls_in_predicate(token_list=statement, database_id=1, default_schema='my_schema')).strip() == expected.strip()


@pytest.mark.parametrize(
    'rls,table,expected',
    [('id=42', 'users', 'users.id=42'), ('users.id=42', 'users', 'users.id=42'), ('schema.users.id=42', 'users', 'schema.users.id=42'), ('false', 'users', 'false')],
)
def test_add_table_name(rls: str, table: str, expected: str) -> None:
    condition = sqlparse.parse(rls)[0]
    add_table_name(condition, table)
    assert str(condition) == expected


def test_get_rls_for_table(mocker: MockerFixture) -> None:
    """
    Tests for ``get_rls_for_table``.
    """
    candidate = Identifier([Token(Name, 'some_table')])
    dataset = mocker.patch('superset.db').session.query().filter().one_or_none()
    dataset.__str__.return_value = 'some_table'
    dataset.get_sqla_row_level_filters.return_value = [text('organization_id = 1')]
    assert str(get_rls_for_table(candidate, 1, 'public')) == 'some_table.organization_id = 1'
    dataset.get_sqla_row_level_filters.return_value = [text('organization_id = 1'), text("foo = 'bar'")]
    assert str(get_rls_for_table(candidate, 1, 'public')) == "some_table.organization_id = 1 AND some_table.foo = 'bar'"
    dataset.get_sqla_row_level_filters.return_value = []
    assert get_rls_for_table(candidate, 1, 'public') is None


def test_extract_table_references(mocker: MockerFixture) -> None:
    """
    Test the ``extract_table_references`` helper function.
    """
    assert extract_table_references('SELECT 1', 'trino') == set()
    assert extract_table_references('SELECT 1 FROM some_table', 'trino') == {Table(table='some_table', schema=None, catalog=None)}
    assert extract_table_references('SELECT {{ jinja }} FROM some_table', 'trino') == {Table(table='some_table', schema=None, catalog=None)}
    assert extract_table_references('SELECT 1 FROM some_catalog.some_schema.some_table', 'trino') == {Table(table='some_table', schema='some_schema', catalog='some_catalog')}
    assert extract_table_references('SELECT 1 FROM `some_catalog`.`some_schema`.`some_table`', 'mysql') == {Table(table='some_table', schema='some_schema', catalog='some_catalog')}
    assert extract_table_references('SELECT 1 FROM "some_catalog".some_schema."some_table"', 'trino') == {Table(table='some_table', schema='some_schema', catalog='some_catalog')}
    assert extract_table_references('SELECT * FROM some_table JOIN other_table ON some_table.id = other_table.id', 'trino') == {Table(table='some_table', schema=None, catalog=None), Table(table='other_table', schema=None, catalog=None)}
    logger = mocker.patch('superset.sql_parse.logger')
    sql = 'SELECT * FROM table UNION ALL SELECT * FROM other_table'
    assert extract_table_references(sql, 'trino') == {Table(table='table', schema=None, catalog=None), Table(table='other_table', schema=None, catalog=None)}
    logger.warning.assert_called_once()
    logger = mocker.patch('superset.migrations.shared.utils.logger')
    sql = 'SELECT * FROM table UNION ALL SELECT * FROM other_table'
    assert extract_table_references(sql, 'trino', show_warning=False) == {Table(table='table', schema=None, catalog=None), Table(table='other_table', schema=None, catalog=None)}
    logger.warning.assert_not_called()


def test_is_select() -> None:
    """
    Test `is_select`.
    """
    assert not ParsedQuery('SELECT 1; DROP DATABASE superset').is_select()
    assert ParsedQuery('with base as(select id from table1 union all select id from table2) select * from base').is_select()
    assert ParsedQuery('\nWITH t AS (\n    SELECT 1 UNION ALL SELECT 2\n)\nSELECT * FROM t').is_select()
    assert not ParsedQuery('').is_select()
    assert not ParsedQuery('USE foo').is_select()
    assert ParsedQuery('USE foo; SELECT * FROM bar').is_select()


@pytest.mark.parametrize('engine', ['hive', 'presto', 'trino'])
@pytest.mark.parametrize(
    'macro,expected',
    [
        ("latest_partition('foo.bar')", {Table(table='bar', schema='foo')}),
        ("latest_partition(' foo.bar ')", {Table(table='bar', schema='foo')}),
        ("latest_partition('foo.%s'|format('bar'))", {Table(table='bar', schema='foo')}),
        ("latest_sub_partition('foo.bar', baz='qux')", {Table(table='bar', schema='foo')}),
        ("latest_partition('foo.%s'|format(str('bar')))", set()),
        ("latest_partition('foo.{}'.format('bar'))", set()),
    ],
)
def test_extract_tables_from_jinja_sql(mocker: MockerFixture, engine: str, macro: str, expected: set[Table]) -> None:
    assert extract_tables_from_jinja_sql(sql=f"'{{{{ {engine}.{macro} }}}}'", database=mocker.Mock()) == expected


@mock.patch.dict('superset.extensions.feature_flag_manager._feature_flags', {'ENABLE_TEMPLATE_PROCESSING': False}, clear=True)
def test_extract_tables_from_jinja_sql_disabled(mocker: MockerFixture) -> None:
    """
    Test the function when the feature flag is disabled.
    """
    database = mocker.Mock()
    database.db_engine_spec.engine = 'mssql'
    assert extract_tables_from_jinja_sql(sql='SELECT 1 FROM t', database=database) == {Table('t')}