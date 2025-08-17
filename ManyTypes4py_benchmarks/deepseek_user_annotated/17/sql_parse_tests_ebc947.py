from typing import Optional, Set

from pytest_mock import MockerFixture
from sqlalchemy import text
from sqlparse.sql import Identifier, Token, TokenList
from sqlparse.tokens import Name

from superset.exceptions import (
    QueryClauseValidationException,
    SupersetSecurityException,
)
from superset.sql.parse import Table
from superset.sql_parse import (
    add_table_name,
    check_sql_functions_exist,
    extract_table_references,
    extract_tables_from_jinja_sql,
    get_rls_for_table,
    has_table_query,
    insert_rls_as_subquery,
    insert_rls_in_predicate,
    ParsedQuery,
    sanitize_clause,
    strip_comments_from_sql,
)


def extract_tables(query: str, engine: str = "base") -> Set[Table]:
    """
    Helper function to extract tables referenced in a query.
    """
    return ParsedQuery(query, engine=engine).tables


def test_table() -> None:
    """
    Test the ``Table`` class and its string conversion.

    Special characters in the table, schema, or catalog name should be escaped correctly.
    """
    assert str(Table("tbname")) == "tbname"
    assert str(Table("tbname", "schemaname")) == "schemaname.tbname"
    assert (
        str(Table("tbname", "schemaname", "catalogname"))
        == "catalogname.schemaname.tbname"
    )
    assert (
        str(Table("table.name", "schema/name", "catalog\nname"))
        == "catalog%0Aname.schema%2Fname.table%2Ename"
    )


def test_extract_tables() -> None:
    """
    Test that referenced tables are parsed correctly from the SQL.
    """
    assert extract_tables("SELECT * FROM tbname") == {Table("tbname")}
    assert extract_tables("SELECT * FROM tbname foo") == {Table("tbname")}
    assert extract_tables("SELECT * FROM tbname AS foo") == {Table("tbname")}

    # underscore
    assert extract_tables("SELECT * FROM tb_name") == {Table("tb_name")}

    # quotes
    assert extract_tables('SELECT * FROM "tbname"') == {Table("tbname")}

    # unicode
    assert extract_tables('SELECT * FROM "tb_name" WHERE city = "Lübeck"') == {
        Table("tb_name")
    }

    # columns
    assert extract_tables("SELECT field1, field2 FROM tb_name") == {Table("tb_name")}
    assert extract_tables("SELECT t1.f1, t2.f2 FROM t1, t2") == {
        Table("t1"),
        Table("t2"),
    }

    # named table
    assert extract_tables("SELECT a.date, a.field FROM left_table a LIMIT 10") == {
        Table("left_table")
    }

    assert extract_tables(
        "SELECT FROM (SELECT FROM forbidden_table) AS forbidden_table;"
    ) == {Table("forbidden_table")}

    assert extract_tables(
        "select * from (select * from forbidden_table) forbidden_table"
    ) == {Table("forbidden_table")}


def test_extract_tables_subselect() -> None:
    """
    Test that tables inside subselects are parsed correctly.
    """
    assert extract_tables(
        """
SELECT sub.*
FROM (
    SELECT *
        FROM s1.t1
        WHERE day_of_week = 'Friday'
    ) sub, s2.t2
WHERE sub.resolution = 'NONE'
"""
    ) == {Table("t1", "s1"), Table("t2", "s2")}

    assert extract_tables(
        """
SELECT sub.*
FROM (
    SELECT *
    FROM s1.t1
    WHERE day_of_week = 'Friday'
) sub
WHERE sub.resolution = 'NONE'
"""
    ) == {Table("t1", "s1")}

    assert extract_tables(
        """
SELECT * FROM t1
WHERE s11 > ANY (
    SELECT COUNT(*) /* no hint */ FROM t2
    WHERE NOT EXISTS (
        SELECT * FROM t3
        WHERE ROW(5*t2.s1,77)=(
            SELECT 50,11*s1 FROM t4
        )
    )
)
"""
    ) == {Table("t1"), Table("t2"), Table("t3"), Table("t4")}


def test_extract_tables_select_in_expression() -> None:
    """
    Test that parser works with ``SELECT``s used as expressions.
    """
    assert extract_tables("SELECT f1, (SELECT count(1) FROM t2) FROM t1") == {
        Table("t1"),
        Table("t2"),
    }
    assert extract_tables("SELECT f1, (SELECT count(1) FROM t2) as f2 FROM t1") == {
        Table("t1"),
        Table("t2"),
    }


def test_extract_tables_parenthesis() -> None:
    """
    Test that parenthesis are parsed correctly.
    """
    assert extract_tables("SELECT f1, (x + y) AS f2 FROM t1") == {Table("t1")}


def test_extract_tables_with_schema() -> None:
    """
    Test that schemas are parsed correctly.
    """
    assert extract_tables("SELECT * FROM schemaname.tbname") == {
        Table("tbname", "schemaname")
    }
    assert extract_tables('SELECT * FROM "schemaname"."tbname"') == {
        Table("tbname", "schemaname")
    }
    assert extract_tables('SELECT * FROM "schemaname"."tbname" foo') == {
        Table("tbname", "schemaname")
    }
    assert extract_tables('SELECT * FROM "schemaname"."tbname" AS foo') == {
        Table("tbname", "schemaname")
    }


def test_extract_tables_union() -> None:
    """
    Test that ``UNION`` queries work as expected.
    """
    assert extract_tables("SELECT * FROM t1 UNION SELECT * FROM t2") == {
        Table("t1"),
        Table("t2"),
    }
    assert extract_tables("SELECT * FROM t1 UNION ALL SELECT * FROM t2") == {
        Table("t1"),
        Table("t2"),
    }
    assert extract_tables("SELECT * FROM t1 INTERSECT ALL SELECT * FROM t2") == {
        Table("t1"),
        Table("t2"),
    }


def test_extract_tables_select_from_values() -> None:
    """
    Test that selecting from values returns no tables.
    """
    assert extract_tables("SELECT * FROM VALUES (13, 42)") == set()


def test_extract_tables_select_array() -> None:
    """
    Test that queries selecting arrays work as expected.
    """
    assert extract_tables(
        """
SELECT ARRAY[1, 2, 3] AS my_array
FROM t1 LIMIT 10
"""
    ) == {Table("t1")}


def test_extract_tables_select_if() -> None:
    """
    Test that queries with an ``IF`` work as expected.
    """
    assert extract_tables(
        """
SELECT IF(CARDINALITY(my_array) >= 3, my_array[3], NULL)
FROM t1 LIMIT 10
"""
    ) == {Table("t1")}


def test_extract_tables_with_catalog() -> None:
    """
    Test that catalogs are parsed correctly.
    """
    assert extract_tables("SELECT * FROM catalogname.schemaname.tbname") == {
        Table("tbname", "schemaname", "catalogname")
    }


def test_extract_tables_illdefined() -> None:
    """
    Test that ill-defined tables return an empty set.
    """
    with pytest.raises(SupersetSecurityException) as excinfo:
        extract_tables("SELECT * FROM schemaname.")
    assert (
        str(excinfo.value)
        == "You may have an error in your SQL statement. Error parsing near '.' at line 1:25"
    )

    with pytest.raises(SupersetSecurityException) as excinfo:
        extract_tables("SELECT * FROM catalogname.schemaname.")
    assert (
        str(excinfo.value)
        == "You may have an error in your SQL statement. Error parsing near '.' at line 1:37"
    )

    with pytest.raises(SupersetSecurityException) as excinfo:
        extract_tables("SELECT * FROM catalogname..")
    assert (
        str(excinfo.value)
        == "You may have an error in your SQL statement. Error parsing near '.' at line 1:27"
    )

    with pytest.raises(SupersetSecurityException) as excinfo:
        extract_tables('SELECT * FROM "tbname')
    assert (
        str(excinfo.value)
        == "You may have an error in your SQL statement. Unable to tokenize script"
    )

    # odd edge case that works
    assert extract_tables("SELECT * FROM catalogname..tbname") == {
        Table(table="tbname", schema=None, catalog="catalogname")
    }


def test_extract_tables_show_tables_from() -> None:
    """
    Test ``SHOW TABLES FROM``.
    """
    assert extract_tables("SHOW TABLES FROM s1 like '%order%'", "mysql") == set()


def test_extract_tables_show_columns_from() -> None:
    """
    Test ``SHOW COLUMNS FROM``.
    """
    assert extract_tables("SHOW COLUMNS FROM t1") == {Table("t1")}


def test_extract_tables_where_subquery() -> None:
    """
    Test that tables in a ``WHERE`` subquery are parsed correctly.
    """
    assert extract_tables(
        """
SELECT name
FROM t1
WHERE regionkey = (SELECT max(regionkey) FROM t2)
"""
    ) == {Table("t1"), Table("t2")}

    assert extract_tables(
        """
SELECT name
FROM t1
WHERE regionkey IN (SELECT regionkey FROM t2)
"""
    ) == {Table("t1"), Table("t2")}

    assert extract_tables(
        """
SELECT name
FROM t1
WHERE EXISTS (SELECT 1 FROM t2 WHERE t1.regionkey = t2.regionkey);
"""
    ) == {Table("t1"), Table("t2")}


def test_extract_tables_describe() -> None:
    """
    Test ``DESCRIBE``.
    """
    assert extract_tables("DESCRIBE t1") == {Table("t1")}


def test_extract_tables_show_partitions() -> None:
    """
    Test ``SHOW PARTITIONS``.
    """
    assert extract_tables(
        """
SHOW PARTITIONS FROM orders
WHERE ds >= '2013-01-01' ORDER BY ds DESC
"""
    ) == {Table("orders")}


def test_extract_tables_join() -> None:
    """
    Test joins.
    """
    assert extract_tables("SELECT t1.*, t2.* FROM t1 JOIN t2 ON t1.a = t2.a;") == {
        Table("t1"),
        Table("t2"),
    }

    assert extract_tables(
        """
SELECT a.date, b.name
FROM left_table a
JOIN (
    SELECT
        CAST((b.year) as VARCHAR) date,
        name
    FROM right_table
) b
ON a.date = b.date
"""
    ) == {Table("left_table"), Table("right_table")}

    assert extract_tables(
        """
SELECT a.date, b.name
FROM left_table a
LEFT INNER JOIN (
    SELECT
        CAST((b.year) as VARCHAR) date,
        name
    FROM right_table
) b
ON a.date = b.date
"""
    ) == {Table("left_table"), Table("right_table")}

    assert extract_tables(
        """
SELECT a.date, b.name
FROM left_table a
RIGHT OUTER JOIN (
    SELECT
        CAST((b.year) as VARCHAR) date,
        name
    FROM right_table
) b
ON a.date = b.date
"""
    ) == {Table("left_table"), Table("right_table")}

    assert extract_tables(
        """
SELECT a.date, b.name
FROM left_table a
FULL OUTER JOIN (
    SELECT
        CAST((b.year) as VARCHAR) date,
        name
        FROM right_table
) b
ON a.date = b.date
"""
    ) == {Table("left_table"), Table("right_table")}


def test_extract_tables_semi_join() -> None:
    """
    Test ``LEFT SEMI JOIN``.
    """
    assert extract_tables(
        """
SELECT a.date, b.name
FROM left_table a
LEFT SEMI JOIN (
    SELECT
        CAST((b.year) as VARCHAR) date,
        name
    FROM right_table
) b
ON a.data = b.date
"""
    ) == {Table("left_table"), Table("right_table")}


def test_extract_tables_combinations() -> None:
    """
    Test a complex case with nested queries.
    """
    assert extract_tables(
        """
SELECT * FROM t1
WHERE s11 > ANY (
    SELECT * FROM t1 UNION ALL SELECT * FROM (
        SELECT t6.*, t3.* FROM t6 JOIN t3 ON t6.a = t3.a
    ) tmp_join
    WHERE NOT EXISTS (
        SELECT * FROM t3
        WHERE ROW(5*t3.s1,77)=(
            SELECT 50,11*s1 FROM t4
        )
    )
)
"""
    ) == {Table("t1"), Table("t3"), Table("t4"), Table("t6")}

    assert extract_tables(
        """
SELECT * FROM (
    SELECT * FROM (
        SELECT * FROM (
            SELECT * FROM EmployeeS
        ) AS S1
    ) AS S2
) AS S3
"""
    ) == {Table("EmployeeS")}


def test_extract_tables_with() -> None:
    """
    Test ``WITH``.
    """
    assert extract_tables(
        """
WITH
    x AS (SELECT a FROM t1),
    y AS (SELECT a AS b FROM t2),
    z AS (SELECT b AS c FROM t3)
SELECT c FROM z
"""
    ) == {Table("t1"), Table("t2"), Table("t3")}

    assert extract_tables(
        """
WITH
    x AS (SELECT a FROM t1),
    y AS (SELECT a AS b FROM x),
    z AS (SELECT b AS c FROM y)
SELECT c FROM z
"""
    ) == {Table("t1")}


def test_extract_tables_reusing_aliases() -> None:
    """
    Test that the parser follows aliases.
    """
    assert extract_tables(
        """
with q1 as ( select key from q2 where key = '5'),
q2 as ( select key from src where key = '5')
select * from (select key from q1) a
"""
    ) == {Table("src")}

    # weird query with circular dependency
    assert (
        extract_tables(
            """
with src as ( select key from q2 where key = '5'),
q2 as ( select key from src where key = '5')
select * from (select key from src) a
"""
        )
        == set()
    )


def test_extract_tables_multistatement() -> None:
    """
    Test that the parser works with multiple statements.
    """
    assert extract_tables("SELECT * FROM t1; SELECT * FROM t2") == {
        Table("t1"),
        Table("t2"),
    }
    assert extract_tables("SELECT * FROM t1; SELECT * FROM t2;") == {
        Table("t1"),
        Table("t2"),
    }
    assert extract_tables(
        "ADD JAR file:///hive.jar; SELECT * FROM t1;",
        engine="hive",
    ) == {Table("t1")}


def test_extract_tables_complex() -> None:
    """
    Test a few complex queries.
    """
    assert extract_tables(
        """
SELECT sum(m_examples) AS "sum__m_example"
FROM (
    SELECT
        COUNT(DISTINCT id_userid) AS m_examples,
        some_more_info
    FROM my_b_table b
    JOIN my_t_table t ON b.ds=t.ds
    JOIN my_l_table l ON b.uid=l.uid
    WHERE
        b.rid IN (
            SELECT other_col
            FROM inner_table
        )
        AND l.bla IN ('x', 'y')
    GROUP BY 2
    ORDER BY 2 ASC
) AS "meh"
ORDER BY "sum__m_example" DESC
LIMIT 10;
"""
    ) == {
        Table("my_l_table"),
        Table("my_b_table"),
        Table("my_t_table"),
        Table("inner_table"),
    }

    assert extract_tables(
        """
SELECT *
FROM table_a AS a, table_b AS b, table_c as c
WHERE a.id = b.id and b.id = c.id
"""
    ) == {Table("table_a"), Table("table_b"), Table("table_c")}

    assert extract_tables(
        """
SELECT somecol AS somecol
FROM (
    WITH bla AS (
        SELECT col_a
        FROM a
        WHERE
            1=1
            AND column_of_choice NOT IN (
                SELECT interesting_col
                FROM b
            )
    ),
    rb AS (
        SELECT yet_another_column
        FROM (
            SELECT a
            FROM c
            GROUP BY the_other_col
        ) not_table
        LEFT JOIN bla foo
        ON foo.prop = not_table.bad_col0
        WHERE 1=1
        GROUP BY
            not_table.bad_col1 ,
            not_table.bad_col2 ,
        ORDER BY not_table.bad_col_3 DESC ,
            not_table.bad_col4 ,
            not_table.bad_col5
    )
    SELECT random_col
    FROM d
    WHERE 1=1
    UNION ALL SELECT even_more_cols
    FROM e
    WHERE 1=1
    UNION ALL SELECT lets_go_deeper
    FROM f
    WHERE 1=1
    WHERE 2=2
    GROUP BY last_col
    LIMIT 50000
)
"""
    ) == {Table("a"), Table("b"), Table("c"), Table("d"), Table("e"), Table("f")}


def test_extract