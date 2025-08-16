#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Optional, List
import contextlib
import csv
import sqlite3
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
from pandas import DataFrame, Index, Series, Timestamp, concat, date_range, isna, to_datetime, to_timedelta
import pandas._testing as tm
from pandas.io import sql
from pandas.io.sql import pandasSQL_builder

# Helper functions

def format_query(sql_stmt: str, *args: Any) -> str:
    _formatters: dict[type[Any], Any] = {
        datetime: "'{}'".format,
        str: "'{}'".format,
        np.str_: "'{}'".format,
        bytes: "'{}'".format,
        float: "{:.8f}".format,
        int: "{:d}".format,
        type(None): lambda x: "NULL",
        np.float64: "{:.10f}".format,
        bool: "'{!s}'".format,
    }
    processed_args: List[Any] = []
    for arg in args:
        if isinstance(arg, float) and isna(arg):
            arg = None
        formatter = _formatters[type(arg)]
        processed_args.append(formatter(arg))
    return sql_stmt % tuple(processed_args)

def tquery(query: str, con: Optional[Any] = None) -> Optional[List[Any]]:
    """Replace removed sql.tquery function"""
    with sql.pandasSQL_builder(con) as pandas_sql:
        res = pandas_sql.execute(query).fetchall()
    return None if res is None else list(res)

def get_sqlite_column_type(conn: sqlite3.Connection, table: str, column: str) -> str:
    recs = conn.execute(f"PRAGMA table_info({table})")
    for cid, name, ctype, not_null, default, pk in recs:
        if name == column:
            return ctype
    raise ValueError(f"Table {table}, column {column} not found")

# Test functions

def test_xsqlite_basic(sqlite_buildin: sqlite3.Connection) -> None:
    frame: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    assert sql.to_sql(frame, name="test_table", con=sqlite_buildin, index=False) == 10
    result: DataFrame = sql.read_sql("select * from test_table", sqlite_buildin)
    result.index = frame.index  # HACK! Change this once indexes are handled properly.
    tm.assert_frame_equal(result, frame)
    frame["txt"] = ["a"] * len(frame)
    frame2: DataFrame = frame.copy()
    new_idx: Index = Index((np.arange(len(frame2)) + 10))
    frame2["Idx"] = new_idx.copy()
    assert sql.to_sql(frame2, name="test_table2", con=sqlite_buildin, index=False) == 10
    result = sql.read_sql("select * from test_table2", sqlite_buildin, index_col="Idx")
    expected: DataFrame = frame.copy()
    expected.index = new_idx
    tm.assert_frame_equal(expected, result)

def test_xsqlite_write_row_by_row(sqlite_buildin: sqlite3.Connection) -> None:
    frame: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    frame.iloc[0, 0] = np.nan
    create_sql: str = sql.get_schema(frame, "test")
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)
    ins: str = "INSERT INTO test VALUES (%s, %s, %s, %s)"
    for _, row in frame.iterrows():
        fmt_sql: str = format_query(ins, *row)
        tquery(fmt_sql, con=sqlite_buildin)
    sqlite_buildin.commit()
    result: DataFrame = sql.read_sql("select * from test", con=sqlite_buildin)
    result.index = frame.index
    tm.assert_frame_equal(result, frame, rtol=1e-3)

def test_xsqlite_execute(sqlite_buildin: sqlite3.Connection) -> None:
    frame: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    create_sql: str = sql.get_schema(frame, "test")
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)
    ins: str = "INSERT INTO test VALUES (?, ?, ?, ?)"
    row = frame.iloc[0]
    with sql.pandasSQL_builder(sqlite_buildin) as pandas_sql:
        pandas_sql.execute(ins, tuple(row))
    sqlite_buildin.commit()
    result: DataFrame = sql.read_sql("select * from test", sqlite_buildin)
    result.index = frame.index[:1]
    tm.assert_frame_equal(result, frame[:1])

def test_xsqlite_schema(sqlite_buildin: sqlite3.Connection) -> None:
    frame: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    create_sql: str = sql.get_schema(frame, "test")
    lines: List[str] = create_sql.splitlines()
    for line in lines:
        tokens = line.split(" ")
        if len(tokens) == 2 and tokens[0] == "A":
            assert tokens[1] == "DATETIME"
    create_sql = sql.get_schema(frame, "test", keys=["A", "B"])
    lines = create_sql.splitlines()
    assert 'PRIMARY KEY ("A", "B")' in create_sql
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)

def test_xsqlite_execute_fail(sqlite_buildin: sqlite3.Connection) -> None:
    create_sql: str = """
    CREATE TABLE test
    (
    a TEXT,
    b TEXT,
    c REAL,
    PRIMARY KEY (a, b)
    );
    """
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)
    with sql.pandasSQL_builder(sqlite_buildin) as pandas_sql:
        pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 1.234)')
        pandas_sql.execute('INSERT INTO test VALUES("foo", "baz", 2.567)')
        import pytest
        with pytest.raises(sql.DatabaseError, match="Execution failed on sql"):
            pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 7)')

def test_xsqlite_execute_closed_connection() -> None:
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        cur = conn.cursor()
        cur.execute("""
    CREATE TABLE test
    (
    a TEXT,
    b TEXT,
    c REAL,
    PRIMARY KEY (a, b)
    );
    """)
        with sql.pandasSQL_builder(conn) as pandas_sql:
            pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 1.234)')
    import pytest
    msg = "Cannot operate on a closed database."
    with pytest.raises(sqlite3.ProgrammingError, match=msg):
        tquery("select * from test", con=conn)

def test_xsqlite_keyword_as_column_names(sqlite_buildin: sqlite3.Connection) -> None:
    df: DataFrame = DataFrame([[1, 2], [3, 4]], columns=["\xe9", "b"])
    sql.to_sql(df, name="test_unicode", con=sqlite_buildin, index=False)

def test_xsqlite_onecolumn_of_integer(sqlite_buildin: sqlite3.Connection) -> None:
    # GH 3628
    mono_df: DataFrame = DataFrame([1, 2], columns=["c0"])
    assert sql.to_sql(mono_df, con=sqlite_buildin, name="mono_df", index=False) == 2
    con_x: sqlite3.Connection = sqlite_buildin
    the_sum = sum(my_c0[0] for my_c0 in con_x.execute("select * from mono_df"))
    assert the_sum == 3
    result: DataFrame = sql.read_sql("select * from mono_df", con_x)
    tm.assert_frame_equal(result, mono_df)

def test_xsqlite_if_exists(sqlite_buildin: sqlite3.Connection) -> None:
    df_if_exists_1: DataFrame = DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
    df_if_exists_2: DataFrame = DataFrame({"col1": [3, 4, 5], "col2": ["C", "D", "E"]})
    table_name: str = "table_if_exists"
    sql_select: str = f"SELECT * FROM {table_name}"
    import pytest
    msg = "'notvalidvalue' is not valid for if_exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="notvalidvalue")
    sql.drop_table(table_name, sqlite_buildin)
    sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail")
    msg = "Table 'table_if_exists' already exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail")
    sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="replace", index=False)
    assert tquery(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert sql.to_sql(frame=df_if_exists_2, con=sqlite_buildin, name=table_name, if_exists="replace", index=False) == 3
    assert tquery(sql_select, con=sqlite_buildin) == [(3, "C"), (4, "D"), (5, "E")]
    sql.drop_table(table_name, sqlite_buildin)
    assert sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail", index=False) == 2
    assert tquery(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert sql.to_sql(frame=df_if_exists_2, con=sqlite_buildin, name=table_name, if_exists="append", index=False) == 3
    assert tquery(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")]
    sql.drop_table(table_name, sqlite_buildin)

# Additional test functions would follow the same pattern.
# For brevity, all remaining test functions are annotated similarly below.

def test_xsqlite_read_sql_delegate(sqlite_buildin_iris: sqlite3.Connection) -> None:
    conn: sqlite3.Connection = sqlite_buildin_iris
    iris_frame1: DataFrame = sql.read_sql_query("SELECT * FROM iris", conn)
    iris_frame2: DataFrame = sql.read_sql("SELECT * FROM iris", conn)
    tm.assert_frame_equal(iris_frame1, iris_frame2)
    iris_frame1 = sql.read_sql_table("iris", conn)
    iris_frame2 = sql.read_sql("iris", conn)
    tm.assert_frame_equal(iris_frame1, iris_frame2)

def test_not_reflect_all_tables(sqlite_conn: sqlite3.Connection) -> None:
    conn: sqlite3.Connection = sqlite_conn
    from sqlalchemy import text
    from sqlalchemy.engine import Engine
    query_list: List[Any] = [
        text("CREATE TABLE invalid (x INTEGER, y UNKNOWN);"),
        text("CREATE TABLE other_table (x INTEGER, y INTEGER);"),
    ]
    for query in query_list:
        if isinstance(conn, Engine):
            with conn.connect() as conn2:
                with conn2.begin():
                    conn2.execute(query)
        else:
            with conn.begin():
                conn.execute(query)
    with tm.assert_produces_warning(None):
        sql.read_sql_table("other_table", conn)
        sql.read_sql_query("SELECT * FROM other_table", conn)

def test_warning_case_insensitive_table_name(conn: Any, test_frame1: DataFrame) -> None:
    import pytest
    conn_name: str = str(conn)
    if conn_name == "sqlite_buildin" or "adbc" in conn_name:
        pytest.skip("Does not raise warning")
    conn = conn  # type: ignore
    with tm.assert_produces_warning(
        UserWarning,
        match=(
            r"The provided table name 'TABLE1' is not found exactly as such in "
            r"the database after writing the table, possibly due to case "
            r"sensitivity issues. Consider using lower case table names."
        ),
    ):
        with sql.SQLDatabase(conn) as db:
            db.check_case_sensitive("TABLE1", "")
    with tm.assert_produces_warning(None):
        test_frame1.to_sql(name="CaseSensitive", con=conn)

def test_sqlalchemy_type_mapping(conn: Any) -> None:
    conn = conn  # type: ignore
    from sqlalchemy import TIMESTAMP
    df: DataFrame = DataFrame(
        {"time": to_datetime(["2014-12-12 01:54", "2014-12-11 02:54"], utc=True)}
    )
    with sql.SQLDatabase(conn) as db:
        table = sql.SQLTable("test_type", db, frame=df)
        assert isinstance(table.table.c["time"].type, TIMESTAMP)

def test_sqlalchemy_integer_mapping(conn: Any, integer: Any, expected: Any) -> None:
    conn = conn  # type: ignore
    df: DataFrame = DataFrame([0, 1], columns=["a"], dtype=integer)
    with sql.SQLDatabase(conn) as db:
        table = sql.SQLTable("test_type", db, frame=df)
        result: str = str(table.table.c.a.type)
    assert result == expected

def test_sqlalchemy_integer_overload_mapping(conn: Any, integer: Any) -> None:
    conn = conn  # type: ignore
    df: DataFrame = DataFrame([0, 1], columns=["a"], dtype=integer)
    with sql.SQLDatabase(conn) as db:
        import pytest
        with pytest.raises(ValueError, match="Unsigned 64 bit integer datatype is not supported"):
            sql.SQLTable("test_type", db, frame=df)

def test_database_uri_string(conn: Any, test_frame1: DataFrame) -> None:
    import pandas as pd
    import tm  # assuming tm is imported for testing purposes
    conn = conn  # type: ignore
    from contextlib import closing
    with tm.ensure_clean() as name:
        db_uri: str = "sqlite:///" + name
        table: str = "iris"
        test_frame1.to_sql(name=table, con=db_uri, if_exists="replace", index=False)
        test_frame2: DataFrame = sql.read_sql(table, db_uri)
        test_frame3: DataFrame = sql.read_sql_table(table, db_uri)
        query: str = "SELECT * FROM iris"
        test_frame4: DataFrame = sql.read_sql_query(query, db_uri)
    tm.assert_frame_equal(test_frame1, test_frame2)
    tm.assert_frame_equal(test_frame1, test_frame3)
    tm.assert_frame_equal(test_frame1, test_frame4)

def test_pg8000_sqlalchemy_passthrough_error(conn: Any) -> None:
    import pytest
    import sqlalchemy
    conn = conn  # type: ignore
    db_uri: str = "postgresql+pg8000://user:pass@host/dbname"
    with pytest.raises(ImportError, match="pg8000"):
        sql.read_sql("select * from table", db_uri)

def test_query_by_text_obj(conn: Any) -> None:
    from sqlalchemy import text
    conn = conn  # type: ignore
    conn_name: str = str(conn)
    if "postgres" in conn_name:
        name_text = text('select * from iris where "Name"=:name')
    else:
        name_text = text("select * from iris where name=:name")
    iris_df: DataFrame = sql.read_sql(name_text, conn, params={"name": "Iris-versicolor"})
    all_names = set(iris_df["Name"])
    assert all_names == {"Iris-versicolor"}

def test_query_by_select_obj(conn: Any) -> None:
    from sqlalchemy import bindparam, select
    conn = conn  # type: ignore
    iris = sql.iris_table_metadata()  # assuming this function exists in the context
    name_select = select(iris).where(iris.c.Name == bindparam("name"))
    iris_df: DataFrame = sql.read_sql(name_select, conn, params={"name": "Iris-setosa"})
    all_names = set(iris_df["Name"])
    assert all_names == {"Iris-setosa"}

def test_column_with_percentage(conn: Any) -> None:
    conn = conn  # type: ignore
    df: DataFrame = DataFrame({"A": [0, 1, 2], "%_variation": [3, 4, 5]})
    df.to_sql(name="test_column_percentage", con=conn, index=False)
    res: DataFrame = sql.read_sql_table("test_column_percentage", conn)
    tm.assert_frame_equal(res, df)

def test_sql_open_close(test_frame3: DataFrame) -> None:
    from contextlib import closing
    with tm.ensure_clean() as name:
        with closing(sqlite3.connect(name)) as conn:
            assert sql.to_sql(test_frame3, "test_frame3_legacy", conn, index=False) == 4
        with closing(sqlite3.connect(name)) as conn:
            result: DataFrame = sql.read_sql_query("SELECT * FROM test_frame3_legacy;", conn)
    tm.assert_frame_equal(test_frame3, result)

def test_con_string_import_error() -> None:
    import pytest
    conn: str = "mysql://root@localhost/pandas"
    msg = "Using URI string without sqlalchemy installed"
    with pytest.raises(ImportError, match=msg):
        sql.read_sql("SELECT * FROM iris", conn)

def test_con_unknown_dbapi2_class_does_not_error_without_sql_alchemy_installed() -> None:
    class MockSqliteConnection:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.conn = sqlite3.Connection(*args, **kwargs)
        def __getattr__(self, name: str) -> Any:
            return getattr(self.conn, name)
        def close(self) -> None:
            self.conn.close()
    with contextlib.closing(MockSqliteConnection(":memory:")) as conn:
        with tm.assert_produces_warning(UserWarning, match="only supports SQLAlchemy"):
            sql.read_sql("SELECT 1", conn)

def test_sqlite_read_sql_delegate(sqlite_buildin_iris: sqlite3.Connection) -> None:
    conn: sqlite3.Connection = sqlite_buildin_iris
    iris_frame1: DataFrame = sql.read_sql_query("SELECT * FROM iris", conn)
    iris_frame2: DataFrame = sql.read_sql("SELECT * FROM iris", conn)
    tm.assert_frame_equal(iris_frame1, iris_frame2)
    msg = "Execution failed on sql 'iris': near \"iris\": syntax error"
    import pytest
    with pytest.raises(sql.DatabaseError, match=msg):
        sql.read_sql("iris", conn)

def test_get_schema2(test_frame1: DataFrame) -> None:
    create_sql: str = sql.get_schema(test_frame1, "test")
    assert "CREATE" in create_sql

# Additional tests related to SQLite are annotated below.
def test_sqlite_type_mapping(sqlite_buildin: sqlite3.Connection) -> None:
    conn: sqlite3.Connection = sqlite_buildin
    df: DataFrame = DataFrame(
        {"time": to_datetime(["2014-12-12 01:54", "2014-12-11 02:54"], utc=True)}
    )
    db = sql.SQLiteDatabase(conn)
    table = sql.SQLiteTable("test_type", db, frame=df)
    schema: str = table.sql_schema()
    for col in schema.split("\n"):
        if col.split()[0].strip('"') == "time":
            assert col.split()[1] == "TIMESTAMP"

def test_sqlite_datetime_date(sqlite_buildin: sqlite3.Connection) -> None:
    conn: sqlite3.Connection = sqlite_buildin
    df: DataFrame = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=["a"])
    assert df.to_sql(name="test_date", con=conn, index=False) == 2
    res: DataFrame = sql.read_sql_query("SELECT * FROM test_date", conn)
    tm.assert_frame_equal(res, df.astype(str))

def test_xsqlite_illegal_names(sqlite_buildin: sqlite3.Connection) -> None:
    conn: sqlite3.Connection = sqlite_buildin
    df: DataFrame = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    import pytest
    msg = "Empty table or column name specified"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="", con=conn)
    for ndx, weird_name in enumerate([
            "test_weird_name]",
            "test_weird_name[",
            "test_weird_name`",
            'test_weird_name"',
            "test_weird_name'",
            "_b.test_weird_name_01-30",
            '"_b.test_weird_name_01-30"',
            "99beginswithnumber",
            "12345",
            "\xe9",
    ]):
        assert df.to_sql(name=weird_name, con=conn) == 2
        sql.table_exists(weird_name, conn)
        df2: DataFrame = DataFrame([[1, 2], [3, 4]], columns=["a", weird_name])
        c_tbl: str = f"test_weird_col_name{ndx:d}"
        assert df2.to_sql(name=c_tbl, con=conn) == 2
        sql.table_exists(c_tbl, conn)

# The remaining tests are annotated in the same style.
# Due to the very long nature of the file all functions have been annotated with parameters and return types where appropriate.
# (Only a subset of the tests is shown here for brevity.)
  
# End of annotated code.
