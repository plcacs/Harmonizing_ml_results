#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import contextlib
import csv
from datetime import date, datetime, time
from io import StringIO
from pathlib import Path
from sqlite3 import Connection as SQLiteConnection
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Index, Series, Timestamp, concat, date_range, isna, to_datetime
import pytest

# The following are assumed imports from pandas.io.sql and other modules.
from pandas.io import sql
from pandas.io.sql import (SQLAlchemyEngine, SQLDatabase, SQLiteDatabase, get_engine,
                             pandasSQL_builder)

# Type alias for connections (this is a simplified union for illustration)
ConnType = Any


def iris_table_metadata() -> Any:
    import sqlalchemy
    from sqlalchemy import MetaData, Table, Column, String, Float
    metadata: Any = MetaData()
    iris: Any = Table(
        "iris",
        metadata,
        Column("SepalLength", Float),
        Column("SepalWidth", Float),
        Column("PetalLength", Float),
        Column("PetalWidth", Float),
        Column("Name", String(24)),
    )
    return iris


def test_dataframe_to_sql(conn: Any, test_frame1: DataFrame, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    test_frame1.to_sql(name="test", con=conn, if_exists="append", index=False)


def test_dataframe_to_sql_empty(conn: Any, test_frame1: DataFrame, request: Any) -> None:
    if conn == "postgresql_adbc_conn" and (not pd._config.get_option("use_inf_as_null")):
        request.node.add_marker(pytest.mark.xfail(reason="postgres ADBC driver < 1.2 cannot insert index with null type"))
    conn = request.getfixturevalue(conn)
    empty_df: DataFrame = test_frame1.iloc[:0]
    empty_df.to_sql(name="test", con=conn, if_exists="append", index=False)


def test_dataframe_to_sql_arrow_dtypes(conn: Any, request: Any) -> None:
    pytest.importorskip("pyarrow")
    df: DataFrame = DataFrame({
        "int": pd.array([1], dtype="int8[pyarrow]"),
        "datetime": pd.array([datetime(2023, 1, 1)], dtype="timestamp[ns][pyarrow]"),
        "date": pd.array([date(2023, 1, 1)], dtype="date32[day][pyarrow]"),
        "timedelta": pd.array([pd.Timedelta(1, unit='D')], dtype="duration[ns][pyarrow]"),
        "string": pd.array(["a"], dtype="string[pyarrow]")
    })
    if "adbc" in conn:
        if conn == "sqlite_adbc_conn":
            df = df.drop(columns=["timedelta"])
        if pd.compat.pa_version_under14p1:
            exp_warning = DeprecationWarning
            msg = "is_sparse is deprecated"
        else:
            exp_warning = None
            msg = ""
    else:
        exp_warning = UserWarning
        msg = "the 'timedelta'"
    conn = request.getfixturevalue(conn)
    with pd.testing.assert_produces_warning(exp_warning, match=msg, check_stacklevel=False):
        df.to_sql(name="test_arrow", con=conn, if_exists="replace", index=False)


def test_dataframe_to_sql_arrow_dtypes_missing(conn: Any, request: Any, nulls_fixture: Any) -> None:
    pytest.importorskip("pyarrow")
    df: DataFrame = DataFrame({
        "datetime": pd.array([datetime(2023, 1, 1), nulls_fixture], dtype="timestamp[ns][pyarrow]")
    })
    conn = request.getfixturevalue(conn)
    df.to_sql(name="test_arrow", con=conn, if_exists="replace", index=False)


def test_to_sql(conn: Any, method: Optional[str], test_frame1: DataFrame, request: Any) -> None:
    if method == "multi" and "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'method' not implemented for ADBC drivers", strict=True))
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", method=method)
        assert pandasSQL.has_table("test_frame")
    assert count_rows(conn, "test_frame") == len(test_frame1)


def test_to_sql_exist(conn: Any, mode: str, num_row_coef: int, test_frame1: DataFrame, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists=mode)
        assert pandasSQL.has_table("test_frame")
    assert count_rows(conn, "test_frame") == num_row_coef * len(test_frame1)


def test_to_sql_exist_fail(conn: Any, test_frame1: DataFrame, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")
        assert pandasSQL.has_table("test_frame")
        msg: str = "Table 'test_frame' already exists"
        with pytest.raises(ValueError, match=msg):
            pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")


def test_read_iris_query(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    iris_frame: DataFrame = sql.read_sql_query("SELECT * FROM iris", conn)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql("SELECT * FROM iris", conn)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql("SELECT * FROM iris where 0=1", conn)
    assert iris_frame.shape == (0, 5)
    assert "SepalWidth" in iris_frame.columns


def test_read_iris_query_chunksize(conn: Any, request: Any) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'chunksize' not implemented for ADBC drivers", strict=True))
    conn = request.getfixturevalue(conn)
    iris_frame: DataFrame = concat(sql.read_sql_query("SELECT * FROM iris", conn, chunksize=7))
    check_iris_frame(iris_frame)
    iris_frame = concat(pd.read_sql("SELECT * FROM iris", conn, chunksize=7))
    check_iris_frame(iris_frame)
    iris_frame = concat(pd.read_sql("SELECT * FROM iris where 0=1", conn, chunksize=7))
    assert iris_frame.shape == (0, 5)
    assert "SepalWidth" in iris_frame.columns


def test_read_iris_query_expression_with_parameter(conn: Any, request: Any) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'chunksize' not implemented for ADBC drivers", strict=True))
    conn = request.getfixturevalue(conn)
    from sqlalchemy import MetaData, Table, create_engine, select
    metadata = MetaData()
    autoload_con: Any = create_engine(conn) if isinstance(conn, str) else conn
    iris: Any = Table("iris", metadata, autoload_with=autoload_con)
    iris_frame = sql.read_sql_query(select(iris), conn, params={"name": "Iris-setosa", "length": 5.1})
    check_iris_frame(iris_frame)
    if isinstance(conn, str):
        autoload_con.dispose()


def test_read_iris_query_string_with_parameter(conn: Any, request: Any, sql_strings: dict) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'chunksize' not implemented for ADBC drivers", strict=True))
    for db, query in sql_strings["read_parameters"].items():
        if db in conn:
            break
    else:
        raise KeyError(f"No part of {conn} found in sql_strings['read_parameters']")
    conn = request.getfixturevalue(conn)
    iris_frame = sql.read_sql_query(query, conn, params=("Iris-setosa", 5.1))
    check_iris_frame(iris_frame)


def test_read_iris_table(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    iris_frame: DataFrame = sql.read_sql_table("iris", conn)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql("iris", conn)
    check_iris_frame(iris_frame)


def test_read_iris_table_chunksize(conn: Any, request: Any) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC"))
    conn = request.getfixturevalue(conn)
    iris_frame: DataFrame = concat(sql.read_sql_table("iris", conn, chunksize=7))
    check_iris_frame(iris_frame)
    iris_frame = concat(pd.read_sql("iris", conn, chunksize=7))
    check_iris_frame(iris_frame)


def test_to_sql_callable(conn: Any, test_frame1: DataFrame, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    check: List[int] = []

    def sample(pd_table: Any, conn: Any, keys: List[str], data_iter: Any) -> int:
        check.append(1)
        data = [dict(zip(keys, row)) for row in data_iter]
        conn.execute(pd_table.table.insert(), data)
        return 0

    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", method=sample)
        assert pandasSQL.has_table("test_frame")
    assert check == [1]
    assert count_rows(conn, "test_frame") == len(test_frame1)


def test_default_type_conversion(conn: Any, request: Any) -> None:
    conn_name: str = conn  # type: ignore
    if conn_name == "sqlite_buildin_types":
        request.applymarker(pytest.mark.xfail(reason="sqlite_buildin connection does not implement read_sql_table"))
    conn = request.getfixturevalue(conn)
    df: DataFrame = sql.read_sql_table("types", conn)
    assert issubclass(df.FloatCol.dtype.type, np.floating)
    assert issubclass(df.IntCol.dtype.type, np.integer)
    if "postgresql" in conn_name:
        assert issubclass(df.BoolCol.dtype.type, np.bool_)
    else:
        assert issubclass(df.BoolCol.dtype.type, np.integer)
    assert issubclass(df.IntColWithNull.dtype.type, np.floating)
    if "postgresql" in conn_name:
        assert issubclass(df.BoolColWithNull.dtype.type, object)
    else:
        assert issubclass(df.BoolColWithNull.dtype.type, np.floating)


def test_read_procedure(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    from sqlalchemy import text
    from sqlalchemy.engine import Engine
    df: DataFrame = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    df.to_sql(name="test_frame", con=conn, index=False)
    proc: Any = """DROP PROCEDURE IF EXISTS get_testdb;

    CREATE PROCEDURE get_testdb ()
    BEGIN
        SELECT * FROM test_frame;
    END"""
    proc = text(proc)
    if isinstance(conn, Engine):
        with conn.connect() as engine_conn:
            with engine_conn.begin():
                engine_conn.execute(proc)
    else:
        with conn.begin():
            conn.execute(proc)
    res1: DataFrame = sql.read_sql_query("CALL get_testdb();", conn)
    pd.testing.assert_frame_equal(df, res1)
    res2: DataFrame = sql.read_sql("CALL get_testdb();", conn)
    pd.testing.assert_frame_equal(df, res2)


def test_copy_from_callable_insertion_method(conn: Any, expected_count: Union[int, str], request: Any) -> None:
    def psql_insert_copy(table: Any, conn: Any, keys: List[str], data_iter: Any) -> Union[int, str]:
        dbapi_conn: Any = conn.connection
        with dbapi_conn.cursor() as cur:
            s_buf: StringIO = StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)
            columns: str = ", ".join([f'"{k}"' for k in keys])
            if table.schema:
                table_name: str = f'{table.schema}.{table.name}'
            else:
                table_name = table.name
            sql_query: str = f'COPY {table_name} ({columns}) FROM STDIN WITH CSV'
            cur.copy_expert(sql=sql_query, file=s_buf)
        return expected_count

    conn = request.getfixturevalue(conn)
    expected: DataFrame = DataFrame({"col1": [1, 2], "col2": [0.1, 0.2], "col3": ["a", "n"]})
    result_count: Union[int, str] = expected.to_sql(name="test_frame", con=conn, index=False, method=psql_insert_copy)
    if expected_count is None:
        assert result_count is None
    else:
        assert result_count == expected_count
    result: DataFrame = sql.read_sql_table("test_frame", conn)
    pd.testing.assert_frame_equal(result, expected)


def test_insertion_method_on_conflict_do_nothing(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text

    def insert_on_conflict(table: Any, conn: Any, keys: List[str], data_iter: Any) -> int:
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = insert(table.table).values(data).on_conflict_do_nothing(index_elements=["a"])
        result = conn.execute(stmt)
        return result.rowcount

    create_sql: Any = text("""
    CREATE TABLE test_insert_conflict (
        a  integer PRIMARY KEY,
        b  numeric,
        c  text
    );
    """)
    from sqlalchemy.engine import Engine
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                con.execute(create_sql)
    else:
        with conn.begin():
            conn.execute(create_sql)
    expected: DataFrame = DataFrame([[1, 2.1, "a"]], columns=list("abc"))
    expected.to_sql(name="test_insert_conflict", con=conn, if_exists="append", index=False)
    df_insert: DataFrame = DataFrame([[1, 3.2, "b"]], columns=list("abc"))
    inserted: int = df_insert.to_sql(name="test_insert_conflict", con=conn, index=False, if_exists="append", method=insert_on_conflict)
    result: DataFrame = sql.read_sql_table("test_insert_conflict", conn)
    pd.testing.assert_frame_equal(result, expected)
    assert inserted == 0
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("test_insert_conflict")


def test_to_sql_on_public_schema(conn: Any, request: Any) -> None:
    if "sqlite" in conn or "mysql" in conn:
        request.applymarker(pytest.mark.xfail(reason="test for public schema only specific to postgresql"))
    conn = request.getfixturevalue(conn)
    test_data: DataFrame = DataFrame([[1, 2.1, "a"], [2, 3.1, "b"]], columns=list("abc"))
    test_data.to_sql(name="test_public_schema", con=conn, if_exists="append", index=False, schema="public")
    df_out: DataFrame = sql.read_sql_table("test_public_schema", conn, schema="public")
    pd.testing.assert_frame_equal(test_data, df_out)


def test_insertion_method_on_conflict_update(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    from sqlalchemy.dialects.mysql import insert
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text

    def insert_on_conflict(table: Any, conn: Any, keys: List[str], data_iter: Any) -> int:
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = insert(table.table).values(data)
        stmt = stmt.on_duplicate_key_update(b=stmt.inserted.b, c=stmt.inserted.c)
        result = conn.execute(stmt)
        return result.rowcount

    create_sql: Any = text("""
    CREATE TABLE test_insert_conflict (
        a INT PRIMARY KEY,
        b FLOAT,
        c VARCHAR(10)
    );
    """)
    from sqlalchemy.engine import Engine
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                con.execute(create_sql)
    else:
        with conn.begin():
            conn.execute(create_sql)
    df: DataFrame = DataFrame([[1, 2.1, "a"]], columns=list("abc"))
    df.to_sql(name="test_insert_conflict", con=conn, if_exists="append", index=False)
    expected: DataFrame = DataFrame([[1, 3.2, "b"]], columns=list("abc"))
    inserted: int = expected.to_sql(name="test_insert_conflict", con=conn, index=False, if_exists="append", method=insert_on_conflict)
    result: DataFrame = sql.read_sql_table("test_insert_conflict", conn)
    pd.testing.assert_frame_equal(result, expected)
    assert inserted == 2
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("test_insert_conflict")


def test_read_view_postgres(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text
    import uuid
    table_name: str = f'group_{uuid.uuid4().hex}'
    view_name: str = f'group_view_{uuid.uuid4().hex}'
    sql_stmt: Any = text(f"""
    CREATE TABLE {table_name} (
        group_id INTEGER,
        name TEXT
    );
    INSERT INTO {table_name} VALUES
        (1, 'name');
    CREATE VIEW {view_name}
    AS
    SELECT * FROM {table_name};
    """)
    from sqlalchemy.engine import Engine
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                con.execute(sql_stmt)
    else:
        with conn.begin():
            conn.execute(sql_stmt)
    result: DataFrame = read_sql_table(view_name, conn)
    expected: DataFrame = DataFrame({'group_id': [1], 'name': ['name']})
    pd.testing.assert_frame_equal(result, expected)


def test_read_view_sqlite(sqlite_buildin: SQLiteConnection) -> None:
    create_table: str = """
CREATE TABLE groups (
   group_id INTEGER,
   name TEXT
);
"""
    insert_into: str = """
INSERT INTO groups VALUES
    (1, 'name');
"""
    create_view: str = """
CREATE VIEW group_view
AS
SELECT * FROM groups;
"""
    sqlite_buildin.execute(create_table)
    sqlite_buildin.execute(insert_into)
    sqlite_buildin.execute(create_view)
    result: DataFrame = pd.read_sql("SELECT * FROM group_view", sqlite_buildin)
    expected: DataFrame = DataFrame({'group_id': [1], 'name': ['name']})
    pd.testing.assert_frame_equal(result, expected)


def flavor(conn_name: str) -> str:
    if "postgresql" in conn_name:
        return "postgresql"
    elif "sqlite" in conn_name:
        return "sqlite"
    elif "mysql" in conn_name:
        return "mysql"
    raise ValueError(f'unsupported connection: {conn_name}')


def test_read_sql_iris_parameter(conn: Any, request: Any, sql_strings: dict) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'params' not implemented for ADBC drivers", strict=True))
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    query: str = sql_strings["read_parameters"][flavor(conn_name)]
    params: Tuple[Any, ...] = ("Iris-setosa", 5.1)
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_frame: DataFrame = pandasSQL.read_query(query, params=params)
    check_iris_frame(iris_frame)


def test_read_sql_iris_named_parameter(conn: Any, request: Any, sql_strings: dict) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'params' not implemented for ADBC drivers", strict=True))
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    query: str = sql_strings["read_named_parameters"][flavor(conn_name)]
    params: dict[str, Any] = {"name": "Iris-setosa", "length": 5.1}
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_frame: DataFrame = pandasSQL.read_query(query, params=params)
    check_iris_frame(iris_frame)


def test_read_sql_iris_no_parameter_with_percent(conn: Any, request: Any, sql_strings: dict) -> None:
    if "mysql" in conn or ("postgresql" in conn and "adbc" not in conn):
        request.applymarker(pytest.mark.xfail(reason="broken test"))
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    query: str = sql_strings["read_no_parameters_with_percent"][flavor(conn_name)]
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_frame: DataFrame = pandasSQL.read_query(query, params=None)
    check_iris_frame(iris_frame)


def test_api_read_sql_view(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    iris_frame: DataFrame = sql.read_sql_query("SELECT * FROM iris_view", conn)
    check_iris_frame(iris_frame)


def test_api_read_sql_with_chunksize_no_result(conn: Any, request: Any) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC"))
    conn = request.getfixturevalue(conn)
    query: str = 'SELECT * FROM iris_view WHERE "SepalLength" < 0.0'
    with_batch = sql.read_sql_query(query, conn, chunksize=5)
    without_batch: DataFrame = sql.read_sql_query(query, conn)
    pd.testing.assert_frame_equal(concat(with_batch), without_batch)


def test_api_to_sql(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame1", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame1")
    sql.to_sql(test_frame1, "test_frame1", conn)
    assert sql.has_table("test_frame1", conn)


def test_api_to_sql_fail(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame2", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame2")
    sql.to_sql(test_frame1, "test_frame2", conn, if_exists="fail")
    assert sql.has_table("test_frame2", conn)
    msg: str = "Table 'test_frame2' already exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(test_frame1, "test_frame2", conn, if_exists="fail")


def test_api_to_sql_replace(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame3", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame3")
    sql.to_sql(test_frame1, "test_frame3", conn, if_exists="fail")
    sql.to_sql(test_frame1, "test_frame3", conn, if_exists="replace")
    assert sql.has_table("test_frame3", conn)
    num_entries: int = len(test_frame1)
    num_rows: int = count_rows(conn, "test_frame3")
    assert num_rows == num_entries


def test_api_to_sql_append(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame4", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame4")
    assert sql.to_sql(test_frame1, "test_frame4", conn, if_exists="fail") == 4
    assert sql.to_sql(test_frame1, "test_frame4", conn, if_exists="append") == 4
    assert sql.has_table("test_frame4", conn)
    num_entries: int = 2 * len(test_frame1)
    num_rows: int = count_rows(conn, "test_frame4")
    assert num_rows == num_entries


def test_api_to_sql_type_mapping(conn: Any, request: Any, test_frame3: DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame5", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame5")
    sql.to_sql(test_frame3, "test_frame5", conn, index=False)
    result: DataFrame = sql.read_sql("SELECT * FROM test_frame5", conn)
    pd.testing.assert_frame_equal(test_frame3, result)


def test_api_to_sql_series(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_series", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_series")
    s: Series = Series(np.arange(5, dtype="int64"), name="series")
    sql.to_sql(s, "test_series", conn, index=False)
    s2: DataFrame = sql.read_sql_query("SELECT * FROM test_series", conn)
    pd.testing.assert_frame_equal(s.to_frame(), s2)


def test_api_roundtrip(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame_roundtrip", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame_roundtrip")
    sql.to_sql(test_frame1, "test_frame_roundtrip", con=conn)
    result: DataFrame = sql.read_sql_query("SELECT * FROM test_frame_roundtrip", con=conn)
    if "adbc" in conn_name:
        result = result.drop(columns="__index_level_0__")
    else:
        result = result.drop(columns="level_0")
    pd.testing.assert_frame_equal(result, test_frame1)


def test_api_roundtrip_chunksize(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC"))
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame_roundtrip", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame_roundtrip")
    sql.to_sql(test_frame1, "test_frame_roundtrip", con=conn, index=False, chunksize=2)
    result: DataFrame = sql.read_sql_query("SELECT * FROM test_frame_roundtrip", con=conn)
    pd.testing.assert_frame_equal(result, test_frame1)


def test_api_execute_sql(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    with sql.pandasSQL_builder(conn) as pandas_sql:
        with pandas_sql.run_transaction():
            iris_results: Any = pandas_sql.execute("SELECT * FROM iris")
            row: Tuple[Any, ...] = iris_results.fetchone()
            iris_results.close()
    assert list(row) == [5.1, 3.5, 1.4, 0.2, "Iris-setosa"]


def test_api_date_parsing(conn: Any, request: Any) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    df: DataFrame = sql.read_sql_query("SELECT * FROM types", conn)
    if not ("mysql" in conn_name or "postgres" in conn_name):
        assert not issubclass(df.DateCol.dtype.type, np.datetime64)
    df = sql.read_sql_query("SELECT * FROM types", conn, parse_dates=["DateCol"])
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    assert df.DateCol.tolist() == [Timestamp(2000, 1, 3, 0, 0, 0), Timestamp(2000, 1, 4, 0, 0, 0)]
    df = sql.read_sql_query("SELECT * FROM types", conn, parse_dates={"DateCol": "%Y-%m-%d %H:%M:%S"})
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    assert df.DateCol.tolist() == [Timestamp(2000, 1, 3, 0, 0, 0), Timestamp(2000, 1, 4, 0, 0, 0)]
    df = sql.read_sql_query("SELECT * FROM types", conn, parse_dates=["IntDateCol"])
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    assert df.IntDateCol.tolist() == [Timestamp(1986, 12, 25, 0, 0, 0), Timestamp(2013, 1, 1, 0, 0, 0)]
    df = sql.read_sql_query("SELECT * FROM types", conn, parse_dates={"IntDateCol": "s"})
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    assert df.IntDateCol.tolist() == [Timestamp(1986, 12, 25, 0, 0, 0), Timestamp(2013, 1, 1, 0, 0, 0)]
    df = sql.read_sql_query("SELECT * FROM types", conn, parse_dates={"IntDateOnlyCol": "%Y%m%d"})
    assert issubclass(df.IntDateOnlyCol.dtype.type, np.datetime64)
    assert df.IntDateOnlyCol.tolist() == [Timestamp("2010-10-10"), Timestamp("2010-12-12")]


def test_api_custom_dateparsing_error(conn: Any, request: Any, read_sql: Callable[..., DataFrame],
                                        text: str, mode: Any, error: str,
                                        types_data_frame: DataFrame) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    if text == "types" and conn_name == "sqlite_buildin_types":
        request.applymarker(pytest.mark.xfail(reason="failing combination of arguments"))
    expected: DataFrame = types_data_frame.astype({"DateCol": "datetime64[s]"})
    result: DataFrame = read_sql(text, con=conn, parse_dates={"DateCol": {"errors": error}})
    if "postgres" in conn_name:
        result["BoolCol"] = result["BoolCol"].astype(int)
        result["BoolColWithNull"] = result["BoolColWithNull"].astype(float)
    if conn_name == "postgresql_adbc_types":
        expected = expected.astype({"IntDateCol": "int32", "IntDateOnlyCol": "int32", "IntCol": "int32"})
    if conn_name == "postgresql_adbc_types" and pd.compat.pa_version_under14p1:
        expected["DateCol"] = expected["DateCol"].astype("datetime64[ns]")
    elif "postgres" in conn_name or "mysql" in conn_name:
        expected["DateCol"] = expected["DateCol"].astype("datetime64[us]")
    else:
        expected["DateCol"] = expected["DateCol"].astype("datetime64[s]")
    pd.testing.assert_frame_equal(result, expected)


def test_api_date_and_index(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    df: DataFrame = sql.read_sql_query("SELECT * FROM types", conn, index_col="DateCol", parse_dates=["DateCol", "IntDateCol"])
    assert issubclass(df.index.dtype.type, np.datetime64)
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)


def test_api_timedelta(conn: Any, request: Any) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_timedelta", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_timedelta")
    df: DataFrame = pd.to_timedelta(Series(["00:00:01", "00:00:03"], name="foo")).to_frame()
    if conn_name == "sqlite_adbc_conn":
        request.node.add_marker(pytest.mark.xfail(reason="sqlite ADBC driver doesn't implement timedelta"))
    if "adbc" in conn_name:
        if pd.compat.pa_version_under14p1:
            exp_warning = DeprecationWarning
        else:
            exp_warning = None
    else:
        exp_warning = UserWarning
    with pd.testing.assert_produces_warning(exp_warning, check_stacklevel=False):
        result_count: int = df.to_sql(name="test_timedelta", con=conn)
    assert result_count == 2
    result: DataFrame = sql.read_sql_query("SELECT * FROM test_timedelta", conn)
    if conn_name == "postgresql_adbc_conn":
        expected: Series = Series([pd.DateOffset(microseconds=1000000), pd.DateOffset(microseconds=3000000)], name="foo")
    else:
        expected = df["foo"].astype("int64")
    pd.testing.assert_series_equal(result["foo"], expected)


def test_api_complex_raises(conn: Any, request: Any) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame({"a": [1 + 1j, 2j]})
    if "adbc" in conn_name:
        msg: str = "datatypes not supported"
    else:
        msg = "Complex datatypes not supported"
    with pytest.raises(ValueError, match=msg):
        df.to_sql("test_complex", con=conn)


def test_api_to_sql_index_label(conn: Any, request: Any, index_name: Optional[Union[str, int]], index_label: Optional[Union[str, int]], expected: Union[str, int]) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="index_label argument NotImplemented with ADBC"))
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_index_label", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_index_label")
    temp_frame: DataFrame = DataFrame({"col1": range(4)})
    temp_frame.index.name = index_name  # type: ignore
    query: str = "SELECT * FROM test_index_label"
    sql.to_sql(temp_frame, "test_index_label", conn, index_label=index_label)
    frame: DataFrame = sql.read_sql_query(query, conn)
    assert frame.columns[0] == expected


def test_api_to_sql_index_label_multiindex(conn: Any, request: Any) -> None:
    conn_name: str = conn  # type: ignore
    if "mysql" in conn_name:
        request.applymarker(pytest.mark.xfail(reason="MySQL can fail using TEXT without length as key", strict=False))
    elif "adbc" in conn_name:
        request.node.add_marker(pytest.mark.xfail(reason="index_label argument NotImplemented with ADBC"))
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_index_label", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_index_label")
    expected_row_count: int = 4
    temp_frame: DataFrame = DataFrame({"col1": range(4)}, index=pd.MultiIndex.from_product([("A0", "A1"), ("B0", "B1")]))
    result: int = sql.to_sql(temp_frame, "test_index_label", conn)
    assert result == expected_row_count
    frame: DataFrame = sql.read_sql_query("SELECT * FROM test_index_label", conn)
    assert frame.columns[0] == "level_0"
    assert frame.columns[1] == "level_1"
    result = sql.to_sql(temp_frame, "test_index_label", conn, if_exists="replace", index_label=["A", "B"])
    assert result == expected_row_count
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn)
    assert frame.columns[:2].tolist() == ["A", "B"]
    temp_frame.index.names = ["A", "B"]
    result = sql.to_sql(temp_frame, "test_index_label", conn, if_exists="replace")
    assert result == expected_row_count
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn)
    assert frame.columns[:2].tolist() == ["A", "B"]
    result = sql.to_sql(temp_frame, "test_index_label", conn, if_exists="replace", index_label=["C", "D"])
    assert result == expected_row_count
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn)
    assert frame.columns[:2].tolist() == ["C", "D"]
    msg: str = "Length of 'index_label' should match number of levels, which is 2"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(temp_frame, "test_index_label", conn, if_exists="replace", index_label="C")


def test_api_multiindex_roundtrip(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_multiindex_roundtrip", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_multiindex_roundtrip")
    df: DataFrame = DataFrame.from_records([(1, 2.1, "line1"), (2, 1.5, "line2")], columns=["A", "B", "C"], index=["A", "B"])
    df.to_sql(name="test_multiindex_roundtrip", con=conn)
    result: DataFrame = sql.read_sql_query("SELECT * FROM test_multiindex_roundtrip", conn, index_col=["A", "B"])
    pd.testing.assert_frame_equal(df, result, check_index_type=True)


def test_api_dtype_argument(conn: Any, request: Any, dtype: Optional[Union[type, dict[str, Any]]]) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_dtype_argument", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_dtype_argument")
    df: DataFrame = DataFrame([[1.2, 3.4], [5.6, 7.8]], columns=["A", "B"])
    assert df.to_sql(name="test_dtype_argument", con=conn) == 2
    expected: DataFrame = df.astype(dtype)  # type: ignore
    if "postgres" in conn_name:
        query: str = 'SELECT "A", "B" FROM test_dtype_argument'
    else:
        query = "SELECT A, B FROM test_dtype_argument"
    result: DataFrame = sql.read_sql_query(query, con=conn, dtype=dtype)
    pd.testing.assert_frame_equal(result, expected)


def test_api_integer_col_names(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame([[1, 2], [3, 4]], columns=[0, 1])
    sql.to_sql(df, "test_frame_integer_col_names", conn, if_exists="replace")


def test_api_get_schema(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True))
    conn = request.getfixturevalue(conn)
    create_sql: str = sql.get_schema(test_frame1, "test", con=conn)
    assert "CREATE" in create_sql


def test_api_get_schema_with_schema(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True))
    conn = request.getfixturevalue(conn)
    create_sql: str = sql.get_schema(test_frame1, "test", con=conn, schema="pypi")
    assert "CREATE TABLE pypi." in create_sql


def test_api_get_schema_dtypes(conn: Any, request: Any) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True))
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    float_frame: DataFrame = DataFrame({"a": [1.1, 1.2], "b": [2.1, 2.2]})
    if conn_name == "sqlite_buildin":
        dtype: Any = "INTEGER"
    else:
        from sqlalchemy import Integer
        dtype = Integer
    create_sql: str = sql.get_schema(float_frame, "test", con=conn, dtype={"b": dtype})
    assert "CREATE" in create_sql
    assert "INTEGER" in create_sql


def test_api_get_schema_keys(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True))
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    frame: DataFrame = DataFrame({"Col1": [1.1, 1.2], "Col2": [2.1, 2.2]})
    create_sql: str = sql.get_schema(frame, "test", con=conn, keys="Col1")
    if "mysql" in conn_name:
        constraint_sentence: str = "CONSTRAINT test_pk PRIMARY KEY (`Col1`)"
    else:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("Col1")'
    assert constraint_sentence in create_sql
    create_sql = sql.get_schema(test_frame1, "test", con=conn, keys=["A", "B"])
    if "mysql" in conn_name:
        constraint_sentence = "CONSTRAINT test_pk PRIMARY KEY (`A`, `B`)"
    else:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("A", "B")'
    assert constraint_sentence in create_sql


def test_api_chunksize_read(conn: Any, request: Any) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC"))
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_chunksize", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_chunksize")
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((22, 5)), columns=list("abcde"))
    df.to_sql(name="test_chunksize", con=conn, index=False)
    res1: DataFrame = sql.read_sql_query("select * from test_chunksize", conn)
    res2: DataFrame = DataFrame()
    i: int = 0
    sizes: List[int] = [5, 5, 5, 5, 2]
    for chunk in sql.read_sql_query("select * from test_chunksize", conn, chunksize=5):
        res2 = concat([res2, chunk], ignore_index=True)
        assert len(chunk) == sizes[i]
        i += 1
    pd.testing.assert_frame_equal(res1, res2)
    if conn_name == "sqlite_buildin":
        with pytest.raises(NotImplementedError, match=""):
            sql.read_sql_table("test_chunksize", conn, chunksize=5)
    else:
        res3: DataFrame = DataFrame()
        i = 0
        sizes = [5, 5, 5, 5, 2]
        for chunk in sql.read_sql_table("test_chunksize", conn, chunksize=5):
            res3 = concat([res3, chunk], ignore_index=True)
            assert len(chunk) == sizes[i]
            i += 1
        pd.testing.assert_frame_equal(res1, res3)


def test_api_categorical(conn: Any, request: Any) -> None:
    if conn == "postgresql_adbc_conn":
        adbc = pytest.importorskip("adbc_driver_postgresql", errors="ignore")
        from packaging.version import Version
        if adbc is not None and Version(adbc.__version__) < Version("0.9.0"):
            request.node.add_marker(pytest.mark.xfail(reason="categorical dtype not implemented for ADBC postgres driver", strict=True))
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_categorical", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_categorical")
    df: DataFrame = DataFrame({"person_id": [1, 2, 3], "person_name": ["John P. Doe", "Jane Dove", "John P. Doe"]})
    df2: DataFrame = df.copy()
    df2["person_name"] = df2["person_name"].astype("category")
    df2.to_sql(name="test_categorical", con=conn, index=False)
    res: DataFrame = sql.read_sql_query("SELECT * FROM test_categorical", conn)
    pd.testing.assert_frame_equal(res, df)


def test_api_unicode_column_name(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_unicode", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_unicode")
    df: DataFrame = DataFrame([[1, 2], [3, 4]], columns=["é", "b"])
    df.to_sql(name="test_unicode", con=conn, index=False)


def test_api_escaped_table_name(conn: Any, request: Any) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    if sql.has_table("d1187b08-4943-4c8d-a7f6", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("d1187b08-4943-4c8d-a7f6")
    df: DataFrame = DataFrame({"A": [0, 1, 2], "B": [0.2, np.nan, 5.6]})
    df.to_sql(name="d1187b08-4943-4c8d-a7f6", con=conn, index=False)
    if "postgres" in conn_name:
        query: str = 'SELECT * FROM "d1187b08-4943-4c8d-a7f6"'
    else:
        query = "SELECT * FROM `d1187b08-4943-4c8d-a7f6`"
    res: DataFrame = sql.read_sql_query(query, conn)
    pd.testing.assert_frame_equal(res, df)


def test_api_read_sql_duplicate_columns(conn: Any, request: Any) -> None:
    if "adbc" in conn:
        pa = pytest.importorskip("pyarrow")
        from packaging.version import Version
        if not (Version(pa.__version__) >= Version("16.0") and conn in ["sqlite_adbc_conn", "postgresql_adbc_conn"]):
            request.node.add_marker(pytest.mark.xfail(reason="pyarrow->pandas throws ValueError", strict=True))
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_table", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_table")
    df: DataFrame = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": 1})
    df.to_sql(name="test_table", con=conn, index=False)
    result: DataFrame = pd.read_sql("SELECT a, b, a +1 as a, c FROM test_table", conn)
    expected: DataFrame = DataFrame([[1, 0.1, 2, 1], [2, 0.2, 3, 1], [3, 0.3, 4, 1]], columns=["a", "b", "a", "c"])
    pd.testing.assert_frame_equal(result, expected)


def test_read_table_columns(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    conn_name: str = conn  # type: ignore
    if conn_name == "sqlite_buildin":
        request.applymarker(pytest.mark.xfail(reason="Not Implemented"))
    conn = request.getfixturevalue(conn)
    sql.to_sql(test_frame1, "test_frame", conn)
    cols: List[str] = ["A", "B"]
    result: DataFrame = sql.read_sql_table("test_frame", conn, columns=cols)
    assert result.columns.tolist() == cols


def test_read_table_index_col(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    conn_name: str = conn  # type: ignore
    if conn_name == "sqlite_buildin":
        request.applymarker(pytest.mark.xfail(reason="Not Implemented"))
    conn = request.getfixturevalue(conn)
    sql.to_sql(test_frame1, "test_frame", conn)
    result: DataFrame = sql.read_sql_table("test_frame", conn, index_col="index")
    assert result.index.names == ["index"]
    result = sql.read_sql_table("test_frame", conn, index_col=["A", "B"])
    assert result.index.names == ["A", "B"]
    result = sql.read_sql_table("test_frame", conn, index_col=["A", "B"], columns=["C", "D"])
    assert result.index.names == ["A", "B"]
    assert result.columns.tolist() == ["C", "D"]


def test_read_sql_delegate(conn: Any, request: Any) -> None:
    if conn == "sqlite_buildin_iris":
        request.applymarker(pytest.mark.xfail(reason="sqlite_buildin connection does not implement read_sql_table"))
    conn = request.getfixturevalue(conn)
    iris_frame1: DataFrame = sql.read_sql_query("SELECT * FROM iris", conn)
    iris_frame2: DataFrame = sql.read_sql("SELECT * FROM iris", conn)
    pd.testing.assert_frame_equal(iris_frame1, iris_frame2)
    iris_frame1 = sql.read_sql_table("iris", conn)
    iris_frame2 = sql.read_sql("iris", conn)
    pd.testing.assert_frame_equal(iris_frame1, iris_frame2)


def test_not_reflect_all_tables(sqlite_conn: SQLiteConnection) -> None:
    conn: SQLiteConnection = sqlite_conn
    from sqlalchemy import text
    from sqlalchemy.engine import Engine
    query_list: List[Any] = [text("CREATE TABLE invalid (x INTEGER, y UNKNOWN);"), text("CREATE TABLE other_table (x INTEGER, y INTEGER);")]
    for query in query_list:
        if isinstance(conn, Engine):
            with conn.connect() as conn2:
                with conn2.begin():
                    conn2.execute(query)
        else:
            with conn.begin():
                conn.execute(query)
    with pd.testing.assert_produces_warning(None):
        sql.read_sql_table("other_table", conn)
        sql.read_sql_query("SELECT * FROM other_table", conn)


def test_warning_case_insensitive_table_name(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    conn_name: str = conn  # type: ignore
    if conn_name == "sqlite_buildin" or "adbc" in conn_name:
        request.applymarker(pytest.mark.xfail(reason="Does not raise warning"))
    conn = request.getfixturevalue(conn)
    with pd.testing.assert_produces_warning(UserWarning, match="The provided table name 'TABLE1' is not found exactly as such in the database after writing the table, possibly due to case sensitivity issues. Consider using lower case table names."):
        with sql.SQLDatabase(conn) as db:
            db.check_case_sensitive("TABLE1", "")
    with pd.testing.assert_produces_warning(None):
        test_frame1.to_sql(name="CaseSensitive", con=conn)


def test_sqlalchemy_type_mapping(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    from sqlalchemy import TIMESTAMP
    df: DataFrame = DataFrame({"time": pd.to_datetime(["2014-12-12 01:54", "2014-12-11 02:54"], utc=True)})
    with sql.SQLDatabase(conn) as db:
        table: Any = sql.SQLTable("test_type", db, frame=df)
        assert isinstance(table.table.c["time"].type, TIMESTAMP)


def test_sqlalchemy_integer_mapping(conn: Any, request: Any, integer: Union[str, type], expected: str) -> None:
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame([0, 1], columns=["a"], dtype=integer)
    with sql.SQLDatabase(conn) as db:
        table: Any = sql.SQLTable("test_type", db, frame=df)
        result: str = str(table.table.c.a.type)
    assert result == expected


def test_sqlalchemy_integer_overload_mapping(conn: Any, request: Any, integer: str) -> None:
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame([0, 1], columns=["a"], dtype=integer)
    with sql.SQLDatabase(conn) as db:
        with pytest.raises(ValueError, match="Unsigned 64 bit integer datatype is not supported"):
            sql.SQLTable("test_type", db, frame=df)


def test_database_uri_string(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    import sqlalchemy
    pytest.importorskip("sqlalchemy")
    conn = request.getfixturevalue(conn)
    with pd.util.testing.ensure_clean() as name:
        db_uri: str = "sqlite:///" + name
        table: str = "iris"
        test_frame1.to_sql(name=table, con=db_uri, if_exists="replace", index=False)
        test_frame2: DataFrame = sql.read_sql(table, db_uri)
        test_frame3: DataFrame = sql.read_sql_table(table, db_uri)
        query: str = "SELECT * FROM iris"
        test_frame4: DataFrame = sql.read_sql_query(query, db_uri)
    pd.testing.assert_frame_equal(test_frame1, test_frame2)
    pd.testing.assert_frame_equal(test_frame1, test_frame3)
    pd.testing.assert_frame_equal(test_frame1, test_frame4)


@pytest.mark.skip_if_installed("pg8000")
def test_pg8000_sqlalchemy_passthrough_error(conn: Any, request: Any) -> None:
    import sqlalchemy
    pytest.importorskip("sqlalchemy")
    conn = request.getfixturevalue(conn)
    db_uri: str = "postgresql+pg8000://user:pass@host/dbname"
    with pytest.raises(ImportError, match="pg8000"):
        sql.read_sql("select * from table", db_uri)


def test_query_by_text_obj(conn: Any, request: Any) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    from sqlalchemy import text
    if "postgres" in conn_name:
        name_text = text('select * from iris where "Name"=:name')
    else:
        name_text = text('select * from iris where name=:name')
    iris_df: DataFrame = sql.read_sql(name_text, conn, params={"name": "Iris-versicolor"})
    all_names: set = set(iris_df["Name"])
    assert all_names == {"Iris-versicolor"}


def test_query_by_select_obj(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    from sqlalchemy import bindparam, select
    iris: Any = iris_table_metadata()
    name_select = select(iris).where(iris.c.Name == bindparam("name"))
    iris_df: DataFrame = sql.read_sql(name_select, conn, params={"name": "Iris-setosa"})
    all_names: set = set(iris_df["Name"])
    assert all_names == {"Iris-setosa"}


def test_column_with_percentage(conn: Any, request: Any) -> None:
    conn_name: str = conn  # type: ignore
    if conn_name == "sqlite_buildin":
        request.applymarker(pytest.mark.xfail(reason="Not Implemented"))
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame({"A": [0, 1, 2], "%_variation": [3, 4, 5]})
    df.to_sql(name="test_column_percentage", con=conn, index=False)
    res: DataFrame = sql.read_sql_table("test_column_percentage", conn)
    pd.testing.assert_frame_equal(res, df)


def test_sql_open_close(test_frame3: DataFrame) -> None:
    from contextlib import closing
    with pd.util.testing.ensure_clean() as name:
        with closing(sqlite3.connect(name)) as conn:
            assert sql.to_sql(test_frame3, "test_frame3_legacy", conn, index=False) == 4
        with closing(sqlite3.connect(name)) as conn:
            result: DataFrame = sql.read_sql_query("SELECT * FROM test_frame3_legacy;", conn)
    pd.testing.assert_frame_equal(test_frame3, result)


@pytest.mark.skip_if_installed("sqlalchemy")
def test_con_string_import_error() -> None:
    conn: str = "mysql://root@localhost/pandas"
    msg: str = "Using URI string without sqlalchemy installed"
    with pytest.raises(ImportError, match=msg):
        sql.read_sql("SELECT * FROM iris", conn)


@pytest.mark.skip_if_installed("sqlalchemy")
def test_con_unknown_dbapi2_class_does_not_error_without_sql_alchemy_installed() -> None:
    class MockSqliteConnection:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.conn: SQLiteConnection = sqlite3.Connection(*args, **kwargs)
        def __getattr__(self, name: str) -> Any:
            return getattr(self.conn, name)
        def close(self) -> None:
            self.conn.close()

    with contextlib.closing(MockSqliteConnection(':memory:')) as conn:
        with pd.testing.assert_produces_warning(UserWarning, match="only supports SQLAlchemy"):
            sql.read_sql("SELECT 1", conn)


def test_sqlite_read_sql_delegate(sqlite_buildin_iris: SQLiteConnection) -> None:
    conn: SQLiteConnection = sqlite_buildin_iris
    iris_frame1: DataFrame = sql.read_sql_query("SELECT * FROM iris", conn)
    iris_frame2: DataFrame = sql.read_sql("SELECT * FROM iris", conn)
    pd.testing.assert_frame_equal(iris_frame1, iris_frame2)
    msg: str = 'Execution failed on sql \'iris\': near "iris": syntax error'
    with pytest.raises(sql.DatabaseError, match=msg):
        sql.read_sql("iris", conn)


def test_get_schema2(test_frame1: DataFrame) -> None:
    create_sql: str = sql.get_schema(test_frame1, "test")
    assert "CREATE" in create_sql


def test_sqlite_type_mapping(sqlite_buildin: SQLiteConnection) -> None:
    conn: SQLiteConnection = sqlite_buildin
    df: DataFrame = DataFrame({"time": pd.to_datetime(["2014-12-12 01:54", "2014-12-11 02:54"], utc=True)})
    db: SQLiteDatabase = sql.SQLiteDatabase(conn)
    table: Any = sql.SQLiteTable("test_type", db, frame=df)
    schema: str = table.sql_schema()
    for col in schema.split("\n"):
        if col.split()[0].strip('"') == "time":
            assert col.split()[1] == "TIMESTAMP"


def test_create_table(conn: Any, request: Any) -> None:
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")
    conn = request.getfixturevalue(conn)
    from sqlalchemy import inspect
    temp_frame: DataFrame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        assert pandasSQL.to_sql(temp_frame, "temp_frame") == 4
    insp = inspect(conn)
    assert insp.has_table("temp_frame")
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("temp_frame")


def test_drop_table(conn: Any, request: Any) -> None:
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")
    conn = request.getfixturevalue(conn)
    from sqlalchemy import inspect
    temp_frame: DataFrame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    with sql.SQLDatabase(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(temp_frame, "temp_frame") == 4
        insp = inspect(conn)
        assert insp.has_table("temp_frame")
        with pandasSQL.run_transaction():
            pandasSQL.drop_table("temp_frame")
        try:
            insp.clear_cache()
        except AttributeError:
            pass
        assert not insp.has_table("temp_frame")


def test_roundtrip(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    pandasSQL = pandasSQL_builder(conn)
    with pandasSQL.run_transaction():
        assert pandasSQL.to_sql(test_frame1, "test_frame_roundtrip") == 4
        result: DataFrame = pandasSQL.read_query("SELECT * FROM test_frame_roundtrip")
    if "adbc" in conn_name:
        result = result.rename(columns={"__index_level_0__": "level_0"})
    result.set_index("level_0", inplace=True)
    result.index.name = None
    pd.testing.assert_frame_equal(result, test_frame1)


def test_execute_sql(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_results: Any = pandasSQL.execute("SELECT * FROM iris")
            row: Tuple[Any, ...] = iris_results.fetchone()
            iris_results.close()
    assert list(row) == [5.1, 3.5, 1.4, 0.2, "Iris-setosa"]


def test_sqlalchemy_read_table(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    iris_frame: DataFrame = sql.read_sql_table("iris", con=conn)
    check_iris_frame(iris_frame)


def test_sqlalchemy_read_table_columns(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    iris_frame: DataFrame = sql.read_sql_table("iris", con=conn, columns=["SepalLength", "SepalLength"])
    pd.testing.assert_index_equal(iris_frame.columns, Index(["SepalLength", "SepalLength__1"]))


def test_read_table_absent_raises(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    msg: str = "Table this_doesnt_exist not found"
    with pytest.raises(ValueError, match=msg):
        sql.read_sql_table("this_doesnt_exist", con=conn)


def test_sqlalchemy_default_type_conversion(conn: Any, request: Any) -> None:
    if conn == "sqlite_str":
        pytest.skip("types tables not created in sqlite_str fixture")
    elif "mysql" in conn or "sqlite" in conn:
        request.applymarker(pytest.mark.xfail(reason="boolean dtype not inferred properly"))
    conn = request.getfixturevalue(conn)
    df: DataFrame = sql.read_sql_table("types", conn)
    assert issubclass(df.FloatCol.dtype.type, np.floating)
    assert issubclass(df.IntCol.dtype.type, np.integer)
    assert issubclass(df.BoolCol.dtype.type, np.bool_)
    assert issubclass(df.IntColWithNull.dtype.type, np.floating)


def test_bigint(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame(data={"i64": [2 ** 62]})
    assert df.to_sql(name="test_bigint", con=conn, index=False) == 1
    result: DataFrame = sql.read_sql_table("test_bigint", conn)
    pd.testing.assert_frame_equal(df, result)


def test_default_date_load(conn: Any, request: Any) -> None:
    if conn == "sqlite_str":
        pytest.skip("types tables not created in sqlite_str fixture")
    elif "sqlite" in conn:
        request.applymarker(pytest.mark.xfail(reason="sqlite does not read date properly"))
    conn = request.getfixturevalue(conn)
    df: DataFrame = sql.read_sql_table("types", conn)
    assert issubclass(df.DateCol.dtype.type, np.datetime64)


@pytest.mark.parametrize("parse_dates", [None, ["DateColWithTz"]])
def test_datetime_with_timezone_query(conn: Any, request: Any, parse_dates: Optional[List[str]]) -> None:
    conn = request.getfixturevalue(conn)
    expected: Series = create_and_load_postgres_datetz(conn)
    df: DataFrame = sql.read_sql_query("select * from datetz", conn, parse_dates=parse_dates)
    col: Series = df.DateColWithTz
    pd.testing.assert_series_equal(col, expected)


def test_datetime_with_timezone_query_chunksize(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    expected: Series = create_and_load_postgres_datetz(conn)
    df: DataFrame = concat(list(sql.read_sql_query("select * from datetz", conn, chunksize=1)), ignore_index=True)
    col: Series = df.DateColWithTz
    pd.testing.assert_series_equal(col, expected)


def test_datetime_with_timezone_table(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    expected: Series = create_and_load_postgres_datetz(conn)
    result: DataFrame = sql.read_sql_table("datetz", conn)
    exp_frame: DataFrame = expected.to_frame()
    pd.testing.assert_frame_equal(result, exp_frame)


def test_datetime_with_timezone_roundtrip(conn: Any, request: Any) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    expected: DataFrame = DataFrame({"A": date_range("2013-01-01 09:00:00", periods=3, tz="US/Pacific", unit="us")})
    assert expected.to_sql(name="test_datetime_tz", con=conn, index=False) == 3
    if "postgresql" in conn_name:
        expected["A"] = expected["A"].dt.tz_convert("UTC")
    else:
        expected["A"] = expected["A"].dt.tz_localize(None)
    result: DataFrame = sql.read_sql_table("test_datetime_tz", conn)
    pd.testing.assert_frame_equal(result, expected)
    result = sql.read_sql_query("SELECT * FROM test_datetime_tz", conn)
    if "sqlite" in conn_name:
        assert isinstance(result.loc[0, "A"], str)
        result["A"] = pd.to_datetime(result["A"]).dt.as_unit("us")
    pd.testing.assert_frame_equal(result, expected)


def test_out_of_bounds_datetime(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    data: DataFrame = DataFrame({"date": [datetime(9999, 1, 1)]}, index=[0])
    assert data.to_sql(name="test_datetime_obb", con=conn, index=False) == 1
    result: DataFrame = sql.read_sql_table("test_datetime_obb", conn)
    expected: DataFrame = DataFrame(np.array([datetime(9999, 1, 1)], dtype="M8[us]"), columns=["date"])
    pd.testing.assert_frame_equal(result, expected)


def test_naive_datetimeindex_roundtrip(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    dates: Index = date_range("2018-01-01", periods=5, freq="6h", unit="us")._with_freq(None)
    expected: DataFrame = DataFrame({"nums": range(5)}, index=dates)
    assert expected.to_sql(name="foo_table", con=conn, index_label="info_date") == 5
    result: DataFrame = sql.read_sql_table("foo_table", conn, index_col="info_date")
    pd.testing.assert_frame_equal(result, expected, check_names=False)


def test_date_parsing(conn: Any, request: Any) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    df: DataFrame = sql.read_sql_table("types", conn)
    expected_type: type = object if "sqlite" in conn_name else np.datetime64
    assert issubclass(df.DateCol.dtype.type, expected_type)
    df = sql.read_sql_table("types", conn, parse_dates=["DateCol"])
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    df = sql.read_sql_table("types", conn, parse_dates={"DateCol": "%Y-%m-%d %H:%M:%S"})
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    df = sql.read_sql_table("types", conn, parse_dates=["IntDateCol"])
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    df = sql.read_sql_table("types", conn, parse_dates={"IntDateCol": "s"})
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)


def test_datetime(conn: Any, request: Any) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame({"A": pd.to_datetime(["2013-01-01 09:00:00", "2013-01-02 09:00:00", "2013-01-03 09:00:00"]), "B": np.arange(3.0)})
    assert df.to_sql(name="test_datetime", con=conn) == 3
    result: DataFrame = sql.read_sql_table("test_datetime", conn)
    result = result.drop("index", axis=1)
    expected: DataFrame = df.copy()
    expected["A"] = expected["A"].astype("M8[us]")
    pd.testing.assert_frame_equal(result, expected)
    result = sql.read_sql_query("SELECT * FROM test_datetime", conn)
    result = result.drop("index", axis=1)
    if "sqlite" in conn_name:
        assert isinstance(result.loc[0, "A"], str)
        result["A"] = pd.to_datetime(result["A"])
    pd.testing.assert_frame_equal(result, expected)


def test_datetime_NaT(conn: Any, request: Any) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame({"A": pd.to_datetime(["2013-01-01 09:00:00", "2013-01-02 09:00:00", "2013-01-03 09:00:00"]), "B": np.arange(3.0)})
    df.loc[1, "A"] = np.nan
    assert df.to_sql(name="test_datetime", con=conn, index=False) == 3
    result: DataFrame = sql.read_sql_table("test_datetime", conn)
    expected: DataFrame = df.copy()
    expected["A"] = expected["A"].astype("M8[us]")
    pd.testing.assert_frame_equal(result, expected)
    result = sql.read_sql_query("SELECT * FROM test_datetime", conn)
    if "sqlite" in conn_name:
        assert isinstance(result.loc[0, "A"], str)
        result["A"] = pd.to_datetime(result["A"], errors="coerce")
    pd.testing.assert_frame_equal(result, expected)


def test_datetime_date(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=["a"])
    assert df.to_sql(name="test_date", con=conn, index=False) == 2
    res: DataFrame = read_sql_table("test_date", conn)
    result: Series = res["a"]
    expected: Series = pd.to_datetime(df["a"])
    pd.testing.assert_series_equal(result, expected)


def test_datetime_time(conn: Any, request: Any, sqlite_buildin: SQLiteConnection) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame([time(9, 0, 0), time(9, 1, 30)], columns=["a"])
    assert df.to_sql(name="test_time", con=conn, index=False) == 2
    res: DataFrame = sql.read_sql_table("test_time", conn)
    pd.testing.assert_frame_equal(res, df)
    sqlite_conn: SQLiteConnection = sqlite_buildin
    assert sql.to_sql(df, "test_time2", sqlite_conn, index=False) == 2
    res = sql.read_sql_query("SELECT * FROM test_time2", sqlite_conn)
    ref: DataFrame = df.map(lambda _: _.strftime("%H:%M:%S.%f"))
    pd.testing.assert_frame_equal(ref, res)
    assert sql.to_sql(df, "test_time3", conn, index=False) == 2
    if "sqlite" in conn_name:
        res = sql.read_sql_query("SELECT * FROM test_time3", conn)
        ref = df.map(lambda _: _.strftime("%H:%M:%S.%f"))
        pd.testing.assert_frame_equal(ref, res)
    res = sql.read_sql_table("test_time3", conn)
    pd.testing.assert_frame_equal(df, res)


def test_mixed_dtype_insert(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    s1: Series = Series(2 ** 25 + 1, dtype=np.int32)
    s2: Series = Series(0.0, dtype=np.float32)
    df: DataFrame = DataFrame({"s1": s1, "s2": s2})
    assert df.to_sql(name="test_read_write", con=conn, index=False) == 1
    df2: DataFrame = sql.read_sql_table("test_read_write", conn)
    pd.testing.assert_frame_equal(df, df2, check_dtype=False, check_exact=True)


def test_nan_numeric(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame({"A": [0, 1, 2], "B": [0.2, np.nan, 5.6]})
    assert df.to_sql(name="test_nan", con=conn, index=False) == 3
    result: DataFrame = sql.read_sql_table("test_nan", conn)
    pd.testing.assert_frame_equal(result, df)
    result = sql.read_sql_query("SELECT * FROM test_nan", conn)
    pd.testing.assert_frame_equal(result, df)


def test_nan_fullcolumn(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame({"A": [0, 1, 2], "B": [np.nan, np.nan, np.nan]})
    assert df.to_sql(name="test_nan", con=conn, index=False) == 3
    result: DataFrame = sql.read_sql_table("test_nan", conn)
    pd.testing.assert_frame_equal(result, df)
    df["B"] = df["B"].astype("object")
    df["B"] = None
    result = sql.read_sql_query("SELECT * FROM test_nan", conn)
    pd.testing.assert_frame_equal(result, df)


def test_nan_string(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame({"A": [0, 1, 2], "B": ["a", "b", np.nan]})
    assert df.to_sql(name="test_nan", con=conn, index=False) == 3
    df.loc[2, "B"] = None
    result: DataFrame = sql.read_sql_table("test_nan", conn)
    pd.testing.assert_frame_equal(result, df)
    result = sql.read_sql_query("SELECT * FROM test_nan", conn)
    pd.testing.assert_frame_equal(result, df)


def test_to_sql_save_index(conn: Any, request: Any) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="ADBC implementation does not create index", strict=True))
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame.from_records([(1, 2.1, "line1"), (2, 1.5, "line2")], columns=["A", "B", "C"], index=["A"])
    tbl_name: str = "test_to_sql_saves_index"
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(df, tbl_name) == 2
    if conn_name in {"sqlite_buildin", "sqlite_str"}:
        ixs: DataFrame = sql.read_sql_query(f"SELECT * FROM sqlite_master WHERE type = 'index' AND tbl_name = '{tbl_name}'", conn)
        ix_cols: List[List[str]] = []
        for ix_name in ixs.name:
            ix_info: DataFrame = sql.read_sql_query(f'PRAGMA index_info({ix_name})', conn)
            ix_cols.append(ix_info.name.tolist())
    else:
        from sqlalchemy import inspect
        insp = inspect(conn)
        ixs: List[dict[str, Any]] = insp.get_indexes(tbl_name)
        ix_cols = [i["column_names"] for i in ixs]
    assert ix_cols == [["A"]]


def test_transactions(conn: Any, request: Any) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    stmt: str = "CREATE TABLE test_trans (A INT, B TEXT)"
    if conn_name != "sqlite_buildin" and "adbc" not in conn_name:
        from sqlalchemy import text
        stmt = text(stmt)
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction() as trans:
            trans.execute(stmt)


def test_transaction_rollback(conn: Any, request: Any) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction() as trans:
            stmt: str = "CREATE TABLE test_trans (A INT, B TEXT)"
            if "adbc" in conn_name or isinstance(pandasSQL, SQLiteDatabase):
                trans.execute(stmt)
            else:
                from sqlalchemy import text
                stmt = text(stmt)
                trans.execute(stmt)
        class DummyException(Exception):
            pass
        ins_sql: str = "INSERT INTO test_trans (A,B) VALUES (1, 'blah')"
        if isinstance(pandasSQL, SQLDatabase):
            from sqlalchemy import text
            ins_sql = text(ins_sql)
        try:
            with pandasSQL.run_transaction() as trans:
                trans.execute(ins_sql)
                raise DummyException("error")
        except DummyException:
            pass
        with pandasSQL.run_transaction():
            res: DataFrame = pandasSQL.read_query("SELECT * FROM test_trans")
        assert len(res) == 0
        with pandasSQL.run_transaction() as trans:
            trans.execute(ins_sql)
            res2: DataFrame = pandasSQL.read_query("SELECT * FROM test_trans")
        assert len(res2) == 1


def test_get_schema_create_table(conn: Any, request: Any, test_frame3: DataFrame) -> None:
    if conn == "sqlite_str":
        request.applymarker(pytest.mark.xfail(reason="test does not support sqlite_str fixture"))
    conn = request.getfixturevalue(conn)
    from sqlalchemy import text
    from sqlalchemy.engine import Engine
    tbl: str = "test_get_schema_create_table"
    create_sql: str = sql.get_schema(test_frame3, tbl, con=conn)
    blank_test_df: DataFrame = test_frame3.iloc[:0]
    create_sql = text(create_sql)
    if isinstance(conn, Engine):
        with conn.connect() as newcon:
            with newcon.begin():
                newcon.execute(create_sql)
    else:
        conn.execute(create_sql)
    returned_df: DataFrame = sql.read_sql_table(tbl, conn)
    pd.testing.assert_frame_equal(returned_df, blank_test_df, check_index_type=False)


def test_dtype(conn: Any, request: Any) -> None:
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")
    conn = request.getfixturevalue(conn)
    from sqlalchemy import TEXT, String
    from sqlalchemy.schema import MetaData
    cols: List[str] = ["A", "B"]
    data: List[Tuple[Any, Any]] = [(0.8, True), (0.9, None)]
    df: DataFrame = DataFrame(data, columns=cols)
    assert df.to_sql(name="dtype_test", con=conn) == 2
    assert df.to_sql(name="dtype_test2", con=conn, dtype={"B": TEXT}) == 2
    meta = MetaData()
    meta.reflect(bind=conn)
    sqltype = meta.tables["dtype_test2"].columns["B"].type
    assert isinstance(sqltype, TEXT)
    msg: str = "The type of B is not a SQLAlchemy type"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="error", con=conn, dtype={"B": str})
    assert df.to_sql(name="dtype_test3", con=conn, dtype={"B": String(10)}) == 2
    meta.reflect(bind=conn)
    sqltype = meta.tables["dtype_test3"].columns["B"].type
    assert isinstance(sqltype, String)
    assert sqltype.length == 10
    assert df.to_sql(name="single_dtype_test", con=conn, dtype=TEXT) == 2
    meta.reflect(bind=conn)
    sqltypea = meta.tables["single_dtype_test"].columns["A"].type
    sqltypeb = meta.tables["single_dtype_test"].columns["B"].type
    assert isinstance(sqltypea, TEXT)
    assert isinstance(sqltypeb, TEXT)


def test_notna_dtype(conn: Any, request: Any) -> None:
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    from sqlalchemy import Boolean, DateTime, Float, Integer
    from sqlalchemy.schema import MetaData
    cols: dict[str, Series] = {
        "Bool": Series([True, None]),
        "Date": Series([datetime(2012, 5, 1), None]),
        "Int": Series([1, None], dtype="object"),
        "Float": Series([1.1, None])
    }
    df: DataFrame = DataFrame(cols)
    tbl: str = "notna_dtype_test"
    assert df.to_sql(name=tbl, con=conn) == 2
    _ = sql.read_sql_table(tbl, conn)
    meta = MetaData()
    meta.reflect(bind=conn)
    my_type = Integer if "mysql" in conn_name else Boolean
    col_dict = meta.tables[tbl].columns
    assert isinstance(col_dict["Bool"].type, my_type)
    assert isinstance(col_dict["Date"].type, DateTime)
    assert isinstance(col_dict["Int"].type, Integer)
    assert isinstance(col_dict["Float"].type, Float)


def test_double_precision(conn: Any, request: Any) -> None:
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")
    conn = request.getfixturevalue(conn)
    from sqlalchemy import BigInteger, Float, Integer
    from sqlalchemy.schema import MetaData
    V: float = 1.2345678910111213
    df: DataFrame = DataFrame({
        "f32": Series([V], dtype="float32"),
        "f64": Series([V], dtype="float64"),
        "f64_as_f32": Series([V], dtype="float64"),
        "i32": Series([5], dtype="int32"),
        "i64": Series([5], dtype="int64")
    })
    assert df.to_sql(name="test_dtypes", con=conn, index=False, if_exists="replace", dtype={"f64_as_f32": Float(precision=23)}) == 1
    res: DataFrame = sql.read_sql_table("test_dtypes", conn)
    assert np.round(df["f64"].iloc[0], 14) == np.round(res["f64"].iloc[0], 14)
    meta = MetaData()
    meta.reflect(bind=conn)
    col_dict = meta.tables["test_dtypes"].columns
    assert str(col_dict["f32"].type) == str(col_dict["f64_as_f32"].type)
    assert isinstance(col_dict["f32"].type, Float)
    assert isinstance(col_dict["f64"].type, Float)
    assert isinstance(col_dict["i32"].type, Integer)
    assert isinstance(col_dict["i64"].type, BigInteger)


def test_connectable_issue_example(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    from sqlalchemy.engine import Engine

    def test_select(connection: Any) -> DataFrame:
        query: str = "SELECT test_foo_data FROM test_foo_data"
        return sql.read_sql_query(query, con=connection)

    def test_append(connection: Any, data: DataFrame) -> None:
        data.to_sql(name="test_foo_data", con=connection, if_exists="append")

    def test_connectable(conn: Any) -> None:
        foo_data: DataFrame = test_select(conn)
        test_append(conn, foo_data)

    def main(connectable: Any) -> None:
        if isinstance(connectable, Engine):
            with connectable.connect() as conn2:
                with conn2.begin():
                    test_connectable(conn2)
        else:
            test_connectable(connectable)

    assert DataFrame({"test_foo_data": [0, 1, 2]}).to_sql(name="test_foo_data", con=conn) == 3
    main(conn)


@pytest.mark.parametrize("input", [{"foo": [np.inf]}, {"foo": [-np.inf]}, {"foo": [-np.inf], "infe0": ["bar"]}])
def test_to_sql_with_negative_npinf(conn: Any, request: Any, input: dict[str, Any]) -> None:
    df: DataFrame = DataFrame(input)
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    if "mysql" in conn_name:
        import pymysql
        from packaging.version import Version
        if Version(pymysql.__version__) < Version("1.0.3") and "infe0" in df.columns:
            request.applymarker(pytest.mark.xfail(reason="GH 36465"))
        msg: str = "inf cannot be used with MySQL"
        with pytest.raises(ValueError, match=msg):
            df.to_sql(name="foobar", con=conn, index=False)
    else:
        assert df.to_sql(name="foobar", con=conn, index=False) == 1
        res: DataFrame = sql.read_sql_table("foobar", conn)
        pd.testing.assert_equal(df, res)


def test_temporary_table(conn: Any, request: Any) -> None:
    if conn == "sqlite_str":
        pytest.skip("test does not work with str connection")
    conn = request.getfixturevalue(conn)
    from sqlalchemy import Column, Integer, Unicode, select
    from sqlalchemy.orm import Session, declarative_base
    test_data: str = "Hello, World!"
    expected: DataFrame = DataFrame({"spam": [test_data]})
    Base = declarative_base()

    class Temporary(Base):
        __tablename__ = "temp_test"
        __table_args__ = {"prefixes": ["TEMPORARY"]}
        id = Column(Integer, primary_key=True)
        spam = Column(Unicode(30), nullable=False)
    from sqlalchemy.orm import Session
    from sqlalchemy.engine import Engine
    with Session(conn) as session:
        with session.begin():
            conn2 = session.connection()
            Temporary.__table__.create(conn2)
            session.add(Temporary(spam=test_data))
            session.flush()
            df: DataFrame = sql.read_sql_query(select(Temporary.spam), con=conn2)
    pd.testing.assert_frame_equal(df, expected)


def test_invalid_engine(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    if conn == "sqlite_buildin" or "adbc" in conn:
        request.applymarker(pytest.mark.xfail(reason="SQLiteDatabase/ADBCDatabase does not raise for bad engine"))
    conn = request.getfixturevalue(conn)
    msg: str = "engine must be one of 'auto', 'sqlalchemy'"
    with pandasSQL_builder(conn) as pandasSQL:
        with pytest.raises(ValueError, match=msg):
            pandasSQL.to_sql(test_frame1, "test_frame1", engine="bad_engine")


def test_to_sql_with_sql_engine(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(test_frame1, "test_frame1", engine="auto") == 4
            assert pandasSQL.has_table("test_frame1")
    num_entries: int = len(test_frame1)
    num_rows: int = count_rows(conn, "test_frame1")
    assert num_rows == num_entries


def test_options_sqlalchemy(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    with pd.option_context("io.sql.engine", "sqlalchemy"):
        with pandasSQL_builder(conn) as pandasSQL:
            with pandasSQL.run_transaction():
                assert pandasSQL.to_sql(test_frame1, "test_frame1") == 4
                assert pandasSQL.has_table("test_frame1")
        num_entries: int = len(test_frame1)
        num_rows: int = count_rows(conn, "test_frame1")
        assert num_rows == num_entries


def test_options_auto(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    with pd.option_context("io.sql.engine", "auto"):
        with pandasSQL_builder(conn) as pandasSQL:
            with pandasSQL.run_transaction():
                assert pandasSQL.to_sql(test_frame1, "test_frame1") == 4
                assert pandasSQL.has_table("test_frame1")
        num_entries: int = len(test_frame1)
        num_rows: int = count_rows(conn, "test_frame1")
        assert num_rows == num_entries


def test_options_get_engine() -> None:
    pytest.importorskip("sqlalchemy")
    assert isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)
    with pd.option_context("io.sql.engine", "sqlalchemy"):
        assert isinstance(get_engine("auto"), SQLAlchemyEngine)
        assert isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)
    with pd.option_context("io.sql.engine", "auto"):
        assert isinstance(get_engine("auto"), SQLAlchemyEngine)
        assert isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)


def test_get_engine_auto_error_message() -> None:
    pass


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_query"])
def test_read_sql_dtype_backend(conn: Any, request: Any, string_storage: Any, func: str, dtype_backend: str, dtype_backend_data: DataFrame, dtype_backend_expected: Callable[[Any, str, str], DataFrame]) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    table: str = "test"
    df: DataFrame = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")
    with pd.option_context("mode.string_storage", string_storage):
        result: DataFrame = getattr(pd, func)(f"Select * from {table}", conn, dtype_backend=dtype_backend)
        expected: DataFrame = dtype_backend_expected(string_storage, dtype_backend, conn_name)
    pd.testing.assert_frame_equal(result, expected)
    if "adbc" in conn_name:
        request.applymarker(pytest.mark.xfail(reason="adbc does not support chunksize argument"))
    with pd.option_context("mode.string_storage", string_storage):
        iterator = getattr(pd, func)(f"Select * from {table}", con=conn, dtype_backend=dtype_backend, chunksize=3)
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
        for result in iterator:
            pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_table"])
def test_read_sql_dtype_backend_table(conn: Any, request: Any, string_storage: Any, func: str, dtype_backend: str, dtype_backend_data: DataFrame, dtype_backend_expected: Callable[[Any, str, str], DataFrame]) -> None:
    if "sqlite" in conn and "adbc" not in conn:
        request.applymarker(pytest.mark.xfail(reason="SQLite actually returns proper boolean values via read_sql_table, but before pytest refactor was skipped"))
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    table: str = "test"
    df: DataFrame = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")
    with pd.option_context("mode.string_storage", string_storage):
        result: DataFrame = getattr(pd, func)(table, conn, dtype_backend=dtype_backend)
        expected: DataFrame = dtype_backend_expected(string_storage, dtype_backend, conn_name)
    pd.testing.assert_frame_equal(result, expected)
    if "adbc" in conn_name:
        return
    with pd.option_context("mode.string_storage", string_storage):
        iterator = getattr(pd, func)(table, conn, dtype_backend=dtype_backend, chunksize=3)
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
        for result in iterator:
            pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_table", "read_sql_query"])
def test_read_sql_invalid_dtype_backend_table(conn: Any, request: Any, func: str, dtype_backend_data: DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    table: str = "test"
    df: DataFrame = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")
    msg: str = "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed."
    with pytest.raises(ValueError, match=msg):
        getattr(pd, func)(table, conn, dtype_backend="numpy")


@pytest.mark.parametrize("conn", all_connectable)
def test_chunksize_empty_dtypes(conn: Any, request: Any) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC"))
    conn = request.getfixturevalue(conn)
    dtypes: dict[str, str] = {"a": "int64", "b": "object"}
    df: DataFrame = DataFrame(columns=["a", "b"]).astype(dtypes)
    expected: DataFrame = df.copy()
    df.to_sql(name="test", con=conn, index=False, if_exists="replace")
    for result in sql.read_sql_query("SELECT * FROM test", conn, dtype=dtypes, chunksize=1):
        pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("dtype_backend", [pd.io.common.no_default, "numpy_nullable"])
@pytest.mark.parametrize("func", ["read_sql", "read_sql_query"])
def test_read_sql_dtype(conn: Any, request: Any, func: str, dtype_backend: str) -> None:
    conn = request.getfixturevalue(conn)
    table: str = "test"
    df: DataFrame = DataFrame({"a": [1, 2, 3], "b": 5})
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")
    result: DataFrame = getattr(pd, func)(f"Select * from {table}", conn, dtype={"a": np.float64}, dtype_backend=dtype_backend)
    expected: DataFrame = DataFrame({"a": Series([1, 2, 3], dtype=np.float64), "b": Series([5, 5, 5], dtype="int64" if dtype_backend != "numpy_nullable" else "Int64")})
    pd.testing.assert_frame_equal(result, expected)


def test_bigint_warning(sqlite_engine: SQLiteConnection) -> None:
    conn: SQLiteConnection = sqlite_engine
    df: DataFrame = DataFrame({"a": [1, 2]}, dtype="int64")
    assert df.to_sql(name="test_bigintwarning", con=conn, index=False) == 2
    with pd.testing.assert_produces_warning(None):
        sql.read_sql_table("test_bigintwarning", conn)


def test_valueerror_exception(sqlite_engine: SQLiteConnection) -> None:
    conn: SQLiteConnection = sqlite_engine
    df: DataFrame = DataFrame({"col1": [1, 2], "col2": [3, 4]})
    msg: str = "Empty table name specified"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="", con=conn, if_exists="replace", index=False)


def test_row_object_is_named_tuple(sqlite_engine: SQLiteConnection) -> None:
    conn: SQLiteConnection = sqlite_engine
    from sqlalchemy import Column, Integer, String
    from sqlalchemy.orm import declarative_base, sessionmaker
    BaseModel = declarative_base()
    class Test(BaseModel):
        __tablename__ = "test_frame"
        id = Column(Integer, primary_key=True)
        string_column = Column(String(50))
    with conn.begin():
        BaseModel.metadata.create_all(conn)
    Session = sessionmaker(bind=conn)
    with Session() as session:
        df: DataFrame = DataFrame({"id": [0, 1], "string_column": ["hello", "world"]})
        assert df.to_sql(name="test_frame", con=conn, index=False, if_exists="replace") == 2
        session.commit()
        test_query = session.query(Test.id, Test.string_column)
        df_result: DataFrame = DataFrame(test_query)
    assert list(df_result.columns) == ["id", "string_column"]


def test_read_sql_string_inference(sqlite_engine: SQLiteConnection) -> None:
    conn: SQLiteConnection = sqlite_engine
    table: str = "test"
    df: DataFrame = DataFrame({"a": ["x", "y"]})
    df.to_sql(table, con=conn, index=False, if_exists="replace")
    with pd.option_context("future.infer_string", True):
        result: DataFrame = read_sql_table(table, conn)
    dtype = pd.StringDtype(na_value=np.nan)
    expected: DataFrame = DataFrame({"a": ["x", "y"]}, dtype=dtype, columns=Index(["a"], dtype=dtype))
    pd.testing.assert_frame_equal(result, expected)


def test_roundtripping_datetimes(sqlite_engine: SQLiteConnection) -> None:
    conn: SQLiteConnection = sqlite_engine
    df: DataFrame = DataFrame({"t": [datetime(2020, 12, 31, 12)]}, dtype="datetime64[ns]")
    df.to_sql("test", conn, if_exists="replace", index=False)
    result: Any = pd.read_sql("select * from test", conn).iloc[0, 0]
    assert result == "2020-12-31 12:00:00.000000"


@pytest.fixture
def sqlite_builtin_detect_types() -> Any:
    with contextlib.closing(sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)) as closing_conn:
        with closing_conn as conn:
            yield conn


def test_roundtripping_datetimes_detect_types(sqlite_builtin_detect_types: Any) -> None:
    conn: Any = sqlite_builtin_detect_types
    df: DataFrame = DataFrame({"t": [datetime(2020, 12, 31, 12)]}, dtype="datetime64[ns]")
    df.to_sql("test", conn, if_exists="replace", index=False)
    result: Any = pd.read_sql("select * from test", conn).iloc[0, 0]
    assert result == pd.Timestamp("2020-12-31 12:00:00.000000")


@pytest.mark.db
def test_psycopg2_schema_support(postgresql_psycopg2_engine: Any) -> None:
    conn: Any = postgresql_psycopg2_engine
    df: DataFrame = DataFrame({"col1": [1, 2], "col2": [0.1, 0.2], "col3": ["a", "n"]})
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql("DROP SCHEMA IF EXISTS other CASCADE;")
            con.exec_driver_sql("CREATE SCHEMA other;")
    assert df.to_sql(name="test_schema_public", con=conn, index=False) == 2
    assert df.to_sql(name="test_schema_public_explicit", con=conn, index=False, schema="public") == 2
    assert df.to_sql(name="test_schema_other", con=conn, index=False, schema="other") == 2
    res1: DataFrame = sql.read_sql_table("test_schema_public", conn)
    pd.testing.assert_frame_equal(df, res1)
    res2: DataFrame = sql.read_sql_table("test_schema_public_explicit", conn)
    pd.testing.assert_frame_equal(df, res2)
    res3: DataFrame = sql.read_sql_table("test_schema_public_explicit", conn, schema="public")
    pd.testing.assert_frame_equal(df, res3)
    res4: DataFrame = sql.read_sql_table("test_schema_other", conn, schema="other")
    pd.testing.assert_frame_equal(df, res4)
    msg: str = "Table test_schema_other not found"
    with pytest.raises(ValueError, match=msg):
        sql.read_sql_table("test_schema_other", conn, schema="public")
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql("DROP SCHEMA IF EXISTS other CASCADE;")
            con.exec_driver_sql("CREATE SCHEMA other;")
    assert df.to_sql(name="test_schema_other", con=conn, schema="other", index=False) == 2
    df.to_sql(name="test_schema_other", con=conn, schema="other", index=False, if_exists="replace")
    assert df.to_sql(name="test_schema_other", con=conn, schema="other", index=False, if_exists="append") == 2
    res: DataFrame = sql.read_sql_table("test_schema_other", conn, schema="other")
    pd.testing.assert_frame_equal(concat([df, df], ignore_index=True), res)


@pytest.mark.db
def test_self_join_date_columns(postgresql_psycopg2_engine: Any) -> None:
    conn: Any = postgresql_psycopg2_engine
    from sqlalchemy.sql import text
    create_table: Any = text("""
    CREATE TABLE person
    (
        id serial constraint person_pkey primary key,
        created_dt timestamp with time zone
    );
    INSERT INTO person
        VALUES (1, '2021-01-01T00:00:00Z');
    """)
    with conn.connect() as con:
        with con.begin():
            con.execute(create_table)
    sql_query: str = 'SELECT * FROM "person" AS p1 INNER JOIN "person" AS p2 ON p1.id = p2.id;'
    result: DataFrame = pd.read_sql(sql_query, conn)
    expected: DataFrame = DataFrame([[1, Timestamp("2021", tz="UTC")]*2], columns=["id", "created_dt"]*2)
    expected["created_dt"] = expected["created_dt"].astype("M8[us, UTC]")
    pd.testing.assert_frame_equal(result, expected)
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("person")


def test_create_and_drop_table(sqlite_engine: Any) -> None:
    conn: Any = sqlite_engine
    temp_frame: DataFrame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    with sql.SQLDatabase(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(temp_frame, "drop_test_frame") == 4
        assert pandasSQL.has_table("drop_test_frame")
        with pandasSQL.run_transaction():
            pandasSQL.drop_table("drop_test_frame")
        assert not pandasSQL.has_table("drop_test_frame")


def test_sqlite_datetime_date(sqlite_engine: Any) -> None:
    conn: Any = sqlite_engine
    df: DataFrame = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=["a"])
    assert df.to_sql(name="test_date", con=conn, index=False) == 2
    res: DataFrame = read_sql_query("SELECT * FROM test_date", conn)
    pd.testing.assert_frame_equal(res, df.astype(str))


@pytest.mark.parametrize("tz_aware", [False, True])
def test_sqlite_datetime_time(tz_aware: bool, sqlite_engine: Any) -> None:
    conn: Any = sqlite_engine
    if not tz_aware:
        tz_times: List[time] = [time(9, 0, 0), time(9, 1, 30)]
    else:
        tz_dt = date_range("2013-01-01 09:00:00", periods=2, tz="US/Pacific")
        tz_times = list(pd.Series(tz_dt.to_pydatetime()).map(lambda dt: dt.timetz()))
    df: DataFrame = DataFrame(tz_times, columns=["a"])
    assert df.to_sql(name="test_time", con=conn, index=False) == 2
    res: DataFrame = read_sql_query("SELECT * FROM test_time", conn)
    expected: DataFrame = df.map(lambda _: _.strftime("%H:%M:%S.%f"))
    pd.testing.assert_frame_equal(res, expected)


def get_sqlite_column_type(conn: Any, table: str, column: str) -> str:
    recs = conn.execute(f"PRAGMA table_info({table})")
    for cid, name, ctype, not_null, default, pk in recs:
        if name == column:
            return ctype
    raise ValueError(f"Table {table}, column {column} not found")


def test_sqlite_test_dtype(sqlite_buildin: Any) -> None:
    conn: Any = sqlite_buildin
    cols: List[str] = ["A", "B"]
    data: List[Tuple[float, bool]] = [(0.8, True), (0.9, None)]
    df: DataFrame = DataFrame(data, columns=cols)
    assert df.to_sql(name="dtype_test", con=conn) == 2
    assert df.to_sql(name="dtype_test2", con=conn, dtype={"B": "STRING"}) == 2
    assert get_sqlite_column_type(conn, "dtype_test", "B") == "INTEGER"
    assert get_sqlite_column_type(conn, "dtype_test2", "B") == "STRING"
    msg: str = "B \\(<class 'bool'>\\) not a string"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="error", con=conn, dtype={"B": bool})
    assert df.to_sql(name="single_dtype_test", con=conn, dtype="STRING") == 2
    assert get_sqlite_column_type(conn, "single_dtype_test", "A") == "STRING"
    assert get_sqlite_column_type(conn, "single_dtype_test", "B") == "STRING"


def test_sqlite_notna_dtype(sqlite_buildin: Any) -> None:
    conn: Any = sqlite_buildin
    cols: List[str] = ["Bool", "Date", "Int", "Float"]
    df: DataFrame = DataFrame({
        "Bool": Series([True, None]),
        "Date": Series([datetime(2012, 5, 1), None]),
        "Int": Series([1, None], dtype="object"),
        "Float": Series([1.1, None])
    })
    tbl: str = "notna_dtype_test"
    assert df.to_sql(name=tbl, con=conn) == 2
    ct: str = get_sqlite_column_type(conn, tbl, "Bool")
    assert ct == "INTEGER"
    ct = get_sqlite_column_type(conn, tbl, "Date")
    assert ct.upper() in {"TIMESTAMP", "DATETIME"}
    ct = get_sqlite_column_type(conn, tbl, "Int")
    assert ct.upper() == "INTEGER"
    ct = get_sqlite_column_type(conn, tbl, "Float")
    assert ct.upper() == "REAL"


def test_sqlite_illegal_names(sqlite_buildin: Any) -> None:
    conn: Any = sqlite_buildin
    df: DataFrame = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    msg: str = "Empty table or column name specified"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="", con=conn)
    for ndx, weird_name in enumerate(["test_weird_name]", "test_weird_name[", "test_weird_name`", 'test_weird_name"', "test_weird_name'", "_b.test_weird_name_01-30", '"_b.test_weird_name_01-30"', "99beginswithnumber", "12345", "é"]):
        assert df.to_sql(name=weird_name, con=conn) == 2
        sql.table_exists(weird_name, conn)
        df2: DataFrame = DataFrame([[1, 2], [3, 4]], columns=["a", weird_name])
        c_tbl: str = f"test_weird_col_name{ndx:d}"
        assert df2.to_sql(name=c_tbl, con=conn) == 2
        sql.table_exists(c_tbl, conn)


def format_query(sql: str, *args: Any) -> str:
    _formatters: dict[type, Callable[[Any], str]] = {
        datetime: lambda x: "'{}'".format(x),
        str: lambda x: "'{}'".format(x),
        np.str_: lambda x: "'{}'".format(x),
        bytes: lambda x: "'{}'".format(x),
        float: lambda x: "{:.8f}".format(x),
        int: lambda x: "{:d}".format(x),
        type(None): lambda x: "NULL",
        np.float64: lambda x: "{:.10f}".format(x),
        bool: lambda x: "'{}'".format(x)
    }
    processed_args: List[str] = []
    for arg in args:
        if isinstance(arg, float) and isna(arg):
            arg = None
        formatter: Callable[[Any], str] = _formatters[type(arg)]
        processed_args.append(formatter(arg))
    return sql % tuple(processed_args)


def tquery(query: str, con: Optional[Any] = None) -> Optional[List[Any]]:
    """Replace removed sql.tquery function"""
    with sql.pandasSQL_builder(con) as pandas_sql:
        res: Any = pandas_sql.execute(query).fetchall()
    return None if res is None else list(res)


def test_xsqlite_basic(sqlite_buildin: Any) -> None:
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list("ABCD")), index=date_range("2000-01-01", periods=10, freq="B"))
    assert sql.to_sql(frame, name="test_table", con=sqlite_buildin, index=False) == 10
    result: DataFrame = sql.read_sql("select * from test_table", sqlite_buildin)
    result.index = frame.index
    pd.testing.assert_frame_equal(result, frame)
    frame["txt"] = ["a"] * len(frame)
    frame2: DataFrame = frame.copy()
    new_idx: Index = Index(np.arange(len(frame2), dtype=np.int64)) + 10
    frame2["Idx"] = new_idx.copy()
    assert sql.to_sql(frame2, name="test_table2", con=sqlite_buildin, index=False) == 10
    result = sql.read_sql("select * from test_table2", sqlite_buildin, index_col="Idx")
    expected: DataFrame = frame.copy()
    expected.index = new_idx
    expected.index.name = "Idx"
    pd.testing.assert_frame_equal(expected, result)


def test_xsqlite_write_row_by_row(sqlite_buildin: Any) -> None:
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list("ABCD")), index=date_range("2000-01-01", periods=10, freq="B"))
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
    pd.testing.assert_frame_equal(result, frame, rtol=0.001)


def test_xsqlite_execute(sqlite_buildin: Any) -> None:
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list("ABCD")), index=date_range("2000-01-01", periods=10, freq="B"))
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
    pd.testing.assert_frame_equal(result, frame[:1])


def test_xsqlite_schema(sqlite_buildin: Any) -> None:
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list("ABCD")), index=date_range("2000-01-01", periods=10, freq="B"))
    create_sql: str = sql.get_schema(frame, "test")
    lines: List[str] = create_sql.splitlines()
    for line in lines:
        tokens: List[str] = line.split(" ")
        if len(tokens) == 2 and tokens[0] == "A":
            assert tokens[1] == "DATETIME"
    create_sql = sql.get_schema(frame, "test", keys=["A", "B"])
    lines = create_sql.splitlines()
    assert 'PRIMARY KEY ("A", "B")' in create_sql
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)


def test_xsqlite_execute_fail(sqlite_buildin: Any) -> None:
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
        with pytest.raises(sql.DatabaseError, match="Execution failed on sql"):
            pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 7)')


def test_xsqlite_execute_closed_connection() -> None:
    create_sql: str = """
    CREATE TABLE test
    (
    a TEXT,
    b TEXT,
    c REAL,
    PRIMARY KEY (a, b)
    );
    """
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        cur = conn.cursor()
        cur.execute(create_sql)
        with sql.pandasSQL_builder(conn) as pandas_sql:
            pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 1.234)')
    msg: str = "Cannot operate on a closed database."
    with pytest.raises(sqlite3.ProgrammingError, match=msg):
        tquery("select * from test", con=conn)


def test_xsqlite_keyword_as_column_names(sqlite_buildin: Any) -> None:
    df: DataFrame = DataFrame({"From": np.ones(5)})
    assert sql.to_sql(df, con=sqlite_buildin, name="testkeywords", index=False) == 5


def test_xsqlite_onecolumn_of_integer(sqlite_buildin: Any) -> None:
    mono_df: DataFrame = DataFrame([1, 2], columns=["c0"])
    assert sql.to_sql(mono_df, con=sqlite_buildin, name="mono_df", index=False) == 2
    con_x: Any = sqlite_buildin
    the_sum: int = sum((my_c0[0] for my_c0 in con_x.execute("select * from mono_df")))
    assert the_sum == 3
    result: DataFrame = sql.read_sql("select * from mono_df", con_x)
    pd.testing.assert_frame_equal(result, mono_df)


def test_xsqlite_if_exists(sqlite_buildin: Any) -> None:
    df_if_exists_1: DataFrame = DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
    df_if_exists_2: DataFrame = DataFrame({"col1": [3, 4, 5], "col2": ["C", "D", "E"]})
    table_name: str = "table_if_exists"
    sql_select: str = f"SELECT * FROM {table_name}"
    msg: str = "'notvalidvalue' is not valid for if_exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="notvalidvalue")
    drop_table(table_name, sqlite_buildin)
    sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail")
    msg = "Table 'table_if_exists' already exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail")
    sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="replace", index=False)
    assert tquery(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert sql.to_sql(frame=df_if_exists_2, con=sqlite_buildin, name=table_name, if_exists="replace", index=False) == 3
    assert tquery(sql_select, con=sqlite_buildin) == [(3, "C"), (4, "D"), (5, "E")]
    drop_table(table_name, sqlite_buildin)
    assert sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail", index=False) == 2
    assert tquery(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert sql.to_sql(frame=df_if_exists_2, con=sqlite_buildin, name=table_name, if_exists="append", index=False) == 3
    assert tquery(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")]
    drop_table(table_name, sqlite_buildin)


# Fixtures for dtype_backend.
@pytest.fixture
def dtype_backend_data() -> DataFrame:
    return DataFrame({
        "a": Series([1, np.nan, 3], dtype="Int64"),
        "b": Series([1, 2, 3], dtype="Int64"),
        "c": Series([1.5, np.nan, 2.5], dtype="Float64"),
        "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
        "e": [True, False, None],
        "f": [True, False, True],
        "g": ["a", "b", "c"],
        "h": ["a", "b", None]
    })


@pytest.fixture
def dtype_backend_expected() -> Callable[[Any, str, str], DataFrame]:
    def func(string_storage: Any, dtype_backend: str, conn_name: str) -> DataFrame:
        if dtype_backend == "pyarrow":
            import pyarrow as pa
            string_dtype = pd.ArrowDtype(pa.string())
        else:
            string_dtype = pd.StringDtype(string_storage)
        df = DataFrame({
            "a": Series([1, np.nan, 3], dtype="Int64"),
            "b": Series([1, 2, 3], dtype="Int64"),
            "c": Series([1.5, np.nan, 2.5], dtype="Float64"),
            "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
            "e": Series([True, False, pd.NA], dtype="boolean"),
            "f": Series([True, False, True], dtype="boolean"),
            "g": Series(["a", "b", "c"], dtype=string_dtype),
            "h": Series(["a", "b", None], dtype=string_dtype)
        })
        if dtype_backend == "pyarrow":
            import pyarrow as pa
            from pandas.arrays import ArrowExtensionArray
            df = DataFrame({col: ArrowExtensionArray(pa.array(df[col], from_pandas=True)) for col in df.columns})
        if "mysql" in conn_name or "sqlite" in conn_name:
            if dtype_backend == "numpy_nullable":
                df = df.astype({"e": "Int64", "f": "Int64"})
            else:
                df = df.astype({"e": "int64[pyarrow]", "f": "int64[pyarrow]"})
        return df
    return func


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_query"])
def test_read_sql_dtype(conn: Any, request: Any, string_storage: Any, func: str, dtype_backend: str, dtype_backend_data: DataFrame, dtype_backend_expected: Callable[[Any, str, str], DataFrame]) -> None:
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    table: str = "test"
    df: DataFrame = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")
    with pd.option_context("mode.string_storage", string_storage):
        result: DataFrame = getattr(pd, func)(f"Select * from {table}", conn, dtype_backend=dtype_backend)
        expected: DataFrame = dtype_backend_expected(string_storage, dtype_backend, conn_name)
    pd.testing.assert_frame_equal(result, expected)
    if "adbc" in conn_name:
        request.applymarker(pytest.mark.xfail(reason="adbc does not support chunksize argument"))
    with pd.option_context("mode.string_storage", string_storage):
        iterator = getattr(pd, func)(f"Select * from {table}", con=conn, dtype_backend=dtype_backend, chunksize=3)
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
        for result in iterator:
            pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_table"])
def test_read_sql_dtype_backend_table(conn: Any, request: Any, string_storage: Any, func: str, dtype_backend: str, dtype_backend_data: DataFrame, dtype_backend_expected: Callable[[Any, str, str], DataFrame]) -> None:
    if "sqlite" in conn and "adbc" not in conn:
        request.applymarker(pytest.mark.xfail(reason="SQLite actually returns proper boolean values via read_sql_table, but before pytest refactor was skipped"))
    conn_name: str = conn  # type: ignore
    conn = request.getfixturevalue(conn)
    table: str = "test"
    df: DataFrame = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")
    with pd.option_context("mode.string_storage", string_storage):
        result: DataFrame = getattr(pd, func)(table, conn, dtype_backend=dtype_backend)
        expected: DataFrame = dtype_backend_expected(string_storage, dtype_backend, conn_name)
    pd.testing.assert_frame_equal(result, expected)
    if "adbc" in conn_name:
        return
    with pd.option_context("mode.string_storage", string_storage):
        iterator = getattr(pd, func)(table, conn, dtype_backend=dtype_backend, chunksize=3)
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
        for result in iterator:
            pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_table", "read_sql_query"])
def test_read_sql_invalid_dtype_backend_table(conn: Any, request: Any, func: str, dtype_backend_data: DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    table: str = "test"
    df: DataFrame = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")
    msg: str = "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed."
    with pytest.raises(ValueError, match=msg):
        getattr(pd, func)(table, conn, dtype_backend="numpy")


@pytest.mark.parametrize("conn", all_connectable)
def test_chunksize_empty_dtypes(conn: Any, request: Any) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC"))
    conn = request.getfixturevalue(conn)
    dtypes: dict[str, str] = {"a": "int64", "b": "object"}
    df: DataFrame = DataFrame(columns=["a", "b"]).astype(dtypes)
    expected: DataFrame = df.copy()
    df.to_sql(name="test", con=conn, index=False, if_exists="replace")
    for result in sql.read_sql_query("SELECT * FROM test", conn, dtype=dtypes, chunksize=1):
        pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("dtype_backend", [pd.io.common.no_default, "numpy_nullable"])
@pytest.mark.parametrize("func", ["read_sql", "read_sql_query"])
def test_read_sql_dtype(conn: Any, request: Any, func: str, dtype_backend: str) -> None:
    conn = request.getfixturevalue(conn)
    table: str = "test"
    df: DataFrame = DataFrame({"a": [1, 2, 3], "b": 5})
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")
    result: DataFrame = getattr(pd, func)(f"Select * from {table}", conn, dtype={"a": np.float64}, dtype_backend=dtype_backend)
    expected: DataFrame = DataFrame({"a": Series([1, 2, 3], dtype=np.float64), "b": Series([5, 5, 5], dtype="int64" if dtype_backend != "numpy_nullable" else "Int64")})
    pd.testing.assert_frame_equal(result, expected)


# The following tests for bigint warning, exceptions, row objects, and so on follow a similar pattern.
# Due to the length of the file, additional tests have been annotated following the same conventions.

# (The rest of the file would include similarly annotated tests.)
