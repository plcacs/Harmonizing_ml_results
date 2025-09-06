from __future__ import annotations
import contextlib
import sqlite3
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Timestamp, date_range, to_datetime


def func_q02il3le() -> None:
    conn: Any = "mysql://root@localhost/pandas"
    msg: str = "Using URI string without sqlalchemy installed"
    import pytest  # type: ignore
    from pandas.io import sql
    with pytest.raises(ImportError, match=msg):
        sql.read_sql("SELECT * FROM iris", conn)


def func_zlk0pjfb() -> None:
    from __future__ import annotations

    class MockSqliteConnection:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.conn: sqlite3.Connection = sqlite3.Connection(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.conn, name)

        def func_6fwqzchl(self) -> None:
            self.conn.close()

    import contextlib
    import pytest  # type: ignore
    from pandas.io import sql
    with contextlib.closing(MockSqliteConnection(":memory:")) as conn:
        with pytest.assert_produces_warning(UserWarning, match="only supports SQLAlchemy"):
            sql.read_sql("SELECT 1", conn)


def func_bewnk1er(sqlite_buildin: Any) -> None:
    from pandas.io import sql
    conn: Any = sqlite_buildin
    iris_frame1: DataFrame = sql.read_sql_query("SELECT * FROM iris", conn)
    iris_frame2: DataFrame = sql.read_sql("SELECT * FROM iris", conn)
    pd.testing.assert_frame_equal(iris_frame1, iris_frame2)
    msg: str = 'Execution failed on sql \'iris\': near "iris": syntax error'
    import pytest  # type: ignore
    with pytest.raises(sql.DatabaseError, match=msg):
        sql.read_sql("iris", conn)


def func_b3roysu5(test_frame1: DataFrame) -> None:
    from pandas.io import sql
    create_sql: str = sql.get_schema(test_frame1, "test")
    assert "CREATE" in create_sql


def func_ihvmtq6j(sqlite_buildin: Any) -> None:
    from pandas.io import sql
    conn: Any = sqlite_buildin
    df: DataFrame = DataFrame({"time": to_datetime(["2014-12-12 01:54", "2014-12-11 02:54"], utc=True)})
    db = sql.SQLiteDatabase(conn)
    table = sql.SQLiteTable("test_type", db, frame=df)
    schema: str = table.sql_schema()
    for col in schema.split("\n"):
        if col.split()[0].strip('"') == "time":
            assert col.split()[1] == "TIMESTAMP"


def func_2x5n7nd1(conn: Any, request: Any) -> None:
    from sqlalchemy import inspect
    conn = request.getfixturevalue(conn)
    temp_frame: DataFrame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    from pandas.io import sql
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        assert pandasSQL.to_sql(temp_frame, "temp_frame") == 4
    insp = inspect(conn)
    assert insp.has_table("temp_frame")
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("temp_frame")


def func_h67npkoi(conn: Any, request: Any) -> None:
    if conn == "sqlite_str":
        import pytest  # type: ignore
        pytest.skip("sqlite_str has no inspection system")
    from sqlalchemy import inspect
    conn = request.getfixturevalue(conn)
    temp_frame: DataFrame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    from pandas.io import sql
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


def func_rc9oe3di(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    if conn == "sqlite_str":
        import pytest  # type: ignore
        pytest.skip("sqlite_str has no inspection system")
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    pandasSQL = sql.pandasSQL_builder(conn)
    with pandasSQL.run_transaction():
        assert pandasSQL.to_sql(test_frame1, "test_frame_roundtrip") == 4
        result: DataFrame = pandasSQL.read_query("SELECT * FROM test_frame_roundtrip")
    if "adbc" in conn_name:
        result = result.rename(columns={"__index_level_0__": "level_0"})
    result.set_index("level_0", inplace=True)
    result.index.name = None
    pd.testing.assert_frame_equal(result, test_frame1)


def func_mnbek3fh(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    with sql.pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_results = pandasSQL.execute("SELECT * FROM iris")
            row = iris_results.fetchone()
            iris_results.close()
    assert list(row) == [5.1, 3.5, 1.4, 0.2, "Iris-setosa"]


def func_o2rffas6(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    iris_frame: DataFrame = sql.read_sql_table("iris", con=conn)
    func_3vthw948(iris_frame)


def func_1naikvis(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    iris_frame: DataFrame = sql.read_sql_table("iris", con=conn, columns=["SepalLength", "SepalLength"])
    pd.testing.assert_index_equal(iris_frame.columns, pd.Index(["SepalLength", "SepalLength__1"]))


def func_odiat346(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    msg: str = "Table this_doesnt_exist not found"
    import pytest  # type: ignore
    with pytest.raises(ValueError, match=msg):
        sql.read_sql_table("this_doesnt_exist", con=conn)


def func_m9v278en(conn: Any, request: Any) -> None:
    conn_name: str = conn
    if conn_name == "sqlite_str":
        import pytest  # type: ignore
        pytest.skip("types tables not created in sqlite_str fixture")
    elif "mysql" in conn_name or "sqlite" in conn_name:
        import pytest  # type: ignore
        request.applymarker(pytest.mark.xfail(reason="boolean dtype not inferred properly"))
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    df: DataFrame = sql.read_sql_table("types", conn)
    assert issubclass(df.FloatCol.dtype.type, np.floating)
    assert issubclass(df.IntCol.dtype.type, np.integer)
    assert issubclass(df.BoolCol.dtype.type, np.bool_)
    assert issubclass(df.IntColWithNull.dtype.type, np.floating)
    assert issubclass(df.BoolColWithNull.dtype.type, object)


def func_e4dyicrq(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame(data={"i64": [2**62]})
    assert df.to_sql(name="test_bigint", con=conn, index=False) == 1
    from pandas.io import sql
    result: DataFrame = sql.read_sql_table("test_bigint", conn)
    pd.testing.assert_frame_equal(df, result)


def func_sbn3ztr2(conn: Any, request: Any) -> None:
    conn_name: str = conn
    if conn_name == "sqlite_str":
        import pytest  # type: ignore
        pytest.skip("types tables not created in sqlite_str fixture")
    elif "sqlite" in conn_name:
        import pytest  # type: ignore
        request.applymarker(pytest.mark.xfail(reason="sqlite does not read date properly"))
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    df: DataFrame = sql.read_sql_table("types", conn)
    assert issubclass(df.DateCol.dtype.type, np.datetime64)


def func_0mi8nzf4(conn: Any, request: Any, parse_dates: Optional[Any]) -> None:
    conn = request.getfixturevalue(conn)
    expected: Series = func_ima3t9si(conn)
    from pandas.io import sql
    df: DataFrame = sql.read_sql_query("select * from datetz", conn, parse_dates=parse_dates)
    col: Series = df.DateColWithTz
    pd.testing.assert_series_equal(col, expected)


def func_dgr4b6ou(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    expected: Series = func_ima3t9si(conn)
    from pandas.io import sql
    df: DataFrame = pd.concat(list(sql.read_sql_query("select * from datetz", conn, chunksize=1)), ignore_index=True)
    col: Series = df.DateColWithTz
    pd.testing.assert_series_equal(col, expected)


def func_jrj2yudl(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    expected: Series = func_ima3t9si(conn)
    from pandas.io import sql
    result: DataFrame = sql.read_sql_table("datetz", conn)
    exp_frame: DataFrame = expected.to_frame()
    pd.testing.assert_frame_equal(result, exp_frame)


def func_1bc9cniw(conn: Any, request: Any) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    expected: DataFrame = DataFrame({"A": date_range("2013-01-01 09:00:00", periods=3, tz="US/Pacific", unit="us")})
    assert expected.to_sql(name="test_datetime_tz", con=conn, index=False) == 3
    if "postgresql" in conn_name:
        expected["A"] = expected["A"].dt.tz_convert("UTC")
    else:
        expected["A"] = expected["A"].dt.tz_localize(None)
    from pandas.io import sql
    result: DataFrame = sql.read_sql_table("test_datetime_tz", conn)
    pd.testing.assert_frame_equal(result, expected)
    result2: DataFrame = sql.read_sql_query("SELECT * FROM test_datetime_tz", conn)
    if "sqlite" in conn_name:
        assert isinstance(result2.loc[0, "A"], str)
        result2["A"] = to_datetime(result2["A"]).dt.as_unit("us")
    pd.testing.assert_frame_equal(result2, expected)


def func_572foxv8(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame({"date": [datetime(9999, 1, 1)]}, index=[0])
    assert df.to_sql(name="test_datetime_obb", con=conn, index=False) == 1
    from pandas.io import sql
    result: DataFrame = sql.read_sql_table("test_datetime_obb", conn)
    expected: DataFrame = DataFrame(np.array([datetime(9999, 1, 1)], dtype="M8[us]"), columns=["date"])
    pd.testing.assert_frame_equal(result, expected)


def func_s310stw9(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    dates = date_range("2018-01-01", periods=5, freq="6h", unit="us")._with_freq(None)
    expected: DataFrame = DataFrame({"nums": list(range(5))}, index=dates)
    assert expected.to_sql(name="foo_table", con=conn, index_label="info_date") == 5
    from pandas.io import sql
    result: DataFrame = sql.read_sql_table("foo_table", conn, index_col="info_date")
    pd.testing.assert_frame_equal(result, expected, check_names=False)


def func_80t5nkux(conn: Any, request: Any) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    df: DataFrame = sql.read_sql_table("types", conn)
    expected_type = object if "sqlite" in conn_name else np.datetime64
    assert issubclass(df.DateCol.dtype.type, expected_type)
    df = sql.read_sql_table("types", conn, parse_dates=["DateCol"])
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    df = sql.read_sql_table("types", conn, parse_dates={"DateCol": "%Y-%m-%d %H:%M:%S"})
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    df = sql.read_sql_table("types", conn, parse_dates=["IntDateCol"])
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    df = sql.read_sql_table("types", conn, parse_dates={"IntDateCol": "s"})
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    df = sql.read_sql_table("types", conn, parse_dates={"IntDateCol": {"unit": "s"}})
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)


def func_yxnq13h3(conn: Any, request: Any) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame({"A": date_range("2013-01-01 09:00:00", periods=3),
                               "B": np.arange(3.0)})
    assert df.to_sql(name="test_datetime", con=conn) == 3
    from pandas.io import sql
    result: DataFrame = sql.read_sql_table("test_datetime", conn)
    result = result.drop("index", axis=1)
    expected: DataFrame = df.copy()
    expected["A"] = expected["A"].astype("M8[us]")
    pd.testing.assert_frame_equal(result, expected)
    result2: DataFrame = sql.read_sql_query("SELECT * FROM test_datetime", conn)
    result2 = result2.drop("index", axis=1)
    if "sqlite" in conn_name:
        assert isinstance(result2.loc[0, "A"], str)
        result2["A"] = to_datetime(result2["A"])
    pd.testing.assert_frame_equal(result2, expected)


def func_iflaxbb1(conn: Any, request: Any) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame({"A": date_range("2013-01-01 09:00:00", periods=3),
                               "B": np.arange(3.0)})
    df.loc[1, "A"] = np.nan
    assert df.to_sql(name="test_datetime", con=conn, index=False) == 3
    from pandas.io import sql
    result: DataFrame = sql.read_sql_table("test_datetime", conn)
    expected: DataFrame = df.copy()
    expected["A"] = expected["A"].astype("M8[us]")
    pd.testing.assert_frame_equal(result, expected)
    result2: DataFrame = sql.read_sql_query("SELECT * FROM test_datetime", conn)
    if "sqlite" in conn_name:
        assert isinstance(result2.loc[0, "A"], str)
        result2["A"] = to_datetime(result2["A"], errors="coerce")
    pd.testing.assert_frame_equal(result2, expected)


def func_lumujizb(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    df: DataFrame = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=["a"])
    assert df.to_sql(name="test_date", con=conn, index=False) == 2
    from pandas.io import sql
    res: DataFrame = sql.read_sql_query("SELECT * FROM test_date", conn)
    result: Series = res["a"]
    expected: Series = to_datetime(df["a"])
    pd.testing.assert_series_equal(result, expected)


def func_gulg7jya(conn: Any, request: Any, sqlite_buildin: Any) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    stmt: str = "CREATE TABLE test_trans (A INT, B TEXT)"
    from pandas.io import sql
    if conn_name != "sqlite_buildin" and "adbc" not in conn_name:
        from sqlalchemy import text
        stmt = text(stmt)  # type: ignore
    with sql.pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction() as trans:
            trans.execute(stmt)


def func_p1cvsbba(conn: Any, request: Any) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    with sql.pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction() as trans:
            stmt: str = "CREATE TABLE test_trans (A INT, B TEXT)"
            if "adbc" in conn_name or isinstance(pandasSQL, sql.SQLiteDatabase):
                trans.execute(stmt)
            else:
                from sqlalchemy import text
                stmt = text(stmt)  # type: ignore
                trans.execute(stmt)
        class DummyException(Exception):
            pass
        ins_sql: str = "INSERT INTO test_trans (A,B) VALUES (1, 'blah')"
        if isinstance(pandasSQL, sql.SQLDatabase):
            from sqlalchemy import text
            ins_sql = text(ins_sql)  # type: ignore
        import pytest  # type: ignore
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


def func_96lgbfga(conn: Any, request: Any, test_frame3: DataFrame) -> None:
    if conn == "sqlite_str":
        import pytest  # type: ignore
        pytest.skip("test does not support sqlite_str fixture")
    conn = request.getfixturevalue(conn)
    from sqlalchemy import text
    from sqlalchemy.engine import Engine
    tbl: str = "test_get_schema_create_table"
    create_sql: str = sql.get_schema(test_frame3, tbl, con=conn)
    blank_test_df: DataFrame = test_frame3.iloc[:0]
    create_sql = text(create_sql)  # type: ignore
    if isinstance(conn, Engine):
        with conn.connect() as newcon:
            with newcon.begin():
                newcon.execute(create_sql)
    else:
        conn.execute(create_sql)
    returned_df: DataFrame = sql.read_sql_table(tbl, conn)
    pd.testing.assert_frame_equal(returned_df, blank_test_df, check_index_type=False)


def func_1thul492(conn: Any, request: Any) -> None:
    if conn == "sqlite_str":
        import pytest  # type: ignore
        pytest.skip("sqlite_str has no inspection system")
    conn = request.getfixturevalue(conn)
    from sqlalchemy import TEXT, String
    from sqlalchemy.schema import MetaData
    cols: list[str] = ["A", "B"]
    df: DataFrame = DataFrame({"a": [0.8, 0.9], "b": [True, None]})
    assert df.to_sql(name="dtype_test", con=conn) == 2
    assert df.to_sql(name="dtype_test2", con=conn, dtype={"B": TEXT}) == 2
    meta: MetaData = MetaData()
    meta.reflect(bind=conn)
    sqltype = meta.tables["dtype_test2"].columns["B"].type
    assert isinstance(sqltype, TEXT)
    import pytest  # type: ignore
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


def func_qd5vylwz(conn: Any, request: Any) -> None:
    if conn == "sqlite_str":
        import pytest  # type: ignore
        pytest.skip("sqlite_str has no inspection system")
    conn_name: str = conn
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
    meta: MetaData = MetaData()
    meta.reflect(bind=conn)
    my_type = Integer if "mysql" in conn_name else Boolean
    col_dict = meta.tables[tbl].columns
    assert isinstance(col_dict["Bool"].type, my_type)
    assert isinstance(col_dict["Date"].type, DateTime)
    assert isinstance(col_dict["Int"].type, Integer)
    assert isinstance(col_dict["Float"].type, Float)


def func_vgxvujzs(conn: Any, request: Any) -> None:
    if conn == "sqlite_str":
        import pytest  # type: ignore
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
    meta: MetaData = MetaData()
    meta.reflect(bind=conn)
    col_dict = meta.tables["test_dtypes"].columns
    assert str(col_dict["f32"].type) == str(col_dict["f64_as_f32"].type)
    assert isinstance(col_dict["f32"].type, Float)
    assert isinstance(col_dict["f64"].type, Float)
    assert isinstance(col_dict["i32"].type, Integer)
    assert isinstance(col_dict["i64"].type, BigInteger)


def func_jvql4zpo(conn: Any, request: Any) -> None:
    conn = request.getfixturevalue(conn)
    from sqlalchemy.engine import Engine

    def func_uzqv7l19(connection: Any) -> DataFrame:
        query: str = "SELECT test_foo_data FROM test_foo_data"
        from pandas.io import sql
        return sql.read_sql_query(query, con=connection)

    def func_jg5zws5r(connection: Any, data: DataFrame) -> None:
        from pandas.io import sql
        data.to_sql(name="test_foo_data", con=connection, if_exists="append")

    def func_t455x3us(conn_inner: Any) -> None:
        foo_data: DataFrame = func_uzqv7l19(conn_inner)
        func_jg5zws5r(conn_inner, foo_data)

    def func_fxrwaym8(connectable: Any) -> None:
        from sqlalchemy.engine import Engine
        if isinstance(connectable, Engine):
            with connectable.connect() as conn_inner:
                with conn_inner.begin():
                    func_t455x3us(conn_inner)
        else:
            func_t455x3us(connectable)
    df: DataFrame = DataFrame({"test_foo_data": [0, 1, 2]})
    assert df.to_sql(name="test_foo_data", con=conn) == 3
    func_fxrwaym8(conn)


def func_kc5nd18v(conn: Any, request: Any, input: dict) -> None:
    df: DataFrame = DataFrame(input)
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    if "mysql" in conn_name:
        import pytest  # type: ignore
        pymysql = pytest.importorskip("pymysql")
        from packaging.version import Version
        if Version(pymysql.__version__) < Version("1.0.3") and "infe0" in df.columns:
            mark = pytest.mark.xfail(reason="GH 36465")
            request.applymarker(mark)
        msg: str = "inf cannot be used with MySQL"
        import pytest  # type: ignore
        with pytest.raises(ValueError, match=msg):
            df.to_sql(name="foobar", con=conn, index=False)
    else:
        assert df.to_sql(name="foobar", con=conn, index=False) == 1
        from pandas.io import sql
        res: DataFrame = sql.read_sql_table("foobar", conn)
        pd.testing.assert_equal(df, res)


def func_adscrgfu(conn: Any, request: Any) -> None:
    if conn == "sqlite_str":
        import pytest  # type: ignore
        pytest.skip("test does not work with str connection")
    conn = request.getfixturevalue(conn)
    from sqlalchemy import Column, Integer, Unicode, select
    from sqlalchemy.orm import Session, declarative_base
    test_data: str = "Hello, World!"
    expected: DataFrame = DataFrame({"spam": [test_data]})
    Base = declarative_base()

    class Temporary(Base):  # type: ignore
        __tablename__ = "temp_test"
        __table_args__ = {"prefixes": ["TEMPORARY"]}
        id = Column(Integer, primary_key=True)
        spam = Column(Unicode(30), nullable=False)

    from sqlalchemy.orm import Session
    with Session(conn) as session:
        with session.begin():
            conn_inner = session.connection()
            Temporary.__table__.create(conn_inner)
            session.add(Temporary(spam=test_data))
            session.flush()
            df: DataFrame = sql.read_sql_query(select(Temporary.spam), con=conn_inner)
    pd.testing.assert_frame_equal(df, expected)


def func_tlpcmvqz(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    if conn == "sqlite_buildin" or "adbc" in conn:
        import pytest  # type: ignore
        request.applymarker(pytest.mark.xfail(reason="SQLiteDatabase/ADBCDatabase does not raise for bad engine"))
    conn = request.getfixturevalue(conn)
    msg: str = "engine must be one of 'auto', 'sqlalchemy'"
    from pandas.io import sql
    with sql.pandasSQL_builder(conn) as pandasSQL:
        import pytest  # type: ignore
        with pytest.raises(ValueError, match=msg):
            pandasSQL.to_sql(test_frame1, "test_frame1", engine="bad_engine")


def func_qntlnvom(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    with sql.pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(test_frame1, "test_frame1", engine="auto") == 4
            assert pandasSQL.has_table("test_frame1")
    num_entries: int = len(test_frame1)
    num_rows: int = func_3od90k71(conn, "test_frame1")
    assert num_rows == num_entries


def func_ad76kcc3(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    import pandas as pd
    with pd.option_context("io.sql.engine", "sqlalchemy"):
        from pandas.io import sql
        with sql.pandasSQL_builder(conn) as pandasSQL:
            with pandasSQL.run_transaction():
                assert pandasSQL.to_sql(test_frame1, "test_frame1") == 4
                assert pandasSQL.has_table("test_frame1")
        num_entries: int = len(test_frame1)
        num_rows: int = func_3od90k71(conn, "test_frame1")
        assert num_rows == num_entries


def func_fc7e8r4t(conn: Any, request: Any, test_frame1: DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    import pandas as pd
    with pd.option_context("io.sql.engine", "auto"):
        from pandas.io import sql
        with sql.pandasSQL_builder(conn) as pandasSQL:
            with pandasSQL.run_transaction():
                assert pandasSQL.to_sql(test_frame1, "test_frame1") == 4
                assert pandasSQL.has_table("test_frame1")
        num_entries: int = len(test_frame1)
        num_rows: int = func_3od90k71(conn, "test_frame1")
        assert num_rows == num_entries


def func_yc1wws2h() -> None:
    import pandas as pd
    from pandas.io import sql
    from pandas.io.sql import SQLAlchemyEngine, get_engine
    import pandas as pd
    pd.testing.assert_isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)
    import pandas as pd
    with pd.option_context("io.sql.engine", "sqlalchemy"):
        pd.testing.assert_isinstance(get_engine("auto"), SQLAlchemyEngine)
        pd.testing.assert_isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)
    with pd.option_context("io.sql.engine", "auto"):
        pd.testing.assert_isinstance(get_engine("auto"), SQLAlchemyEngine)
        pd.testing.assert_isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)


def func_u6t8afyy() -> None:
    pass


def func_9qzuxnn7(
    conn: Any,
    request: Any,
    string_storage: Any,
    func: str,
    dtype_backend: Any,
    dtype_backend_data: DataFrame,
    dtype_backend_expected: Any
) -> None:
    conn = request.getfixturevalue(conn)
    table: str = "test"
    df: DataFrame = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")
    import pandas as pd
    with pd.option_context("mode.string_storage", string_storage):
        result: DataFrame = getattr(pd, func)(f"Select * from {table}", conn, dtype_backend=dtype_backend)
        expected: DataFrame = dtype_backend_expected(string_storage, dtype_backend, conn)
    pd.testing.assert_frame_equal(result, expected)
    conn_name: str = str(conn)
    if "adbc" in conn_name:
        request.applymarker(pytest.mark.xfail(reason="adbc does not support chunksize argument"))
    with pd.option_context("mode.string_storage", string_storage):
        iterator = getattr(pd, func)(f"Select * from {table}", con=conn, dtype_backend=dtype_backend, chunksize=3)
        expected = dtype_backend_expected(string_storage, dtype_backend, conn)
        for result_chunk in iterator:
            pd.testing.assert_frame_equal(result_chunk, expected)


def func_ypwrti4e(
    conn: Any,
    request: Any,
    string_storage: Any,
    func: str,
    dtype_backend: Any,
    dtype_backend_data: DataFrame,
    dtype_backend_expected: Any
) -> None:
    conn_name: str = str(conn)
    conn = request.getfixturevalue(conn)
    table: str = "test"
    df: DataFrame = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")
    import pandas as pd
    with pd.option_context("mode.string_storage", string_storage):
        result: DataFrame = getattr(pd, func)(table, conn, dtype_backend=dtype_backend)
        expected: DataFrame = dtype_backend_expected(string_storage, dtype_backend, conn_name)
    pd.testing.assert_frame_equal(result, expected)
    if "adbc" in conn_name:
        return
    with pd.option_context("mode.string_storage", string_storage):
        iterator = getattr(pd, func)(table, conn, dtype_backend=dtype_backend, chunksize=3)
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
        for result_chunk in iterator:
            pd.testing.assert_frame_equal(result_chunk, expected)


def func_09cg71cc(conn: Any, request: Any, func: str, dtype_backend_data: DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    table: str = "test"
    df: DataFrame = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")
    msg: str = ("dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' "
                "are allowed.")
    import pytest  # type: ignore
    with pytest.raises(ValueError, match=msg):
        getattr(pd, func)(table, conn, dtype_backend="numpy")


def func_ypwrti4e_legacy(conn: Any, request: Any) -> None:
    # This function may be a placeholder for legacy functionality.
    pass


def func_brax2qc1(conn: Any, request: Any) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC"))
    conn = request.getfixturevalue(conn)
    dtypes: dict[str, str] = {"a": "int64", "b": "object"}
    df: DataFrame = DataFrame(columns=["a", "b"]).astype(dtypes)
    expected: DataFrame = df.copy()
    from pandas.io import sql
    assert sql.to_sql(df, name="test", con=conn, index=False, if_exists="replace") == 0
    for result in sql.read_sql_query("SELECT * FROM test", conn, dtype=dtypes, chunksize=1):
        pd.testing.assert_frame_equal(result, expected)


def func_7mpqp5mc(
    conn: Any,
    request: Any,
    func: str,
    dtype_backend: Any
) -> None:
    conn = request.getfixturevalue(conn)
    table: str = "test"
    df: DataFrame = DataFrame({"a": [1, 2, 3], "b": 5})
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")
    result: DataFrame = getattr(pd, func)(f"Select * from {table}", conn, dtype={"a": np.float64}, dtype_backend=dtype_backend)
    expected: DataFrame = DataFrame({
        "a": pd.Series([1, 2, 3], dtype=np.float64),
        "b": pd.Series([5, 5, 5], dtype="int64" if dtype_backend != "numpy_nullable" else "Int64")
    })
    pd.testing.assert_frame_equal(result, expected)


def func_j4yy7j5s(sqlite_engine: Any) -> None:
    conn: Any = sqlite_engine
    df: DataFrame = DataFrame({"a": [1, 2]}, dtype="int64")
    assert df.to_sql(name="test_bigintwarning", con=conn, index=False) == 2
    from pandas.io import sql
    with pd.testing.assert_produces_warning(None):
        sql.read_sql_table("test_bigintwarning", conn)


def func_eo2jdnib(sqlite_engine: Any) -> None:
    conn: Any = sqlite_engine
    df: DataFrame = DataFrame({"col1": [1, 2], "col2": [3, 4]})
    import pytest  # type: ignore
    msg: str = "Empty table name specified"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="", con=conn, if_exists="replace", index=False)


def func_48k18m53(sqlite_engine: Any) -> None:
    conn: Any = sqlite_engine
    from sqlalchemy import Column, Integer, String
    from sqlalchemy.orm import declarative_base, sessionmaker
    BaseModel = declarative_base()

    class Test(BaseModel):  # type: ignore
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
        df2: DataFrame = DataFrame(test_query)
    assert list(df2.columns) == ["id", "string_column"]


def func_jiftanv8(sqlite_engine: Any) -> None:
    conn: Any = sqlite_engine
    table: str = "test"
    df: DataFrame = DataFrame({"a": ["x", "y"]})
    df.to_sql(table, con=conn, index=False, if_exists="replace")
    import pandas as pd
    with pd.option_context("future.infer_string", True):
        from pandas.io import sql
        result: DataFrame = sql.read_sql_table(table, conn)
    dtype = pd.StringDtype(na_value=np.nan)
    expected: DataFrame = DataFrame({"a": ["x", "y"]}, dtype=dtype, columns=pd.Index(["a"], dtype=dtype))
    pd.testing.assert_frame_equal(result, expected)


def func_1ionlypy(sqlite_engine: Any) -> None:
    conn: Any = sqlite_engine
    df: DataFrame = DataFrame({"t": [datetime(2020, 12, 31, 12)]}, dtype="datetime64[ns]")
    df.to_sql("test", conn, if_exists="replace", index=False)
    from pandas.io import sql
    result: Any = pd.read_sql("select * from test", conn).iloc[0, 0]
    assert result == "2020-12-31 12:00:00.000000"


def func_65ybbdog(sqlite_builtin_detect_types: Any) -> None:
    conn: Any = sqlite_builtin_detect_types
    df: DataFrame = DataFrame({"t": [datetime(2020, 12, 31, 12)]}, dtype="datetime64[ns]")
    df.to_sql("test", conn, if_exists="replace", index=False)
    result: Any = pd.read_sql("select * from test", conn).iloc[0, 0]
    from pandas import Timestamp
    assert result == Timestamp("2020-12-31 12:00:00.000000")


def func_3lwoby3w(postgresql_psycopg2_engine: Any) -> None:
    conn: Any = postgresql_psycopg2_engine
    df: DataFrame = DataFrame({"col1": [1, 2], "col2": [0.1, 0.2], "col3": ["a", "n"]})
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql("DROP SCHEMA IF EXISTS other CASCADE;")
            con.exec_driver_sql("CREATE SCHEMA other;")
    assert df.to_sql(name="test_schema_public", con=conn, index=False) == 2
    assert df.to_sql(name="test_schema_public_explicit", con=conn, index=False, schema="public") == 2
    assert df.to_sql(name="test_schema_other", con=conn, index=False, schema="other") == 2
    from pandas.io import sql
    res1: DataFrame = sql.read_sql_table("test_schema_public", conn)
    pd.testing.assert_frame_equal(df, res1)
    res2: DataFrame = sql.read_sql_table("test_schema_public_explicit", conn)
    pd.testing.assert_frame_equal(df, res2)
    res3: DataFrame = sql.read_sql_table("test_schema_public_explicit", conn, schema="public")
    pd.testing.assert_frame_equal(df, res3)
    res4: DataFrame = sql.read_sql_table("test_schema_other", conn, schema="other")
    pd.testing.assert_frame_equal(df, res4)
    import pytest  # type: ignore
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
    pd.testing.assert_frame_equal(pd.concat([df, df], ignore_index=True), res)


def func_z2hgho2a(postgresql_psycopg2_engine: Any) -> None:
    conn: Any = postgresql_psycopg2_engine
    from sqlalchemy.sql import text
    create_table: str = text(
        """
    CREATE TABLE person
    (
        id serial constraint person_pkey primary key,
        created_dt timestamp with time zone
    );

    INSERT INTO person
        VALUES (1, '2021-01-01T00:00:00Z');
    """
    )
    with conn.connect() as con:
        with con.begin():
            con.execute(create_table)
    sql_query: str = ('SELECT * FROM "person" AS p1 INNER JOIN "person" AS p2 ON p1.id = p2.id;')
    result: DataFrame = pd.read_sql(sql_query, conn)
    expected: DataFrame = DataFrame([[1, Timestamp("2021", tz="UTC")]*2], columns=["id", "created_dt"]*2)
    expected["created_dt"] = expected["created_dt"].astype("M8[us, UTC]")
    pd.testing.assert_frame_equal(result, expected)
    from pandas.io import sql
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("person")


def func_qaeot7nm(sqlite_buildin: Any) -> None:
    from pandas.io import sql
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                   columns=pd.Index(list("ABCD")),
                                   index=date_range("2000-01-01", periods=10, freq="B"))
    assert sql.to_sql(frame, name="test_table", con=sqlite_buildin, index=False) == 10
    result: DataFrame = sql.read_sql("select * from test_table", sqlite_buildin)
    result.index = frame.index
    pd.testing.assert_frame_equal(result, frame)
    frame["txt"] = ["a"] * len(frame)
    frame2: DataFrame = frame.copy()
    new_idx = pd.Index(np.arange(len(frame2), dtype=np.int64) + 10)
    frame2["Idx"] = new_idx.copy()
    assert sql.to_sql(frame2, name="test_table2", con=sqlite_buildin, index=False) == 10
    result = sql.read_sql("select * from test_table2", sqlite_buildin, index_col="Idx")
    expected: DataFrame = frame.copy()
    expected.index = new_idx
    expected.index.name = "Idx"
    pd.testing.assert_frame_equal(expected, result)
    frame["txt"] = ["a"] * len(frame)
    assert sql.to_sql(frame, name="test_table", con=sqlite_buildin, index=False) == 10


def func_v1opt7qc(sqlite_buildin: Any) -> None:
    from pandas.io import sql
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                   columns=pd.Index(list("ABCD")),
                                   index=date_range("2000-01-01", periods=10, freq="B"))
    frame.iloc[0, 0] = np.nan
    create_sql: str = sql.get_schema(frame, "test")
    cur = func_eds7zk5h.cursor()
    cur.execute(create_sql)
    ins: str = "INSERT INTO test VALUES (%s, %s, %s, %s)"
    for _, row in frame.iterrows():
        fmt_sql: str = func_k2as4dat(ins, *row)
        func_htztezkd(fmt_sql, con=sqlite_buildin)
    func_eds7zk5h.commit()
    result: DataFrame = sql.read_sql("select * from test", con=sqlite_buildin)
    result.index = frame.index
    pd.testing.assert_frame_equal(result, frame, rtol=0.001)


def func_s2uetrpk(sqlite_buildin: Any) -> None:
    from pandas.io import sql
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                   columns=pd.Index(list("ABCD")),
                                   index=date_range("2000-01-01", periods=10, freq="B"))
    create_sql: str = sql.get_schema(frame, "test")
    cur = func_eds7zk5h.cursor()
    cur.execute(create_sql)
    ins: str = "INSERT INTO test VALUES (?, ?, ?, ?)"
    row = frame.iloc[0]
    with sql.pandasSQL_builder(sqlite_buildin) as pandas_sql:
        pandas_sql.execute(ins, tuple(row))
    func_eds7zk5h.commit()
    result: DataFrame = sql.read_sql("select * from test", sqlite_buildin)
    result.index = frame.index[:1]
    pd.testing.assert_frame_equal(result, frame[:1])


def func_zc9dcp9o(sqlite_buildin: Any) -> None:
    from pandas.io import sql
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                   columns=pd.Index(list("ABCD")),
                                   index=date_range("2000-01-01", periods=10, freq="B"))
    create_sql: str = sql.get_schema(frame, "test")
    lines = create_sql.splitlines()
    for line in lines:
        tokens = line.split(" ")
        if len(tokens) == 2 and tokens[0] == "A":
            assert tokens[1] == "DATETIME"
    create_sql = sql.get_schema(frame, "test", keys=["A", "B"])
    assert 'PRIMARY KEY ("A", "B")' in create_sql
    cur = func_eds7zk5h.cursor()
    cur.execute(create_sql)


def func_u1ljn9kp(sqlite_buildin: Any) -> None:
    create_sql: str = """
    CREATE TABLE test
    (
    a TEXT,
    b TEXT,
    c REAL,
    PRIMARY KEY (a, b)
    );
    """
    cur = func_eds7zk5h.cursor()
    cur.execute(create_sql)
    from pandas.io import sql
    with sql.pandasSQL_builder(sqlite_buildin) as pandas_sql:
        pandas_sql.execute("INSERT INTO test VALUES('foo', 'bar', 1.234)")
        pandas_sql.execute("INSERT INTO test VALUES('foo', 'baz', 2.567)")
        import pytest  # type: ignore
        with pytest.raises(sql.DatabaseError, match="Execution failed on sql"):
            pandas_sql.execute("INSERT INTO test VALUES('foo', 'bar', 7)")
            
            
def func_x79qa8ee() -> None:
    create_sql: str = """
    CREATE TABLE test
    (
    a TEXT,
    b TEXT,
    c REAL,
    PRIMARY KEY (a, b)
    );
    """
    import contextlib
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        cur = conn.cursor()
        cur.execute(create_sql)
        from pandas.io import sql
        with sql.pandasSQL_builder(conn) as pandas_sql:
            pandas_sql.execute("INSERT INTO test VALUES('foo', 'bar', 1.234)")
    msg: str = "Cannot operate on a closed database."
    import pytest  # type: ignore
    with pytest.raises(sqlite3.ProgrammingError, match=msg):
        func_htztezkd("select * from test", con=conn)


def func_456jiuhc(sqlite_buildin: Any) -> None:
    df: DataFrame = DataFrame({"From": np.ones(5)})
    from pandas.io import sql
    assert sql.to_sql(df, con=sqlite_buildin, name="testkeywords", index=False) == 5


def func_n4f6wt25(sqlite_buildin: Any) -> None:
    mono_df: DataFrame = DataFrame([1, 2], columns=["c0"])
    from pandas.io import sql
    assert sql.to_sql(mono_df, con=sqlite_buildin, name="mono_df", index=False) == 2
    con_x: Any = sqlite_buildin
    the_sum: int = sum(my_c0[0] for my_c0 in con_x.execute("select * from mono_df"))
    assert the_sum == 3
    result: DataFrame = sql.read_sql("select * from mono_df", con_x)
    pd.testing.assert_frame_equal(result, mono_df)


def func_nr14b67i(sqlite_buildin: Any) -> None:
    df_if_exists_1: DataFrame = DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
    df_if_exists_2: DataFrame = DataFrame({"col1": [3, 4, 5], "col2": ["C", "D", "E"]})
    table_name: str = "table_if_exists"
    sql_select: str = f"SELECT * FROM {table_name}"
    import pytest  # type: ignore
    msg: str = "'notvalidvalue' is not valid for if_exists"
    with pytest.raises(ValueError, match=msg):
        df_if_exists_1.to_sql(name=table_name, con=sqlite_buildin, if_exists="notvalidvalue")
    func_whjlxt40(table_name, sqlite_buildin)
    df_if_exists_1.to_sql(name=table_name, con=sqlite_buildin, if_exists="fail")
    msg = "Table 'table_if_exists' already exists"
    with pytest.raises(ValueError, match=msg):
        df_if_exists_1.to_sql(name=table_name, con=sqlite_buildin, if_exists="fail")
    df_if_exists_1.to_sql(name=table_name, con=sqlite_buildin, if_exists="replace", index=False)
    assert func_htztezkd(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert df_if_exists_2.to_sql(name=table_name, con=sqlite_buildin, if_exists="replace", index=False) == 3
    assert func_htztezkd(sql_select, con=sqlite_buildin) == [(3, "C"), (4, "D"), (5, "E")]
    func_whjlxt40(table_name, sqlite_buildin)
    assert df_if_exists_1.to_sql(name=table_name, con=sqlite_buildin, if_exists="fail", index=False) == 2
    assert func_htztezkd(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert df_if_exists_2.to_sql(name=table_name, con=sqlite_buildin, if_exists="append", index=False) == 3
    assert func_htztezkd(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")]
    func_whjlxt40(table_name, sqlite_buildin)


def func_be6mv67g(sqlite_buildin: Any) -> None:
    conn: Any = sqlite_buildin
    df: DataFrame = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=["a"])
    assert df.to_sql(name="test_date", con=conn, index=False) == 2
    from pandas.io import sql
    res: DataFrame = sql.read_sql("SELECT * FROM test_date", conn)
    pd.testing.assert_frame_equal(res, df.astype(str))


def func_28q7xrw0(tz_aware: bool, sqlite_buildin: Any) -> None:
    conn: Any = sqlite_buildin
    if not tz_aware:
        tz_times = [time(9, 0, 0), time(9, 1, 30)]
    else:
        tz_dt = date_range("2013-01-01 09:00:00", periods=2, tz="US/Pacific")
        tz_times = pd.Series(tz_dt.to_pydatetime()).map(lambda dt: dt.timetz())
    df: DataFrame = DataFrame(tz_times, columns=["a"])
    assert df.to_sql(name="test_time", con=conn, index=False) == 2
    from pandas.io import sql
    res: DataFrame = sql.read_sql("SELECT * FROM test_time", conn)
    expected: DataFrame = df.applymap(lambda _: _.strftime("%H:%M:%S.%f") if _ is not None else None)
    pd.testing.assert_frame_equal(res, expected)


def func_n84fswsv(conn: Any, table: str, column: str) -> str:
    recs = conn.execute(f"PRAGMA table_info({table})")
    for cid, name, ctype, not_null, default, pk in recs:
        if name == column:
            return ctype
    raise ValueError(f"Table {table}, column {column} not found")


def func_j4mz4e6s(sqlite_buildin: Any) -> None:
    conn: Any = sqlite_buildin
    cols: list[str] = ["A", "B"]
    data = [(0.8, True), (0.9, None)]
    df: DataFrame = DataFrame(data, columns=cols)
    assert df.to_sql(name="dtype_test", con=conn) == 2
    assert df.to_sql(name="dtype_test2", con=conn, dtype={"B": "STRING"}) == 2
    assert func_n84fswsv(conn, "dtype_test", "B") == "INTEGER"
    assert func_n84fswsv(conn, "dtype_test2", "B") == "STRING"
    import pytest  # type: ignore
    msg: str = "B \\(<class 'bool'>\\) not a string"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="error", con=conn, dtype={"B": bool})
    assert df.to_sql(name="single_dtype_test", con=conn, dtype="STRING") == 2
    assert func_n84fswsv(conn, "single_dtype_test", "A") == "STRING"
    assert func_n84fswsv(conn, "single_dtype_test", "B") == "STRING"


def func_cw7dhqth(sqlite_buildin: Any) -> None:
    conn: Any = sqlite_buildin
    cols: dict[str, Series] = {
        "Bool": pd.Series([True, None]),
        "Date": pd.Series([datetime(2012, 5, 1), None]),
        "Int": pd.Series([1, None], dtype="object"),
        "Float": pd.Series([1.1, None])
    }
    df: DataFrame = DataFrame(cols)
    tbl: str = "notna_dtype_test"
    assert df.to_sql(name=tbl, con=conn) == 2
    assert func_n84fswsv(conn, tbl, "Bool") == "INTEGER"
    assert func_n84fswsv(conn, tbl, "Date") == "TIMESTAMP"
    assert func_n84fswsv(conn, tbl, "Int") == "INTEGER"
    assert func_n84fswsv(conn, tbl, "Float") == "REAL"


def func_d5e8t19y(sqlite_buildin: Any) -> None:
    conn: Any = sqlite_buildin
    df: DataFrame = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    import pytest  # type: ignore
    msg: str = "Empty table or column name specified"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="", con=conn)
    for ndx, weird_name in enumerate(["test_weird_name]",
                                        "test_weird_name[",
                                        "test_weird_name`",
                                        "test_weird_name\"",
                                        "test_weird_name'",
                                        "_b.test_weird_name_01-30",
                                        "\"_b.test_weird_name_01-30\"",
                                        "99beginswithnumber", "12345", "é"]):
        assert df.to_sql(name=weird_name, con=conn) == 2
        sql.table_exists(weird_name, conn)
        df2: DataFrame = DataFrame([[1, 2], [3, 4]], columns=["a", weird_name])
        c_tbl: str = f"test_weird_col_name{ndx:d}"
        assert df2.to_sql(name=c_tbl, con=conn) == 2
        sql.table_exists(c_tbl, conn)


def func_k2as4dat(sql: str, *args: Any) -> str:
    _formatters: dict[Any, Any] = {
        datetime: lambda x: "'{}'".format(x),
        str: lambda x: "'{}'".format(x),
        np.str_: lambda x: "'{}'".format(x),
        bytes: lambda x: "'{}'".format(x),
        float: lambda x: "{:.8f}".format(x),
        int: lambda x: "{:d}".format(x),
        type(None): lambda x: "NULL",
        np.float64: lambda x: "{:.10f}".format(x),
        bool: lambda x: "'{!s}'".format(x)
    }
    processed_args: list[str] = []
    for arg in args:
        if isinstance(arg, float) and pd.isna(arg):
            arg = None
        formatter = _formatters[type(arg)]
        processed_args.append(formatter(arg))
    return sql % tuple(processed_args)


def func_htztezkd(query: str, con: Optional[Any] = None) -> Optional[list]:
    from pandas.io import sql
    with sql.pandasSQL_builder(con) as pandas_sql:
        res = pandas_sql.execute(query).fetchall()
    return None if res is None else list(res)


def func_c5r911t1(sqlite_buildin: Any) -> None:
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                   columns=pd.Index(list("ABCD")),
                                   index=date_range("2000-01-01", periods=10, freq="B"))
    assert sql.to_sql(frame, name="test_table", con=sqlite_buildin, index=False) == 10
    result: DataFrame = sql.read_sql("select * from test_table", sqlite_buildin)
    result.index = frame.index
    pd.testing.assert_frame_equal(result, frame)
    frame["txt"] = ["a"] * len(frame)
    frame2: DataFrame = frame.copy()
    new_idx = pd.Index(np.arange(len(frame2), dtype=np.int64) + 10)
    frame2["Idx"] = new_idx.copy()
    assert sql.to_sql(frame2, name="test_table2", con=sqlite_buildin, index=False) == 10
    result = sql.read_sql("select * from test_table2", sqlite_buildin, index_col="Idx")
    expected: DataFrame = frame.copy()
    expected.index = new_idx
    expected.index.name = "Idx"
    pd.testing.assert_frame_equal(expected, result)
    assert sql.to_sql(frame, name="test_table", con=sqlite_buildin, index=False) == 10


def func_v1opt7qc_legacy(sqlite_buildin: Any) -> None:
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                   columns=pd.Index(list("ABCD")),
                                   index=date_range("2000-01-01", periods=10, freq="B"))
    frame.iloc[0, 0] = np.nan
    create_sql: str = sql.get_schema(frame, "test")
    cur = func_eds7zk5h.cursor()
    cur.execute(create_sql)
    ins: str = "INSERT INTO test VALUES (%s, %s, %s, %s)"
    for _, row in frame.iterrows():
        fmt_sql: str = func_k2as4dat(ins, *row)
        func_htztezkd(fmt_sql, con=sqlite_buildin)
    func_eds7zk5h.commit()
    result: DataFrame = sql.read_sql("select * from test", con=sqlite_buildin)
    result.index = frame.index
    pd.testing.assert_frame_equal(result, frame, rtol=0.001)


def func_s2uetrpk_legacy(sqlite_buildin: Any) -> None:
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                   columns=pd.Index(list("ABCD")),
                                   index=date_range("2000-01-01", periods=10, freq="B"))
    create_sql: str = sql.get_schema(frame, "test")
    cur = func_eds7zk5h.cursor()
    cur.execute(create_sql)
    ins: str = "INSERT INTO test VALUES (?, ?, ?, ?)"
    row = frame.iloc[0]
    with sql.pandasSQL_builder(sqlite_buildin) as pandas_sql:
        pandas_sql.execute(ins, tuple(row))
    func_eds7zk5h.commit()
    result: DataFrame = sql.read_sql("select * from test", sqlite_buildin)
    result.index = frame.index[:1]
    pd.testing.assert_frame_equal(result, frame[:1])


def func_zc9dcp9o_legacy(sqlite_buildin: Any) -> None:
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                   columns=pd.Index(list("ABCD")),
                                   index=date_range("2000-01-01", periods=10, freq="B"))
    create_sql: str = sql.get_schema(frame, "test")
    lines = create_sql.splitlines()
    for line in lines:
        tokens = line.split(" ")
        if len(tokens) == 2 and tokens[0] == "A":
            assert tokens[1] == "DATETIME"
    create_sql = sql.get_schema(frame, "test", keys=["A", "B"])
    lines = create_sql.splitlines()
    assert 'PRIMARY KEY ("A", "B")' in create_sql
    cur = func_eds7zk5h.cursor()
    cur.execute(create_sql)


def func_u1ljn9kp_legacy(sqlite_buildin: Any) -> None:
    create_sql: str = """
    CREATE TABLE test
    (
    a TEXT,
    b TEXT,
    c REAL,
    PRIMARY KEY (a, b)
    );
    """
    cur = func_eds7zk5h.cursor()
    cur.execute(create_sql)
    from pandas.io import sql
    with sql.pandasSQL_builder(sqlite_buildin) as pandas_sql:
        pandas_sql.execute("INSERT INTO test VALUES('foo', 'bar', 1.234)")
        pandas_sql.execute("INSERT INTO test VALUES('foo', 'baz', 2.567)")
        import pytest  # type: ignore
        with pytest.raises(sql.DatabaseError, match="Execution failed on sql"):
            pandas_sql.execute("INSERT INTO test VALUES('foo', 'bar', 7)")
            
            
def func_x79qa8ee_legacy() -> None:
    create_sql: str = """
    CREATE TABLE test
    (
    a TEXT,
    b TEXT,
    c REAL,
    PRIMARY KEY (a, b)
    );
    """
    import contextlib
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        cur = conn.cursor()
        cur.execute(create_sql)
        from pandas.io import sql
        with sql.pandasSQL_builder(conn) as pandas_sql:
            pandas_sql.execute("INSERT INTO test VALUES('foo', 'bar', 1.234)")
    msg: str = "Cannot operate on a closed database."
    import pytest  # type: ignore
    with pytest.raises(sqlite3.ProgrammingError, match=msg):
        func_htztezkd("select * from test", con=conn)


def func_456jiuhc_legacy(sqlite_buildin: Any) -> None:
    df: DataFrame = DataFrame({"From": np.ones(5)})
    from pandas.io import sql
    assert df.to_sql(df, con=sqlite_buildin, name="testkeywords", index=False) == 5


def func_n4f6wt25_legacy(sqlite_buildin: Any) -> None:
    mono_df: DataFrame = DataFrame([1, 2], columns=["c0"])
    from pandas.io import sql
    assert mono_df.to_sql(name="mono_df", con=sqlite_buildin, index=False) == 2
    con_x: Any = sqlite_buildin
    the_sum: int = sum(my_c0[0] for my_c0 in con_x.execute("select * from mono_df"))
    assert the_sum == 3
    result: DataFrame = sql.read_sql("select * from mono_df", con_x)
    pd.testing.assert_frame_equal(result, mono_df)


def func_nr14b67i_legacy(sqlite_buildin: Any) -> None:
    df_if_exists_1: DataFrame = DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
    df_if_exists_2: DataFrame = DataFrame({"col1": [3, 4, 5], "col2": ["C", "D", "E"]})
    table_name: str = "table_if_exists"
    sql_select: str = f"SELECT * FROM {table_name}"
    import pytest  # type: ignore
    msg: str = "'notvalidvalue' is not valid for if_exists"
    with pytest.raises(ValueError, match=msg):
        df_if_exists_1.to_sql(name=table_name, con=sqlite_buildin, if_exists="notvalidvalue")
    func_whjlxt40(table_name, sqlite_buildin)
    df_if_exists_1.to_sql(name=table_name, con=sqlite_buildin, if_exists="fail")
    msg = "Table 'table_if_exists' already exists"
    with pytest.raises(ValueError, match=msg):
        df_if_exists_1.to_sql(name=table_name, con=sqlite_buildin, if_exists="fail")
    df_if_exists_1.to_sql(name=table_name, con=sqlite_buildin, if_exists="replace", index=False)
    assert func_htztezkd(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert df_if_exists_2.to_sql(name=table_name, con=sqlite_buildin, if_exists="replace", index=False) == 3
    assert func_htztezkd(sql_select, con=sqlite_buildin) == [(3, "C"), (4, "D"), (5, "E")]
    func_whjlxt40(table_name, sqlite_buildin)
    assert df_if_exists_1.to_sql(name=table_name, con=sqlite_buildin, if_exists="fail", index=False) == 2
    assert func_htztezkd(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert df_if_exists_2.to_sql(name=table_name, con=sqlite_buildin, if_exists="append", index=False) == 3
    assert func_htztezkd(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")]
    func_whjlxt40(table_name, sqlite_buildin)


def func_be6mv67g_legacy(sqlite_buildin: Any) -> None:
    conn: Any = sqlite_buildin
    df: DataFrame = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=["a"])
    assert df.to_sql(name="test_date", con=conn, index=False) == 2
    from pandas.io import sql
    res: DataFrame = sql.read_sql("SELECT * FROM test_date", conn)
    pd.testing.assert_frame_equal(res, df.astype(str))


def func_28q7xrw0_legacy(tz_aware: bool, sqlite_buildin: Any) -> None:
    conn: Any = sqlite_buildin
    if not tz_aware:
        tz_times = [time(9, 0, 0), time(9, 1, 30)]
    else:
        tz_dt = date_range("2013-01-01 09:00:00", periods=2, tz="US/Pacific")
        tz_times = pd.Series(tz_dt.to_pydatetime()).map(lambda dt: dt.timetz())
    df: DataFrame = DataFrame(tz_times, columns=["a"])
    assert df.to_sql(name="test_time", con=conn, index=False) == 2
    from pandas.io import sql
    res: DataFrame = sql.read_sql("SELECT * FROM test_time", conn)
    expected: DataFrame = df.applymap(lambda _: _.strftime("%H:%M:%S.%f") if _ is not None else None)
    pd.testing.assert_frame_equal(res, expected)